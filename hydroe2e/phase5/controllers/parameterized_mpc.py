"""
参数化本地MPC控制器 (Parameterized Local MPC)

基于 phase2/controllers/improved_admm.py 改造
关键改进:
1. 物理参数化: area, flow_efficiency 可在线调整
2. 动态轨迹跟踪: 支持L3下发的参考轨迹
3. 场景自适应: 根据场景调整物理模型

数学模型:
Z_{k+1} = Z_k + (eta * Q_in - Q_out) * dt / A
其中:
- eta: 流速效率系数 (结冰时降低到0.7-0.8)
- A: 水面面积参数 (可因水位变化而改变)
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicalParameters:
    """物理参数配置"""
    area: float = 10000.0           # 水面面积 [m²]
    flow_efficiency: float = 1.0    # 流速效率 (结冰/淤积时<1)
    manning_n: float = 0.025        # 曼宁系数
    channel_slope: float = 0.0001   # 渠道坡度
    hydraulic_radius: float = 2.0   # 水力半径 [m]

    def get_effective_flow_capacity(self) -> float:
        """计算有效过流能力 (曼宁公式)"""
        # Q = (1/n) * A * R^(2/3) * S^(1/2)
        # 简化为效率系数
        return self.flow_efficiency


@dataclass
class ScenarioPhysics:
    """场景物理参数映射"""

    @staticmethod
    def get_parameters(scenario_id: str) -> Dict[str, float]:
        """根据场景ID获取物理参数调整"""
        mappings = {
            # 结冰场景
            'ICE_FORMATION': {
                'flow_efficiency': 0.75,
                'manning_n': 0.035  # 冰盖增加糙率
            },
            'ICE_STABLE': {
                'flow_efficiency': 0.8,
                'manning_n': 0.032
            },
            'ICE_BREAKUP': {
                'flow_efficiency': 0.7,  # 冰塞风险
                'manning_n': 0.04
            },

            # 淤积场景
            'SEDIMENT_HIGH': {
                'flow_efficiency': 0.85,
                'manning_n': 0.03
            },

            # 水草生长
            'VEGETATION_GROWTH': {
                'flow_efficiency': 0.9,
                'manning_n': 0.028
            },

            # 正常
            'NORMAL': {
                'flow_efficiency': 1.0,
                'manning_n': 0.025
            }
        }
        return mappings.get(scenario_id, mappings['NORMAL'])


class ParameterizedLocalMPC:
    """
    参数化本地MPC控制器

    关键特性:
    1. 物理模型参数可在线调整 (area, flow_efficiency)
    2. 支持动态参考轨迹跟踪 (从L3接收)
    3. CVXPY问题缓存加速求解
    """

    def __init__(self,
                 pool_id: int,
                 horizon: int = 5,
                 dt: float = 3600.0,
                 physical_params: PhysicalParameters = None):
        """
        初始化参数化本地MPC

        Args:
            pool_id: 渠池ID
            horizon: 预测时域
            dt: 时间步长 [s]
            physical_params: 物理参数
        """
        self.pool_id = pool_id
        self.N = horizon
        self.dt = dt

        # 物理参数 (可动态调整)
        self.physical_params = physical_params or PhysicalParameters()

        # 控制权重 (可动态调整)
        self.W_level = 10.0
        self.W_smooth = 5.0
        self.W_coupling = 1.0
        self.W_trajectory = 15.0  # 新增: 轨迹跟踪权重

        # 约束 (可动态调整)
        self.Z_ref = 3.0  # 固定参考 (向后兼容)
        self.Z_min = 0.5
        self.Z_max = 8.0
        self.Q_min = 0.0
        self.Q_max = 20.0
        self.delta_Q_max = 2.0

        # ADMM变量
        self.z = np.zeros(horizon)
        self.u = np.zeros(horizon)

        # 参考轨迹 (从L3接收)
        self._reference_trajectory: Optional[np.ndarray] = None

        # 构建参数化问题
        self._build_parameterized_problem()

    def _build_parameterized_problem(self):
        """构建参数化优化问题"""
        N = self.N

        # CVXPY决策变量
        self.Q_in_var = cp.Variable(N)
        self.Q_out_var = cp.Variable(N)
        self.Z_var = cp.Variable(N)

        # 标量参数
        self.Z_curr_param = cp.Parameter(name='Z_curr')
        self.Q_in_prev_param = cp.Parameter(name='Q_in_prev')
        self.rho_param = cp.Parameter(nonneg=True, name='rho')

        # 关键改进: 物理参数作为CVXPY参数
        self.area_param = cp.Parameter(pos=True, name='area')
        self.flow_efficiency_param = cp.Parameter(pos=True, name='flow_efficiency')

        # 向量参数 (ADMM)
        self.z_param = cp.Parameter(N, name='z_consensus')
        self.u_param = cp.Parameter(N, name='u_dual')

        # 关键改进: 动态参考轨迹参数
        self.Z_ref_trajectory_param = cp.Parameter(N, name='Z_ref_trajectory')

        # 使用轨迹跟踪的开关
        self.use_trajectory_param = cp.Parameter(name='use_trajectory')

        # 初始化参数默认值
        self.area_param.value = self.physical_params.area
        self.flow_efficiency_param.value = self.physical_params.flow_efficiency
        self.Z_ref_trajectory_param.value = np.ones(N) * self.Z_ref
        self.use_trajectory_param.value = 0.0  # 默认不使用轨迹

        # 构建约束
        self.constraints = []
        self._build_constraints()

        # 构建目标函数
        self._build_objective()

    def _build_constraints(self):
        """构建约束 (物理模型参数化)"""
        N = self.N

        for k in range(N):
            # 水力学约束 (参数化)
            # Z_{k+1} = Z_k + (eta * Q_in - Q_out) * dt / A
            if k == 0:
                self.constraints.append(
                    self.Z_var[k] == self.Z_curr_param +
                    (self.flow_efficiency_param * self.Q_in_prev_param - self.Q_out_var[k]) *
                    self.dt / self.area_param
                )
            else:
                self.constraints.append(
                    self.Z_var[k] == self.Z_var[k-1] +
                    (self.flow_efficiency_param * self.Q_in_var[k-1] - self.Q_out_var[k]) *
                    self.dt / self.area_param
                )

            # 状态约束
            self.constraints.extend([
                self.Z_var[k] >= self.Z_min,
                self.Z_var[k] <= self.Z_max,
                self.Q_in_var[k] >= self.Q_min,
                self.Q_in_var[k] <= self.Q_max,
                self.Q_out_var[k] >= self.Q_min,
                self.Q_out_var[k] <= self.Q_max
            ])

            # 流量变化率约束
            if k == 0:
                self.constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_prev_param) <= self.delta_Q_max
                )
            else:
                self.constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_var[k-1]) <= self.delta_Q_max
                )

    def _build_objective(self):
        """构建目标函数 (支持动态轨迹跟踪)"""
        cost = 0

        # 水位跟踪目标
        for k in range(self.N):
            # 混合跟踪: 固定参考 + 动态轨迹
            # 当 use_trajectory = 1 时跟踪轨迹, = 0 时跟踪固定参考
            trajectory_cost = self.W_trajectory * cp.square(
                self.Z_var[k] - self.Z_ref_trajectory_param[k]
            )
            fixed_cost = self.W_level * cp.square(self.Z_var[k] - self.Z_ref)

            # 简化: 直接使用轨迹 (当轨迹设置时)
            cost += trajectory_cost

        # 流量平滑
        for k in range(self.N):
            if k == 0:
                cost += self.W_smooth * cp.square(
                    self.Q_in_var[k] - self.Q_in_prev_param
                )
            else:
                cost += self.W_smooth * cp.square(
                    self.Q_in_var[k] - self.Q_in_var[k-1]
                )

            if k > 0:
                cost += self.W_smooth * cp.square(
                    self.Q_out_var[k] - self.Q_out_var[k-1]
                )

        # ADMM增广拉格朗日项
        for k in range(self.N):
            cost += self.u_param[k] * (self.Q_out_var[k] - self.z_param[k])
            cost += (self.rho_param / 2) * cp.square(
                self.Q_out_var[k] - self.z_param[k]
            )

        self.objective = cp.Minimize(cost)
        self.problem = cp.Problem(self.objective, self.constraints)

    def set_physical_parameters(self,
                                area: float = None,
                                flow_efficiency: float = None,
                                manning_n: float = None):
        """
        在线更新物理参数

        Args:
            area: 水面面积 [m²]
            flow_efficiency: 流速效率系数 [0-1]
            manning_n: 曼宁糙率系数
        """
        if area is not None:
            self.physical_params.area = area
            self.area_param.value = area

        if flow_efficiency is not None:
            self.physical_params.flow_efficiency = flow_efficiency
            self.flow_efficiency_param.value = flow_efficiency

        if manning_n is not None:
            self.physical_params.manning_n = manning_n

    def set_scenario_physics(self, scenario_id: str):
        """
        根据场景ID自动设置物理参数

        Args:
            scenario_id: 场景标识 (如 'ICE_FORMATION', 'NORMAL')
        """
        params = ScenarioPhysics.get_parameters(scenario_id)
        self.set_physical_parameters(
            flow_efficiency=params.get('flow_efficiency'),
            manning_n=params.get('manning_n')
        )

    def set_reference_trajectory(self, trajectory: np.ndarray):
        """
        设置参考轨迹 (从L3接收)

        Args:
            trajectory: 参考水位轨迹 [N]
        """
        if len(trajectory) >= self.N:
            self._reference_trajectory = trajectory[:self.N]
        else:
            # 补齐
            self._reference_trajectory = np.concatenate([
                trajectory,
                np.ones(self.N - len(trajectory)) * trajectory[-1]
            ])

        self.Z_ref_trajectory_param.value = self._reference_trajectory
        self.use_trajectory_param.value = 1.0

    def clear_reference_trajectory(self):
        """清除参考轨迹 (回退到固定参考)"""
        self._reference_trajectory = None
        self.Z_ref_trajectory_param.value = np.ones(self.N) * self.Z_ref
        self.use_trajectory_param.value = 0.0

    def solve(self,
              current_level: float,
              q_in_prev: float,
              z: np.ndarray,
              u: np.ndarray,
              rho: float) -> Tuple[float, float, bool]:
        """
        求解本地MPC (带ADMM)

        Args:
            current_level: 当前水位
            q_in_prev: 上一步入流
            z: 一致性变量
            u: 对偶变量
            rho: 惩罚参数

        Returns:
            (最优入流, 最优出流, 是否成功)
        """
        # 更新参数
        self.Z_curr_param.value = current_level
        self.Q_in_prev_param.value = q_in_prev
        self.rho_param.value = rho
        self.z_param.value = z
        self.u_param.value = u

        # 确保物理参数已设置
        if self.area_param.value is None:
            self.area_param.value = self.physical_params.area
        if self.flow_efficiency_param.value is None:
            self.flow_efficiency_param.value = self.physical_params.flow_efficiency

        # 求解
        try:
            self.problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

            if self.problem.status == "optimal":
                return (
                    float(self.Q_in_var.value[0]),
                    float(self.Q_out_var.value[0]),
                    True
                )
            else:
                return q_in_prev, 5.0, False
        except Exception as e:
            logger.info(f"池{self.pool_id}求解失败: {e}")
            return q_in_prev, 5.0, False

    def get_output_sequence(self) -> np.ndarray:
        """获取完整出流序列"""
        if self.Q_out_var.value is not None:
            return np.array(self.Q_out_var.value)
        return np.zeros(self.N)

    def get_level_prediction(self) -> np.ndarray:
        """获取水位预测序列"""
        if self.Z_var.value is not None:
            return np.array(self.Z_var.value)
        return np.ones(self.N) * self.Z_ref

    def get_physical_state(self) -> Dict:
        """获取当前物理参数状态"""
        return {
            'area': self.physical_params.area,
            'flow_efficiency': self.physical_params.flow_efficiency,
            'manning_n': self.physical_params.manning_n,
            'has_trajectory': self._reference_trajectory is not None
        }


class ParameterizedDistributedMPC:
    """
    参数化分布式MPC (快速ADMM)

    改进:
    1. 使用ParameterizedLocalMPC
    2. 支持场景驱动的物理参数注入
    3. 支持L3参考轨迹跟踪
    """

    def __init__(self,
                 num_pools: int,
                 horizon: int = 5,
                 dt: float = 3600.0,
                 physical_params_list: List[PhysicalParameters] = None):
        """
        初始化参数化分布式MPC

        Args:
            num_pools: 渠池数量
            horizon: 预测时域
            dt: 时间步长
            physical_params_list: 各池物理参数列表
        """
        self.num_pools = num_pools
        self.horizon = horizon
        self.dt = dt

        # ADMM参数
        self.rho = 1.0
        self.alpha = 1.6  # 过松弛
        self.max_iterations = 20
        self.tolerance = 1e-3

        # 物理参数
        if physical_params_list is None:
            physical_params_list = [PhysicalParameters() for _ in range(num_pools)]

        # 创建参数化本地控制器
        self.local_controllers = [
            ParameterizedLocalMPC(
                pool_id=i,
                horizon=horizon,
                dt=dt,
                physical_params=physical_params_list[i]
            )
            for i in range(num_pools)
        ]

        # ADMM一致性变量
        self.z = [np.ones(horizon) * 5.0 for _ in range(num_pools - 1)]
        self.u = [np.zeros(horizon) for _ in range(num_pools - 1)]

        # 收敛历史
        self.convergence_history = []

    def set_reference_trajectories(self, trajectories: np.ndarray):
        """
        设置所有池的参考轨迹 (从L3接收)

        Args:
            trajectories: [num_pools x horizon] 参考水位轨迹
        """
        for i, controller in enumerate(self.local_controllers):
            if i < trajectories.shape[0]:
                controller.set_reference_trajectory(trajectories[i, :])

    def set_scenario_physics(self, scenario_id: str, pool_ids: List[int] = None):
        """
        根据场景设置物理参数

        Args:
            scenario_id: 场景标识
            pool_ids: 受影响的池ID列表 (默认全部)
        """
        if pool_ids is None:
            pool_ids = list(range(self.num_pools))

        for i in pool_ids:
            self.local_controllers[i].set_scenario_physics(scenario_id)

    def set_flow_efficiency(self, pool_id: int, efficiency: float):
        """
        直接设置指定池的流速效率

        Args:
            pool_id: 池ID
            efficiency: 流速效率 [0-1]
        """
        if 0 <= pool_id < self.num_pools:
            self.local_controllers[pool_id].set_physical_parameters(
                flow_efficiency=efficiency
            )

    def solve(self,
              current_levels: List[float],
              q_in_prevs: List[float],
              q_out_forecasts: List[List[float]] = None) -> Tuple[List[Tuple], Dict]:
        """
        求解分布式MPC

        Args:
            current_levels: 各池当前水位
            q_in_prevs: 各池上一步入流
            q_out_forecasts: 各池出流预测 (可选)

        Returns:
            (各池最优控制, 求解信息)
        """
        start_time = time.time()

        # 初始化
        q_in_solutions = [5.0] * self.num_pools
        q_out_solutions = [5.0] * self.num_pools
        q_out_sequences = [np.ones(self.horizon) * 5.0 for _ in range(self.num_pools)]

        # ADMM迭代
        for iteration in range(self.max_iterations):
            q_out_old = [seq.copy() for seq in q_out_sequences]

            # 1. x-update: 并行求解各池
            for i in range(self.num_pools):
                z_i = self.z[i] if i < self.num_pools - 1 else np.ones(self.horizon) * 5.0
                u_i = self.u[i] if i < self.num_pools - 1 else np.zeros(self.horizon)

                q_in, q_out, success = self.local_controllers[i].solve(
                    current_level=current_levels[i],
                    q_in_prev=q_in_prevs[i],
                    z=z_i,
                    u=u_i,
                    rho=self.rho
                )

                q_in_solutions[i] = q_in
                q_out_solutions[i] = q_out
                q_out_sequences[i] = self.local_controllers[i].get_output_sequence()

            # 2. z-update: 过松弛
            for i in range(self.num_pools - 1):
                x_hat = self.alpha * q_out_sequences[i] + (1 - self.alpha) * self.z[i]
                self.z[i] = 0.5 * (x_hat + q_out_sequences[i])

            # 3. u-update
            for i in range(self.num_pools - 1):
                residual = q_out_sequences[i] - self.z[i]
                self.u[i] += self.rho * residual

            # 4. 收敛检查
            primal_res = self._compute_primal_residual(q_out_sequences)
            dual_res = self._compute_dual_residual(q_out_sequences, q_out_old)

            self.convergence_history.append({
                'iteration': iteration + 1,
                'primal_residual': primal_res,
                'dual_residual': dual_res,
                'rho': self.rho
            })

            if primal_res < self.tolerance and dual_res < self.tolerance:
                break

            # 5. 自适应rho
            self._adaptive_rho(primal_res, dual_res)

        solve_time = time.time() - start_time

        info = {
            'converged': primal_res < self.tolerance and dual_res < self.tolerance,
            'iterations': iteration + 1,
            'solve_time': solve_time,
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'final_rho': self.rho,
            'physical_states': [c.get_physical_state() for c in self.local_controllers]
        }

        return list(zip(q_in_solutions, q_out_solutions)), info

    def _compute_primal_residual(self, q_out_sequences):
        """计算原始残差"""
        residual = 0.0
        for i in range(self.num_pools - 1):
            r = q_out_sequences[i] - self.z[i]
            residual += np.linalg.norm(r)
        return residual

    def _compute_dual_residual(self, q_out_new, q_out_old):
        """计算对偶残差"""
        residual = 0.0
        for i in range(self.num_pools - 1):
            s = self.rho * (q_out_new[i] - q_out_old[i])
            residual += np.linalg.norm(s)
        return residual

    def _adaptive_rho(self, primal_res, dual_res):
        """自适应更新rho"""
        mu = 10.0
        if primal_res > mu * dual_res:
            self.rho *= 2.0
            for i in range(self.num_pools - 1):
                self.u[i] /= 2.0
        elif dual_res > mu * primal_res:
            self.rho /= 2.0
            for i in range(self.num_pools - 1):
                self.u[i] *= 2.0
        self.rho = np.clip(self.rho, 0.01, 100.0)


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*15 + "参数化MPC演示")
    logger.info("="*70)

    # 创建参数化分布式MPC
    controller = ParameterizedDistributedMPC(num_pools=3, horizon=5)

    logger.info(f"\n✓ 参数化MPC已初始化")

    # 测试1: 正常模式
    logger.info(f"\n{'='*70}")
    logger.info("测试1: 正常模式 (flow_efficiency=1.0)")
    logger.info('='*70)

    current_levels = [3.0, 3.0, 3.0]
    q_in_prevs = [5.0, 5.0, 5.0]

    solutions, info = controller.solve(current_levels, q_in_prevs)

    logger.info(f"\n结果:")
    logger.info(f"  收敛: {info['converged']}")
    logger.info(f"  迭代: {info['iterations']}")
    for i, (q_in, q_out) in enumerate(solutions):
        state = info['physical_states'][i]
        logger.info(f"  池{i}: q_in={q_in:.2f}, q_out={q_out:.2f}, eta={state['flow_efficiency']}")

    # 测试2: 结冰场景
    logger.info(f"\n{'='*70}")
    logger.info("测试2: 结冰场景 (flow_efficiency=0.75)")
    logger.info('='*70)

    # 设置结冰物理参数
    controller.set_scenario_physics('ICE_FORMATION')

    solutions, info = controller.solve(current_levels, q_in_prevs)

    logger.info(f"\n结果:")
    logger.info(f"  收敛: {info['converged']}")
    for i, (q_in, q_out) in enumerate(solutions):
        state = info['physical_states'][i]
        logger.info(f"  池{i}: q_in={q_in:.2f}, q_out={q_out:.2f}, eta={state['flow_efficiency']:.2f}")

    # 测试3: L3轨迹跟踪
    logger.info(f"\n{'='*70}")
    logger.info("测试3: L3参考轨迹跟踪")
    logger.info('='*70)

    # 恢复正常物理参数
    controller.set_scenario_physics('NORMAL')

    # 设置预泄参考轨迹 (水位逐步降低)
    ref_trajectory = np.array([
        [3.0, 2.8, 2.6, 2.4, 2.2],  # 池0
        [3.0, 2.9, 2.8, 2.7, 2.6],  # 池1
        [3.0, 2.9, 2.9, 2.8, 2.8],  # 池2
    ])
    controller.set_reference_trajectories(ref_trajectory)

    solutions, info = controller.solve(current_levels, q_in_prevs)

    logger.info(f"\n结果:")
    logger.info(f"  收敛: {info['converged']}")
    for i, (q_in, q_out) in enumerate(solutions):
        pred_level = controller.local_controllers[i].get_level_prediction()
        logger.info(f"  池{i}: q_in={q_in:.2f}, q_out={q_out:.2f}")
        logger.info(f"       预测水位: {', '.join([f'{l:.2f}' for l in pred_level])} m")

    logger.info("\n" + "="*70)
    logger.info("演示完成!")
    logger.info("="*70)

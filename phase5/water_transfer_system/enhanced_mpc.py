"""
增强参数化MPC - 支持在线热重构
Enhanced Parameterized MPC with Hot Reconfiguration

关键特性:
1. 在线热重构: 不停机修改目标函数和约束
2. 角色感知: 根据PoolRole自动调整MPC参数
3. 指令驱动: 应用ControlDirective修改权重和约束
4. 平滑切换: 场景切换时的平滑过渡

数学模型:
min Σ(W_z * (Z-Z_ref)² + W_q * (Q-Q_ref)² + W_s * ΔQ²)
s.t. Z[k+1] = Z[k] + (η*Q_in - Q_out)*dt/A
     Z_min ≤ Z ≤ Z_max
     Q_min ≤ Q ≤ Q_max
     |ΔQ| ≤ ΔQ_max
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import time

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

from .core_types import PoolRole, ControlDirective, ScenarioType

logger = logging.getLogger(__name__)


# ==============================================================================
# MPC配置
# ==============================================================================

@dataclass
class MPCWeights:
    """MPC权重配置"""
    W_level: float = 10.0       # 水位跟踪权重
    W_flow: float = 5.0         # 流量跟踪权重
    W_smooth: float = 2.0       # 平滑权重
    W_coupling: float = 1.0     # 耦合权重
    W_trajectory: float = 15.0  # 轨迹跟踪权重
    W_energy: float = 0.1       # 能耗权重


@dataclass
class MPCConstraints:
    """MPC约束配置"""
    Z_min: float = 0.5          # 最小水位 [m]
    Z_max: float = 8.0          # 最大水位 [m]
    Q_min: float = 0.0          # 最小流量 [m³/s]
    Q_max: float = 400.0        # 最大流量 [m³/s]
    delta_Q_max: float = 50.0   # 最大流量变化率 [m³/s/step]
    gate_min: float = 0.0       # 最小闸门开度
    gate_max: float = 1.0       # 最大闸门开度


@dataclass
class MPCPhysics:
    """MPC物理参数"""
    area: float = 100000.0      # 水面面积 [m²]
    flow_efficiency: float = 1.0  # 流速效率
    delay_steps: int = 0        # 延迟步数
    manning_n: float = 0.014    # 曼宁糙率


# ==============================================================================
# 角色-参数映射
# ==============================================================================

class RoleParameterMapper:
    """
    角色-MPC参数映射器

    根据PoolRole自动调整MPC权重和约束
    """

    # 角色对应的权重调整
    WEIGHT_MAPPING: Dict[PoolRole, Dict[str, float]] = {
        PoolRole.TRANSMIT: {
            'W_level': 1.0, 'W_flow': 1.0, 'W_smooth': 1.0
        },
        PoolRole.ISOLATE: {
            'W_level': 0.0, 'W_flow': 1e6, 'W_smooth': 0.1
        },
        PoolRole.BUFFER: {
            'W_level': 0.5, 'W_flow': 0.1, 'W_smooth': 2.0
        },
        PoolRole.DRAIN: {
            'W_level': 2.0, 'W_flow': 0.5, 'W_smooth': 1.0
        },
        PoolRole.SOURCE: {
            'W_level': 0.5, 'W_flow': 2.0, 'W_smooth': 0.8
        },
        PoolRole.THROTTLE: {
            'W_level': 1.0, 'W_flow': 0.5, 'W_smooth': 3.0
        },
        PoolRole.STABLE: {
            'W_level': 1.5, 'W_flow': 0.5, 'W_smooth': 5.0
        },
        PoolRole.WAIT: {
            'W_level': 1.5, 'W_flow': 0.3, 'W_smooth': 2.0
        },
        PoolRole.FEED: {
            'W_level': 0.8, 'W_flow': 1.5, 'W_smooth': 1.0
        },
        PoolRole.HOLD: {
            'W_level': 2.0, 'W_flow': 0.2, 'W_smooth': 2.0
        },
        PoolRole.PASS: {
            'W_level': 1.0, 'W_flow': 1.5, 'W_smooth': 0.8
        },
        PoolRole.PASSIVE: {
            'W_level': 1.0, 'W_flow': 0.1, 'W_smooth': 1.5
        },
        PoolRole.BRAKE: {
            'W_level': 1.5, 'W_flow': 2.0, 'W_smooth': 1.0
        },
        PoolRole.COMPENSATE: {
            'W_level': 0.8, 'W_flow': 1.5, 'W_smooth': 1.0
        },
        PoolRole.STORE: {
            'W_level': 0.3, 'W_flow': 0.5, 'W_smooth': 2.0
        },
        PoolRole.CONSUME: {
            'W_level': 1.2, 'W_flow': 1.0, 'W_smooth': 1.0
        },
        PoolRole.BYPASS: {
            'W_level': 1.0, 'W_flow': 0.5, 'W_smooth': 1.0
        },
        PoolRole.ISLAND: {
            'W_level': 1.5, 'W_flow': 0.3, 'W_smooth': 2.0
        },
        PoolRole.MAINT: {
            'W_level': 0.5, 'W_flow': 1e6, 'W_smooth': 0.1
        },
        PoolRole.FIX: {
            'W_level': 0.0, 'W_flow': 1e6, 'W_smooth': 0.1
        },
    }

    # 角色对应的约束调整
    CONSTRAINT_MAPPING: Dict[PoolRole, Dict[str, Any]] = {
        PoolRole.ISOLATE: {
            'Q_max': 0, 'force_Q_out': 0
        },
        PoolRole.BUFFER: {
            'Z_max_relax': 0.3  # 放宽上限
        },
        PoolRole.DRAIN: {
            'Z_min_relax': -0.2  # 放宽下限
        },
        PoolRole.STABLE: {
            'delta_Q_max_factor': 0.2  # 限制变化率
        },
        PoolRole.MAINT: {
            'Q_max': 0, 'force_Q_out': 0, 'force_Q_in': 0
        },
        PoolRole.FIX: {
            'Q_max': 0, 'force_Q_out': 0
        },
    }

    @classmethod
    def get_weights_for_role(cls, role: PoolRole, base_weights: MPCWeights) -> MPCWeights:
        """获取角色对应的权重"""
        factors = cls.WEIGHT_MAPPING.get(role, cls.WEIGHT_MAPPING[PoolRole.TRANSMIT])

        return MPCWeights(
            W_level=base_weights.W_level * factors.get('W_level', 1.0),
            W_flow=base_weights.W_flow * factors.get('W_flow', 1.0),
            W_smooth=base_weights.W_smooth * factors.get('W_smooth', 1.0),
            W_coupling=base_weights.W_coupling,
            W_trajectory=base_weights.W_trajectory,
            W_energy=base_weights.W_energy,
        )

    @classmethod
    def get_constraints_for_role(cls, role: PoolRole,
                                  base_constraints: MPCConstraints) -> MPCConstraints:
        """获取角色对应的约束"""
        adjustments = cls.CONSTRAINT_MAPPING.get(role, {})

        return MPCConstraints(
            Z_min=base_constraints.Z_min - adjustments.get('Z_min_relax', 0),
            Z_max=base_constraints.Z_max + adjustments.get('Z_max_relax', 0),
            Q_min=base_constraints.Q_min,
            Q_max=adjustments.get('Q_max', base_constraints.Q_max),
            delta_Q_max=base_constraints.delta_Q_max * adjustments.get('delta_Q_max_factor', 1.0),
            gate_min=base_constraints.gate_min,
            gate_max=base_constraints.gate_max,
        )


# ==============================================================================
# 增强参数化MPC
# ==============================================================================

class EnhancedParameterizedMPC:
    """
    增强参数化MPC

    特性:
    1. 参数化问题构建 (支持热重构)
    2. 角色感知权重调整
    3. 指令驱动约束修改
    4. 平滑场景切换
    """

    def __init__(self,
                 pool_id: int,
                 horizon: int = 10,
                 dt: float = 900.0,
                 physics: MPCPhysics = None,
                 weights: MPCWeights = None,
                 constraints: MPCConstraints = None):
        """
        初始化增强MPC

        Args:
            pool_id: 渠池ID
            horizon: 预测时域
            dt: 时间步长 [s]
            physics: 物理参数
            weights: 权重配置
            constraints: 约束配置
        """
        self.pool_id = pool_id
        self.N = horizon
        self.dt = dt

        # 配置
        self.physics = physics or MPCPhysics()
        self.base_weights = weights or MPCWeights()
        self.base_constraints = constraints or MPCConstraints()

        # 当前有效配置
        self.active_weights = MPCWeights(**vars(self.base_weights))
        self.active_constraints = MPCConstraints(**vars(self.base_constraints))

        # 角色状态
        self.current_role: PoolRole = PoolRole.TRANSMIT
        self.active_directive: Optional[ControlDirective] = None

        # 参考值
        self.Z_ref = 4.0
        self.reference_trajectory: Optional[np.ndarray] = None

        # CVXPY问题
        self._problem_built = False
        if CVXPY_AVAILABLE:
            self._build_problem()

        # 状态
        self.last_solve_time = 0.0
        self.last_solve_status = "not_solved"

        # 场景切换平滑
        self._transition_steps = 0
        self._transition_target_weights: Optional[MPCWeights] = None

    def _build_problem(self):
        """构建CVXPY参数化问题"""
        if not CVXPY_AVAILABLE:
            logger.warning("CVXPY不可用，使用简化求解器")
            return

        N = self.N

        # 决策变量
        self.Q_in_var = cp.Variable(N, name='Q_in')
        self.Q_out_var = cp.Variable(N, name='Q_out')
        self.Z_var = cp.Variable(N, name='Z')

        # 参数
        self.Z_curr_param = cp.Parameter(name='Z_curr')
        self.Q_in_prev_param = cp.Parameter(name='Q_in_prev')

        # 权重参数 (可热更新)
        self.W_level_param = cp.Parameter(nonneg=True, name='W_level')
        self.W_flow_param = cp.Parameter(nonneg=True, name='W_flow')
        self.W_smooth_param = cp.Parameter(nonneg=True, name='W_smooth')

        # 物理参数
        self.area_param = cp.Parameter(pos=True, name='area')
        self.efficiency_param = cp.Parameter(pos=True, name='efficiency')

        # 约束参数
        self.Z_min_param = cp.Parameter(name='Z_min')
        self.Z_max_param = cp.Parameter(name='Z_max')
        self.Q_max_param = cp.Parameter(nonneg=True, name='Q_max')
        self.delta_Q_max_param = cp.Parameter(nonneg=True, name='delta_Q_max')

        # 参考轨迹参数
        self.Z_ref_param = cp.Parameter(N, name='Z_ref')

        # 初始化参数默认值
        self._set_parameter_values()

        # 构建约束
        constraints = []

        for k in range(N):
            # 水力学约束
            if k == 0:
                constraints.append(
                    self.Z_var[k] == self.Z_curr_param +
                    (self.efficiency_param * self.Q_in_prev_param - self.Q_out_var[k]) *
                    self.dt / self.area_param
                )
            else:
                constraints.append(
                    self.Z_var[k] == self.Z_var[k-1] +
                    (self.efficiency_param * self.Q_in_var[k-1] - self.Q_out_var[k]) *
                    self.dt / self.area_param
                )

            # 状态约束
            constraints.extend([
                self.Z_var[k] >= self.Z_min_param,
                self.Z_var[k] <= self.Z_max_param,
                self.Q_in_var[k] >= 0,
                self.Q_in_var[k] <= self.Q_max_param,
                self.Q_out_var[k] >= 0,
                self.Q_out_var[k] <= self.Q_max_param,
            ])

            # 流量变化率约束
            if k == 0:
                constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_prev_param) <= self.delta_Q_max_param
                )
            else:
                constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_var[k-1]) <= self.delta_Q_max_param
                )

        # 目标函数
        cost = 0

        # 水位跟踪
        for k in range(N):
            cost += self.W_level_param * cp.square(self.Z_var[k] - self.Z_ref_param[k])

        # 流量平滑
        for k in range(N):
            if k == 0:
                cost += self.W_smooth_param * cp.square(
                    self.Q_in_var[k] - self.Q_in_prev_param
                )
            else:
                cost += self.W_smooth_param * cp.square(
                    self.Q_in_var[k] - self.Q_in_var[k-1]
                )

            if k > 0:
                cost += self.W_smooth_param * cp.square(
                    self.Q_out_var[k] - self.Q_out_var[k-1]
                )

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self._problem_built = True

        logger.debug(f"池{self.pool_id} MPC问题构建完成")

    def _set_parameter_values(self):
        """设置参数值"""
        if not CVXPY_AVAILABLE:
            return

        # 权重
        self.W_level_param.value = self.active_weights.W_level
        self.W_flow_param.value = self.active_weights.W_flow
        self.W_smooth_param.value = self.active_weights.W_smooth

        # 物理
        self.area_param.value = self.physics.area
        self.efficiency_param.value = self.physics.flow_efficiency

        # 约束
        self.Z_min_param.value = self.active_constraints.Z_min
        self.Z_max_param.value = self.active_constraints.Z_max
        self.Q_max_param.value = self.active_constraints.Q_max
        self.delta_Q_max_param.value = self.active_constraints.delta_Q_max

        # 参考
        if self.reference_trajectory is not None:
            self.Z_ref_param.value = self.reference_trajectory
        else:
            self.Z_ref_param.value = np.ones(self.N) * self.Z_ref

    def apply_role(self, role: PoolRole, smooth_transition: bool = True):
        """
        应用角色 (热重构)

        Args:
            role: 目标角色
            smooth_transition: 是否平滑过渡
        """
        if role == self.current_role:
            return

        logger.info(f"池{self.pool_id}: 角色切换 {self.current_role.value} -> {role.value}")

        # 计算新权重
        new_weights = RoleParameterMapper.get_weights_for_role(role, self.base_weights)
        new_constraints = RoleParameterMapper.get_constraints_for_role(role, self.base_constraints)

        if smooth_transition:
            # 设置过渡
            self._transition_steps = 5
            self._transition_target_weights = new_weights
        else:
            # 立即切换
            self.active_weights = new_weights

        self.active_constraints = new_constraints
        self.current_role = role

        # 更新参数
        self._set_parameter_values()

    def apply_directive(self, directive: ControlDirective):
        """
        应用控制指令

        Args:
            directive: 控制指令
        """
        if directive.pool_id != self.pool_id:
            return

        self.active_directive = directive

        # 应用角色
        self.apply_role(directive.role)

        # 应用权重乘数
        if directive.weight_multipliers:
            for key, factor in directive.weight_multipliers.items():
                if key == 'W_Q' and hasattr(self.active_weights, 'W_flow'):
                    self.active_weights.W_flow *= factor
                elif key == 'W_Z' and hasattr(self.active_weights, 'W_level'):
                    self.active_weights.W_level *= factor
                elif key == 'W_smooth' and hasattr(self.active_weights, 'W_smooth'):
                    self.active_weights.W_smooth *= factor

        # 应用硬约束
        if directive.hard_constraints:
            if 'Q_out_max' in directive.hard_constraints:
                self.active_constraints.Q_max = directive.hard_constraints['Q_out_max']
            if 'Z_max' in directive.hard_constraints:
                val = directive.hard_constraints['Z_max']
                if isinstance(val, (int, float)):
                    self.active_constraints.Z_max = val
            if 'delta_Q_max' in directive.hard_constraints:
                self.active_constraints.delta_Q_max = directive.hard_constraints['delta_Q_max']

        # 应用目标偏移
        self.Z_ref += directive.target_bias

        # 更新参数
        self._set_parameter_values()

        logger.info(f"池{self.pool_id}: 应用指令 - 角色={directive.role.value}, "
                   f"偏移={directive.target_bias:.2f}")

    def set_reference_trajectory(self, trajectory: np.ndarray):
        """设置参考轨迹"""
        if len(trajectory) >= self.N:
            self.reference_trajectory = trajectory[:self.N]
        else:
            self.reference_trajectory = np.concatenate([
                trajectory,
                np.ones(self.N - len(trajectory)) * trajectory[-1]
            ])

        if CVXPY_AVAILABLE and hasattr(self, 'Z_ref_param'):
            self.Z_ref_param.value = self.reference_trajectory

    def set_physics(self, area: float = None, efficiency: float = None):
        """设置物理参数"""
        if area is not None:
            self.physics.area = area
        if efficiency is not None:
            self.physics.flow_efficiency = efficiency

        if CVXPY_AVAILABLE:
            if area is not None:
                self.area_param.value = area
            if efficiency is not None:
                self.efficiency_param.value = efficiency

    def solve(self,
              current_level: float,
              q_in_prev: float) -> Tuple[float, float, bool]:
        """
        求解MPC

        Args:
            current_level: 当前水位 [m]
            q_in_prev: 上一步入流 [m³/s]

        Returns:
            (最优入流, 最优出流, 是否成功)
        """
        start_time = time.time()

        # 处理平滑过渡
        if self._transition_steps > 0 and self._transition_target_weights:
            alpha = 1.0 - self._transition_steps / 5.0
            self.active_weights.W_level = (
                (1 - alpha) * self.active_weights.W_level +
                alpha * self._transition_target_weights.W_level
            )
            self.active_weights.W_smooth = (
                (1 - alpha) * self.active_weights.W_smooth +
                alpha * self._transition_target_weights.W_smooth
            )
            self._transition_steps -= 1
            self._set_parameter_values()

        if CVXPY_AVAILABLE and self._problem_built:
            return self._solve_cvxpy(current_level, q_in_prev)
        else:
            return self._solve_simple(current_level, q_in_prev)

    def _solve_cvxpy(self, current_level: float, q_in_prev: float) -> Tuple[float, float, bool]:
        """使用CVXPY求解"""
        self.Z_curr_param.value = current_level
        self.Q_in_prev_param.value = q_in_prev

        try:
            self.problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

            if self.problem.status == "optimal":
                self.last_solve_status = "optimal"
                self.last_solve_time = time.time()
                return (
                    float(self.Q_in_var.value[0]),
                    float(self.Q_out_var.value[0]),
                    True
                )
            else:
                # CVXPY未找到最优解，回退到简化求解器
                logger.warning(f"池{self.pool_id} CVXPY状态={self.problem.status}，使用简化求解器")
                return self._solve_simple(current_level, q_in_prev)

        except Exception as e:
            logger.warning(f"池{self.pool_id} CVXPY求解异常: {e}，使用简化求解器")
            return self._solve_simple(current_level, q_in_prev)

    def _solve_simple(self, current_level: float, q_in_prev: float) -> Tuple[float, float, bool]:
        """简化求解 (无CVXPY时使用)"""
        # 简单PI控制
        Z_error = current_level - self.Z_ref
        Q_adjustment = -self.active_weights.W_level * Z_error * 0.01

        Q_out = q_in_prev + Q_adjustment
        Q_out = np.clip(Q_out, self.active_constraints.Q_min, self.active_constraints.Q_max)

        # 根据角色调整
        if self.current_role == PoolRole.ISOLATE:
            Q_out = 0
        elif self.current_role == PoolRole.DRAIN:
            Q_out *= 1.1
        elif self.current_role == PoolRole.BUFFER:
            Q_out *= 0.9

        Q_in = q_in_prev  # 入流保持

        self.last_solve_status = "simple"
        return float(Q_in), float(Q_out), True

    def get_state(self) -> Dict[str, Any]:
        """获取MPC状态"""
        return {
            'pool_id': self.pool_id,
            'role': self.current_role.value,
            'weights': vars(self.active_weights),
            'constraints': vars(self.active_constraints),
            'physics': vars(self.physics),
            'Z_ref': self.Z_ref,
            'last_status': self.last_solve_status,
        }


# ==============================================================================
# 热重构MPC管理器
# ==============================================================================

class HotReconfigurableMPC:
    """
    热重构MPC管理器

    管理多个渠池的MPC控制器，支持整体重构
    """

    def __init__(self,
                 num_pools: int,
                 horizon: int = 10,
                 dt: float = 900.0):
        """
        初始化

        Args:
            num_pools: 渠池数量
            horizon: 预测时域
            dt: 时间步长
        """
        self.num_pools = num_pools
        self.horizon = horizon
        self.dt = dt

        # 创建MPC控制器
        self.controllers: Dict[int, EnhancedParameterizedMPC] = {}
        for i in range(num_pools):
            self.controllers[i] = EnhancedParameterizedMPC(
                pool_id=i,
                horizon=horizon,
                dt=dt
            )

        # 当前场景
        self.current_scenario: Optional[ScenarioType] = None

        logger.info(f"热重构MPC初始化: {num_pools} 控制器")

    def reconfigure_for_scenario(self,
                                 scenario: ScenarioType,
                                 center_pool: int,
                                 directives: Dict[int, ControlDirective] = None):
        """
        为场景重构所有MPC

        Args:
            scenario: 场景类型
            center_pool: 事故中心池
            directives: 指令字典
        """
        self.current_scenario = scenario

        # 应用指令
        if directives:
            for pool_id, directive in directives.items():
                if pool_id in self.controllers:
                    self.controllers[pool_id].apply_directive(directive)

        logger.info(f"MPC重构完成: 场景={scenario.value}, 中心={center_pool}")

    def apply_plan(self, plan: Any):
        """应用控制计划"""
        for pool_id, directive in plan.directives.items():
            if pool_id in self.controllers:
                self.controllers[pool_id].apply_directive(directive)

    def solve_all(self,
                  current_levels: np.ndarray,
                  current_flows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解所有MPC

        Args:
            current_levels: 当前水位数组
            current_flows: 当前流量数组

        Returns:
            (入流数组, 出流数组)
        """
        Q_in = np.zeros(self.num_pools)
        Q_out = np.zeros(self.num_pools)

        for i in range(self.num_pools):
            controller = self.controllers[i]

            q_in, q_out, success = controller.solve(
                current_level=current_levels[i],
                q_in_prev=current_flows[i]
            )

            Q_in[i] = q_in
            Q_out[i] = q_out

        return Q_in, Q_out

    def set_reference_trajectories(self, trajectories: np.ndarray):
        """设置所有池的参考轨迹"""
        for i in range(min(self.num_pools, trajectories.shape[0])):
            self.controllers[i].set_reference_trajectory(trajectories[i, :])

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        role_counts = {}
        for controller in self.controllers.values():
            role = controller.current_role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            'num_pools': self.num_pools,
            'current_scenario': self.current_scenario.value if self.current_scenario else None,
            'role_distribution': role_counts,
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" " * 15 + "增强参数化MPC测试")
    print("="*70)

    # 测试1: 基本MPC
    print(f"\n{'='*70}")
    print("测试1: 基本MPC求解")
    print('='*70)

    mpc = EnhancedParameterizedMPC(
        pool_id=30,
        horizon=10,
        dt=900.0
    )

    q_in, q_out, success = mpc.solve(
        current_level=4.0,
        q_in_prev=100.0
    )

    print(f"  求解状态: {mpc.last_solve_status}")
    print(f"  入流: {q_in:.2f} m³/s")
    print(f"  出流: {q_out:.2f} m³/s")

    # 测试2: 角色切换
    print(f"\n{'='*70}")
    print("测试2: 角色切换 (TRANSMIT -> ISOLATE)")
    print('='*70)

    print(f"  切换前: 角色={mpc.current_role.value}")
    print(f"  切换前: W_level={mpc.active_weights.W_level:.1f}, "
          f"W_flow={mpc.active_weights.W_flow:.1f}")

    mpc.apply_role(PoolRole.ISOLATE)

    print(f"  切换后: 角色={mpc.current_role.value}")
    print(f"  切换后: W_level={mpc.active_weights.W_level:.1f}, "
          f"W_flow={mpc.active_weights.W_flow:.1f}")

    q_in, q_out, success = mpc.solve(current_level=4.0, q_in_prev=100.0)
    print(f"  出流: {q_out:.2f} m³/s (应为0)")

    # 测试3: 指令应用
    print(f"\n{'='*70}")
    print("测试3: 指令应用")
    print('='*70)

    mpc2 = EnhancedParameterizedMPC(pool_id=31, horizon=10)

    directive = ControlDirective(
        pool_id=31,
        role=PoolRole.BUFFER,
        target_bias=0.5,
        weight_multipliers={'W_Q': 0.1, 'W_Z': 0.5},
    )

    mpc2.apply_directive(directive)

    print(f"  角色: {mpc2.current_role.value}")
    print(f"  Z_ref: {mpc2.Z_ref:.2f} (包含偏移)")
    print(f"  W_level: {mpc2.active_weights.W_level:.2f}")

    # 测试4: 热重构管理器
    print(f"\n{'='*70}")
    print("测试4: 热重构MPC管理器")
    print('='*70)

    manager = HotReconfigurableMPC(num_pools=60, horizon=10)

    # 重构为污染场景
    directives = {
        29: ControlDirective(pool_id=29, role=PoolRole.BUFFER),
        30: ControlDirective(pool_id=30, role=PoolRole.ISOLATE),
        31: ControlDirective(pool_id=31, role=PoolRole.DRAIN),
    }

    manager.reconfigure_for_scenario(
        scenario=ScenarioType.S3_POLLUTION,
        center_pool=30,
        directives=directives
    )

    summary = manager.get_summary()
    print(f"  场景: {summary['current_scenario']}")
    print(f"  角色分布: {summary['role_distribution']}")

    # 求解
    levels = np.ones(60) * 4.0
    flows = np.ones(60) * 100.0

    Q_in, Q_out = manager.solve_all(levels, flows)

    print(f"  池30出流: {Q_out[30]:.2f} (应为0)")
    print(f"  平均出流: {np.mean(Q_out):.2f}")

    # 测试5: 角色-参数映射
    print(f"\n{'='*70}")
    print("测试5: 角色-参数映射")
    print('='*70)

    base_weights = MPCWeights()
    base_constraints = MPCConstraints()

    for role in [PoolRole.TRANSMIT, PoolRole.ISOLATE, PoolRole.BUFFER,
                 PoolRole.DRAIN, PoolRole.STABLE]:
        weights = RoleParameterMapper.get_weights_for_role(role, base_weights)
        constraints = RoleParameterMapper.get_constraints_for_role(role, base_constraints)
        print(f"  {role.value}: W_level={weights.W_level:.1f}, "
              f"W_flow={weights.W_flow:.1f}, Q_max={constraints.Q_max:.1f}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)

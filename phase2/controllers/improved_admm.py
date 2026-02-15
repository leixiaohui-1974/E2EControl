"""
改进的ADMM分布式优化算法
提升收敛速度和鲁棒性
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ADMMParameters:
    """ADMM算法参数"""
    rho: float = 1.0                    # 惩罚参数
    alpha: float = 1.6                  # 过松弛参数（1.0-1.8）
    adaptive_rho: bool = True           # 自适应rho
    rho_incr: float = 2.0              # rho增长因子
    rho_decr: float = 2.0              # rho降低因子
    mu: float = 10.0                   # 自适应阈值
    tau_incr: float = 2.0              # 增长判断比例
    tau_decr: float = 2.0              # 降低判断比例
    max_iterations: int = 20           # 最大迭代次数
    tolerance: float = 1e-3            # 收敛容差
    abs_tolerance: float = 1e-4        # 绝对容差
    rel_tolerance: float = 1e-3        # 相对容差


class ImprovedLocalMPC:
    """改进的本地MPC控制器"""
    
    def __init__(self, pool_id: int, horizon: int = 5, dt: float = 3600.0, 
                 area: float = 10000.0):
        """
        初始化本地MPC
        
        Args:
            pool_id: 渠池ID
            horizon: 预测时域
            dt: 时间步长
            area: 渠池面积
        """
        self.pool_id = pool_id
        self.N = horizon
        self.dt = dt
        self.area = area
        
        # 权重
        self.W_level = 10.0
        self.W_smooth = 5.0
        self.W_coupling = 1.0  # 耦合惩罚权重
        
        # 目标和约束
        self.Z_ref = 3.0
        self.Z_min = 0.5
        self.Z_max = 8.0
        self.Q_min = 0.0
        self.Q_max = 20.0
        self.delta_Q_max = 2.0
        
        # ADMM变量
        self.z = np.zeros(horizon)  # 一致性变量
        self.u = np.zeros(horizon)  # 对偶变量
        
        # 缓存CVXPY问题以加速
        self._build_problem()
    
    def _build_problem(self):
        """预先构建优化问题（加速求解）"""
        # CVXPY变量
        self.Q_in_var = cp.Variable(self.N)
        self.Q_out_var = cp.Variable(self.N)
        self.Z_var = cp.Variable(self.N)
        
        # 参数（可以在每次求解时更新）
        self.Z_curr_param = cp.Parameter()
        self.Q_in_prev_param = cp.Parameter()
        self.rho_param = cp.Parameter(nonneg=True)
        self.z_param = cp.Parameter(self.N)
        self.u_param = cp.Parameter(self.N)
        
        # 约束（固定部分）
        self.constraints = []
        for k in range(self.N):
            # 水力学
            if k == 0:
                self.constraints.append(
                    self.Z_var[k] == self.Z_curr_param + 
                    (self.Q_in_prev_param - self.Q_out_var[k]) * self.dt / self.area
                )
            else:
                self.constraints.append(
                    self.Z_var[k] == self.Z_var[k-1] + 
                    (self.Q_in_var[k-1] - self.Q_out_var[k]) * self.dt / self.area
                )
            
            # 约束
            self.constraints.extend([
                self.Z_var[k] >= self.Z_min,
                self.Z_var[k] <= self.Z_max,
                self.Q_in_var[k] >= self.Q_min,
                self.Q_in_var[k] <= self.Q_max,
                self.Q_out_var[k] >= self.Q_min,
                self.Q_out_var[k] <= self.Q_max
            ])
            
            if k == 0:
                self.constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_prev_param) <= self.delta_Q_max
                )
            else:
                self.constraints.append(
                    cp.abs(self.Q_in_var[k] - self.Q_in_var[k-1]) <= self.delta_Q_max
                )
        
        # 目标函数（会在求解时更新）
        self._build_objective()
    
    def _build_objective(self):
        """构建目标函数"""
        cost = 0
        
        # 水位跟踪
        for k in range(self.N):
            cost += self.W_level * cp.square(self.Z_var[k] - self.Z_ref)
        
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
            # 线性项
            cost += self.u_param[k] * (self.Q_out_var[k] - self.z_param[k])
            # 二次惩罚项
            cost += (self.rho_param / 2) * cp.square(
                self.Q_out_var[k] - self.z_param[k]
            )
        
        self.objective = cp.Minimize(cost)
        self.problem = cp.Problem(self.objective, self.constraints)
    
    def solve(self, current_level: float, q_in_prev: float,
             z: np.ndarray, u: np.ndarray, rho: float) -> Tuple[float, float, bool]:
        """
        求解本地MPC（带ADMM）
        
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
        
        # 求解
        try:
            self.problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)
            
            if self.problem.status == "optimal":
                return (float(self.Q_in_var.value[0]), 
                       float(self.Q_out_var.value[0]), 
                       True)
            else:
                return q_in_prev, 5.0, False
        except Exception as e:
            logger.info(f"池{self.pool_id}求解失败: {e}")
            return q_in_prev, 5.0, False
    
    def get_output_sequence(self) -> np.ndarray:
        """获取完整的出流序列"""
        if self.Q_out_var.value is not None:
            return np.array(self.Q_out_var.value)
        else:
            return np.zeros(self.N)


class ImprovedDistributedMPC:
    """改进的分布式MPC（快速ADMM）"""
    
    def __init__(self, num_pools: int, horizon: int = 5, 
                 dt: float = 3600.0, params: ADMMParameters = None):
        """
        初始化改进的分布式MPC
        
        Args:
            num_pools: 渠池数量
            horizon: 预测时域
            dt: 时间步长
            params: ADMM参数
        """
        self.num_pools = num_pools
        self.horizon = horizon
        self.params = params or ADMMParameters()
        
        # 创建本地控制器
        self.local_controllers = [
            ImprovedLocalMPC(pool_id=i, horizon=horizon, dt=dt)
            for i in range(num_pools)
        ]
        
        # ADMM变量
        self.z = [np.ones(horizon) * 5.0 for _ in range(num_pools - 1)]
        self.u = [np.zeros(horizon) for _ in range(num_pools - 1)]
        
        # 自适应rho
        self.rho = self.params.rho
        
        # 历史记录
        self.convergence_history = []
    
    def solve(self, 
              current_levels: List[float],
              q_in_prevs: List[float],
              q_out_forecasts: List[List[float]]) -> Tuple[List[Tuple], Dict]:
        """
        求解分布式MPC
        
        Args:
            current_levels: 各池当前水位
            q_in_prevs: 各池上一步入流
            q_out_forecasts: 各池出流预测
            
        Returns:
            (各池最优控制列表, 求解信息字典)
        """
        start_time = time.time()
        
        # 初始化
        q_in_solutions = [5.0] * self.num_pools
        q_out_solutions = [5.0] * self.num_pools
        q_out_sequences = [np.ones(self.horizon) * 5.0 for _ in range(self.num_pools)]
        
        # ADMM迭代
        for iteration in range(self.params.max_iterations):
            q_out_old = [seq.copy() for seq in q_out_sequences]
            
            # 1. x-update: 每个池独立优化
            for i in range(self.num_pools):
                # 确定z和u
                if i < self.num_pools - 1:
                    z_i = self.z[i]
                    u_i = self.u[i]
                else:
                    z_i = np.ones(self.horizon) * 5.0
                    u_i = np.zeros(self.horizon)
                
                # 求解
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
            
            # 2. z-update: 更新一致性变量（过松弛）
            for i in range(self.num_pools - 1):
                # 获取相邻池的出流和入流
                x_out_i = q_out_sequences[i]
                x_in_i1 = q_out_sequences[i] if i + 1 >= self.num_pools else q_out_sequences[i]
                
                # 过松弛
                alpha = self.params.alpha
                x_hat = alpha * x_out_i + (1 - alpha) * self.z[i]
                
                # 更新z（这里简化为平均）
                self.z[i] = 0.5 * (x_hat + x_in_i1)
            
            # 3. u-update: 更新对偶变量
            for i in range(self.num_pools - 1):
                residual = q_out_sequences[i] - self.z[i]
                self.u[i] += self.rho * residual
            
            # 4. 收敛检查
            primal_residual = self._compute_primal_residual(q_out_sequences)
            dual_residual = self._compute_dual_residual(q_out_sequences, q_out_old)
            
            # 记录历史
            self.convergence_history.append({
                'iteration': iteration + 1,
                'primal_residual': primal_residual,
                'dual_residual': dual_residual,
                'rho': self.rho
            })
            
            # 检查收敛
            if self._check_convergence(primal_residual, dual_residual, q_out_sequences):
                solve_time = time.time() - start_time
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'solve_time': solve_time,
                    'primal_residual': primal_residual,
                    'dual_residual': dual_residual,
                    'final_rho': self.rho
                }
                return list(zip(q_in_solutions, q_out_solutions)), info
            
            # 5. 自适应调整rho
            if self.params.adaptive_rho:
                self._update_rho(primal_residual, dual_residual)
        
        # 未收敛
        solve_time = time.time() - start_time
        info = {
            'converged': False,
            'iterations': self.params.max_iterations,
            'solve_time': solve_time,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'final_rho': self.rho
        }
        return list(zip(q_in_solutions, q_out_solutions)), info
    
    def _compute_primal_residual(self, q_out_sequences: List[np.ndarray]) -> float:
        """计算原始残差"""
        residual = 0.0
        for i in range(self.num_pools - 1):
            r = q_out_sequences[i] - self.z[i]
            residual += np.linalg.norm(r)
        return residual
    
    def _compute_dual_residual(self, q_out_new: List[np.ndarray],
                              q_out_old: List[np.ndarray]) -> float:
        """计算对偶残差"""
        residual = 0.0
        for i in range(self.num_pools - 1):
            s = self.rho * (q_out_new[i] - q_out_old[i])
            residual += np.linalg.norm(s)
        return residual
    
    def _check_convergence(self, primal_res: float, dual_res: float,
                          q_out_sequences: List[np.ndarray]) -> bool:
        """检查是否收敛"""
        # 绝对容差
        eps_abs = self.params.abs_tolerance
        
        # 相对容差
        eps_rel = self.params.rel_tolerance
        
        # 原始容差
        x_norm = sum(np.linalg.norm(x) for x in q_out_sequences)
        z_norm = sum(np.linalg.norm(z) for z in self.z)
        eps_pri = np.sqrt(self.num_pools * self.horizon) * eps_abs + eps_rel * max(x_norm, z_norm)
        
        # 对偶容差
        u_norm = sum(np.linalg.norm(u) for u in self.u)
        eps_dual = np.sqrt(self.num_pools * self.horizon) * eps_abs + eps_rel * self.rho * u_norm
        
        return primal_res <= eps_pri and dual_res <= eps_dual
    
    def _update_rho(self, primal_res: float, dual_res: float):
        """自适应更新rho"""
        if primal_res > self.params.mu * dual_res:
            # 原始残差大，增大rho
            self.rho *= self.params.rho_incr
            # 更新对偶变量
            for i in range(self.num_pools - 1):
                self.u[i] /= self.params.rho_incr
        elif dual_res > self.params.mu * primal_res:
            # 对偶残差大，减小rho
            self.rho /= self.params.rho_decr
            # 更新对偶变量
            for i in range(self.num_pools - 1):
                self.u[i] *= self.params.rho_decr
        
        # 限制rho范围
        self.rho = np.clip(self.rho, 0.01, 100.0)
    
    def get_convergence_plot_data(self) -> Dict:
        """获取收敛曲线数据"""
        if not self.convergence_history:
            return {}
        
        return {
            'iterations': [h['iteration'] for h in self.convergence_history],
            'primal_residual': [h['primal_residual'] for h in self.convergence_history],
            'dual_residual': [h['dual_residual'] for h in self.convergence_history],
            'rho': [h['rho'] for h in self.convergence_history]
        }


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "改进ADMM分布式MPC演示")
    logger.info("="*70)
    
    # 创建控制器（启用自适应rho）
    params = ADMMParameters(
        rho=1.0,
        alpha=1.6,
        adaptive_rho=True,
        max_iterations=20
    )
    
    controller = ImprovedDistributedMPC(num_pools=3, horizon=5, params=params)
    
    # 测试求解
    logger.info("\n1. 测试求解性能...")
    current_levels = [3.1, 2.9, 3.0]
    q_in_prevs = [5.0, 5.0, 5.0]
    q_out_forecasts = [[5.0]*5, [5.0]*5, [5.0]*5]
    
    solutions, info = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
    
    logger.info(f"\n   收敛状态: {'✓ 收敛' if info['converged'] else '✗ 未收敛'}")
    logger.info(f"   迭代次数: {info['iterations']}")
    logger.info(f"   求解时间: {info['solve_time']*1000:.2f} ms")
    logger.info(f"   原始残差: {info['primal_residual']:.6f}")
    logger.info(f"   对偶残差: {info['dual_residual']:.6f}")
    logger.info(f"   最终rho: {info['final_rho']:.3f}")
    
    logger.info("\n2. 最优控制:")
    for i, (q_in, q_out) in enumerate(solutions):
        logger.info(f"   池{i}: 入流={q_in:.2f} m³/s, 出流={q_out:.2f} m³/s")
    
    logger.info("\n3. 收敛历史:")
    conv_data = controller.get_convergence_plot_data()
    if conv_data:
        for i, (iter, pr, dr, rho) in enumerate(zip(
            conv_data['iterations'][:5],
            conv_data['primal_residual'][:5],
            conv_data['dual_residual'][:5],
            conv_data['rho'][:5]
        )):
            logger.info(f"   迭代{iter}: pr={pr:.4f}, dr={dr:.4f}, rho={rho:.3f}")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)

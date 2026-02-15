"""
分布式MPC控制器
使用ADMM算法实现多池协同优化
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LocalMPC:
    """本地MPC控制器（单池）"""
    
    def __init__(self, pool_id: int, horizon: int = 5, dt: float = 3600.0, area: float = 10000.0):
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
        
        # 目标水位
        self.Z_ref = 3.0
        
        # 约束
        self.Z_min = 0.5
        self.Z_max = 8.0
        self.Q_min = 0.0
        self.Q_max = 20.0
        self.delta_Q_max = 2.0
    
    def solve(self, 
              current_level: float,
              q_in_prev: float,
              q_out_forecast: List[float],
              coupling_vars: Dict = None) -> Tuple[float, float]:
        """
        求解本地MPC
        
        Args:
            current_level: 当前水位
            q_in_prev: 上一步入流
            q_out_forecast: 出流预测
            coupling_vars: 耦合变量（来自邻居）
            
        Returns:
            (最优入流, 最优出流)
        """
        # CVXPY变量
        Q_in = cp.Variable(self.N)   # 入流序列
        Q_out = cp.Variable(self.N)  # 出流序列
        Z = cp.Variable(self.N)      # 水位序列
        
        # 目标函数
        cost = 0
        constraints = []
        
        Z_curr = current_level
        Q_in_prev = q_in_prev
        
        for k in range(self.N):
            # 水力学约束
            # Z(k+1) = Z(k) + (Q_in(k) - Q_out(k)) * dt / area
            if k == 0:
                constraints.append(
                    Z[k] == Z_curr + (Q_in_prev - Q_out[k]) * self.dt / self.area
                )
            else:
                constraints.append(
                    Z[k] == Z[k-1] + (Q_in[k-1] - Q_out[k]) * self.dt / self.area
                )
            
            # 目标：水位跟踪
            cost += self.W_level * cp.square(Z[k] - self.Z_ref)
            
            # 目标：入流平滑
            if k == 0:
                cost += self.W_smooth * cp.square(Q_in[k] - Q_in_prev)
            else:
                cost += self.W_smooth * cp.square(Q_in[k] - Q_in[k-1])
            
            # 目标：出流平滑
            if k > 0:
                cost += self.W_smooth * cp.square(Q_out[k] - Q_out[k-1])
            
            # 约束：水位范围
            constraints.append(Z[k] >= self.Z_min)
            constraints.append(Z[k] <= self.Z_max)
            
            # 约束：流量范围
            constraints.append(Q_in[k] >= self.Q_min)
            constraints.append(Q_in[k] <= self.Q_max)
            constraints.append(Q_out[k] >= self.Q_min)
            constraints.append(Q_out[k] <= self.Q_max)
            
            # 约束：流量变化率
            if k == 0:
                constraints.append(cp.abs(Q_in[k] - Q_in_prev) <= self.delta_Q_max)
            else:
                constraints.append(cp.abs(Q_in[k] - Q_in[k-1]) <= self.delta_Q_max)
        
        # 耦合约束（ADMM）
        if coupling_vars is not None:
            # 本池的出流应该等于下游池的入流
            if 'downstream_q_in' in coupling_vars:
                lam = coupling_vars.get('lambda', 0)  # 拉格朗日乘子
                rho = coupling_vars.get('rho', 1.0)   # 惩罚参数
                
                # 增广拉格朗日项
                for k in range(self.N):
                    q_down = coupling_vars['downstream_q_in'][k]
                    cost += lam * (Q_out[k] - q_down)
                    cost += (rho / 2) * cp.square(Q_out[k] - q_down)
        
        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == "optimal":
                return Q_in.value[0], Q_out.value[0]
            else:
                # 失败时返回安全值
                return q_in_prev, q_out_forecast[0] if q_out_forecast else 5.0
                
        except Exception as e:
            logger.info(f"池{self.pool_id} MPC求解失败: {e}")
            return q_in_prev, q_out_forecast[0] if q_out_forecast else 5.0


class DistributedMPCController:
    """分布式MPC控制器（ADMM算法）"""
    
    def __init__(self, num_pools: int, horizon: int = 5, dt: float = 3600.0):
        """
        初始化分布式MPC
        
        Args:
            num_pools: 渠池数量
            horizon: 预测时域
            dt: 时间步长
        """
        self.num_pools = num_pools
        self.horizon = horizon
        
        # 创建本地控制器
        self.local_controllers = [
            LocalMPC(pool_id=i, horizon=horizon, dt=dt)
            for i in range(num_pools)
        ]
        
        # ADMM参数
        self.rho = 1.0           # 惩罚参数
        self.max_iterations = 10  # 最大迭代次数
        self.tolerance = 1e-3    # 收敛容差
        
        # 对偶变量（拉格朗日乘子）
        self.dual_vars = [np.zeros(horizon) for _ in range(num_pools - 1)]
    
    def solve(self, 
              current_levels: List[float],
              q_in_prevs: List[float],
              q_out_forecasts: List[List[float]]) -> List[Tuple[float, float]]:
        """
        求解分布式MPC
        
        Args:
            current_levels: 各池当前水位
            q_in_prevs: 各池上一步入流
            q_out_forecasts: 各池出流预测
            
        Returns:
            各池最优控制 [(q_in_0, q_out_0), (q_in_1, q_out_1), ...]
        """
        # 初始化
        q_in_solutions = [5.0] * self.num_pools
        q_out_solutions = [5.0] * self.num_pools
        
        # ADMM迭代
        for iter in range(self.max_iterations):
            q_in_old = q_in_solutions.copy()
            q_out_old = q_out_solutions.copy()
            
            # 1. 每个池独立优化（并行）
            for i in range(self.num_pools):
                # 构造耦合变量
                coupling = {}
                
                # 如果不是最后一个池，考虑下游耦合
                if i < self.num_pools - 1:
                    coupling['downstream_q_in'] = [q_in_solutions[i+1]] * self.horizon
                    coupling['lambda'] = self.dual_vars[i][0]
                    coupling['rho'] = self.rho
                
                # 求解本地MPC
                q_in, q_out = self.local_controllers[i].solve(
                    current_level=current_levels[i],
                    q_in_prev=q_in_prevs[i],
                    q_out_forecast=q_out_forecasts[i],
                    coupling_vars=coupling if coupling else None
                )
                
                q_in_solutions[i] = q_in
                q_out_solutions[i] = q_out
            
            # 2. 更新对偶变量
            for i in range(self.num_pools - 1):
                # 不一致性：池i的出流 vs 池i+1的入流
                residual = q_out_solutions[i] - q_in_solutions[i+1]
                self.dual_vars[i][0] += self.rho * residual
            
            # 3. 检查收敛
            q_in_change = max(abs(q_in_solutions[i] - q_in_old[i]) for i in range(self.num_pools))
            q_out_change = max(abs(q_out_solutions[i] - q_out_old[i]) for i in range(self.num_pools))
            
            if max(q_in_change, q_out_change) < self.tolerance:
                logger.info(f"ADMM收敛于第{iter+1}次迭代")
                break
        
        # 返回结果
        return list(zip(q_in_solutions, q_out_solutions))


# 示例使用
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("分布式MPC控制器测试")
    logger.info("="*60)
    
    # 创建3池系统的分布式MPC
    controller = DistributedMPCController(num_pools=3, horizon=5)
    
    # 当前状态
    current_levels = [3.0, 3.0, 3.0]
    q_in_prevs = [5.0, 5.0, 5.0]
    q_out_forecasts = [[5.0]*5, [5.0]*5, [5.0]*5]
    
    # 求解
    logger.info("\n求解分布式MPC...")
    solutions = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
    
    logger.info("\n最优控制:")
    for i, (q_in, q_out) in enumerate(solutions):
        logger.info(f"  池{i}: 入流={q_in:.2f} m³/s, 出流={q_out:.2f} m³/s")
    
    logger.info("\n" + "="*60)

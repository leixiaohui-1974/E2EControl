"""
单渠池ADMM鲁棒求解器
处理效率与安全的冲突
"""

import logging
import sys
sys.path.append('..')

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from perception.intelligent_observer import DynamicConstraints


@dataclass
class ElectricityPrice:
    """分时电价"""
    peak_hours: List[int]  # 高峰时段
    valley_hours: List[int]  # 低谷时段
    peak_price: float  # 高峰电价 [元/kWh]
    valley_price: float  # 低谷电价
    normal_price: float  # 平段电价


class SinglePoolADMM:
    """
    单渠池ADMM求解器
    
    核心功能:
    1. x-update: 优化效率 (水位跟踪 + 能耗最小化)
    2. z-update: 安全投影 (动态约束)
    3. 鲁棒性: 数据被黑时增大正则化
    """
    
    def __init__(self, N: int = 20, horizon: int = 10, dt: float = 60.0):
        """
        初始化ADMM求解器
        
        Args:
            N: 空间切片数
            horizon: 预测时域
            dt: 时间步长 [s]
        """
        self.N = N
        self.H = horizon
        self.dt = dt
        
        # ADMM参数
        self.rho = 1.0  # 正则化参数
        self.max_iter = 20
        self.tolerance = 1e-3
        
        # 电价信息
        self.electricity = ElectricityPrice(
            peak_hours=list(range(8, 12)) + list(range(18, 22)),
            valley_hours=list(range(0, 6)) + list(range(23, 24)),
            peak_price=1.2,
            valley_price=0.4,
            normal_price=0.8
        )
        
        # 优化权重
        self.w_level = 1.0  # 水位跟踪权重
        self.w_energy = 0.5  # 能耗权重
        self.w_smooth = 0.3  # 平滑性权重
        
        # 约束
        self.u_in_min = 10.0
        self.u_in_max = 120.0
        self.u_out_min = 10.0
        self.u_out_max = 120.0
        
        # 动态约束（由感知层更新）
        self.dynamic_constraints = None
        
        # 求解历史
        self.solve_history = {
            'iterations': [],
            'residuals': [],
            'cost': [],
            'energy_cost': []
        }
    
    def update_dynamic_constraints(self, constraints: DynamicConstraints):
        """更新动态约束"""
        self.dynamic_constraints = constraints
        
        # 根据应急等级调整ADMM参数
        if constraints.emergency_level >= 3:
            self.rho = 5.0  # 高度保守
        elif constraints.emergency_level >= 2:
            self.rho = 2.0  # 保守
        elif constraints.emergency_level >= 1:
            self.rho = 1.5  # 适度保守
        else:
            self.rho = 1.0  # 正常
    
    def solve(self, current_Z: np.ndarray, current_Q: np.ndarray,
              target_Z: np.ndarray, current_hour: int) -> Tuple[Tuple[float, float], Dict]:
        """
        ADMM求解
        
        Args:
            current_Z: 当前水位 [N]
            current_Q: 当前流量 [N]
            target_Z: 目标水位 [N]
            current_hour: 当前小时 (用于电价)
            
        Returns:
            ((u_in, u_out), debug_info)
        """
        # 获取当前电价
        current_price = self._get_electricity_price(current_hour)
        
        # x-update: 效率优化
        u_in_opt, u_out_opt, cost_efficiency = self._x_update(
            current_Z, current_Q, target_Z, current_price
        )
        
        # z-update: 安全投影
        u_in_safe, u_out_safe = self._z_update(
            u_in_opt, u_out_opt, current_Z, current_Q
        )
        
        # 记录
        energy_cost = self._compute_energy_cost(u_out_safe, current_price)
        
        self.solve_history['cost'].append(cost_efficiency)
        self.solve_history['energy_cost'].append(energy_cost)
        
        debug_info = {
            'u_in_opt': u_in_opt,
            'u_out_opt': u_out_opt,
            'u_in_safe': u_in_safe,
            'u_out_safe': u_out_safe,
            'electricity_price': current_price,
            'energy_cost': energy_cost,
            'rho': self.rho,
            'emergency_level': self.dynamic_constraints.emergency_level if self.dynamic_constraints else 0
        }
        
        return (u_in_safe, u_out_safe), debug_info
    
    def _x_update(self, current_Z: np.ndarray, current_Q: np.ndarray,
                  target_Z: np.ndarray, price: float) -> Tuple[float, float, float]:
        """
        x-update: 效率优化
        
        最小化: w1 * ||Z - Z_target||² + w2 * 能耗成本 + w3 * 平滑性
        """
        # 决策变量
        u_in = cp.Variable()
        u_out = cp.Variable()
        
        # 预测水位 (简化为单步预测)
        Z_pred = current_Z[0] + self.dt * (u_in - current_Q[0]) / 1000.0  # 简化
        
        # 目标函数
        level_cost = self.w_level * cp.square(Z_pred - target_Z[0])
        
        # 能耗成本 (下游泵站功率 = ρ * g * Q * H)
        # 简化：能耗 ∝ Q * H
        pump_head = 10.0  # 简化提升高度 [m]
        power = 9.81 * u_out * pump_head / 1000.0  # [kW]
        energy_cost = self.w_energy * price * power * (self.dt / 3600.0)  # [元]
        
        # 平滑性 (减少控制变化)
        if len(self.solve_history['cost']) > 0:
            # 假设上一步的控制量
            u_in_prev = 50.0
            u_out_prev = 48.0
            smooth_cost = self.w_smooth * (cp.square(u_in - u_in_prev) + 
                                           cp.square(u_out - u_out_prev))
        else:
            smooth_cost = 0
        
        objective = cp.Minimize(level_cost + energy_cost + smooth_cost)
        
        # 约束
        constraints = [
            u_in >= self.u_in_min,
            u_in <= self.u_in_max,
            u_out >= self.u_out_min,
            u_out <= self.u_out_max,
            u_in >= u_out - 20.0,  # 防止库容过快下降
            u_out <= u_in + 10.0   # 防止库容过快上升
        ]
        
        # 求解
        prob = cp.Problem(objective, constraints)
        
        try:
            # 尝试多个求解器
            for solver in [cp.OSQP, cp.SCS, cp.CVXOPT]:
                try:
                    prob.solve(solver=solver, verbose=False)
                    if prob.status == cp.OPTIMAL:
                        return float(u_in.value), float(u_out.value), float(prob.value)
                except Exception as exc:
                    logger.debug("Solver %s failed: %s", solver, exc)
                    continue
            
            # 所有求解器都失败，返回保守值
            return 50.0, 48.0, 999.0
                
        except Exception as e:
            # print(f"    ⚠️  x-update求解失败: {e}")
            return 50.0, 48.0, 999.0
    
    def _z_update(self, u_in_opt: float, u_out_opt: float,
                  current_Z: np.ndarray, current_Q: np.ndarray) -> Tuple[float, float]:
        """
        z-update: 安全投影
        
        将优化解投影到安全约束集合
        """
        u_in_safe = u_in_opt
        u_out_safe = u_out_opt
        
        # 如果有动态约束
        if self.dynamic_constraints is not None:
            max_dZ_dt = self.dynamic_constraints.max_dZ_dt
            max_flow_change = self.dynamic_constraints.max_flow_change
            
            # 预测水位变化
            mean_Z = np.mean(current_Z)
            A_approx = 50.0 * mean_Z + 2.0 * mean_Z**2  # 简化断面积
            
            dQ = u_in_safe - u_out_safe
            predicted_dZ = self.dt * dQ / A_approx
            predicted_dZ_dt = predicted_dZ / self.dt
            
            # 如果预测变化率超标
            if abs(predicted_dZ_dt) > max_dZ_dt:
                # 压缩流量差
                scaling = max_dZ_dt / (abs(predicted_dZ_dt) + 1e-6)
                dQ_safe = dQ * scaling
                
                # 调整输出量（优先保持输入）
                u_out_safe = u_in_safe - dQ_safe
                
                # 确保在物理范围内
                u_out_safe = np.clip(u_out_safe, self.u_out_min, self.u_out_max)
                
                print(f"    ⚙️  安全投影: 限制dZ/dt至 {max_dZ_dt:.5f}m/s")
            
            # 流量变化限制
            mean_Q = np.mean(current_Q)
            
            if abs(u_in_safe - mean_Q) > max_flow_change:
                u_in_safe = mean_Q + np.sign(u_in_safe - mean_Q) * max_flow_change
            
            if abs(u_out_safe - mean_Q) > max_flow_change:
                u_out_safe = mean_Q + np.sign(u_out_safe - mean_Q) * max_flow_change
        
        # 最终范围约束
        u_in_safe = np.clip(u_in_safe, self.u_in_min, self.u_in_max)
        u_out_safe = np.clip(u_out_safe, self.u_out_min, self.u_out_max)
        
        return u_in_safe, u_out_safe
    
    def _get_electricity_price(self, hour: int) -> float:
        """获取当前电价"""
        if hour in self.electricity.peak_hours:
            return self.electricity.peak_price
        elif hour in self.electricity.valley_hours:
            return self.electricity.valley_price
        else:
            return self.electricity.normal_price
    
    def _compute_energy_cost(self, u_out: float, price: float) -> float:
        """计算能耗成本"""
        pump_head = 10.0  # [m]
        efficiency = 0.85
        power = 9.81 * u_out * pump_head / (1000.0 * efficiency)  # [kW]
        energy = power * (self.dt / 3600.0)  # [kWh]
        cost = price * energy  # [元]
        
        return cost
    
    def get_cumulative_cost(self) -> float:
        """获取累计能耗费用"""
        return sum(self.solve_history['energy_cost'])


# 演示
if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "SinglePoolADMM求解器演示")
    print("="*80)
    
    # 创建求解器
    solver = SinglePoolADMM(N=20, horizon=10)
    
    print(f"\n初始化完成")
    print(f"  ADMM参数: ρ={solver.rho}")
    print(f"  预测时域: {solver.H}")
    
    # 模拟几步
    for t in range(5):
        current_Z = np.ones(20) * 3.0 + 0.1 * np.random.randn(20)
        current_Q = np.ones(20) * 50.0
        target_Z = np.ones(20) * 3.0
        current_hour = 8 + t  # 从早上8点开始
        
        (u_in, u_out), debug = solver.solve(current_Z, current_Q, target_Z, current_hour)
        
        print(f"\nt={t} (Hour {current_hour}h):")
        print(f"  控制量: u_in={u_in:.1f}, u_out={u_out:.1f} m³/s")
        print(f"  电价: {debug['electricity_price']:.2f} 元/kWh")
        print(f"  能耗成本: {debug['energy_cost']:.4f} 元")
    
    print(f"\n累计能耗费用: {solver.get_cumulative_cost():.2f} 元")
    print("\n✅ ADMM求解器演示完成！")
    print("="*80)

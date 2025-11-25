"""
多目标优化MPC控制器
支持Pareto优化和权重自适应
"""

import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectiveType(Enum):
    """优化目标类型"""
    LEVEL_TRACKING = "level_tracking"      # 水位跟踪
    FLOW_SMOOTHNESS = "flow_smoothness"    # 流量平滑
    ENERGY_COST = "energy_cost"            # 能耗成本
    WATER_DELIVERY = "water_delivery"      # 供水保证
    SAFETY_MARGIN = "safety_margin"        # 安全裕度


@dataclass
class OptimizationWeights:
    """优化权重"""
    level_tracking: float = 10.0
    flow_smoothness: float = 5.0
    energy_cost: float = 0.3
    water_delivery: float = 2.0
    safety_margin: float = 1.0


class MultiObjectiveMPC:
    """多目标MPC控制器"""
    
    def __init__(self, pool_id: int, horizon: int = 10, dt: float = 3600.0,
                 area: float = 10000.0):
        """
        初始化多目标MPC
        
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
        
        # 默认权重
        self.weights = OptimizationWeights()
        
        # 目标水位
        self.Z_ref = 3.0
        
        # 约束
        self.Z_min = 0.5
        self.Z_max = 8.0
        self.Q_min = 0.0
        self.Q_max = 20.0
        self.delta_Q_max = 2.0
        
        # 历史成本（用于自适应权重）
        self.cost_history = {obj: [] for obj in ObjectiveType}
        
        # Pareto前沿
        self.pareto_front = []
    
    def solve(self, 
              current_level: float,
              q_in_prev: float,
              demand_forecast: List[float],
              mode: str = 'weighted_sum') -> Tuple[float, float, Dict]:
        """
        求解多目标优化
        
        Args:
            current_level: 当前水位
            q_in_prev: 上一步入流
            demand_forecast: 需求预测
            mode: 优化模式 ('weighted_sum', 'pareto', 'adaptive')
            
        Returns:
            (最优入流, 最优出流, 目标函数值字典)
        """
        if mode == 'weighted_sum':
            return self._solve_weighted_sum(current_level, q_in_prev, demand_forecast)
        elif mode == 'pareto':
            return self._solve_pareto(current_level, q_in_prev, demand_forecast)
        elif mode == 'adaptive':
            return self._solve_adaptive(current_level, q_in_prev, demand_forecast)
        else:
            raise ValueError(f"未知的优化模式: {mode}")
    
    def _solve_weighted_sum(self, current_level: float, q_in_prev: float,
                           demand_forecast: List[float]) -> Tuple[float, float, Dict]:
        """加权求和方法"""
        # CVXPY变量
        Q_in = cp.Variable(self.N)
        Q_out = cp.Variable(self.N)
        Z = cp.Variable(self.N)
        
        # 分目标成本
        costs = {}
        constraints = []
        
        Z_curr = current_level
        Q_in_prev = q_in_prev
        
        # 构建各目标
        for k in range(self.N):
            # 水力学约束
            if k == 0:
                constraints.append(
                    Z[k] == Z_curr + (Q_in_prev - Q_out[k]) * self.dt / self.area
                )
            else:
                constraints.append(
                    Z[k] == Z[k-1] + (Q_in[k-1] - Q_out[k]) * self.dt / self.area
                )
            
            # 约束
            constraints.extend([
                Z[k] >= self.Z_min,
                Z[k] <= self.Z_max,
                Q_in[k] >= self.Q_min,
                Q_in[k] <= self.Q_max,
                Q_out[k] >= self.Q_min,
                Q_out[k] <= self.Q_max
            ])
            
            if k == 0:
                constraints.append(cp.abs(Q_in[k] - Q_in_prev) <= self.delta_Q_max)
            else:
                constraints.append(cp.abs(Q_in[k] - Q_in[k-1]) <= self.delta_Q_max)
        
        # 目标1: 水位跟踪
        J1 = sum(cp.square(Z[k] - self.Z_ref) for k in range(self.N))
        costs[ObjectiveType.LEVEL_TRACKING] = J1
        
        # 目标2: 流量平滑
        J2 = 0
        for k in range(self.N):
            if k == 0:
                J2 += cp.square(Q_in[k] - Q_in_prev)
            else:
                J2 += cp.square(Q_in[k] - Q_in[k-1])
            if k > 0:
                J2 += cp.square(Q_out[k] - Q_out[k-1])
        costs[ObjectiveType.FLOW_SMOOTHNESS] = J2
        
        # 目标3: 能耗成本（简化为流量变化）
        J3 = sum(cp.abs(Q_in[k] - Q_out[k]) for k in range(self.N))
        costs[ObjectiveType.ENERGY_COST] = J3
        
        # 目标4: 供水保证（跟踪需求）
        J4 = sum(cp.square(Q_out[k] - demand_forecast[k]) for k in range(self.N))
        costs[ObjectiveType.WATER_DELIVERY] = J4
        
        # 目标5: 安全裕度（远离约束边界）
        J5 = 0
        for k in range(self.N):
            # 水位裕度
            J5 += cp.inv_pos(Z[k] - self.Z_min + 0.1)
            J5 += cp.inv_pos(self.Z_max - Z[k] + 0.1)
        costs[ObjectiveType.SAFETY_MARGIN] = J5
        
        # 加权总目标
        total_cost = (
            self.weights.level_tracking * J1 +
            self.weights.flow_smoothness * J2 +
            self.weights.energy_cost * J3 +
            self.weights.water_delivery * J4 +
            self.weights.safety_margin * J5
        )
        
        # 求解
        prob = cp.Problem(cp.Minimize(total_cost), constraints)
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status == "optimal":
                # 计算各目标实际值
                cost_values = {
                    obj: float(cost.value) if hasattr(cost, 'value') else 0.0
                    for obj, cost in costs.items()
                }
                
                return Q_in.value[0], Q_out.value[0], cost_values
            else:
                return q_in_prev, demand_forecast[0], {}
        except Exception as e:
            print(f"优化失败: {e}")
            return q_in_prev, demand_forecast[0], {}
    
    def _solve_pareto(self, current_level: float, q_in_prev: float,
                     demand_forecast: List[float]) -> Tuple[float, float, Dict]:
        """
        Pareto优化方法
        
        使用ε-约束法生成Pareto前沿，然后选择最优解
        """
        # 简化实现：在不同权重下求解，找到Pareto前沿
        pareto_solutions = []
        
        # 尝试不同的权重组合
        weight_combinations = [
            OptimizationWeights(10.0, 5.0, 0.3, 2.0, 1.0),  # 平衡
            OptimizationWeights(20.0, 3.0, 0.1, 1.0, 0.5),  # 重视水位
            OptimizationWeights(5.0, 10.0, 0.5, 1.0, 0.5),  # 重视平滑
            OptimizationWeights(5.0, 5.0, 1.0, 5.0, 0.2),   # 重视供水
        ]
        
        for weights in weight_combinations:
            self.weights = weights
            q_in, q_out, costs = self._solve_weighted_sum(
                current_level, q_in_prev, demand_forecast
            )
            pareto_solutions.append((q_in, q_out, costs))
        
        # 选择Pareto最优解（这里简化为选择水位跟踪最好的）
        best_solution = min(pareto_solutions, 
                           key=lambda x: x[2].get(ObjectiveType.LEVEL_TRACKING, float('inf')))
        
        return best_solution[0], best_solution[1], best_solution[2]
    
    def _solve_adaptive(self, current_level: float, q_in_prev: float,
                       demand_forecast: List[float]) -> Tuple[float, float, Dict]:
        """
        自适应权重方法
        
        根据历史性能动态调整权重
        """
        # 如果有历史记录，分析性能
        if len(self.cost_history[ObjectiveType.LEVEL_TRACKING]) > 5:
            # 计算各目标的平均成本
            avg_costs = {
                obj: np.mean(history[-5:])
                for obj, history in self.cost_history.items()
                if len(history) > 0
            }
            
            # 自适应调整权重：成本高的目标增加权重
            if ObjectiveType.LEVEL_TRACKING in avg_costs:
                level_cost = avg_costs[ObjectiveType.LEVEL_TRACKING]
                if level_cost > 1.0:  # 水位偏差大
                    self.weights.level_tracking = min(20.0, self.weights.level_tracking * 1.2)
                elif level_cost < 0.1:  # 水位控制好
                    self.weights.level_tracking = max(5.0, self.weights.level_tracking * 0.9)
            
            if ObjectiveType.WATER_DELIVERY in avg_costs:
                delivery_cost = avg_costs[ObjectiveType.WATER_DELIVERY]
                if delivery_cost > 2.0:  # 供水偏差大
                    self.weights.water_delivery = min(5.0, self.weights.water_delivery * 1.3)
        
        # 使用调整后的权重求解
        q_in, q_out, costs = self._solve_weighted_sum(
            current_level, q_in_prev, demand_forecast
        )
        
        # 记录历史
        for obj, cost in costs.items():
            self.cost_history[obj].append(cost)
            # 保留最近50个
            if len(self.cost_history[obj]) > 50:
                self.cost_history[obj].pop(0)
        
        return q_in, q_out, costs
    
    def set_weights(self, weights: OptimizationWeights):
        """设置优化权重"""
        self.weights = weights
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if not self.cost_history[ObjectiveType.LEVEL_TRACKING]:
            return {}
        
        metrics = {}
        for obj, history in self.cost_history.items():
            if len(history) > 0:
                metrics[obj.value] = {
                    'mean': float(np.mean(history)),
                    'std': float(np.std(history)),
                    'min': float(np.min(history)),
                    'max': float(np.max(history))
                }
        
        return metrics


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "多目标MPC演示")
    print("="*70)
    
    # 创建控制器
    controller = MultiObjectiveMPC(pool_id=0, horizon=10)
    
    # 测试不同优化模式
    current_level = 3.2
    q_in_prev = 5.0
    demand = [5.0 + np.sin(k*0.5) for k in range(10)]
    
    print("\n1. 加权求和模式...")
    q_in, q_out, costs = controller.solve(current_level, q_in_prev, demand, 
                                          mode='weighted_sum')
    print(f"   最优控制: 入流={q_in:.2f}, 出流={q_out:.2f}")
    print(f"   目标成本:")
    for obj, cost in costs.items():
        print(f"     {obj.value}: {cost:.4f}")
    
    print("\n2. Pareto优化模式...")
    q_in, q_out, costs = controller.solve(current_level, q_in_prev, demand,
                                          mode='pareto')
    print(f"   Pareto最优: 入流={q_in:.2f}, 出流={q_out:.2f}")
    
    print("\n3. 自适应权重模式...")
    for i in range(5):
        q_in, q_out, costs = controller.solve(current_level, q_in_prev, demand,
                                              mode='adaptive')
        current_level += (q_in - q_out) * controller.dt / controller.area
        q_in_prev = q_in
        print(f"   步骤{i+1}: 入流={q_in:.2f}, 水位={current_level:.3f}")
    
    print("\n4. 性能指标...")
    metrics = controller.get_performance_metrics()
    for obj_name, stats in metrics.items():
        print(f"   {obj_name}:")
        print(f"     均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

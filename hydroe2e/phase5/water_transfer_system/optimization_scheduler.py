"""
优化调度与水量分配系统
Optimization Scheduling and Water Allocation System

功能:
1. 多目标优化 - 水位、流量、能耗、供水保证
2. 水量分配 - 优先级分配、公平分配、比例分配
3. 约束处理 - 容量、流量、时间约束
4. 调度策略 - 日调度、周调度、应急调度
5. 协调优化 - 区域协调、全线优化

作者: AI Assistant
日期: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum, auto
import math
from collections import defaultdict


# ============ 类型定义 ============

class OptimizationObjective(Enum):
    """优化目标"""
    LEVEL_TRACKING = auto()     # 水位跟踪
    FLOW_SMOOTHNESS = auto()    # 流量平稳
    ENERGY_EFFICIENCY = auto()  # 能耗最小
    WATER_DELIVERY = auto()     # 供水保证
    STORAGE_BALANCE = auto()    # 蓄量平衡
    SAFETY_MARGIN = auto()      # 安全裕度


class AllocationStrategy(Enum):
    """分配策略"""
    PRIORITY = auto()           # 优先级分配
    PROPORTIONAL = auto()       # 比例分配
    EQUAL = auto()              # 均等分配
    DEMAND_BASED = auto()       # 需求导向
    EFFICIENCY_BASED = auto()   # 效率导向


class ScheduleType(Enum):
    """调度类型"""
    HOURLY = auto()             # 小时调度
    DAILY = auto()              # 日调度
    WEEKLY = auto()             # 周调度
    EMERGENCY = auto()          # 应急调度
    MAINTENANCE = auto()        # 检修调度


class ConstraintType(Enum):
    """约束类型"""
    LEVEL_BOUND = auto()        # 水位约束
    FLOW_BOUND = auto()         # 流量约束
    GATE_BOUND = auto()         # 闸门约束
    RATE_LIMIT = auto()         # 变化速率约束
    BALANCE = auto()            # 平衡约束
    TEMPORAL = auto()           # 时间约束


@dataclass
class WaterDemand:
    """用水需求"""
    demand_id: str
    location: str               # 取水口位置
    pool_id: int
    volume: float               # 需求量 [m³]
    flow_rate: float            # 需求流量 [m³/s]
    priority: int               # 优先级 1-10 (1最高)
    start_time: float           # 开始时间
    end_time: float             # 结束时间
    flexibility: float = 0.0    # 弹性度 0-1 (可延迟比例)
    min_flow: float = 0.0       # 最小流量
    category: str = "general"   # 类别 (agricultural, industrial, municipal)


@dataclass
class WaterSupply:
    """供水能力"""
    supply_id: str
    location: str
    pool_id: int
    max_flow: float             # 最大流量 [m³/s]
    available_volume: float     # 可用水量 [m³]
    cost_per_unit: float = 0.0  # 单位成本
    reliability: float = 1.0    # 可靠性


@dataclass
class OptimizationConstraint:
    """优化约束"""
    constraint_id: str
    constraint_type: ConstraintType
    target: str                 # 约束对象 (pool_id, gate_id, etc.)
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    penalty_weight: float = 1.0  # 违反惩罚权重
    is_hard: bool = True        # 硬约束/软约束


@dataclass
class ScheduleSlot:
    """调度时段"""
    slot_id: str
    start_time: float
    end_time: float
    target_levels: Dict[int, float]     # 目标水位
    target_flows: Dict[int, float]      # 目标流量
    gate_positions: Dict[int, float]    # 闸门开度
    allocations: Dict[str, float]       # 分配量


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    objective_value: float
    schedule: List[ScheduleSlot]
    allocations: Dict[str, float]
    constraint_violations: List[Tuple[str, float]]
    iterations: int
    computation_time: float


# ============ 多目标优化器 ============

class ObjectiveFunction:
    """目标函数"""

    def __init__(self):
        self.weights: Dict[OptimizationObjective, float] = {
            OptimizationObjective.LEVEL_TRACKING: 1.0,
            OptimizationObjective.FLOW_SMOOTHNESS: 0.5,
            OptimizationObjective.ENERGY_EFFICIENCY: 0.3,
            OptimizationObjective.WATER_DELIVERY: 2.0,
            OptimizationObjective.STORAGE_BALANCE: 0.5,
            OptimizationObjective.SAFETY_MARGIN: 0.8,
        }

        # 目标参考值
        self.target_levels: Dict[int, float] = {}
        self.target_flows: Dict[int, float] = {}

    def set_weights(self, weights: Dict[OptimizationObjective, float]):
        """设置权重"""
        self.weights.update(weights)

    def set_targets(self, levels: Dict[int, float], flows: Dict[int, float]):
        """设置目标值"""
        self.target_levels = levels
        self.target_flows = flows

    def evaluate(self, state: Dict[str, Any]) -> float:
        """评估目标函数值"""
        total = 0.0

        # 水位跟踪
        if OptimizationObjective.LEVEL_TRACKING in self.weights:
            level_cost = self._level_tracking_cost(state)
            total += self.weights[OptimizationObjective.LEVEL_TRACKING] * level_cost

        # 流量平稳
        if OptimizationObjective.FLOW_SMOOTHNESS in self.weights:
            flow_cost = self._flow_smoothness_cost(state)
            total += self.weights[OptimizationObjective.FLOW_SMOOTHNESS] * flow_cost

        # 能耗
        if OptimizationObjective.ENERGY_EFFICIENCY in self.weights:
            energy_cost = self._energy_cost(state)
            total += self.weights[OptimizationObjective.ENERGY_EFFICIENCY] * energy_cost

        # 供水保证
        if OptimizationObjective.WATER_DELIVERY in self.weights:
            delivery_cost = self._delivery_cost(state)
            total += self.weights[OptimizationObjective.WATER_DELIVERY] * delivery_cost

        # 蓄量平衡
        if OptimizationObjective.STORAGE_BALANCE in self.weights:
            balance_cost = self._storage_balance_cost(state)
            total += self.weights[OptimizationObjective.STORAGE_BALANCE] * balance_cost

        # 安全裕度
        if OptimizationObjective.SAFETY_MARGIN in self.weights:
            safety_cost = self._safety_margin_cost(state)
            total += self.weights[OptimizationObjective.SAFETY_MARGIN] * safety_cost

        return total

    def _level_tracking_cost(self, state: Dict) -> float:
        """水位跟踪成本"""
        cost = 0.0
        levels = state.get('levels', {})

        for pool_id, target in self.target_levels.items():
            actual = levels.get(pool_id, target)
            cost += (actual - target) ** 2

        return cost

    def _flow_smoothness_cost(self, state: Dict) -> float:
        """流量平稳成本"""
        cost = 0.0
        flow_changes = state.get('flow_changes', {})

        for gate_id, change in flow_changes.items():
            cost += change ** 2

        return cost

    def _energy_cost(self, state: Dict) -> float:
        """能耗成本"""
        cost = 0.0
        gate_movements = state.get('gate_movements', {})

        for gate_id, movement in gate_movements.items():
            cost += abs(movement) * 0.1  # 闸门移动耗能

        return cost

    def _delivery_cost(self, state: Dict) -> float:
        """供水缺口成本"""
        cost = 0.0
        deliveries = state.get('deliveries', {})
        demands = state.get('demands', {})

        for demand_id, demand in demands.items():
            delivered = deliveries.get(demand_id, 0)
            shortfall = max(0, demand - delivered)
            cost += shortfall ** 2

        return cost

    def _storage_balance_cost(self, state: Dict) -> float:
        """蓄量平衡成本"""
        cost = 0.0
        storages = state.get('storages', {})

        if len(storages) < 2:
            return 0.0

        mean_storage = sum(storages.values()) / len(storages)
        for pool_id, storage in storages.items():
            cost += (storage - mean_storage) ** 2

        return cost

    def _safety_margin_cost(self, state: Dict) -> float:
        """安全裕度成本"""
        cost = 0.0
        levels = state.get('levels', {})
        limits = state.get('level_limits', {})

        for pool_id, level in levels.items():
            if pool_id in limits:
                min_level, max_level = limits[pool_id]
                # 接近边界时增加成本
                margin_to_min = level - min_level
                margin_to_max = max_level - level

                if margin_to_min < 0.5:
                    cost += (0.5 - margin_to_min) ** 2
                if margin_to_max < 0.5:
                    cost += (0.5 - margin_to_max) ** 2

        return cost


class ConstraintHandler:
    """约束处理器"""

    def __init__(self):
        self.constraints: List[OptimizationConstraint] = []
        self.violation_history: List[Tuple[str, float]] = []

    def add_constraint(self, constraint: OptimizationConstraint):
        """添加约束"""
        self.constraints.append(constraint)

    def clear_constraints(self):
        """清除约束"""
        self.constraints.clear()

    def check_feasibility(self, state: Dict) -> Tuple[bool, List[Tuple[str, float]]]:
        """检查可行性"""
        violations = []

        for constraint in self.constraints:
            violation = self._check_constraint(constraint, state)
            if violation > 0:
                violations.append((constraint.constraint_id, violation))

                if constraint.is_hard:
                    return False, violations

        return len(violations) == 0, violations

    def compute_penalty(self, state: Dict) -> float:
        """计算违反惩罚"""
        total_penalty = 0.0

        for constraint in self.constraints:
            violation = self._check_constraint(constraint, state)
            if violation > 0:
                total_penalty += constraint.penalty_weight * violation ** 2

        return total_penalty

    def project_to_feasible(self, state: Dict) -> Dict:
        """投影到可行域"""
        projected = state.copy()

        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.LEVEL_BOUND:
                key = f'level_{constraint.target}'
                if key in projected:
                    value = projected[key]
                    if constraint.lower_bound is not None:
                        value = max(value, constraint.lower_bound)
                    if constraint.upper_bound is not None:
                        value = min(value, constraint.upper_bound)
                    projected[key] = value

            elif constraint.constraint_type == ConstraintType.FLOW_BOUND:
                key = f'flow_{constraint.target}'
                if key in projected:
                    value = projected[key]
                    if constraint.lower_bound is not None:
                        value = max(value, constraint.lower_bound)
                    if constraint.upper_bound is not None:
                        value = min(value, constraint.upper_bound)
                    projected[key] = value

            elif constraint.constraint_type == ConstraintType.GATE_BOUND:
                key = f'gate_{constraint.target}'
                if key in projected:
                    value = projected[key]
                    if constraint.lower_bound is not None:
                        value = max(value, constraint.lower_bound)
                    if constraint.upper_bound is not None:
                        value = min(value, constraint.upper_bound)
                    projected[key] = value

        return projected

    def _check_constraint(self, constraint: OptimizationConstraint,
                          state: Dict) -> float:
        """检查单个约束违反程度"""
        violation = 0.0

        if constraint.constraint_type == ConstraintType.LEVEL_BOUND:
            value = state.get('levels', {}).get(int(constraint.target), 0)
        elif constraint.constraint_type == ConstraintType.FLOW_BOUND:
            value = state.get('flows', {}).get(int(constraint.target), 0)
        elif constraint.constraint_type == ConstraintType.GATE_BOUND:
            value = state.get('gates', {}).get(int(constraint.target), 0)
        else:
            return 0.0

        if constraint.lower_bound is not None and value < constraint.lower_bound:
            violation += constraint.lower_bound - value
        if constraint.upper_bound is not None and value > constraint.upper_bound:
            violation += value - constraint.upper_bound

        return violation


class MultiObjectiveOptimizer:
    """多目标优化器"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools
        self.objective = ObjectiveFunction()
        self.constraint_handler = ConstraintHandler()

        # 优化参数
        self.max_iterations = 100
        self.tolerance = 1e-6
        self.step_size = 0.1

    def optimize(self, initial_state: Dict,
                 horizon: int = 24) -> OptimizationResult:
        """执行优化"""
        import time
        start_time = time.time()

        # 初始化
        current_state = initial_state.copy()
        best_state = current_state.copy()
        best_cost = float('inf')

        iterations = 0
        for iteration in range(self.max_iterations):
            iterations += 1

            # 计算目标值和梯度
            cost = self.objective.evaluate(current_state)
            penalty = self.constraint_handler.compute_penalty(current_state)
            total_cost = cost + penalty

            # 更新最优
            if total_cost < best_cost:
                best_cost = total_cost
                best_state = current_state.copy()

            # 梯度下降步
            gradient = self._compute_gradient(current_state)
            for key in current_state:
                if key in gradient:
                    current_state[key] -= self.step_size * gradient[key]

            # 投影到可行域
            current_state = self.constraint_handler.project_to_feasible(current_state)

            # 收敛检查
            if abs(total_cost - best_cost) < self.tolerance:
                break

        # 检查约束违反
        _, violations = self.constraint_handler.check_feasibility(best_state)

        computation_time = time.time() - start_time

        # 构建调度结果
        schedule = self._build_schedule(best_state, horizon)

        return OptimizationResult(
            success=len(violations) == 0,
            objective_value=best_cost,
            schedule=schedule,
            allocations={},
            constraint_violations=violations,
            iterations=iterations,
            computation_time=computation_time,
        )

    def _compute_gradient(self, state: Dict) -> Dict[str, float]:
        """计算梯度（数值微分）"""
        gradient = {}
        epsilon = 1e-6

        base_cost = self.objective.evaluate(state)

        for key, value in state.items():
            if isinstance(value, (int, float)):
                perturbed = state.copy()
                perturbed[key] = value + epsilon
                perturbed_cost = self.objective.evaluate(perturbed)
                gradient[key] = (perturbed_cost - base_cost) / epsilon

        return gradient

    def _build_schedule(self, state: Dict, horizon: int) -> List[ScheduleSlot]:
        """构建调度计划"""
        schedule = []

        for t in range(horizon):
            slot = ScheduleSlot(
                slot_id=f"SLOT_{t:03d}",
                start_time=float(t),
                end_time=float(t + 1),
                target_levels={i: state.get('levels', {}).get(i, 2.5)
                               for i in range(self.num_pools)},
                target_flows={i: state.get('flows', {}).get(i, 10.0)
                              for i in range(self.num_pools)},
                gate_positions={i: state.get('gates', {}).get(i, 0.5)
                                for i in range(self.num_pools + 1)},
                allocations={},
            )
            schedule.append(slot)

        return schedule


# ============ 水量分配器 ============

class WaterAllocator:
    """水量分配器"""

    def __init__(self):
        self.demands: List[WaterDemand] = []
        self.supplies: List[WaterSupply] = []
        self.strategy = AllocationStrategy.PRIORITY

    def add_demand(self, demand: WaterDemand):
        """添加需求"""
        self.demands.append(demand)

    def add_supply(self, supply: WaterSupply):
        """添加供给"""
        self.supplies.append(supply)

    def clear(self):
        """清除需求和供给"""
        self.demands.clear()
        self.supplies.clear()

    def set_strategy(self, strategy: AllocationStrategy):
        """设置分配策略"""
        self.strategy = strategy

    def allocate(self, available_water: float,
                 current_time: float) -> Dict[str, float]:
        """分配水量"""
        if self.strategy == AllocationStrategy.PRIORITY:
            return self._priority_allocation(available_water, current_time)
        elif self.strategy == AllocationStrategy.PROPORTIONAL:
            return self._proportional_allocation(available_water, current_time)
        elif self.strategy == AllocationStrategy.EQUAL:
            return self._equal_allocation(available_water, current_time)
        elif self.strategy == AllocationStrategy.DEMAND_BASED:
            return self._demand_based_allocation(available_water, current_time)
        elif self.strategy == AllocationStrategy.EFFICIENCY_BASED:
            return self._efficiency_allocation(available_water, current_time)
        else:
            return self._priority_allocation(available_water, current_time)

    def _priority_allocation(self, available: float,
                             current_time: float) -> Dict[str, float]:
        """优先级分配"""
        allocations = {}
        remaining = available

        # 按优先级排序（优先级数值小的优先）
        active_demands = [d for d in self.demands
                          if d.start_time <= current_time <= d.end_time]
        sorted_demands = sorted(active_demands, key=lambda x: x.priority)

        for demand in sorted_demands:
            needed = demand.volume
            allocated = min(needed, remaining)
            allocations[demand.demand_id] = allocated
            remaining -= allocated

            if remaining <= 0:
                break

        return allocations

    def _proportional_allocation(self, available: float,
                                  current_time: float) -> Dict[str, float]:
        """比例分配"""
        allocations = {}

        active_demands = [d for d in self.demands
                          if d.start_time <= current_time <= d.end_time]

        total_demand = sum(d.volume for d in active_demands)

        if total_demand == 0:
            return allocations

        ratio = min(1.0, available / total_demand)

        for demand in active_demands:
            allocations[demand.demand_id] = demand.volume * ratio

        return allocations

    def _equal_allocation(self, available: float,
                          current_time: float) -> Dict[str, float]:
        """均等分配"""
        allocations = {}

        active_demands = [d for d in self.demands
                          if d.start_time <= current_time <= d.end_time]

        if not active_demands:
            return allocations

        per_demand = available / len(active_demands)

        for demand in active_demands:
            allocations[demand.demand_id] = min(per_demand, demand.volume)

        return allocations

    def _demand_based_allocation(self, available: float,
                                  current_time: float) -> Dict[str, float]:
        """需求导向分配（考虑弹性）"""
        allocations = {}
        remaining = available

        active_demands = [d for d in self.demands
                          if d.start_time <= current_time <= d.end_time]

        # 先满足刚性需求
        rigid_demands = sorted(
            [d for d in active_demands if d.flexibility < 0.3],
            key=lambda x: x.priority
        )

        for demand in rigid_demands:
            needed = demand.min_flow * (demand.end_time - demand.start_time)
            allocated = min(needed, remaining)
            allocations[demand.demand_id] = allocated
            remaining -= allocated

        # 再分配弹性需求
        flexible_demands = [d for d in active_demands if d.flexibility >= 0.3]
        if flexible_demands and remaining > 0:
            per_flexible = remaining / len(flexible_demands)
            for demand in flexible_demands:
                allocations[demand.demand_id] = min(per_flexible, demand.volume)

        return allocations

    def _efficiency_allocation(self, available: float,
                                current_time: float) -> Dict[str, float]:
        """效率导向分配（最小化损失）"""
        allocations = {}

        active_demands = [d for d in self.demands
                          if d.start_time <= current_time <= d.end_time]

        # 按效率（优先级/需求量）排序
        sorted_demands = sorted(
            active_demands,
            key=lambda x: x.priority / max(x.volume, 0.001)
        )

        remaining = available
        for demand in sorted_demands:
            allocated = min(demand.volume, remaining)
            allocations[demand.demand_id] = allocated
            remaining -= allocated

            if remaining <= 0:
                break

        return allocations

    def get_satisfaction_rate(self, allocations: Dict[str, float]) -> float:
        """计算满足率"""
        total_demand = sum(d.volume for d in self.demands)
        total_allocated = sum(allocations.values())

        if total_demand == 0:
            return 1.0

        return total_allocated / total_demand

    def get_allocation_summary(self, allocations: Dict[str, float]) -> Dict[str, Any]:
        """获取分配汇总"""
        total_demand = sum(d.volume for d in self.demands)
        total_allocated = sum(allocations.values())

        by_category = defaultdict(float)
        by_priority = defaultdict(float)

        for demand in self.demands:
            allocated = allocations.get(demand.demand_id, 0)
            by_category[demand.category] += allocated
            by_priority[demand.priority] += allocated

        return {
            'total_demand': total_demand,
            'total_allocated': total_allocated,
            'satisfaction_rate': total_allocated / max(total_demand, 0.001),
            'by_category': dict(by_category),
            'by_priority': dict(by_priority),
            'num_demands': len(self.demands),
            'num_satisfied': sum(1 for d in self.demands
                                 if allocations.get(d.demand_id, 0) >= d.volume * 0.9),
        }


# ============ 调度策略生成器 ============

class ScheduleGenerator:
    """调度策略生成器"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools

        # 默认参数
        self.target_levels = {i: 2.5 for i in range(num_pools)}
        self.level_limits = {i: (1.0, 4.0) for i in range(num_pools)}
        self.flow_limits = {i: (0.0, 100.0) for i in range(num_pools)}

    def generate_daily_schedule(self,
                                demands: List[WaterDemand],
                                initial_levels: Dict[int, float],
                                upstream_forecast: List[float]) -> List[ScheduleSlot]:
        """生成日调度"""
        schedule = []

        current_levels = initial_levels.copy()

        for hour in range(24):
            # 确定该时段的目标
            target_levels = self._compute_target_levels(
                hour, current_levels, demands
            )

            # 确定流量目标
            target_flows = self._compute_target_flows(
                hour, demands, upstream_forecast
            )

            # 确定闸门开度
            gate_positions = self._compute_gate_positions(
                current_levels, target_levels, target_flows
            )

            slot = ScheduleSlot(
                slot_id=f"H{hour:02d}",
                start_time=float(hour),
                end_time=float(hour + 1),
                target_levels=target_levels,
                target_flows=target_flows,
                gate_positions=gate_positions,
                allocations={},
            )

            schedule.append(slot)

            # 更新水位预测
            current_levels = self._predict_levels(
                current_levels, target_flows, 1.0
            )

        return schedule

    def generate_emergency_schedule(self,
                                    emergency_type: str,
                                    current_levels: Dict[int, float],
                                    affected_pools: List[int]) -> List[ScheduleSlot]:
        """生成应急调度"""
        schedule = []

        if emergency_type == 'flood':
            # 洪水应急：快速降低水位
            for step in range(12):  # 12小时应急
                target_levels = {}
                for pool_id in range(self.num_pools):
                    if pool_id in affected_pools:
                        # 逐步降低到安全水位
                        target_levels[pool_id] = max(
                            self.level_limits[pool_id][0] + 0.5,
                            current_levels.get(pool_id, 2.5) - 0.2
                        )
                    else:
                        target_levels[pool_id] = self.target_levels[pool_id]

                gate_positions = {i: 0.8 if i in affected_pools else 0.5
                                  for i in range(self.num_pools + 1)}

                slot = ScheduleSlot(
                    slot_id=f"EMG_FLOOD_{step:02d}",
                    start_time=float(step),
                    end_time=float(step + 1),
                    target_levels=target_levels,
                    target_flows={i: 30.0 if i in affected_pools else 10.0
                                  for i in range(self.num_pools)},
                    gate_positions=gate_positions,
                    allocations={},
                )
                schedule.append(slot)

        elif emergency_type == 'drought':
            # 干旱应急：节水调度
            for step in range(24):
                target_levels = {}
                for pool_id in range(self.num_pools):
                    # 保持较低水位以减少蒸发
                    target_levels[pool_id] = (
                        self.level_limits[pool_id][0] +
                        0.3 * (self.level_limits[pool_id][1] -
                               self.level_limits[pool_id][0])
                    )

                slot = ScheduleSlot(
                    slot_id=f"EMG_DROUGHT_{step:02d}",
                    start_time=float(step),
                    end_time=float(step + 1),
                    target_levels=target_levels,
                    target_flows={i: 5.0 for i in range(self.num_pools)},
                    gate_positions={i: 0.3 for i in range(self.num_pools + 1)},
                    allocations={},
                )
                schedule.append(slot)

        elif emergency_type == 'pollution':
            # 污染应急：隔离调度
            for step in range(6):
                target_flows = {}
                gate_positions = {}

                for pool_id in range(self.num_pools):
                    if pool_id in affected_pools:
                        target_flows[pool_id] = 0.0  # 停止流动
                    else:
                        target_flows[pool_id] = 10.0

                for gate_id in range(self.num_pools + 1):
                    if gate_id in affected_pools or gate_id - 1 in affected_pools:
                        gate_positions[gate_id] = 0.0  # 关闭闸门
                    else:
                        gate_positions[gate_id] = 0.5

                slot = ScheduleSlot(
                    slot_id=f"EMG_POLLUTION_{step:02d}",
                    start_time=float(step),
                    end_time=float(step + 1),
                    target_levels=self.target_levels.copy(),
                    target_flows=target_flows,
                    gate_positions=gate_positions,
                    allocations={},
                )
                schedule.append(slot)

        return schedule

    def _compute_target_levels(self, hour: int,
                               current_levels: Dict[int, float],
                               demands: List[WaterDemand]) -> Dict[int, float]:
        """计算目标水位"""
        target_levels = self.target_levels.copy()

        # 根据需求调整
        for demand in demands:
            if demand.start_time <= hour < demand.end_time:
                pool_id = demand.pool_id
                # 有大需求时保持较高水位
                if demand.flow_rate > 20:
                    target_levels[pool_id] = min(
                        self.level_limits[pool_id][1] - 0.3,
                        target_levels[pool_id] + 0.3
                    )

        # 日内调节：白天用水高峰保持较高水位
        if 8 <= hour <= 20:
            for pool_id in target_levels:
                target_levels[pool_id] = min(
                    self.level_limits[pool_id][1] - 0.2,
                    target_levels[pool_id] + 0.2
                )

        return target_levels

    def _compute_target_flows(self, hour: int,
                              demands: List[WaterDemand],
                              upstream_forecast: List[float]) -> Dict[int, float]:
        """计算目标流量"""
        target_flows = {i: 10.0 for i in range(self.num_pools)}

        # 上游来水
        if hour < len(upstream_forecast):
            base_flow = upstream_forecast[hour]
        else:
            base_flow = 50.0

        target_flows[0] = base_flow

        # 根据需求调整
        for demand in demands:
            if demand.start_time <= hour < demand.end_time:
                pool_id = demand.pool_id
                target_flows[pool_id] = max(
                    target_flows[pool_id],
                    demand.flow_rate
                )

        return target_flows

    def _compute_gate_positions(self, current_levels: Dict[int, float],
                                target_levels: Dict[int, float],
                                target_flows: Dict[int, float]) -> Dict[int, float]:
        """计算闸门开度"""
        gate_positions = {}

        for gate_id in range(self.num_pools + 1):
            if gate_id == 0:
                # 首闸：根据上游来水调节
                gate_positions[gate_id] = 0.6
            elif gate_id == self.num_pools:
                # 末闸：根据下游需求调节
                gate_positions[gate_id] = 0.5
            else:
                # 中间闸门：根据水位差调节
                pool_upstream = gate_id - 1
                pool_downstream = gate_id

                level_up = current_levels.get(pool_upstream, 2.5)
                level_down = current_levels.get(pool_downstream, 2.5)
                target_up = target_levels.get(pool_upstream, 2.5)

                # 水位高于目标则开大，低于则关小
                error = level_up - target_up
                base_position = 0.5 + 0.1 * error

                gate_positions[gate_id] = max(0.1, min(0.9, base_position))

        return gate_positions

    def _predict_levels(self, current_levels: Dict[int, float],
                        target_flows: Dict[int, float],
                        dt: float) -> Dict[int, float]:
        """预测水位变化"""
        new_levels = {}

        As = 5000.0  # 水面面积

        for pool_id in range(self.num_pools):
            inflow = target_flows.get(pool_id, 10.0)
            outflow = target_flows.get(pool_id + 1, 10.0) if pool_id < self.num_pools - 1 else inflow

            current = current_levels.get(pool_id, 2.5)
            dh = (inflow - outflow) * dt * 3600 / As  # 1小时
            new_levels[pool_id] = current + dh

            # 限幅
            new_levels[pool_id] = max(
                self.level_limits[pool_id][0],
                min(new_levels[pool_id], self.level_limits[pool_id][1])
            )

        return new_levels


# ============ 综合优化调度系统 ============

class OptimizationSchedulingSystem:
    """综合优化调度系统"""

    def __init__(self, num_pools: int = 5):
        self.num_pools = num_pools

        # 子系统
        self.optimizer = MultiObjectiveOptimizer(num_pools)
        self.allocator = WaterAllocator()
        self.scheduler = ScheduleGenerator(num_pools)

        # 当前状态
        self.current_schedule: List[ScheduleSlot] = []
        self.current_allocations: Dict[str, float] = {}

        # 配置
        self.schedule_type = ScheduleType.DAILY

    def configure_objectives(self, weights: Dict[OptimizationObjective, float]):
        """配置优化目标权重"""
        self.optimizer.objective.set_weights(weights)

    def add_constraint(self, constraint: OptimizationConstraint):
        """添加约束"""
        self.optimizer.constraint_handler.add_constraint(constraint)

    def add_demand(self, demand: WaterDemand):
        """添加需求"""
        self.allocator.add_demand(demand)

    def set_allocation_strategy(self, strategy: AllocationStrategy):
        """设置分配策略"""
        self.allocator.set_strategy(strategy)

    def generate_schedule(self,
                          initial_state: Dict[str, Any],
                          forecast: Dict[str, List[float]],
                          schedule_type: ScheduleType = ScheduleType.DAILY) -> OptimizationResult:
        """生成调度计划"""
        self.schedule_type = schedule_type

        # 设置优化目标
        self.optimizer.objective.set_targets(
            levels={i: 2.5 for i in range(self.num_pools)},
            flows={i: 10.0 for i in range(self.num_pools)},
        )

        # 运行优化
        result = self.optimizer.optimize(initial_state)

        if result.success:
            self.current_schedule = result.schedule

            # 计算水量分配
            available_water = sum(
                forecast.get('upstream', [50.0] * 24)
            ) * 3600  # 转换为体积

            self.current_allocations = self.allocator.allocate(
                available_water,
                initial_state.get('current_time', 0)
            )

            result.allocations = self.current_allocations

        return result

    def generate_emergency_schedule(self,
                                    emergency_type: str,
                                    current_state: Dict[str, Any],
                                    affected_pools: List[int]) -> List[ScheduleSlot]:
        """生成应急调度"""
        current_levels = {
            i: current_state.get('levels', {}).get(i, 2.5)
            for i in range(self.num_pools)
        }

        schedule = self.scheduler.generate_emergency_schedule(
            emergency_type, current_levels, affected_pools
        )

        self.current_schedule = schedule
        return schedule

    def get_current_slot(self, current_time: float) -> Optional[ScheduleSlot]:
        """获取当前时段调度"""
        for slot in self.current_schedule:
            if slot.start_time <= current_time < slot.end_time:
                return slot
        return None

    def evaluate_schedule(self, actual_state: Dict[str, Any]) -> Dict[str, Any]:
        """评估调度执行情况"""
        if not self.current_schedule:
            return {'error': 'No schedule available'}

        current_time = actual_state.get('current_time', 0)
        slot = self.get_current_slot(current_time)

        if not slot:
            return {'error': 'No slot for current time'}

        # 计算偏差
        level_errors = {}
        for pool_id, target in slot.target_levels.items():
            actual = actual_state.get('levels', {}).get(pool_id, target)
            level_errors[pool_id] = actual - target

        flow_errors = {}
        for pool_id, target in slot.target_flows.items():
            actual = actual_state.get('flows', {}).get(pool_id, target)
            flow_errors[pool_id] = actual - target

        gate_errors = {}
        for gate_id, target in slot.gate_positions.items():
            actual = actual_state.get('gates', {}).get(gate_id, target)
            gate_errors[gate_id] = actual - target

        return {
            'slot_id': slot.slot_id,
            'level_errors': level_errors,
            'flow_errors': flow_errors,
            'gate_errors': gate_errors,
            'max_level_error': max(abs(e) for e in level_errors.values()) if level_errors else 0,
            'max_flow_error': max(abs(e) for e in flow_errors.values()) if flow_errors else 0,
            'schedule_adherence': 1 - (
                sum(abs(e) for e in level_errors.values()) / max(len(level_errors), 1) / 2.5
            ),
        }

    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配汇总"""
        return self.allocator.get_allocation_summary(self.current_allocations)

    def get_schedule_summary(self) -> Dict[str, Any]:
        """获取调度汇总"""
        if not self.current_schedule:
            return {'error': 'No schedule available'}

        return {
            'schedule_type': self.schedule_type.name,
            'num_slots': len(self.current_schedule),
            'time_range': (
                self.current_schedule[0].start_time,
                self.current_schedule[-1].end_time
            ),
            'avg_target_level': sum(
                sum(s.target_levels.values()) / len(s.target_levels)
                for s in self.current_schedule
            ) / len(self.current_schedule),
            'avg_target_flow': sum(
                sum(s.target_flows.values()) / len(s.target_flows)
                for s in self.current_schedule
            ) / len(self.current_schedule),
        }


# ============ 导出 ============

__all__ = [
    # 枚举
    'OptimizationObjective',
    'AllocationStrategy',
    'ScheduleType',
    'ConstraintType',

    # 数据类
    'WaterDemand',
    'WaterSupply',
    'OptimizationConstraint',
    'ScheduleSlot',
    'OptimizationResult',

    # 优化器
    'ObjectiveFunction',
    'ConstraintHandler',
    'MultiObjectiveOptimizer',

    # 分配器
    'WaterAllocator',

    # 调度器
    'ScheduleGenerator',

    # 综合系统
    'OptimizationSchedulingSystem',
]

"""
高保真水力学仿真器
High-Fidelity Hydraulic Simulator

核心功能:
1. 基于IDZ模型的动态仿真
2. 闸门动态响应模拟
3. 长时间连续运行支持
4. 与级联控制系统集成
5. 场景注入与回放

物理模型:
- 圣维南方程简化 (IDZ: Integrator Delay Zero)
- 水位-流量动态关系
- 闸门流量特性曲线
- 渠道糙率与断面参数
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
from collections import deque
import json

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity,
    CanalPoolConfig, SpecialStructure, StructureType,
)
from .local_pool_scenarios import L1ScenarioType, L1ScenarioEvent

logger = logging.getLogger(__name__)


# ==============================================================================
# 物理常数与参数
# ==============================================================================

class PhysicalConstants:
    """物理常数"""
    GRAVITY = 9.81              # 重力加速度 [m/s²]
    WATER_DENSITY = 1000.0      # 水密度 [kg/m³]
    MANNING_N = 0.014           # 曼宁糙率系数 (混凝土衬砌)


@dataclass
class PoolPhysicalParams:
    """渠池物理参数"""
    pool_id: int
    length: float = 5000.0          # 渠池长度 [m]
    bottom_width: float = 15.0      # 底宽 [m]
    side_slope: float = 2.0         # 边坡比 (水平:垂直)
    bed_slope: float = 1e-4         # 底坡
    manning_n: float = 0.014        # 曼宁系数

    # IDZ模型参数
    delay_time: float = 300.0       # 延迟时间 [s]
    integrator_gain: float = 1e-4   # 积分增益 [m/m³]

    # 水位约束
    min_level: float = 0.5          # 最小水位 [m]
    max_level: float = 6.0          # 最大水位 [m]
    target_level: float = 3.0       # 目标水位 [m]

    # 库容参数
    def get_surface_area(self, level: float) -> float:
        """计算水面面积"""
        top_width = self.bottom_width + 2 * self.side_slope * level
        return self.length * (self.bottom_width + top_width) / 2

    def get_storage(self, level: float) -> float:
        """计算库容"""
        top_width = self.bottom_width + 2 * self.side_slope * level
        cross_area = (self.bottom_width + top_width) * level / 2
        return self.length * cross_area


@dataclass
class GateParams:
    """闸门参数"""
    gate_id: str
    pool_upstream: int
    pool_downstream: int

    # 闸门尺寸
    width: float = 10.0             # 闸门宽度 [m]
    max_opening: float = 3.0        # 最大开度 [m]

    # 动态特性
    max_rate: float = 0.01          # 最大开启速率 [m/s]
    response_time: float = 60.0     # 响应时间常数 [s]

    # 流量系数
    discharge_coef: float = 0.6     # 流量系数

    # 当前状态
    current_opening: float = 1.5    # 当前开度 [m]
    target_opening: float = 1.5     # 目标开度 [m]


# ==============================================================================
# 渠池状态
# ==============================================================================

@dataclass
class PoolState:
    """渠池状态"""
    pool_id: int
    timestamp: float = 0.0

    # 水力状态
    water_level: float = 3.0        # 水位 [m]
    inflow: float = 50.0            # 入流 [m³/s]
    outflow: float = 50.0           # 出流 [m³/s]
    storage: float = 0.0            # 库容 [m³]

    # 闸门状态
    upstream_gate_opening: float = 1.5   # 上游闸门开度 [m]
    downstream_gate_opening: float = 1.5  # 下游闸门开度 [m]

    # 水质状态
    quality_index: float = 1.0      # 水质指数 [0-1]
    pollution_concentration: float = 0.0  # 污染浓度

    # 环境状态
    rainfall: float = 0.0           # 降雨强度 [mm/h]
    evaporation: float = 0.0        # 蒸发量 [mm/h]
    temperature: float = 15.0       # 水温 [°C]

    # 异常状态
    has_anomaly: bool = False
    anomaly_type: Optional[str] = None


@dataclass
class SimulationState:
    """仿真状态"""
    current_time: float = 0.0
    step_count: int = 0
    pool_states: Dict[int, PoolState] = field(default_factory=dict)
    gate_states: Dict[str, float] = field(default_factory=dict)

    # 历史记录
    history_length: int = 1000
    level_history: Dict[int, deque] = field(default_factory=dict)
    flow_history: Dict[int, deque] = field(default_factory=dict)


# ==============================================================================
# IDZ动态模型
# ==============================================================================

class IDZDynamicModel:
    """
    IDZ动态模型

    基于Integrator-Delay-Zero模型的渠池水力学仿真
    """

    def __init__(self, pool_params: PoolPhysicalParams):
        self.params = pool_params
        self.delay_steps = 0
        self._buffer_dt: Optional[float] = None
        self.delay_buffer: deque = deque(maxlen=1)

        # 状态
        self.current_level = pool_params.target_level
        self.current_inflow = 50.0
        self.current_outflow = 50.0
        self._configure_delay_buffer(dt=60.0, initial_inflow=self.current_inflow)

    def _configure_delay_buffer(self, dt: float, initial_inflow: Optional[float] = None) -> None:
        """Rebuild the inflow delay buffer when the effective delay resolution changes."""
        dt = max(float(dt), 1e-9)
        delay_steps = max(0, int(np.ceil(self.params.delay_time / dt)))
        initial = self.current_inflow if initial_inflow is None else float(initial_inflow)

        if self._buffer_dt == dt and self.delay_steps == delay_steps:
            return

        self.delay_steps = delay_steps
        self._buffer_dt = dt
        self.delay_buffer = deque(
            [initial] * (self.delay_steps + 1),
            maxlen=self.delay_steps + 1,
        )

    def step(self, dt: float, inflow: float, outflow: float) -> float:
        """
        执行一步仿真

        Args:
            dt: 时间步长 [s]
            inflow: 入流量 [m³/s]
            outflow: 出流量 [m³/s]

        Returns:
            新水位 [m]
        """
        self._configure_delay_buffer(dt, initial_inflow=self.current_inflow)
        self.delay_buffer.append(float(inflow))
        delayed_inflow = self.delay_buffer[0]

        # 计算水量平衡
        dV = (delayed_inflow - outflow) * dt

        # 计算水面面积
        A_s = self.params.get_surface_area(self.current_level)

        # 水位变化
        dZ = dV / A_s if A_s > 0 else 0

        # 更新水位
        new_level = self.current_level + dZ

        # 应用约束
        new_level = max(self.params.min_level, min(self.params.max_level, new_level))

        self.current_level = new_level
        self.current_inflow = delayed_inflow
        self.current_outflow = outflow

        return new_level

    def get_delayed_flow(self, flow_history: deque, delay_steps: int) -> float:
        """获取延迟流量"""
        if len(flow_history) > delay_steps:
            return flow_history[-delay_steps]
        elif len(flow_history) > 0:
            return flow_history[0]
        else:
            return self.current_inflow

    def get_delayed_inflow(self) -> float:
        """获取当前内部延迟缓冲对应的有效入流。"""
        return float(self.delay_buffer[0]) if self.delay_buffer else self.current_inflow


# ==============================================================================
# 闸门动态模型
# ==============================================================================

class GateDynamicModel:
    """
    闸门动态模型

    模拟闸门开度调节过程
    """

    def __init__(self, gate_params: GateParams):
        self.params = gate_params
        self.current_opening = gate_params.current_opening
        self.target_opening = gate_params.target_opening

        # 状态
        self.is_moving = False
        self.move_direction = 0  # 1: opening, -1: closing, 0: stopped

    def set_target(self, target: float):
        """设置目标开度"""
        self.target_opening = max(0, min(self.params.max_opening, target))
        if abs(self.target_opening - self.current_opening) > 0.001:
            self.is_moving = True
            self.move_direction = 1 if self.target_opening > self.current_opening else -1

    def step(self, dt: float) -> float:
        """
        执行一步仿真

        Returns:
            当前开度 [m]
        """
        if not self.is_moving:
            return self.current_opening

        # 计算最大移动量
        max_delta = self.params.max_rate * dt

        # 计算实际移动量
        delta = self.target_opening - self.current_opening
        if abs(delta) <= max_delta:
            self.current_opening = self.target_opening
            self.is_moving = False
            self.move_direction = 0
        else:
            self.current_opening += self.move_direction * max_delta

        # 应用约束
        self.current_opening = max(0, min(self.params.max_opening, self.current_opening))

        return self.current_opening

    def calculate_flow(self, upstream_level: float, downstream_level: float) -> float:
        """
        计算过闸流量

        使用闸孔出流公式
        """
        if self.current_opening <= 0:
            return 0.0

        h_up = upstream_level
        h_down = downstream_level
        e = self.current_opening  # 闸门开度
        b = self.params.width     # 闸门宽度
        C_d = self.params.discharge_coef

        # 判断流态
        if h_down < e:
            # 自由出流
            Q = C_d * b * e * np.sqrt(2 * PhysicalConstants.GRAVITY * h_up)
        else:
            # 淹没出流
            dh = max(0.01, h_up - h_down)
            Q = C_d * b * e * np.sqrt(2 * PhysicalConstants.GRAVITY * dh)

        return max(0, Q)


# ==============================================================================
# 全线水力学仿真器
# ==============================================================================

class FullLineHydraulicSimulator:
    """
    全线水力学仿真器

    仿真60个渠池的水力学动态
    """

    def __init__(self, num_pools: int = 60, dt: float = 60.0):
        self.num_pools = num_pools
        self.dt = dt  # 仿真步长 [s]

        # 渠池模型
        self.pool_params: Dict[int, PoolPhysicalParams] = {}
        self.pool_models: Dict[int, IDZDynamicModel] = {}

        # 闸门模型
        self.gate_params: Dict[str, GateParams] = {}
        self.gate_models: Dict[str, GateDynamicModel] = {}

        # 仿真状态
        self.state = SimulationState()

        # 边界条件 (需要在初始化状态前设置)
        self.upstream_inflow = 50.0     # 上游来水 [m³/s]
        self.downstream_level = 2.5     # 下游水位 [m]

        # 初始化
        self._initialize_pools()
        self._initialize_gates()
        self._initialize_state()

        # 场景注入
        self.active_scenarios: Dict[int, L1ScenarioEvent] = {}

        # 回调
        self.state_callback: Optional[Callable] = None

    def _initialize_pools(self):
        """初始化渠池"""
        for i in range(self.num_pools):
            # 根据位置调整参数
            params = PoolPhysicalParams(
                pool_id=i,
                length=5000 + np.random.uniform(-500, 500),  # 4.5-5.5km
                bottom_width=15.0 + np.random.uniform(-2, 2),
                delay_time=300 + i * 5,  # 延迟递增
            )
            self.pool_params[i] = params
            self.pool_models[i] = IDZDynamicModel(params)

    def _initialize_gates(self):
        """初始化闸门"""
        for i in range(self.num_pools):
            gate_id = f"GATE_{i}"
            params = GateParams(
                gate_id=gate_id,
                pool_upstream=i,
                pool_downstream=i + 1 if i < self.num_pools - 1 else -1,
            )
            self.gate_params[gate_id] = params
            self.gate_models[gate_id] = GateDynamicModel(params)

    def _initialize_state(self):
        """初始化状态"""
        self.state.current_time = 0.0
        self.state.step_count = 0

        for i in range(self.num_pools):
            # 初始水位：从上游到下游递减
            initial_level = 3.0 - i * 0.01
            self.state.pool_states[i] = PoolState(
                pool_id=i,
                water_level=initial_level,
                inflow=self.upstream_inflow,
                outflow=self.upstream_inflow,
            )
            self.state.level_history[i] = deque(maxlen=self.state.history_length)
            self.state.flow_history[i] = deque(maxlen=self.state.history_length)

        for gate_id in self.gate_models:
            self.state.gate_states[gate_id] = 1.5  # 初始开度

    def set_gate_opening(self, pool_id: int, opening: float):
        """设置闸门目标开度"""
        gate_id = f"GATE_{pool_id}"
        if gate_id in self.gate_models:
            self.gate_models[gate_id].set_target(opening)

    def set_upstream_inflow(self, inflow: float):
        """设置上游来水"""
        self.upstream_inflow = max(0, inflow)

    def inject_scenario(self, pool_id: int, scenario: L1ScenarioEvent):
        """注入场景"""
        self.active_scenarios[pool_id] = scenario
        logger.info(f"注入场景: 池{pool_id}, {scenario.scenario_type.value}")

    def remove_scenario(self, pool_id: int):
        """移除场景"""
        if pool_id in self.active_scenarios:
            del self.active_scenarios[pool_id]

    def step(self) -> Dict[int, PoolState]:
        """
        执行一步仿真

        Returns:
            所有池的状态
        """
        new_states = {}

        # 1. 更新闸门
        for gate_id, model in self.gate_models.items():
            opening = model.step(self.dt)
            self.state.gate_states[gate_id] = opening

        # 2. 计算流量 (从上游到下游)
        flows = [self.upstream_inflow]  # 第一个池的入流

        for i in range(self.num_pools):
            gate_id = f"GATE_{i}"
            if i < self.num_pools - 1:
                # 计算过闸流量
                up_level = self.state.pool_states[i].water_level
                down_level = self.state.pool_states[i + 1].water_level
                flow = self.gate_models[gate_id].calculate_flow(up_level, down_level)
            else:
                # 最后一个池，流向下游
                up_level = self.state.pool_states[i].water_level
                flow = self.gate_models[gate_id].calculate_flow(up_level, self.downstream_level)

            flows.append(flow)

        # 3. 更新水位
        for i in range(self.num_pools):
            inflow = flows[i]
            outflow = flows[i + 1]

            # 应用场景影响
            if i in self.active_scenarios:
                inflow, outflow = self._apply_scenario_effect(i, inflow, outflow)

            # IDZ模型更新
            new_level = self.pool_models[i].step(self.dt, inflow, outflow)

            # 更新状态
            pool_state = PoolState(
                pool_id=i,
                timestamp=self.state.current_time + self.dt,
                water_level=new_level,
                inflow=inflow,
                outflow=outflow,
                storage=self.pool_params[i].get_storage(new_level),
                upstream_gate_opening=self.state.gate_states.get(f"GATE_{i-1}", 0) if i > 0 else 0,
                downstream_gate_opening=self.state.gate_states.get(f"GATE_{i}", 0),
            )

            # 更新水质 (简化模型)
            if i in self.active_scenarios:
                scenario = self.active_scenarios[i]
                if scenario.scenario_type in [
                    L1ScenarioType.L1_POLLUTION_DETECTED,
                    L1ScenarioType.L1_POLLUTION_TRACKING,
                ]:
                    pool_state.quality_index = 0.5
                    pool_state.pollution_concentration = 0.3
                    pool_state.has_anomaly = True
                    pool_state.anomaly_type = "pollution"

            new_states[i] = pool_state

            # 记录历史
            self.state.level_history[i].append(new_level)
            self.state.flow_history[i].append(outflow)

        # 4. 更新仿真状态
        self.state.pool_states = new_states
        self.state.current_time += self.dt
        self.state.step_count += 1

        # 5. 回调
        if self.state_callback:
            self.state_callback(self.state)

        return new_states

    def _apply_scenario_effect(self,
                                pool_id: int,
                                inflow: float,
                                outflow: float) -> Tuple[float, float]:
        """应用场景影响"""
        scenario = self.active_scenarios[pool_id]

        if scenario.scenario_type == L1ScenarioType.L1_LEVEL_RAPID_RISE:
            # 水位快速上涨：增加入流
            inflow *= 1.5
        elif scenario.scenario_type == L1ScenarioType.L1_LEVEL_RAPID_DROP:
            # 水位快速下降：减少入流
            inflow *= 0.5
        elif scenario.scenario_type == L1ScenarioType.L1_GATE_STUCK:
            # 闸门卡住：出流固定
            pass  # 保持当前出流
        elif scenario.scenario_type == L1ScenarioType.L1_DISCHARGE_EMERGENCY:
            # 紧急退水：增加出流
            outflow *= 1.5

        return inflow, outflow

    def run(self, duration: float, callback: Optional[Callable] = None) -> List[SimulationState]:
        """
        运行仿真

        Args:
            duration: 仿真时长 [s]
            callback: 每步回调函数

        Returns:
            状态历史
        """
        if callback:
            self.state_callback = callback

        num_steps = int(duration / self.dt)
        history = []

        for _ in range(num_steps):
            self.step()
            # 保存状态快照
            if self.state.step_count % 10 == 0:  # 每10步保存一次
                history.append(self._snapshot_state())

        return history

    def _snapshot_state(self) -> Dict[str, Any]:
        """创建状态快照"""
        return {
            'time': self.state.current_time,
            'step': self.state.step_count,
            'levels': {i: s.water_level for i, s in self.state.pool_states.items()},
            'flows': {i: s.outflow for i, s in self.state.pool_states.items()},
            'gates': dict(self.state.gate_states),
        }

    def get_pool_state(self, pool_id: int) -> Optional[PoolState]:
        """获取池状态"""
        return self.state.pool_states.get(pool_id)

    def get_all_levels(self) -> Dict[int, float]:
        """获取所有水位"""
        return {i: s.water_level for i, s in self.state.pool_states.items()}

    def get_all_flows(self) -> Dict[int, float]:
        """获取所有流量"""
        return {i: s.outflow for i, s in self.state.pool_states.items()}


# ==============================================================================
# 场景回放系统
# ==============================================================================

@dataclass
class SimulationRecord:
    """仿真记录"""
    record_id: str
    start_time: float
    end_time: float
    dt: float
    num_pools: int

    # 时序数据
    timestamps: List[float] = field(default_factory=list)
    level_data: Dict[int, List[float]] = field(default_factory=dict)
    flow_data: Dict[int, List[float]] = field(default_factory=dict)
    gate_data: Dict[str, List[float]] = field(default_factory=dict)

    # 场景事件
    scenario_events: List[Dict] = field(default_factory=list)

    # 控制动作
    control_actions: List[Dict] = field(default_factory=list)


class SimulationRecorder:
    """仿真记录器"""

    def __init__(self, simulator: FullLineHydraulicSimulator):
        self.simulator = simulator
        self.is_recording = False
        self.current_record: Optional[SimulationRecord] = None

    def start_recording(self, record_id: str):
        """开始记录"""
        self.current_record = SimulationRecord(
            record_id=record_id,
            start_time=self.simulator.state.current_time,
            end_time=0.0,
            dt=self.simulator.dt,
            num_pools=self.simulator.num_pools,
        )
        for i in range(self.simulator.num_pools):
            self.current_record.level_data[i] = []
            self.current_record.flow_data[i] = []
        for gate_id in self.simulator.gate_models:
            self.current_record.gate_data[gate_id] = []

        self.is_recording = True
        self.simulator.state_callback = self._record_callback

    def stop_recording(self) -> SimulationRecord:
        """停止记录"""
        self.is_recording = False
        self.simulator.state_callback = None
        if self.current_record:
            self.current_record.end_time = self.simulator.state.current_time
        return self.current_record

    def _record_callback(self, state: SimulationState):
        """记录回调"""
        if not self.is_recording or not self.current_record:
            return

        self.current_record.timestamps.append(state.current_time)

        for i, pool_state in state.pool_states.items():
            self.current_record.level_data[i].append(pool_state.water_level)
            self.current_record.flow_data[i].append(pool_state.outflow)

        for gate_id, opening in state.gate_states.items():
            self.current_record.gate_data[gate_id].append(opening)

    def record_scenario_event(self, pool_id: int, scenario: L1ScenarioEvent):
        """记录场景事件"""
        if self.current_record:
            self.current_record.scenario_events.append({
                'time': self.simulator.state.current_time,
                'pool_id': pool_id,
                'scenario_type': scenario.scenario_type.value,
                'severity': scenario.severity.value,
            })

    def record_control_action(self, pool_id: int, action_type: str, value: float):
        """记录控制动作"""
        if self.current_record:
            self.current_record.control_actions.append({
                'time': self.simulator.state.current_time,
                'pool_id': pool_id,
                'action_type': action_type,
                'value': value,
            })


class SimulationReplayer:
    """仿真回放器"""

    def __init__(self, record: SimulationRecord):
        self.record = record
        self.current_index = 0

    def reset(self):
        """重置回放"""
        self.current_index = 0

    def get_state_at(self, time: float) -> Optional[Dict[str, Any]]:
        """获取指定时间的状态"""
        if not self.record.timestamps:
            return None

        # 找到最近的时间点
        for i, t in enumerate(self.record.timestamps):
            if t >= time:
                return self._get_state_at_index(i)

        return self._get_state_at_index(len(self.record.timestamps) - 1)

    def _get_state_at_index(self, index: int) -> Dict[str, Any]:
        """获取指定索引的状态"""
        return {
            'time': self.record.timestamps[index],
            'levels': {i: self.record.level_data[i][index]
                       for i in range(self.record.num_pools)
                       if i in self.record.level_data and index < len(self.record.level_data[i])},
            'flows': {i: self.record.flow_data[i][index]
                      for i in range(self.record.num_pools)
                      if i in self.record.flow_data and index < len(self.record.flow_data[i])},
            'gates': {g: self.record.gate_data[g][index]
                      for g in self.record.gate_data
                      if index < len(self.record.gate_data[g])},
        }

    def step(self) -> Optional[Dict[str, Any]]:
        """单步回放"""
        if self.current_index >= len(self.record.timestamps):
            return None

        state = self._get_state_at_index(self.current_index)
        self.current_index += 1
        return state


# ==============================================================================
# 性能指标计算
# ==============================================================================

class PerformanceMetrics:
    """性能指标"""

    @staticmethod
    def calculate_level_rmse(target: float, levels: List[float]) -> float:
        """计算水位RMSE"""
        if not levels:
            return 0.0
        errors = [(l - target) ** 2 for l in levels]
        return np.sqrt(np.mean(errors))

    @staticmethod
    def calculate_level_mae(target: float, levels: List[float]) -> float:
        """计算水位MAE"""
        if not levels:
            return 0.0
        errors = [abs(l - target) for l in levels]
        return np.mean(errors)

    @staticmethod
    def calculate_flow_balance(inflows: List[float], outflows: List[float]) -> float:
        """计算流量平衡度"""
        if not inflows or not outflows:
            return 1.0
        total_in = sum(inflows)
        total_out = sum(outflows)
        if total_in == 0:
            return 1.0
        return abs(total_in - total_out) / total_in

    @staticmethod
    def calculate_settling_time(levels: List[float],
                                 target: float,
                                 tolerance: float = 0.1) -> Optional[int]:
        """计算调节时间 (步数)"""
        for i, level in enumerate(levels):
            if abs(level - target) <= tolerance:
                # 检查是否稳定
                remaining = levels[i:]
                if all(abs(l - target) <= tolerance for l in remaining[:10]):
                    return i
        return None

    @staticmethod
    def calculate_overshoot(levels: List[float], target: float) -> float:
        """计算超调量"""
        if not levels:
            return 0.0
        max_dev = max(abs(l - target) for l in levels)
        return max_dev


class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self, simulator: FullLineHydraulicSimulator):
        self.simulator = simulator

    def analyze_pool(self, pool_id: int) -> Dict[str, float]:
        """分析单个池的性能"""
        history = list(self.simulator.state.level_history.get(pool_id, []))
        target = self.simulator.pool_params[pool_id].target_level

        return {
            'rmse': PerformanceMetrics.calculate_level_rmse(target, history),
            'mae': PerformanceMetrics.calculate_level_mae(target, history),
            'overshoot': PerformanceMetrics.calculate_overshoot(history, target),
            'settling_time': PerformanceMetrics.calculate_settling_time(history, target),
        }

    def analyze_all(self) -> Dict[int, Dict[str, float]]:
        """分析所有池的性能"""
        return {i: self.analyze_pool(i) for i in range(self.simulator.num_pools)}

    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        all_metrics = self.analyze_all()

        # 计算统计信息
        rmse_values = [m['rmse'] for m in all_metrics.values()]
        mae_values = [m['mae'] for m in all_metrics.values()]

        return {
            'pool_metrics': all_metrics,
            'summary': {
                'avg_rmse': np.mean(rmse_values),
                'max_rmse': np.max(rmse_values),
                'avg_mae': np.mean(mae_values),
                'max_mae': np.max(mae_values),
            },
            'simulation_info': {
                'current_time': self.simulator.state.current_time,
                'step_count': self.simulator.state.step_count,
                'num_pools': self.simulator.num_pools,
            },
        }

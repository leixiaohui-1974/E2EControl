"""
L1层现地控制器 - 分钟级自主响应控制
L1 Layer Local Controller - Minute-Level Autonomous Response Control

核心功能:
1. 场景事件接收与处理
2. 动作指令生成与执行
3. 状态监控与反馈
4. L2层协调接口
5. 安全约束检查

时间尺度: 1-15分钟
控制目标: 快速响应本地异常，确保安全运行
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum, auto
import logging
import time
from collections import deque
import threading

from .core_types import PoolRole, ScenarioType, ScenarioSeverity
from .local_pool_scenarios import (
    L1ScenarioType, L1ActionType, L1ScenarioEvent, L1ActionCommand,
    PollutionTracker, SlopePanelMonitor, DischargeManager, DischargeType,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# L1控制器状态
# ==============================================================================

class L1ControllerState(Enum):
    """L1控制器状态"""
    IDLE = "空闲"
    MONITORING = "监控中"
    RESPONDING = "响应中"
    EXECUTING = "执行中"
    COORDINATING = "协调中"
    EMERGENCY = "紧急状态"
    MAINTENANCE = "维护状态"


@dataclass
class L1PoolState:
    """L1渠池状态"""
    pool_id: int

    # 水力状态
    water_level: float = 3.0           # 水位 [m]
    upstream_flow: float = 50.0        # 上游流量 [m³/s]
    downstream_flow: float = 50.0      # 下游流量 [m³/s]
    gate_position: float = 0.5         # 闸门开度 [0-1]

    # 环境状态
    water_quality: float = 1.0         # 水质指标 [0-1, 1=正常]
    groundwater_level: float = 2.5     # 地下水位 [m]
    rainfall_1h: float = 0.0           # 1小时降雨 [mm]
    temperature: float = 15.0          # 温度 [°C]

    # 设备状态
    gate_operational: bool = True      # 闸门可操作
    sensors_operational: bool = True   # 传感器正常
    communication_ok: bool = True      # 通信正常

    # 控制状态
    controller_state: L1ControllerState = L1ControllerState.MONITORING
    active_scenarios: List[str] = field(default_factory=list)
    pending_commands: List[str] = field(default_factory=list)

    # 时间戳
    last_update: float = 0.0


@dataclass
class L1ControlResult:
    """L1控制结果"""
    success: bool
    pool_id: int
    action_type: L1ActionType

    # 执行详情
    target_value: float = 0.0
    achieved_value: float = 0.0
    execution_time: float = 0.0

    # 状态
    message: str = ""
    requires_l2_coordination: bool = False
    escalate_to_l2: bool = False

    # 后续动作
    follow_up_actions: List[L1ActionCommand] = field(default_factory=list)


# ==============================================================================
# L1响应策略
# ==============================================================================

class L1ResponseStrategy:
    """
    L1响应策略

    定义不同场景类型的响应动作序列
    """

    # 场景-动作映射表
    SCENARIO_ACTIONS: Dict[L1ScenarioType, List[L1ActionType]] = {
        # 污染场景
        L1ScenarioType.L1_POLLUTION_DETECTED: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.MONITOR_ENHANCE,
            L1ActionType.ISOLATE_DOWNSTREAM,
        ],
        L1ScenarioType.L1_POLLUTION_TRACKING: [
            L1ActionType.MONITOR_ENHANCE,
            L1ActionType.COORDINATE_DOWNSTREAM,
        ],
        L1ScenarioType.L1_POLLUTION_TRACING: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.ISOLATE_UPSTREAM,
        ],
        L1ScenarioType.L1_POLLUTION_ISOLATION: [
            L1ActionType.GATE_CLOSE,
            L1ActionType.ISOLATE_BOTH,
        ],

        # 边坡场景
        L1ScenarioType.L1_SLOPE_GROUNDWATER: [
            L1ActionType.MONITOR_ENHANCE,
            L1ActionType.ALARM_TRIGGER,
        ],
        L1ScenarioType.L1_SLOPE_RAINFALL: [
            L1ActionType.MONITOR_ENHANCE,
        ],
        L1ScenarioType.L1_SLOPE_PANEL_FLOAT: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_OPEN,
            L1ActionType.DRAIN_START,
        ],
        L1ScenarioType.L1_SLOPE_INSTABILITY: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.DRAIN_START,
            L1ActionType.ISOLATE_BOTH,
        ],

        # 退水场景
        L1ScenarioType.L1_DISCHARGE_EMERGENCY: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.DRAIN_START,
            L1ActionType.COORDINATE_UPSTREAM,
        ],
        L1ScenarioType.L1_DISCHARGE_MAINTENANCE: [
            L1ActionType.DRAIN_START,
            L1ActionType.COORDINATE_UPSTREAM,
            L1ActionType.COORDINATE_DOWNSTREAM,
        ],

        # 水位场景
        L1ScenarioType.L1_LEVEL_HIGH: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_OPEN,
        ],
        L1ScenarioType.L1_LEVEL_LOW: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_ADJUST,
        ],
        L1ScenarioType.L1_LEVEL_RAPID_RISE: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_OPEN,
            L1ActionType.COORDINATE_UPSTREAM,
        ],
        L1ScenarioType.L1_LEVEL_RAPID_DROP: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_ADJUST,
            L1ActionType.COORDINATE_UPSTREAM,
        ],

        # 闸门场景
        L1ScenarioType.L1_GATE_STUCK: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.COORDINATE_UPSTREAM,
            L1ActionType.COORDINATE_DOWNSTREAM,
        ],
        L1ScenarioType.L1_GATE_LEAK: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.MONITOR_ENHANCE,
        ],
        L1ScenarioType.L1_GATE_CONTROL_FAIL: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.GATE_LOCK,
        ],

        # 渗漏场景
        L1ScenarioType.L1_LEAKAGE_MINOR: [
            L1ActionType.MONITOR_ENHANCE,
        ],
        L1ScenarioType.L1_LEAKAGE_MODERATE: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.MONITOR_ENHANCE,
        ],
        L1ScenarioType.L1_LEAKAGE_SEVERE: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.DRAIN_START,
        ],
        L1ScenarioType.L1_LEAKAGE_PIPE_BURST: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.ISOLATE_BOTH,
            L1ActionType.DRAIN_START,
        ],

        # 冰凌场景
        L1ScenarioType.L1_ICE_FORMATION: [
            L1ActionType.MONITOR_ENHANCE,
        ],
        L1ScenarioType.L1_ICE_ACCUMULATION: [
            L1ActionType.ALARM_TRIGGER,
            L1ActionType.GATE_ADJUST,
        ],
        L1ScenarioType.L1_ICE_BLOCKAGE: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.GATE_OPEN,
        ],
        L1ScenarioType.L1_ICE_DAM: [
            L1ActionType.ALARM_ESCALATE,
            L1ActionType.DRAIN_START,
        ],
    }

    # 严重程度-优先级映射
    SEVERITY_PRIORITY: Dict[ScenarioSeverity, int] = {
        ScenarioSeverity.LOW: 3,
        ScenarioSeverity.MEDIUM: 5,
        ScenarioSeverity.HIGH: 8,
        ScenarioSeverity.CRITICAL: 10,
    }

    @classmethod
    def get_response_actions(cls,
                            scenario_type: L1ScenarioType,
                            severity: ScenarioSeverity) -> List[L1ActionType]:
        """获取响应动作列表"""
        actions = cls.SCENARIO_ACTIONS.get(scenario_type, [L1ActionType.MONITOR_ENHANCE])

        # 严重程度高时添加上报动作
        if severity in [ScenarioSeverity.HIGH, ScenarioSeverity.CRITICAL]:
            if L1ActionType.ALARM_ESCALATE not in actions:
                actions = [L1ActionType.ALARM_ESCALATE] + list(actions)

        return actions

    @classmethod
    def get_priority(cls, severity: ScenarioSeverity) -> int:
        """获取优先级"""
        return cls.SEVERITY_PRIORITY.get(severity, 5)


# ==============================================================================
# L1控制器
# ==============================================================================

class L1Controller:
    """
    L1层现地控制器

    功能:
    1. 接收和处理L1场景事件
    2. 生成和执行控制动作
    3. 与L2层协调
    4. 状态监控和反馈
    """

    def __init__(self,
                 pool_id: int,
                 num_pools: int = 60,
                 control_interval: float = 60.0):  # 控制间隔 [s]
        self.pool_id = pool_id
        self.num_pools = num_pools
        self.control_interval = control_interval

        # 状态
        self.state = L1PoolState(pool_id=pool_id)

        # 事件队列
        self.event_queue: deque = deque(maxlen=100)
        self.active_events: Dict[str, L1ScenarioEvent] = {}

        # 命令队列
        self.command_queue: deque = deque(maxlen=50)
        self.executing_commands: Dict[str, L1ActionCommand] = {}

        # 历史记录
        self.event_history: deque = deque(maxlen=1000)
        self.action_history: deque = deque(maxlen=1000)

        # 子模块
        self.pollution_tracker = PollutionTracker(num_pools)
        self.slope_monitor = SlopePanelMonitor(num_pools)
        self.discharge_manager = DischargeManager(num_pools)

        # 回调
        self.l2_callback: Optional[Callable] = None

        # 安全约束
        self.level_min = 0.5
        self.level_max = 6.0
        self.gate_rate_max = 0.1  # 每分钟最大开度变化

    def set_l2_callback(self, callback: Callable):
        """设置L2协调回调"""
        self.l2_callback = callback

    def update_state(self,
                     water_level: float = None,
                     upstream_flow: float = None,
                     downstream_flow: float = None,
                     gate_position: float = None,
                     water_quality: float = None,
                     groundwater_level: float = None,
                     rainfall_1h: float = None,
                     temperature: float = None):
        """更新状态"""
        if water_level is not None:
            self.state.water_level = water_level
        if upstream_flow is not None:
            self.state.upstream_flow = upstream_flow
        if downstream_flow is not None:
            self.state.downstream_flow = downstream_flow
        if gate_position is not None:
            self.state.gate_position = gate_position
        if water_quality is not None:
            self.state.water_quality = water_quality
        if groundwater_level is not None:
            self.state.groundwater_level = groundwater_level
        if rainfall_1h is not None:
            self.state.rainfall_1h = rainfall_1h
        if temperature is not None:
            self.state.temperature = temperature

        self.state.last_update = time.time()

    def receive_event(self, event: L1ScenarioEvent) -> bool:
        """接收场景事件"""
        if event.pool_id != self.pool_id:
            return False

        self.event_queue.append(event)
        self.active_events[event.event_id] = event
        self.event_history.append((time.time(), event))

        logger.info(f"L1控制器[{self.pool_id}]接收事件: {event.scenario_type.value}")

        # 立即处理高优先级事件
        if event.severity in [ScenarioSeverity.HIGH, ScenarioSeverity.CRITICAL]:
            self._process_event_immediate(event)

        return True

    def _process_event_immediate(self, event: L1ScenarioEvent):
        """立即处理紧急事件"""
        self.state.controller_state = L1ControllerState.RESPONDING

        # 获取响应动作
        actions = L1ResponseStrategy.get_response_actions(
            event.scenario_type, event.severity
        )

        # 生成命令
        for action_type in actions:
            cmd = self._create_command(event, action_type)
            self.command_queue.append(cmd)

        # 如果需要上报L2
        if L1ActionType.ALARM_ESCALATE in actions:
            self._escalate_to_l2(event)

    def _create_command(self,
                       event: L1ScenarioEvent,
                       action_type: L1ActionType) -> L1ActionCommand:
        """创建控制命令"""
        priority = L1ResponseStrategy.get_priority(event.severity)

        cmd = L1ActionCommand(
            command_id=f"CMD_{self.pool_id}_{int(time.time())}_{action_type.name}",
            action_type=action_type,
            pool_id=self.pool_id,
            priority=priority,
            start_time=time.time(),
        )

        # 根据动作类型设置参数
        if action_type == L1ActionType.GATE_OPEN:
            cmd.gate_position = min(1.0, self.state.gate_position + 0.3)
            cmd.gate_rate = self.gate_rate_max
        elif action_type == L1ActionType.GATE_CLOSE:
            cmd.gate_position = max(0.0, self.state.gate_position - 0.3)
            cmd.gate_rate = self.gate_rate_max
        elif action_type == L1ActionType.GATE_ADJUST:
            # 根据水位调整
            if self.state.water_level > (self.level_max + self.level_min) / 2:
                cmd.gate_position = min(1.0, self.state.gate_position + 0.1)
            else:
                cmd.gate_position = max(0.0, self.state.gate_position - 0.1)
        elif action_type == L1ActionType.DRAIN_START:
            cmd.target_flow = 30.0  # 退水流量
            cmd.level_constraint_min = self.level_min

        return cmd

    def _escalate_to_l2(self, event: L1ScenarioEvent):
        """上报L2层"""
        if self.l2_callback:
            self.state.controller_state = L1ControllerState.COORDINATING
            self.l2_callback(self.pool_id, event)
            logger.warning(f"L1控制器[{self.pool_id}]上报L2: {event.scenario_type.value}")

    def process_events(self) -> List[L1ControlResult]:
        """处理事件队列"""
        results = []

        while self.event_queue:
            event = self.event_queue.popleft()

            # 检测场景类型并触发相应处理
            if event.scenario_type in [
                L1ScenarioType.L1_POLLUTION_DETECTED,
                L1ScenarioType.L1_POLLUTION_TRACKING,
            ]:
                result = self._handle_pollution_event(event)
            elif event.scenario_type in [
                L1ScenarioType.L1_SLOPE_GROUNDWATER,
                L1ScenarioType.L1_SLOPE_PANEL_FLOAT,
            ]:
                result = self._handle_slope_event(event)
            elif event.scenario_type in [
                L1ScenarioType.L1_DISCHARGE_EMERGENCY,
                L1ScenarioType.L1_DISCHARGE_MAINTENANCE,
            ]:
                result = self._handle_discharge_event(event)
            else:
                result = self._handle_generic_event(event)

            results.append(result)

        return results

    def _handle_pollution_event(self, event: L1ScenarioEvent) -> L1ControlResult:
        """处理污染事件"""
        # 更新追踪器
        if event.scenario_type == L1ScenarioType.L1_POLLUTION_DETECTED:
            self.pollution_tracker.detect_pollution(
                pool_id=event.pool_id,
                concentration=event.measured_value,
                timestamp=event.timestamp,
                threshold=event.threshold_value
            )

        # 生成响应动作
        actions = L1ResponseStrategy.get_response_actions(
            event.scenario_type, event.severity
        )

        follow_ups = []
        for action_type in actions:
            cmd = self._create_command(event, action_type)
            follow_ups.append(cmd)

        return L1ControlResult(
            success=True,
            pool_id=self.pool_id,
            action_type=actions[0] if actions else L1ActionType.MONITOR_ENHANCE,
            message=f"污染事件处理: {event.scenario_type.value}",
            requires_l2_coordination=event.severity in [ScenarioSeverity.HIGH, ScenarioSeverity.CRITICAL],
            follow_up_actions=follow_ups,
        )

    def _handle_slope_event(self, event: L1ScenarioEvent) -> L1ControlResult:
        """处理边坡事件"""
        # 更新监测器
        self.slope_monitor.update_condition(
            pool_id=event.pool_id,
            groundwater_level=event.groundwater_level,
            canal_level=self.state.water_level,
            rainfall_1h=event.rainfall_intensity,
        )

        # 获取推荐动作
        recommended = self.slope_monitor.recommend_action(event.pool_id)

        actions = L1ResponseStrategy.get_response_actions(
            event.scenario_type, event.severity
        )

        follow_ups = []
        for action_type in actions:
            cmd = self._create_command(event, action_type)
            follow_ups.append(cmd)

        # 添加推荐动作
        if recommended:
            follow_ups.append(recommended)

        return L1ControlResult(
            success=True,
            pool_id=self.pool_id,
            action_type=actions[0] if actions else L1ActionType.MONITOR_ENHANCE,
            message=f"边坡事件处理: {event.scenario_type.value}",
            requires_l2_coordination=event.severity == ScenarioSeverity.CRITICAL,
            follow_up_actions=follow_ups,
        )

    def _handle_discharge_event(self, event: L1ScenarioEvent) -> L1ControlResult:
        """处理退水事件"""
        # 确定退水类型
        if event.scenario_type == L1ScenarioType.L1_DISCHARGE_EMERGENCY:
            discharge_type = DischargeType.EMERGENCY_POLLUTION
            priority = 9
        else:
            discharge_type = DischargeType.PLANNED_MAINTENANCE
            priority = 5

        # 请求退水
        request, discharge_event = self.discharge_manager.request_discharge(
            pool_id=event.pool_id,
            discharge_type=discharge_type,
            target_level=event.measured_value if event.measured_value > 0 else 1.0,
            current_level=self.state.water_level,
            priority=priority,
        )

        # 生成退水命令
        commands = self.discharge_manager.generate_discharge_commands(request.request_id)

        return L1ControlResult(
            success=True,
            pool_id=self.pool_id,
            action_type=L1ActionType.DRAIN_START,
            message=f"退水事件处理: {discharge_type.value}",
            requires_l2_coordination=True,
            follow_up_actions=commands,
        )

    def _handle_generic_event(self, event: L1ScenarioEvent) -> L1ControlResult:
        """处理通用事件"""
        actions = L1ResponseStrategy.get_response_actions(
            event.scenario_type, event.severity
        )

        follow_ups = []
        for action_type in actions:
            cmd = self._create_command(event, action_type)
            follow_ups.append(cmd)

        return L1ControlResult(
            success=True,
            pool_id=self.pool_id,
            action_type=actions[0] if actions else L1ActionType.MONITOR_ENHANCE,
            message=f"事件处理: {event.scenario_type.value}",
            follow_up_actions=follow_ups,
        )

    def execute_commands(self) -> List[L1ControlResult]:
        """执行命令队列"""
        results = []

        # 按优先级排序
        commands = sorted(self.command_queue, key=lambda c: -c.priority)
        self.command_queue.clear()

        for cmd in commands:
            result = self._execute_command(cmd)
            results.append(result)
            self.action_history.append((time.time(), cmd, result))

        return results

    def _execute_command(self, cmd: L1ActionCommand) -> L1ControlResult:
        """执行单个命令"""
        self.state.controller_state = L1ControllerState.EXECUTING

        success = True
        achieved_value = 0.0
        message = ""

        if cmd.action_type in [L1ActionType.GATE_OPEN, L1ActionType.GATE_CLOSE, L1ActionType.GATE_ADJUST]:
            # 检查安全约束
            if self._check_gate_safety(cmd.gate_position):
                self.state.gate_position = cmd.gate_position
                achieved_value = cmd.gate_position
                message = f"闸门调整到 {cmd.gate_position:.2f}"
            else:
                success = False
                message = "闸门调整超出安全约束"

        elif cmd.action_type == L1ActionType.GATE_LOCK:
            self.state.gate_operational = False
            message = "闸门已锁定"

        elif cmd.action_type in [L1ActionType.DRAIN_START, L1ActionType.DRAIN_ADJUST]:
            achieved_value = cmd.target_flow
            message = f"退水启动，流量 {cmd.target_flow:.1f} m³/s"

        elif cmd.action_type == L1ActionType.DRAIN_STOP:
            achieved_value = 0
            message = "退水停止"

        elif cmd.action_type in [L1ActionType.ALARM_TRIGGER, L1ActionType.ALARM_ESCALATE]:
            message = f"告警触发: {cmd.action_type.value}"

        elif cmd.action_type == L1ActionType.MONITOR_ENHANCE:
            message = "加强监测模式启动"

        elif cmd.action_type in [L1ActionType.ISOLATE_UPSTREAM, L1ActionType.ISOLATE_DOWNSTREAM, L1ActionType.ISOLATE_BOTH]:
            message = f"隔离动作: {cmd.action_type.value}"

        elif cmd.action_type in [L1ActionType.COORDINATE_UPSTREAM, L1ActionType.COORDINATE_DOWNSTREAM]:
            message = f"协调请求: {cmd.action_type.value}"

        cmd.is_executed = True
        cmd.execution_result = message

        self.state.controller_state = L1ControllerState.MONITORING

        return L1ControlResult(
            success=success,
            pool_id=self.pool_id,
            action_type=cmd.action_type,
            target_value=cmd.gate_position if cmd.action_type in [L1ActionType.GATE_OPEN, L1ActionType.GATE_CLOSE] else cmd.target_flow,
            achieved_value=achieved_value,
            message=message,
        )

    def _check_gate_safety(self, target_position: float) -> bool:
        """检查闸门安全约束"""
        # 检查水位约束
        if self.state.water_level > self.level_max * 0.95:
            # 高水位只能开大闸门
            return target_position >= self.state.gate_position
        elif self.state.water_level < self.level_min * 1.1:
            # 低水位只能关小闸门
            return target_position <= self.state.gate_position

        # 检查变化率约束
        rate = abs(target_position - self.state.gate_position)
        if rate > self.gate_rate_max * 2:  # 允许一定余量
            return False

        return True

    def control_step(self, dt: float = 60.0) -> Dict[str, Any]:
        """执行一个控制步"""
        start_time = time.time()

        # 1. 检测本地场景
        local_events = self._detect_local_scenarios()
        for event in local_events:
            self.receive_event(event)

        # 2. 处理事件
        event_results = self.process_events()

        # 3. 执行命令
        command_results = self.execute_commands()

        # 4. 更新状态
        self.state.active_scenarios = [e.event_id for e in self.active_events.values()]

        return {
            'pool_id': self.pool_id,
            'control_time': time.time() - start_time,
            'events_processed': len(event_results),
            'commands_executed': len(command_results),
            'state': self.state.controller_state.value,
            'water_level': self.state.water_level,
            'gate_position': self.state.gate_position,
        }

    def _detect_local_scenarios(self) -> List[L1ScenarioEvent]:
        """检测本地场景"""
        events = []

        # 1. 水位检测
        if self.state.water_level > self.level_max * 0.9:
            events.append(L1ScenarioEvent(
                event_id=f"LEVEL_HIGH_{self.pool_id}_{int(time.time())}",
                scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
                pool_id=self.pool_id,
                severity=ScenarioSeverity.HIGH if self.state.water_level > self.level_max * 0.95 else ScenarioSeverity.MEDIUM,
                measured_value=self.state.water_level,
                threshold_value=self.level_max * 0.9,
            ))
        elif self.state.water_level < self.level_min * 1.2:
            events.append(L1ScenarioEvent(
                event_id=f"LEVEL_LOW_{self.pool_id}_{int(time.time())}",
                scenario_type=L1ScenarioType.L1_LEVEL_LOW,
                pool_id=self.pool_id,
                severity=ScenarioSeverity.HIGH if self.state.water_level < self.level_min * 1.1 else ScenarioSeverity.MEDIUM,
                measured_value=self.state.water_level,
                threshold_value=self.level_min * 1.2,
            ))

        # 2. 边坡检测
        slope_event = self.slope_monitor.update_condition(
            pool_id=self.pool_id,
            groundwater_level=self.state.groundwater_level,
            canal_level=self.state.water_level,
            rainfall_1h=self.state.rainfall_1h,
        )
        if slope_event:
            events.append(slope_event)

        # 3. 水质检测
        if self.state.water_quality < 0.7:
            pollution_event = self.pollution_tracker.detect_pollution(
                pool_id=self.pool_id,
                concentration=1.0 - self.state.water_quality,
                timestamp=time.time(),
                threshold=0.3,
            )
            if pollution_event:
                events.append(pollution_event)

        return events


# ==============================================================================
# L1控制器管理器
# ==============================================================================

class L1ControllerManager:
    """
    L1控制器管理器

    管理多个渠池的L1控制器
    """

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools
        self.controllers: Dict[int, L1Controller] = {}

        # 创建控制器
        for i in range(num_pools):
            self.controllers[i] = L1Controller(pool_id=i, num_pools=num_pools)

        # L2协调回调
        self.l2_coordinator_callback: Optional[Callable] = None

    def set_l2_coordinator(self, callback: Callable):
        """设置L2协调器回调"""
        self.l2_coordinator_callback = callback
        for controller in self.controllers.values():
            controller.set_l2_callback(callback)

    def update_pool_state(self, pool_id: int, **kwargs):
        """更新渠池状态"""
        if pool_id in self.controllers:
            self.controllers[pool_id].update_state(**kwargs)

    def broadcast_event(self, event: L1ScenarioEvent):
        """广播事件到相关控制器"""
        pool_id = event.pool_id
        if pool_id in self.controllers:
            self.controllers[pool_id].receive_event(event)

    def control_step_all(self, dt: float = 60.0) -> Dict[int, Dict]:
        """所有控制器执行一步"""
        results = {}
        for pool_id, controller in self.controllers.items():
            results[pool_id] = controller.control_step(dt)
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        active_events = 0
        emergency_pools = []

        for pool_id, controller in self.controllers.items():
            active_events += len(controller.active_events)
            if controller.state.controller_state == L1ControllerState.EMERGENCY:
                emergency_pools.append(pool_id)

        return {
            'total_pools': self.num_pools,
            'active_events': active_events,
            'emergency_pools': emergency_pools,
            'timestamp': time.time(),
        }

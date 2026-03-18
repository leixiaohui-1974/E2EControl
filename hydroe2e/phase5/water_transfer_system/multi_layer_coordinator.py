"""
多层协同控制器 - L1/L2/L3全线联动
Multi-Layer Coordinator - Full-Line L1/L2/L3 Integration

核心功能:
1. 三层架构统一管理 (L3全局 + L2区域 + L1现地)
2. 跨层事件传递与协调
3. 多场景并发处理
4. 智能决策融合
5. 全线协同控制

时间尺度:
- L3: 日/周级 (全局调度优化)
- L2: 小时级 (区域协调)
- L1: 分钟级 (现地自主响应)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum, auto
import logging
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan, ScenarioEvent, RegionConfig,
)
from .local_pool_scenarios import (
    L1ScenarioType, L1ActionType, L1ScenarioEvent, L1ActionCommand,
)
from .l1_controller import (
    L1Controller, L1ControllerManager, L1ControlResult,
    L1ControllerState, L1ResponseStrategy,
)
from .l2_l1_coordinator import (
    L2L1Coordinator, FullLineCoordinatorManager,
    CoordinationType, CoordinationRequest, CoordinationResponse,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 多层事件类型
# ==============================================================================

class MultiLayerEventType(Enum):
    """多层事件类型"""
    # L1层事件
    L1_LOCAL_ANOMALY = "L1现地异常"
    L1_SENSOR_ALERT = "L1传感器告警"
    L1_GATE_ACTION = "L1闸门动作"

    # L2层事件
    L2_REGIONAL_COORDINATION = "L2区域协调"
    L2_CROSS_POOL_SPREAD = "L2跨池扩散"
    L2_EMERGENCY_RESPONSE = "L2应急响应"

    # L3层事件
    L3_GLOBAL_OPTIMIZATION = "L3全局优化"
    L3_DEMAND_CHANGE = "L3需求变化"
    L3_MAINTENANCE_PLAN = "L3检修计划"

    # 跨层事件
    CROSS_LAYER_ESCALATION = "跨层上报"
    CROSS_LAYER_DIRECTIVE = "跨层指令"
    CROSS_REGION_COORDINATION = "跨区域协调"


class DecisionPriority(Enum):
    """决策优先级"""
    SAFETY_CRITICAL = 10      # 安全关键
    EMERGENCY = 9             # 紧急
    HIGH = 7                  # 高
    NORMAL = 5                # 正常
    LOW = 3                   # 低
    OPTIMIZATION = 1          # 优化


@dataclass
class MultiLayerEvent:
    """多层事件"""
    event_id: str
    event_type: MultiLayerEventType
    source_layer: int  # 1, 2, 3
    source_pool: Optional[int] = None
    source_region: Optional[int] = None

    # 事件内容
    scenario_type: Optional[ScenarioType] = None
    l1_scenario_type: Optional[L1ScenarioType] = None
    severity: ScenarioSeverity = ScenarioSeverity.MEDIUM
    priority: DecisionPriority = DecisionPriority.NORMAL

    # 影响范围
    affected_pools: List[int] = field(default_factory=list)
    affected_regions: List[int] = field(default_factory=list)

    # 时间
    timestamp: float = 0.0
    deadline: float = 0.0

    # 状态
    is_processed: bool = False
    requires_escalation: bool = False

    # 附加数据
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiLayerDecision:
    """多层决策"""
    decision_id: str
    source_event_id: str
    decision_layer: int  # 1, 2, 3

    # 决策内容
    l1_commands: Dict[int, List[L1ActionCommand]] = field(default_factory=dict)
    l2_coordinations: List[CoordinationRequest] = field(default_factory=list)
    l3_directives: List[ControlDirective] = field(default_factory=list)

    # 优先级
    priority: DecisionPriority = DecisionPriority.NORMAL

    # 执行状态
    is_executed: bool = False
    execution_results: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# L3全局调度层
# ==============================================================================

class L3GlobalScheduler:
    """
    L3全局调度层

    职责:
    1. 全线供水调度优化
    2. 检修计划协调
    3. 需求变化响应
    4. 跨区域协调决策
    """

    def __init__(self, num_pools: int = 60, num_regions: int = 6):
        self.num_pools = num_pools
        self.num_regions = num_regions

        # 全局状态
        self.global_state = {
            'total_flow': 0.0,
            'average_level': 3.0,
            'active_emergencies': 0,
            'maintenance_pools': set(),
        }

        # 调度计划
        self.active_plans: Dict[str, ControlPlan] = {}
        self.pending_directives: deque = deque(maxlen=100)

        # 需求预测
        self.demand_forecast: Dict[int, float] = {}  # region_id -> demand

        # 优化参数
        self.optimization_interval = 3600.0  # 1小时
        self.last_optimization = 0.0

    def receive_l2_report(self, region_id: int, report: Dict[str, Any]):
        """接收L2层报告"""
        # 更新区域状态
        if 'emergency_mode' in report and report['emergency_mode']:
            self.global_state['active_emergencies'] += 1

        # 检查是否需要全局协调
        if report.get('requires_global_coordination', False):
            self._create_global_coordination(region_id, report)

    def _create_global_coordination(self, region_id: int, report: Dict[str, Any]):
        """创建全局协调"""
        # 为区域内的池创建指令
        start_pool = region_id * 10
        for pool_id in range(start_pool, min(start_pool + 10, self.num_pools)):
            directive = ControlDirective(
                pool_id=pool_id,
                role=PoolRole.BUFFER,  # 全局协调使用缓冲角色
                priority=8,
                remarks=f"GLOBAL_COORDINATION_region_{region_id}",
            )
            self.pending_directives.append(directive)

    def optimize_global_schedule(self) -> Dict[str, Any]:
        """全局调度优化"""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return {'optimized': False}

        self.last_optimization = current_time

        # 计算全局优化目标
        optimization_result = {
            'optimized': True,
            'timestamp': current_time,
            'target_flows': {},
            'target_levels': {},
        }

        # 简化的流量分配优化
        base_flow = 50.0  # 基础流量
        for region_id in range(self.num_regions):
            demand = self.demand_forecast.get(region_id, 1.0)
            optimization_result['target_flows'][region_id] = base_flow * demand

        return optimization_result

    def process_maintenance_plan(self,
                                  pool_id: int,
                                  start_time: float,
                                  duration: float) -> ControlPlan:
        """处理检修计划"""
        plan_id = f"MAINT_{pool_id}_{int(start_time)}"

        plan = ControlPlan(
            plan_id=plan_id,
            scenario=ScenarioType.S7_PLANNED_MAINT,
            timestamp=start_time,
            duration=duration,
            priority=5,
        )

        # 添加检修池指令
        directive = ControlDirective(
            pool_id=pool_id,
            role=PoolRole.DRAIN,  # 检修时退水
            priority=5,
            remarks="MAINTENANCE",
        )
        plan.add_directive(directive)

        self.active_plans[plan_id] = plan
        self.global_state['maintenance_pools'].add(pool_id)

        return plan

    def get_global_directives(self) -> List[ControlDirective]:
        """获取全局指令"""
        directives = list(self.pending_directives)
        self.pending_directives.clear()
        return directives


# ==============================================================================
# 多层协同控制器
# ==============================================================================

class MultiLayerCoordinator:
    """
    多层协同控制器

    整合L1/L2/L3三层控制，实现全线协同
    """

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools

        # 三层控制器
        self.l3_scheduler = L3GlobalScheduler(num_pools)
        self.l2_coordinator = FullLineCoordinatorManager(num_pools)
        # L1控制器通过L2协调器管理

        # 事件队列
        self.event_queue: deque = deque(maxlen=500)
        self.processed_events: deque = deque(maxlen=1000)

        # 决策队列
        self.decision_queue: deque = deque(maxlen=200)
        self.executed_decisions: deque = deque(maxlen=500)

        # 跨层通信
        self.l1_to_l2_queue: deque = deque(maxlen=200)
        self.l2_to_l3_queue: deque = deque(maxlen=100)
        self.l3_to_l2_queue: deque = deque(maxlen=100)
        self.l2_to_l1_queue: deque = deque(maxlen=200)

        # 统计
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'decisions_made': 0,
            'l1_commands_issued': 0,
            'l2_coordinations': 0,
            'l3_directives': 0,
        }

        # 设置跨层回调
        self._setup_cross_layer_callbacks()

    def _setup_cross_layer_callbacks(self):
        """设置跨层回调"""
        # L2协调器已经管理了L1控制器的回调
        pass

    def receive_event(self, event: MultiLayerEvent):
        """接收多层事件"""
        self.event_queue.append(event)
        self.stats['events_received'] += 1

        # 根据优先级立即处理紧急事件
        if event.priority in [DecisionPriority.SAFETY_CRITICAL, DecisionPriority.EMERGENCY]:
            self._process_urgent_event(event)

    def _process_urgent_event(self, event: MultiLayerEvent):
        """处理紧急事件"""
        logger.warning(f"紧急事件处理: {event.event_type.value}, 源层: L{event.source_layer}")

        # 直接生成决策
        decision = self._make_decision(event)
        if decision:
            self._execute_decision(decision)

    def inject_l1_event(self, pool_id: int, l1_event: L1ScenarioEvent):
        """注入L1事件"""
        # 转换为多层事件
        ml_event = MultiLayerEvent(
            event_id=f"ML_{l1_event.event_id}",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=pool_id,
            l1_scenario_type=l1_event.scenario_type,
            severity=l1_event.severity,
            priority=self._severity_to_priority(l1_event.severity),
            affected_pools=[pool_id],
            timestamp=time.time(),
        )

        self.receive_event(ml_event)

        # 同时发送到L2协调器
        self.l2_coordinator.broadcast_event(l1_event)

    def _severity_to_priority(self, severity: ScenarioSeverity) -> DecisionPriority:
        """严重程度转优先级"""
        mapping = {
            ScenarioSeverity.LOW: DecisionPriority.LOW,
            ScenarioSeverity.MEDIUM: DecisionPriority.NORMAL,
            ScenarioSeverity.HIGH: DecisionPriority.HIGH,
            ScenarioSeverity.CRITICAL: DecisionPriority.EMERGENCY,
        }
        return mapping.get(severity, DecisionPriority.NORMAL)

    def inject_l2_coordination(self,
                                region_id: int,
                                coord_type: CoordinationType,
                                affected_pools: List[int]):
        """注入L2协调请求"""
        ml_event = MultiLayerEvent(
            event_id=f"ML_L2_{region_id}_{int(time.time())}",
            event_type=MultiLayerEventType.L2_REGIONAL_COORDINATION,
            source_layer=2,
            source_region=region_id,
            priority=DecisionPriority.HIGH,
            affected_pools=affected_pools,
            affected_regions=[region_id],
            timestamp=time.time(),
            data={'coord_type': coord_type},
        )

        self.receive_event(ml_event)

    def inject_l3_directive(self, directive: ControlDirective):
        """注入L3指令"""
        ml_event = MultiLayerEvent(
            event_id=f"ML_L3_{directive.directive_id}",
            event_type=MultiLayerEventType.L3_GLOBAL_OPTIMIZATION,
            source_layer=3,
            priority=DecisionPriority.NORMAL,
            affected_regions=directive.affected_regions if hasattr(directive, 'affected_regions') else [],
            timestamp=time.time(),
            data={'directive': directive},
        )

        self.receive_event(ml_event)

    def process_events(self) -> List[MultiLayerDecision]:
        """处理事件队列"""
        decisions = []

        # 按优先级排序
        events = sorted(self.event_queue, key=lambda e: -e.priority.value)
        self.event_queue.clear()

        for event in events:
            decision = self._make_decision(event)
            if decision:
                decisions.append(decision)
                self.decision_queue.append(decision)

            event.is_processed = True
            self.processed_events.append(event)
            self.stats['events_processed'] += 1

        return decisions

    def _make_decision(self, event: MultiLayerEvent) -> Optional[MultiLayerDecision]:
        """生成决策"""
        decision_id = f"DEC_{event.event_id}_{int(time.time())}"

        decision = MultiLayerDecision(
            decision_id=decision_id,
            source_event_id=event.event_id,
            decision_layer=event.source_layer,
            priority=event.priority,
        )

        # 根据事件类型生成决策
        if event.source_layer == 1:
            decision = self._make_l1_decision(event, decision)
        elif event.source_layer == 2:
            decision = self._make_l2_decision(event, decision)
        elif event.source_layer == 3:
            decision = self._make_l3_decision(event, decision)

        self.stats['decisions_made'] += 1
        return decision

    def _make_l1_decision(self,
                          event: MultiLayerEvent,
                          decision: MultiLayerDecision) -> MultiLayerDecision:
        """生成L1决策"""
        if event.l1_scenario_type and event.source_pool is not None:
            # 获取响应动作
            actions = L1ResponseStrategy.get_response_actions(
                event.l1_scenario_type,
                event.severity
            )

            # 生成命令
            commands = []
            for action_type in actions:
                cmd = L1ActionCommand(
                    command_id=f"{decision.decision_id}_{action_type.name}",
                    action_type=action_type,
                    pool_id=event.source_pool,
                    priority=event.priority.value,
                )
                commands.append(cmd)

            decision.l1_commands[event.source_pool] = commands

            # 检查是否需要上报L2
            if event.severity in [ScenarioSeverity.HIGH, ScenarioSeverity.CRITICAL]:
                coord_type = self._map_l1_to_l2_coordination(event.l1_scenario_type)
                if coord_type:
                    request = CoordinationRequest(
                        request_id=f"COORD_{decision.decision_id}",
                        coordination_type=coord_type,
                        source_pool=event.source_pool,
                        priority=event.priority.value,
                        affected_pools=event.affected_pools,
                    )
                    decision.l2_coordinations.append(request)

        return decision

    def _make_l2_decision(self,
                          event: MultiLayerEvent,
                          decision: MultiLayerDecision) -> MultiLayerDecision:
        """生成L2决策"""
        coord_type = event.data.get('coord_type')

        if coord_type and event.affected_pools:
            # 根据协调类型生成命令
            for pool_id in event.affected_pools:
                commands = self._generate_coordination_commands(
                    pool_id, coord_type, event.priority
                )
                decision.l1_commands[pool_id] = commands

        # 检查是否需要上报L3
        if event.priority in [DecisionPriority.SAFETY_CRITICAL, DecisionPriority.EMERGENCY]:
            if len(event.affected_regions) > 1:
                # 为跨区域协调创建指令
                for pool_id in event.affected_pools[:3]:  # 取前3个影响池
                    directive = ControlDirective(
                        pool_id=pool_id,
                        role=PoolRole.BUFFER,
                        priority=event.priority.value,
                        remarks=f"CROSS_REGION_COORDINATION_{decision.decision_id}",
                    )
                    decision.l3_directives.append(directive)

        return decision

    def _make_l3_decision(self,
                          event: MultiLayerEvent,
                          decision: MultiLayerDecision) -> MultiLayerDecision:
        """生成L3决策"""
        # L3决策下发到L2
        for region_id in event.affected_regions:
            request = CoordinationRequest(
                request_id=f"L3_COORD_{decision.decision_id}_{region_id}",
                coordination_type=CoordinationType.FLOW_ADJUSTMENT,
                source_pool=region_id * 10,  # 区域起始池
                priority=event.priority.value,
                affected_pools=list(range(region_id * 10, (region_id + 1) * 10)),
            )
            decision.l2_coordinations.append(request)

        return decision

    def _map_l1_to_l2_coordination(self,
                                    l1_type: L1ScenarioType) -> Optional[CoordinationType]:
        """映射L1场景到L2协调类型"""
        mapping = {
            L1ScenarioType.L1_POLLUTION_DETECTED: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_POLLUTION_TRACKING: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_DISCHARGE_EMERGENCY: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_SLOPE_INSTABILITY: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_LEVEL_RAPID_RISE: CoordinationType.FLOOD_CONTROL,
            L1ScenarioType.L1_ICE_BLOCKAGE: CoordinationType.ICE_CONTROL,
            L1ScenarioType.L1_GATE_STUCK: CoordinationType.GATE_FAILURE,
        }
        return mapping.get(l1_type)

    def _generate_coordination_commands(self,
                                         pool_id: int,
                                         coord_type: CoordinationType,
                                         priority: DecisionPriority) -> List[L1ActionCommand]:
        """生成协调命令"""
        commands = []

        if coord_type == CoordinationType.POLLUTION_SPREAD:
            commands.append(L1ActionCommand(
                command_id=f"POLL_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.ISOLATE_DOWNSTREAM,
                pool_id=pool_id,
                priority=priority.value,
            ))
        elif coord_type == CoordinationType.EMERGENCY_DISCHARGE:
            commands.append(L1ActionCommand(
                command_id=f"DRAIN_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.DRAIN_START,
                pool_id=pool_id,
                priority=priority.value,
                target_flow=50.0,
            ))
        elif coord_type == CoordinationType.FLOOD_CONTROL:
            commands.append(L1ActionCommand(
                command_id=f"FLOOD_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.GATE_OPEN,
                pool_id=pool_id,
                priority=priority.value,
                gate_position=0.9,
            ))
        elif coord_type == CoordinationType.ICE_CONTROL:
            commands.append(L1ActionCommand(
                command_id=f"ICE_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.GATE_ADJUST,
                pool_id=pool_id,
                priority=priority.value,
                gate_position=0.6,
            ))
        else:
            commands.append(L1ActionCommand(
                command_id=f"GEN_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.MONITOR_ENHANCE,
                pool_id=pool_id,
                priority=priority.value,
            ))

        return commands

    def execute_decisions(self) -> Dict[str, Any]:
        """执行决策队列"""
        results = {
            'executed': 0,
            'l1_commands': 0,
            'l2_coordinations': 0,
            'l3_directives': 0,
        }

        while self.decision_queue:
            decision = self.decision_queue.popleft()
            exec_result = self._execute_decision(decision)

            results['executed'] += 1
            results['l1_commands'] += exec_result.get('l1_commands', 0)
            results['l2_coordinations'] += exec_result.get('l2_coordinations', 0)
            results['l3_directives'] += exec_result.get('l3_directives', 0)

            self.executed_decisions.append(decision)

        return results

    def _execute_decision(self, decision: MultiLayerDecision) -> Dict[str, Any]:
        """执行单个决策"""
        result = {
            'l1_commands': 0,
            'l2_coordinations': 0,
            'l3_directives': 0,
        }

        # 执行L1命令
        for pool_id, commands in decision.l1_commands.items():
            region_id = self.l2_coordinator.get_region_for_pool(pool_id)
            if region_id in self.l2_coordinator.coordinators:
                coordinator = self.l2_coordinator.coordinators[region_id]
                if pool_id in coordinator.l1_manager.controllers:
                    controller = coordinator.l1_manager.controllers[pool_id]
                    for cmd in commands:
                        controller.command_queue.append(cmd)
                        result['l1_commands'] += 1
                        self.stats['l1_commands_issued'] += 1

        # 执行L2协调
        for coord_request in decision.l2_coordinations:
            region_id = self.l2_coordinator.get_region_for_pool(coord_request.source_pool)
            if region_id in self.l2_coordinator.coordinators:
                coordinator = self.l2_coordinator.coordinators[region_id]
                coordinator.pending_requests.append(coord_request)
                result['l2_coordinations'] += 1
                self.stats['l2_coordinations'] += 1

        # 执行L3指令
        for directive in decision.l3_directives:
            self.l3_scheduler.pending_directives.append(directive)
            result['l3_directives'] += 1
            self.stats['l3_directives'] += 1

        decision.is_executed = True
        decision.execution_results = result

        return result

    def coordination_step(self, dt: float = 60.0) -> Dict[str, Any]:
        """执行一个协同控制步"""
        start_time = time.time()

        # 1. 处理事件
        decisions = self.process_events()

        # 2. 执行决策
        exec_result = self.execute_decisions()

        # 3. L2协调步
        l2_results = self.l2_coordinator.coordination_step_all(dt)

        # 4. L3调度 (每小时)
        l3_result = self.l3_scheduler.optimize_global_schedule()

        return {
            'control_time': time.time() - start_time,
            'decisions_made': len(decisions),
            'execution_result': exec_result,
            'l2_results': l2_results,
            'l3_optimized': l3_result.get('optimized', False),
            'stats': self.stats.copy(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'pending_events': len(self.event_queue),
            'pending_decisions': len(self.decision_queue),
            'l2_status': self.l2_coordinator.get_system_status(),
            'l3_status': self.l3_scheduler.global_state,
            'stats': self.stats.copy(),
        }


# ==============================================================================
# 场景组合生成器
# ==============================================================================

class ScenarioCombinationGenerator:
    """
    场景组合生成器

    支持全线-现地多层场景组合
    """

    # 场景组合模板
    COMBINATION_TEMPLATES = {
        'POLLUTION_CASCADE': {
            'name': '污染级联',
            'description': 'L1检测污染 -> L2协调隔离 -> L3调整调度',
            'layers': [1, 2, 3],
            'l1_scenarios': [L1ScenarioType.L1_POLLUTION_DETECTED],
            'l2_coordinations': [CoordinationType.POLLUTION_SPREAD],
            'l3_actions': ['FLOW_ADJUSTMENT'],
        },
        'FLOOD_EMERGENCY': {
            'name': '防洪应急',
            'description': '多池水位上涨 -> 区域协调泄洪 -> 全线流量调整',
            'layers': [1, 2, 3],
            'l1_scenarios': [L1ScenarioType.L1_LEVEL_HIGH, L1ScenarioType.L1_LEVEL_RAPID_RISE],
            'l2_coordinations': [CoordinationType.FLOOD_CONTROL],
            'l3_actions': ['EMERGENCY_SCHEDULE'],
        },
        'ICE_CONTROL': {
            'name': '冰凌防控',
            'description': '冰凌形成 -> 流速调节 -> 全线协调',
            'layers': [1, 2],
            'l1_scenarios': [L1ScenarioType.L1_ICE_FORMATION, L1ScenarioType.L1_ICE_ACCUMULATION],
            'l2_coordinations': [CoordinationType.ICE_CONTROL],
            'l3_actions': [],
        },
        'MAINTENANCE_DISCHARGE': {
            'name': '检修退水',
            'description': 'L3计划 -> L2协调 -> L1执行',
            'layers': [3, 2, 1],
            'l1_scenarios': [L1ScenarioType.L1_DISCHARGE_MAINTENANCE],
            'l2_coordinations': [CoordinationType.MAINTENANCE_DISCHARGE],
            'l3_actions': ['MAINTENANCE_PLAN'],
        },
        'GATE_FAILURE_CASCADE': {
            'name': '闸门故障级联',
            'description': '闸门故障 -> 相邻池补偿 -> 区域协调',
            'layers': [1, 2],
            'l1_scenarios': [L1ScenarioType.L1_GATE_STUCK, L1ScenarioType.L1_GATE_CONTROL_FAIL],
            'l2_coordinations': [CoordinationType.GATE_FAILURE],
            'l3_actions': [],
        },
        'SLOPE_EMERGENCY': {
            'name': '边坡应急',
            'description': '边坡失稳 -> 紧急退水 -> 区域协调',
            'layers': [1, 2],
            'l1_scenarios': [L1ScenarioType.L1_SLOPE_INSTABILITY, L1ScenarioType.L1_SLOPE_PANEL_FLOAT],
            'l2_coordinations': [CoordinationType.EMERGENCY_DISCHARGE],
            'l3_actions': [],
        },
        'LEAKAGE_RESPONSE': {
            'name': '渗漏响应',
            'description': '渗漏检测 -> 退水处理 -> 检修协调',
            'layers': [1, 2, 3],
            'l1_scenarios': [L1ScenarioType.L1_LEAKAGE_SEVERE, L1ScenarioType.L1_LEAKAGE_PIPE_BURST],
            'l2_coordinations': [CoordinationType.EMERGENCY_DISCHARGE],
            'l3_actions': ['MAINTENANCE_PLAN'],
        },
        'MULTI_REGION_COORDINATION': {
            'name': '跨区域协调',
            'description': '多区域同时异常 -> 全线协调',
            'layers': [1, 2, 3],
            'l1_scenarios': [L1ScenarioType.L1_LEVEL_HIGH],
            'l2_coordinations': [CoordinationType.LEVEL_BALANCE],
            'l3_actions': ['GLOBAL_OPTIMIZATION'],
        },
    }

    def __init__(self, num_pools: int = 60, num_regions: int = 6):
        self.num_pools = num_pools
        self.num_regions = num_regions

    def generate_combination(self,
                             template_name: str,
                             source_pool: int,
                             severity: ScenarioSeverity = ScenarioSeverity.HIGH) -> List[MultiLayerEvent]:
        """生成场景组合"""
        if template_name not in self.COMBINATION_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.COMBINATION_TEMPLATES[template_name]
        events = []
        base_time = time.time()

        # 生成L1事件
        if 1 in template['layers']:
            for i, l1_type in enumerate(template['l1_scenarios']):
                event = MultiLayerEvent(
                    event_id=f"COMB_{template_name}_L1_{i}_{int(base_time)}",
                    event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
                    source_layer=1,
                    source_pool=source_pool,
                    l1_scenario_type=l1_type,
                    severity=severity,
                    priority=DecisionPriority.HIGH if severity == ScenarioSeverity.CRITICAL else DecisionPriority.NORMAL,
                    affected_pools=self._get_affected_pools(source_pool, l1_type),
                    timestamp=base_time + i * 60,
                )
                events.append(event)

        # 生成L2协调事件
        if 2 in template['layers']:
            region_id = source_pool // 10
            for i, coord_type in enumerate(template['l2_coordinations']):
                event = MultiLayerEvent(
                    event_id=f"COMB_{template_name}_L2_{i}_{int(base_time)}",
                    event_type=MultiLayerEventType.L2_REGIONAL_COORDINATION,
                    source_layer=2,
                    source_region=region_id,
                    priority=DecisionPriority.HIGH,
                    affected_pools=self._get_affected_pools(source_pool, None),
                    affected_regions=[region_id],
                    timestamp=base_time + (len(template['l1_scenarios']) + i) * 60,
                    data={'coord_type': coord_type},
                )
                events.append(event)

        # 生成L3指令事件
        if 3 in template['layers'] and template['l3_actions']:
            for i, action in enumerate(template['l3_actions']):
                event = MultiLayerEvent(
                    event_id=f"COMB_{template_name}_L3_{i}_{int(base_time)}",
                    event_type=MultiLayerEventType.L3_GLOBAL_OPTIMIZATION,
                    source_layer=3,
                    priority=DecisionPriority.NORMAL,
                    affected_regions=list(range(self.num_regions)),
                    timestamp=base_time + (len(template['l1_scenarios']) + len(template['l2_coordinations']) + i) * 60,
                    data={'action': action},
                )
                events.append(event)

        return events

    def _get_affected_pools(self,
                            source_pool: int,
                            l1_type: Optional[L1ScenarioType]) -> List[int]:
        """获取影响的渠池"""
        if l1_type in [L1ScenarioType.L1_POLLUTION_DETECTED, L1ScenarioType.L1_POLLUTION_TRACKING]:
            # 污染影响下游
            return list(range(source_pool, min(source_pool + 5, self.num_pools)))
        elif l1_type in [L1ScenarioType.L1_ICE_FORMATION, L1ScenarioType.L1_ICE_BLOCKAGE]:
            # 冰凌影响上下游
            return list(range(max(0, source_pool - 3), min(source_pool + 3, self.num_pools)))
        else:
            # 默认影响相邻池
            return list(range(max(0, source_pool - 2), min(source_pool + 3, self.num_pools)))

    def generate_multi_region_combination(self,
                                           template_name: str,
                                           regions: List[int],
                                           severity: ScenarioSeverity = ScenarioSeverity.HIGH) -> List[MultiLayerEvent]:
        """生成多区域场景组合"""
        all_events = []

        for region_id in regions:
            source_pool = region_id * 10 + 5  # 区域中间位置
            events = self.generate_combination(template_name, source_pool, severity)
            all_events.extend(events)

        return all_events

    def generate_random_combination(self, num_events: int = 5) -> List[MultiLayerEvent]:
        """生成随机场景组合"""
        events = []
        templates = list(self.COMBINATION_TEMPLATES.keys())

        for _ in range(num_events):
            template = np.random.choice(templates)
            source_pool = np.random.randint(0, self.num_pools)
            severity = np.random.choice([
                ScenarioSeverity.LOW,
                ScenarioSeverity.MEDIUM,
                ScenarioSeverity.HIGH,
            ], p=[0.3, 0.5, 0.2])

            events.extend(self.generate_combination(template, source_pool, severity))

        return events


# ==============================================================================
# 智能决策引擎
# ==============================================================================

class IntelligentDecisionEngine:
    """
    智能决策引擎

    基于规则和学习的多层决策支持
    """

    def __init__(self, coordinator: MultiLayerCoordinator):
        self.coordinator = coordinator

        # 决策规则库
        self.rules: Dict[str, Dict] = {}
        self._init_rules()

        # 决策历史
        self.decision_history: deque = deque(maxlen=1000)

        # 学习参数 (简化)
        self.action_weights: Dict[str, float] = {}

    def _init_rules(self):
        """初始化决策规则"""
        self.rules = {
            'POLLUTION_ISOLATION': {
                'condition': lambda e: e.l1_scenario_type == L1ScenarioType.L1_POLLUTION_DETECTED,
                'actions': [L1ActionType.ISOLATE_DOWNSTREAM, L1ActionType.MONITOR_ENHANCE],
                'escalation': True,
            },
            'FLOOD_RESPONSE': {
                'condition': lambda e: e.l1_scenario_type in [L1ScenarioType.L1_LEVEL_HIGH, L1ScenarioType.L1_LEVEL_RAPID_RISE],
                'actions': [L1ActionType.GATE_OPEN, L1ActionType.COORDINATE_DOWNSTREAM],
                'escalation': True,
            },
            'ICE_MANAGEMENT': {
                'condition': lambda e: e.l1_scenario_type in [L1ScenarioType.L1_ICE_FORMATION, L1ScenarioType.L1_ICE_BLOCKAGE],
                'actions': [L1ActionType.GATE_ADJUST, L1ActionType.MONITOR_ENHANCE],
                'escalation': False,
            },
            'GATE_FAILURE_RECOVERY': {
                'condition': lambda e: e.l1_scenario_type in [L1ScenarioType.L1_GATE_STUCK, L1ScenarioType.L1_GATE_CONTROL_FAIL],
                'actions': [L1ActionType.GATE_LOCK, L1ActionType.ALARM_ESCALATE],
                'escalation': True,
            },
        }

    def evaluate_event(self, event: MultiLayerEvent) -> Dict[str, Any]:
        """评估事件"""
        evaluation = {
            'event_id': event.event_id,
            'matched_rules': [],
            'recommended_actions': [],
            'escalation_needed': False,
            'priority_adjustment': 0,
        }

        for rule_name, rule in self.rules.items():
            if rule['condition'](event):
                evaluation['matched_rules'].append(rule_name)
                evaluation['recommended_actions'].extend(rule['actions'])
                if rule['escalation']:
                    evaluation['escalation_needed'] = True

        # 去重
        evaluation['recommended_actions'] = list(set(evaluation['recommended_actions']))

        return evaluation

    def make_intelligent_decision(self, event: MultiLayerEvent) -> Optional[MultiLayerDecision]:
        """智能决策"""
        evaluation = self.evaluate_event(event)

        if not evaluation['matched_rules']:
            return None

        decision_id = f"INT_DEC_{event.event_id}_{int(time.time())}"

        decision = MultiLayerDecision(
            decision_id=decision_id,
            source_event_id=event.event_id,
            decision_layer=event.source_layer,
            priority=event.priority,
        )

        # 生成L1命令
        if event.source_pool is not None:
            commands = []
            for action_type in evaluation['recommended_actions']:
                cmd = L1ActionCommand(
                    command_id=f"{decision_id}_{action_type.name}",
                    action_type=action_type,
                    pool_id=event.source_pool,
                    priority=event.priority.value,
                )
                commands.append(cmd)
            decision.l1_commands[event.source_pool] = commands

        # 生成L2协调
        if evaluation['escalation_needed']:
            coord_type = self._get_coordination_type(event)
            if coord_type:
                request = CoordinationRequest(
                    request_id=f"INT_COORD_{decision_id}",
                    coordination_type=coord_type,
                    source_pool=event.source_pool or 0,
                    priority=event.priority.value,
                    affected_pools=event.affected_pools,
                )
                decision.l2_coordinations.append(request)

        # 记录决策历史
        self.decision_history.append({
            'event': event,
            'evaluation': evaluation,
            'decision': decision,
            'timestamp': time.time(),
        })

        return decision

    def _get_coordination_type(self, event: MultiLayerEvent) -> Optional[CoordinationType]:
        """获取协调类型"""
        if event.l1_scenario_type:
            mapping = {
                L1ScenarioType.L1_POLLUTION_DETECTED: CoordinationType.POLLUTION_SPREAD,
                L1ScenarioType.L1_LEVEL_HIGH: CoordinationType.FLOOD_CONTROL,
                L1ScenarioType.L1_ICE_BLOCKAGE: CoordinationType.ICE_CONTROL,
                L1ScenarioType.L1_GATE_STUCK: CoordinationType.GATE_FAILURE,
            }
            return mapping.get(event.l1_scenario_type)
        return None

    def update_weights(self, decision_id: str, success: bool):
        """更新权重 (简化学习)"""
        # 查找决策记录
        for record in self.decision_history:
            if record['decision'].decision_id == decision_id:
                for rule_name in record['evaluation']['matched_rules']:
                    if rule_name not in self.action_weights:
                        self.action_weights[rule_name] = 1.0
                    # 简单的权重更新
                    if success:
                        self.action_weights[rule_name] *= 1.1
                    else:
                        self.action_weights[rule_name] *= 0.9
                break

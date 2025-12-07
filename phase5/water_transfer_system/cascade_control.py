"""
分层级联控制与事件驱动上报系统
Hierarchical Cascade Control and Event-Driven Escalation System

核心功能:
1. 控制效果评估 (每层判断是否控制住)
2. 逐级上报机制 (L1→L2→L3)
3. 上层干预决策 (接管/增援/协调)
4. 闭环控制监控
5. 失控场景处理

设计原则:
- L1优先自主处理，控制不住才上报
- L2接管后协调多池，仍控制不住上报L3
- L3全局干预，调用更多资源
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum, auto
import logging
import time
from collections import deque

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan,
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
from .multi_layer_coordinator import (
    MultiLayerCoordinator, MultiLayerEvent, MultiLayerEventType,
    DecisionPriority, ScenarioCombinationGenerator,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 控制状态与效果枚举
# ==============================================================================

class ControlEffectiveness(Enum):
    """控制有效性"""
    EFFECTIVE = "有效控制"           # 完全控制住
    PARTIALLY_EFFECTIVE = "部分有效"  # 部分控制，趋势好转
    INEFFECTIVE = "无效"             # 未能控制
    DETERIORATING = "恶化中"          # 情况恶化
    CRITICAL_FAILURE = "严重失控"     # 严重失控


class EscalationReason(Enum):
    """上报原因"""
    CONTROL_TIMEOUT = "控制超时"
    THRESHOLD_EXCEEDED = "超阈值"
    RAPID_DETERIORATION = "快速恶化"
    RESOURCE_EXHAUSTED = "资源耗尽"
    CROSS_BOUNDARY_SPREAD = "跨边界扩散"
    CASCADING_FAILURE = "级联故障"
    MANUAL_ESCALATION = "人工上报"


class InterventionType(Enum):
    """干预类型"""
    TAKEOVER = "接管控制"         # 上层完全接管
    REINFORCE = "增援资源"        # 增加资源支援
    COORDINATE = "协调邻域"       # 协调相邻区域
    ADJUST_TARGET = "调整目标"    # 放松控制目标
    EMERGENCY_SHUTDOWN = "紧急停机"  # 紧急隔离


# ==============================================================================
# 控制效果评估器
# ==============================================================================

@dataclass
class ControlMetrics:
    """控制指标"""
    pool_id: int
    timestamp: float

    # 水位指标
    level_error: float = 0.0          # 水位偏差 [m]
    level_rate: float = 0.0           # 水位变化率 [m/min]
    level_trend: str = "stable"       # 趋势: rising/falling/stable

    # 流量指标
    flow_error: float = 0.0           # 流量偏差 [m³/s]
    flow_balance: float = 0.0         # 流量平衡 (入-出)

    # 水质指标
    quality_index: float = 1.0        # 水质指数 [0-1]
    pollution_level: float = 0.0      # 污染浓度

    # 设备状态
    gate_response: float = 1.0        # 闸门响应率 [0-1]
    sensor_reliability: float = 1.0   # 传感器可靠性

    # 控制动作
    action_count: int = 0             # 已执行动作数
    failed_actions: int = 0           # 失败动作数

    def get_effectiveness_score(self) -> float:
        """计算控制有效性分数 [0-1]"""
        score = 1.0

        # 水位偏差惩罚
        if abs(self.level_error) > 0.5:
            score -= 0.3
        elif abs(self.level_error) > 0.2:
            score -= 0.1

        # 恶化趋势惩罚
        if self.level_trend == "rising" and self.level_error > 0:
            score -= 0.2
        elif self.level_trend == "falling" and self.level_error < 0:
            score -= 0.2

        # 水质惩罚
        if self.quality_index < 0.5:
            score -= 0.3
        elif self.quality_index < 0.8:
            score -= 0.1

        # 动作失败惩罚
        if self.action_count > 0:
            fail_rate = self.failed_actions / self.action_count
            score -= fail_rate * 0.2

        return max(0.0, min(1.0, score))


class ControlEffectEvaluator:
    """
    控制效果评估器

    评估每层控制器的控制效果，判断是否需要上报
    """

    # 评估阈值
    THRESHOLDS = {
        'level_error_critical': 1.0,      # 水位偏差临界值 [m]
        'level_error_warning': 0.5,       # 水位偏差警告值
        'level_rate_critical': 0.1,       # 水位变化率临界值 [m/min]
        'quality_critical': 0.3,          # 水质临界值
        'control_timeout': 300,           # 控制超时 [s]
        'effectiveness_threshold': 0.4,   # 有效性阈值
        'deterioration_count': 3,         # 连续恶化次数
    }

    def __init__(self, pool_id: int):
        self.pool_id = pool_id

        # 历史记录
        self.metrics_history: deque = deque(maxlen=100)
        self.effectiveness_history: deque = deque(maxlen=20)

        # 当前状态
        self.current_metrics: Optional[ControlMetrics] = None
        self.current_effectiveness: ControlEffectiveness = ControlEffectiveness.EFFECTIVE

        # 控制开始时间
        self.control_start_time: Optional[float] = None
        self.last_evaluation_time: float = 0.0

    def update_metrics(self, metrics: ControlMetrics):
        """更新指标"""
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        self.last_evaluation_time = time.time()

    def evaluate(self) -> Tuple[ControlEffectiveness, Optional[EscalationReason]]:
        """评估控制效果"""
        if not self.current_metrics:
            return ControlEffectiveness.EFFECTIVE, None

        metrics = self.current_metrics
        score = metrics.get_effectiveness_score()

        # 检查各种失控条件
        escalation_reason = None

        # 1. 检查阈值超标
        if abs(metrics.level_error) > self.THRESHOLDS['level_error_critical']:
            effectiveness = ControlEffectiveness.CRITICAL_FAILURE
            escalation_reason = EscalationReason.THRESHOLD_EXCEEDED
        elif metrics.quality_index < self.THRESHOLDS['quality_critical']:
            effectiveness = ControlEffectiveness.CRITICAL_FAILURE
            escalation_reason = EscalationReason.THRESHOLD_EXCEEDED

        # 2. 检查控制超时
        elif self.control_start_time and \
                (time.time() - self.control_start_time) > self.THRESHOLDS['control_timeout']:
            if score < self.THRESHOLDS['effectiveness_threshold']:
                effectiveness = ControlEffectiveness.INEFFECTIVE
                escalation_reason = EscalationReason.CONTROL_TIMEOUT
            else:
                effectiveness = ControlEffectiveness.PARTIALLY_EFFECTIVE

        # 3. 检查恶化趋势
        elif self._check_deterioration():
            effectiveness = ControlEffectiveness.DETERIORATING
            escalation_reason = EscalationReason.RAPID_DETERIORATION

        # 4. 根据分数判断
        elif score >= 0.8:
            effectiveness = ControlEffectiveness.EFFECTIVE
        elif score >= 0.5:
            effectiveness = ControlEffectiveness.PARTIALLY_EFFECTIVE
        else:
            effectiveness = ControlEffectiveness.INEFFECTIVE
            escalation_reason = EscalationReason.THRESHOLD_EXCEEDED

        self.current_effectiveness = effectiveness
        self.effectiveness_history.append(effectiveness)

        return effectiveness, escalation_reason

    def _check_deterioration(self) -> bool:
        """检查是否持续恶化"""
        if len(self.metrics_history) < 3:
            return False

        recent = list(self.metrics_history)[-3:]
        deteriorating = 0

        for i in range(1, len(recent)):
            # 检查水位偏差是否在增大
            if abs(recent[i].level_error) > abs(recent[i - 1].level_error) * 1.1:
                deteriorating += 1
            # 检查水质是否在下降
            if recent[i].quality_index < recent[i - 1].quality_index * 0.95:
                deteriorating += 1

        return deteriorating >= self.THRESHOLDS['deterioration_count']

    def start_control(self):
        """开始控制计时"""
        self.control_start_time = time.time()

    def reset(self):
        """重置评估器"""
        self.control_start_time = None
        self.metrics_history.clear()
        self.effectiveness_history.clear()


# ==============================================================================
# 上报事件
# ==============================================================================

@dataclass
class EscalationEvent:
    """上报事件"""
    event_id: str
    source_layer: int  # 1, 2, 3
    source_pool: int
    source_region: Optional[int] = None

    # 上报原因
    reason: EscalationReason = EscalationReason.THRESHOLD_EXCEEDED
    effectiveness: ControlEffectiveness = ControlEffectiveness.INEFFECTIVE

    # 场景信息
    scenario_type: Optional[L1ScenarioType] = None
    severity: ScenarioSeverity = ScenarioSeverity.HIGH

    # 指标
    metrics: Optional[ControlMetrics] = None
    control_duration: float = 0.0       # 已控制时长 [s]
    action_attempts: int = 0            # 动作尝试次数

    # 请求
    requested_intervention: InterventionType = InterventionType.TAKEOVER
    urgency: int = 8                    # 紧急程度 1-10

    # 时间
    timestamp: float = 0.0

    # 附加信息
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionDecision:
    """干预决策"""
    decision_id: str
    escalation_event_id: str
    target_layer: int  # 被干预的层
    intervention_type: InterventionType

    # 干预内容
    commands: List[L1ActionCommand] = field(default_factory=list)
    coordinations: List[CoordinationRequest] = field(default_factory=list)
    directives: List[ControlDirective] = field(default_factory=list)

    # 目标调整
    relaxed_constraints: Dict[str, float] = field(default_factory=dict)
    extended_deadline: float = 0.0

    # 资源
    additional_resources: Dict[str, Any] = field(default_factory=dict)

    # 状态
    is_executed: bool = False
    execution_result: str = ""


# ==============================================================================
# 上层干预决策器
# ==============================================================================

class UpperLayerInterventionDecider:
    """
    上层干预决策器

    接收下层上报，决定干预策略
    """

    # 干预策略规则
    INTERVENTION_RULES = {
        # (原因, 有效性) -> 干预类型
        (EscalationReason.CONTROL_TIMEOUT, ControlEffectiveness.INEFFECTIVE): InterventionType.REINFORCE,
        (EscalationReason.THRESHOLD_EXCEEDED, ControlEffectiveness.CRITICAL_FAILURE): InterventionType.TAKEOVER,
        (EscalationReason.RAPID_DETERIORATION, ControlEffectiveness.DETERIORATING): InterventionType.COORDINATE,
        (EscalationReason.CROSS_BOUNDARY_SPREAD, ControlEffectiveness.INEFFECTIVE): InterventionType.COORDINATE,
        (EscalationReason.CASCADING_FAILURE, ControlEffectiveness.CRITICAL_FAILURE): InterventionType.EMERGENCY_SHUTDOWN,
        (EscalationReason.RESOURCE_EXHAUSTED, ControlEffectiveness.PARTIALLY_EFFECTIVE): InterventionType.REINFORCE,
    }

    def __init__(self, layer: int):  # 2 or 3
        self.layer = layer

        # 干预历史
        self.intervention_history: deque = deque(maxlen=200)

        # 活跃干预
        self.active_interventions: Dict[str, InterventionDecision] = {}

    def decide_intervention(self, escalation: EscalationEvent) -> InterventionDecision:
        """决定干预策略"""
        decision_id = f"INT_{self.layer}_{escalation.source_pool}_{int(time.time())}"

        # 查找匹配规则
        key = (escalation.reason, escalation.effectiveness)
        intervention_type = self.INTERVENTION_RULES.get(key, InterventionType.REINFORCE)

        # 根据紧急程度调整
        if escalation.urgency >= 9:
            if intervention_type == InterventionType.REINFORCE:
                intervention_type = InterventionType.TAKEOVER
            elif intervention_type == InterventionType.COORDINATE:
                intervention_type = InterventionType.TAKEOVER

        decision = InterventionDecision(
            decision_id=decision_id,
            escalation_event_id=escalation.event_id,
            target_layer=escalation.source_layer,
            intervention_type=intervention_type,
        )

        # 生成干预内容
        self._generate_intervention_content(escalation, decision)

        self.active_interventions[decision_id] = decision
        self.intervention_history.append(decision)

        logger.warning(f"L{self.layer}决定干预L{escalation.source_layer}: "
                       f"{intervention_type.value}, 池{escalation.source_pool}")

        return decision

    def _generate_intervention_content(self,
                                        escalation: EscalationEvent,
                                        decision: InterventionDecision):
        """生成干预内容"""
        pool_id = escalation.source_pool

        if decision.intervention_type == InterventionType.TAKEOVER:
            # 接管：生成更强力的控制命令
            decision.commands = self._generate_takeover_commands(pool_id, escalation)
            decision.relaxed_constraints = {'level_tolerance': 0.5}

        elif decision.intervention_type == InterventionType.REINFORCE:
            # 增援：调用更多资源
            decision.commands = self._generate_reinforce_commands(pool_id, escalation)
            decision.additional_resources = {
                'extra_pools': self._get_adjacent_pools(pool_id),
                'priority_boost': 2,
            }

        elif decision.intervention_type == InterventionType.COORDINATE:
            # 协调：协调相邻区域
            coord_type = self._map_scenario_to_coordination(escalation.scenario_type)
            if coord_type:
                request = CoordinationRequest(
                    request_id=f"COORD_{decision.decision_id}",
                    coordination_type=coord_type,
                    source_pool=pool_id,
                    priority=escalation.urgency,
                    affected_pools=self._get_affected_pools(pool_id, escalation),
                )
                decision.coordinations.append(request)

        elif decision.intervention_type == InterventionType.ADJUST_TARGET:
            # 调整目标：放松约束
            decision.relaxed_constraints = {
                'level_tolerance': 1.0,
                'flow_tolerance': 10.0,
                'quality_threshold': 0.5,
            }
            decision.extended_deadline = 600.0  # 延长10分钟

        elif decision.intervention_type == InterventionType.EMERGENCY_SHUTDOWN:
            # 紧急停机：隔离
            decision.commands = self._generate_shutdown_commands(pool_id)

    def _generate_takeover_commands(self,
                                     pool_id: int,
                                     escalation: EscalationEvent) -> List[L1ActionCommand]:
        """生成接管命令"""
        commands = []

        # 强制闸门调整
        if escalation.metrics and escalation.metrics.level_error > 0:
            # 水位过高，开大闸门
            commands.append(L1ActionCommand(
                command_id=f"TAKE_OPEN_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.GATE_OPEN,
                pool_id=pool_id,
                priority=10,
                gate_position=0.9,
            ))
        else:
            # 水位过低，关小闸门
            commands.append(L1ActionCommand(
                command_id=f"TAKE_CLOSE_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.GATE_ADJUST,
                pool_id=pool_id,
                priority=10,
                gate_position=0.3,
            ))

        # 加强监测
        commands.append(L1ActionCommand(
            command_id=f"TAKE_MONITOR_{pool_id}_{int(time.time())}",
            action_type=L1ActionType.MONITOR_ENHANCE,
            pool_id=pool_id,
            priority=10,
        ))

        return commands

    def _generate_reinforce_commands(self,
                                      pool_id: int,
                                      escalation: EscalationEvent) -> List[L1ActionCommand]:
        """生成增援命令"""
        commands = []

        # 上下游协调
        adjacent = self._get_adjacent_pools(pool_id)
        for adj_pool in adjacent:
            commands.append(L1ActionCommand(
                command_id=f"REINF_ADJ_{adj_pool}_{int(time.time())}",
                action_type=L1ActionType.COORDINATE_UPSTREAM if adj_pool < pool_id else L1ActionType.COORDINATE_DOWNSTREAM,
                pool_id=adj_pool,
                priority=8,
            ))

        return commands

    def _generate_shutdown_commands(self, pool_id: int) -> List[L1ActionCommand]:
        """生成紧急停机命令"""
        commands = [
            L1ActionCommand(
                command_id=f"SHUT_ISO_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.ISOLATE_BOTH,
                pool_id=pool_id,
                priority=10,
            ),
            L1ActionCommand(
                command_id=f"SHUT_ALARM_{pool_id}_{int(time.time())}",
                action_type=L1ActionType.ALARM_ESCALATE,
                pool_id=pool_id,
                priority=10,
            ),
        ]
        return commands

    def _get_adjacent_pools(self, pool_id: int, num_pools: int = 60) -> List[int]:
        """获取相邻池"""
        adjacent = []
        if pool_id > 0:
            adjacent.append(pool_id - 1)
        if pool_id < num_pools - 1:
            adjacent.append(pool_id + 1)
        return adjacent

    def _get_affected_pools(self,
                            pool_id: int,
                            escalation: EscalationEvent,
                            num_pools: int = 60) -> List[int]:
        """获取影响池"""
        if escalation.scenario_type in [
            L1ScenarioType.L1_POLLUTION_DETECTED,
            L1ScenarioType.L1_POLLUTION_TRACKING,
        ]:
            # 污染影响下游
            return list(range(pool_id, min(pool_id + 5, num_pools)))
        else:
            # 默认上下游各2个
            start = max(0, pool_id - 2)
            end = min(pool_id + 3, num_pools)
            return list(range(start, end))

    def _map_scenario_to_coordination(self,
                                       scenario_type: Optional[L1ScenarioType]) -> Optional[CoordinationType]:
        """映射场景到协调类型"""
        if not scenario_type:
            return CoordinationType.FLOW_ADJUSTMENT

        mapping = {
            L1ScenarioType.L1_POLLUTION_DETECTED: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_POLLUTION_TRACKING: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_DISCHARGE_EMERGENCY: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_LEVEL_RAPID_RISE: CoordinationType.FLOOD_CONTROL,
            L1ScenarioType.L1_ICE_BLOCKAGE: CoordinationType.ICE_CONTROL,
            L1ScenarioType.L1_GATE_STUCK: CoordinationType.GATE_FAILURE,
        }
        return mapping.get(scenario_type, CoordinationType.FLOW_ADJUSTMENT)


# ==============================================================================
# 级联控制系统
# ==============================================================================

class CascadeControlSystem:
    """
    级联控制系统

    整合L1/L2/L3，实现:
    1. 各层自主控制
    2. 控制效果评估
    3. 失控上报
    4. 上层干预
    5. 闭环监控
    """

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools

        # 三层控制器
        self.multi_layer = MultiLayerCoordinator(num_pools)

        # 控制效果评估器 (每个池一个)
        self.evaluators: Dict[int, ControlEffectEvaluator] = {
            i: ControlEffectEvaluator(i) for i in range(num_pools)
        }

        # 上层干预决策器
        self.l2_decider = UpperLayerInterventionDecider(layer=2)
        self.l3_decider = UpperLayerInterventionDecider(layer=3)

        # 上报队列
        self.l1_to_l2_escalations: deque = deque(maxlen=100)
        self.l2_to_l3_escalations: deque = deque(maxlen=50)

        # 干预结果
        self.intervention_results: deque = deque(maxlen=200)

        # 统计
        self.stats = {
            'l1_escalations': 0,
            'l2_escalations': 0,
            'l2_interventions': 0,
            'l3_interventions': 0,
            'successful_controls': 0,
            'failed_controls': 0,
        }

        # 设置回调
        self._setup_escalation_callbacks()

    def _setup_escalation_callbacks(self):
        """设置上报回调"""
        # L1控制器的上报回调已在L2协调器中设置
        pass

    def update_pool_metrics(self, pool_id: int, metrics: ControlMetrics):
        """更新池指标"""
        if pool_id in self.evaluators:
            self.evaluators[pool_id].update_metrics(metrics)

    def evaluate_and_escalate(self, pool_id: int) -> Optional[EscalationEvent]:
        """评估并上报"""
        if pool_id not in self.evaluators:
            return None

        evaluator = self.evaluators[pool_id]
        effectiveness, reason = evaluator.evaluate()

        # 检查是否需要上报
        if effectiveness in [
            ControlEffectiveness.INEFFECTIVE,
            ControlEffectiveness.DETERIORATING,
            ControlEffectiveness.CRITICAL_FAILURE,
        ]:
            return self._create_escalation(pool_id, evaluator, effectiveness, reason)

        return None

    def _create_escalation(self,
                           pool_id: int,
                           evaluator: ControlEffectEvaluator,
                           effectiveness: ControlEffectiveness,
                           reason: Optional[EscalationReason]) -> EscalationEvent:
        """创建上报事件"""
        control_duration = 0.0
        if evaluator.control_start_time:
            control_duration = time.time() - evaluator.control_start_time

        escalation = EscalationEvent(
            event_id=f"ESC_L1_{pool_id}_{int(time.time())}",
            source_layer=1,
            source_pool=pool_id,
            source_region=pool_id // 10,
            reason=reason or EscalationReason.THRESHOLD_EXCEEDED,
            effectiveness=effectiveness,
            metrics=evaluator.current_metrics,
            control_duration=control_duration,
            urgency=self._calculate_urgency(effectiveness, reason),
            timestamp=time.time(),
        )

        self.l1_to_l2_escalations.append(escalation)
        self.stats['l1_escalations'] += 1

        logger.warning(f"L1[{pool_id}]上报L2: {reason.value if reason else '未知'}, "
                       f"效果: {effectiveness.value}")

        return escalation

    def _calculate_urgency(self,
                           effectiveness: ControlEffectiveness,
                           reason: Optional[EscalationReason]) -> int:
        """计算紧急程度"""
        urgency = 5

        if effectiveness == ControlEffectiveness.CRITICAL_FAILURE:
            urgency = 10
        elif effectiveness == ControlEffectiveness.DETERIORATING:
            urgency = 8
        elif effectiveness == ControlEffectiveness.INEFFECTIVE:
            urgency = 7

        if reason == EscalationReason.CASCADING_FAILURE:
            urgency = min(10, urgency + 2)
        elif reason == EscalationReason.RAPID_DETERIORATION:
            urgency = min(10, urgency + 1)

        return urgency

    def process_l1_escalations(self) -> List[InterventionDecision]:
        """处理L1上报"""
        decisions = []

        while self.l1_to_l2_escalations:
            escalation = self.l1_to_l2_escalations.popleft()

            # L2决定干预
            decision = self.l2_decider.decide_intervention(escalation)
            decisions.append(decision)
            self.stats['l2_interventions'] += 1

            # 执行干预
            self._execute_intervention(decision)

        return decisions

    def process_l2_escalations(self) -> List[InterventionDecision]:
        """处理L2上报"""
        decisions = []

        while self.l2_to_l3_escalations:
            escalation = self.l2_to_l3_escalations.popleft()

            # L3决定干预
            decision = self.l3_decider.decide_intervention(escalation)
            decisions.append(decision)
            self.stats['l3_interventions'] += 1

            # 执行干预
            self._execute_intervention(decision)

        return decisions

    def _execute_intervention(self, decision: InterventionDecision):
        """执行干预决策"""
        # 执行L1命令
        for cmd in decision.commands:
            region_id = self.multi_layer.l2_coordinator.get_region_for_pool(cmd.pool_id)
            if region_id in self.multi_layer.l2_coordinator.coordinators:
                coordinator = self.multi_layer.l2_coordinator.coordinators[region_id]
                if cmd.pool_id in coordinator.l1_manager.controllers:
                    controller = coordinator.l1_manager.controllers[cmd.pool_id]
                    controller.command_queue.append(cmd)

        # 执行L2协调
        for coord in decision.coordinations:
            region_id = self.multi_layer.l2_coordinator.get_region_for_pool(coord.source_pool)
            if region_id in self.multi_layer.l2_coordinator.coordinators:
                coordinator = self.multi_layer.l2_coordinator.coordinators[region_id]
                coordinator.pending_requests.append(coord)

        decision.is_executed = True
        decision.execution_result = "executed"

        self.intervention_results.append({
            'decision': decision,
            'timestamp': time.time(),
        })

    def escalate_l2_to_l3(self,
                          region_id: int,
                          reason: EscalationReason,
                          affected_pools: List[int]):
        """L2上报L3"""
        escalation = EscalationEvent(
            event_id=f"ESC_L2_{region_id}_{int(time.time())}",
            source_layer=2,
            source_pool=affected_pools[0] if affected_pools else 0,
            source_region=region_id,
            reason=reason,
            effectiveness=ControlEffectiveness.INEFFECTIVE,
            urgency=9,
            timestamp=time.time(),
            data={'affected_pools': affected_pools},
        )

        self.l2_to_l3_escalations.append(escalation)
        self.stats['l2_escalations'] += 1

        logger.warning(f"L2[region{region_id}]上报L3: {reason.value}")

    def control_step(self, dt: float = 60.0) -> Dict[str, Any]:
        """执行一个控制步"""
        start_time = time.time()

        # 1. 多层协同步
        ml_result = self.multi_layer.coordination_step(dt)

        # 2. 评估各池控制效果
        escalations = []
        for pool_id in range(self.num_pools):
            esc = self.evaluate_and_escalate(pool_id)
            if esc:
                escalations.append(esc)

        # 3. 处理L1上报
        l2_decisions = self.process_l1_escalations()

        # 4. 处理L2上报
        l3_decisions = self.process_l2_escalations()

        # 5. 更新统计
        return {
            'control_time': time.time() - start_time,
            'ml_result': ml_result,
            'escalations': len(escalations),
            'l2_interventions': len(l2_decisions),
            'l3_interventions': len(l3_decisions),
            'stats': self.stats.copy(),
        }

    def inject_control_failure(self,
                                pool_id: int,
                                scenario_type: L1ScenarioType,
                                severity: ScenarioSeverity = ScenarioSeverity.CRITICAL):
        """注入控制失败场景 (用于测试)"""
        # 创建失控指标
        metrics = ControlMetrics(
            pool_id=pool_id,
            timestamp=time.time(),
            level_error=1.5,  # 超阈值
            level_rate=0.15,  # 快速变化
            level_trend="rising",
            quality_index=0.4,  # 水质差
            failed_actions=5,
            action_count=10,
        )

        self.update_pool_metrics(pool_id, metrics)

        # 启动控制计时
        self.evaluators[pool_id].start_control()

        # 评估将触发上报
        self.evaluate_and_escalate(pool_id)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        ineffective_pools = []
        for pool_id, evaluator in self.evaluators.items():
            if evaluator.current_effectiveness in [
                ControlEffectiveness.INEFFECTIVE,
                ControlEffectiveness.CRITICAL_FAILURE,
            ]:
                ineffective_pools.append(pool_id)

        return {
            'total_pools': self.num_pools,
            'ineffective_pools': ineffective_pools,
            'pending_l1_escalations': len(self.l1_to_l2_escalations),
            'pending_l2_escalations': len(self.l2_to_l3_escalations),
            'active_l2_interventions': len(self.l2_decider.active_interventions),
            'active_l3_interventions': len(self.l3_decider.active_interventions),
            'stats': self.stats.copy(),
        }


# ==============================================================================
# 扩展场景类型
# ==============================================================================

class ExtendedL1Scenarios:
    """
    扩展的L1场景类型

    增加更多控制失败相关场景
    """

    # 控制失败场景
    CONTROL_FAILURE_SCENARIOS = {
        'GATE_JAMMED_OPEN': {
            'description': '闸门卡死在开启位置',
            'l1_type': L1ScenarioType.L1_GATE_STUCK,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.RESOURCE_EXHAUSTED,
        },
        'GATE_JAMMED_CLOSED': {
            'description': '闸门卡死在关闭位置',
            'l1_type': L1ScenarioType.L1_GATE_STUCK,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.RESOURCE_EXHAUSTED,
        },
        'SENSOR_FAILURE_BLIND': {
            'description': '传感器故障导致盲区',
            'l1_type': L1ScenarioType.L1_GATE_CONTROL_FAIL,
            'severity': ScenarioSeverity.HIGH,
            'escalation_reason': EscalationReason.RESOURCE_EXHAUSTED,
        },
        'POLLUTION_RAPID_SPREAD': {
            'description': '污染快速扩散超出控制',
            'l1_type': L1ScenarioType.L1_POLLUTION_TRACKING,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.CROSS_BOUNDARY_SPREAD,
        },
        'LEVEL_UNCONTROLLABLE_RISE': {
            'description': '水位不可控上涨',
            'l1_type': L1ScenarioType.L1_LEVEL_RAPID_RISE,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.RAPID_DETERIORATION,
        },
        'LEVEL_UNCONTROLLABLE_DROP': {
            'description': '水位不可控下降',
            'l1_type': L1ScenarioType.L1_LEVEL_RAPID_DROP,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.RAPID_DETERIORATION,
        },
        'CASCADING_GATE_FAILURES': {
            'description': '级联闸门故障',
            'l1_type': L1ScenarioType.L1_GATE_CONTROL_FAIL,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.CASCADING_FAILURE,
        },
        'MULTI_POOL_FLOOD': {
            'description': '多池同时超限',
            'l1_type': L1ScenarioType.L1_LEVEL_HIGH,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.CROSS_BOUNDARY_SPREAD,
        },
        'ICE_JAM_BLOCKING': {
            'description': '冰塞完全阻塞',
            'l1_type': L1ScenarioType.L1_ICE_DAM,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.THRESHOLD_EXCEEDED,
        },
        'LEAKAGE_UNCONTAINED': {
            'description': '渗漏无法控制',
            'l1_type': L1ScenarioType.L1_LEAKAGE_PIPE_BURST,
            'severity': ScenarioSeverity.CRITICAL,
            'escalation_reason': EscalationReason.THRESHOLD_EXCEEDED,
        },
    }

    @classmethod
    def get_scenario(cls, scenario_name: str) -> Optional[Dict]:
        """获取场景定义"""
        return cls.CONTROL_FAILURE_SCENARIOS.get(scenario_name)

    @classmethod
    def generate_failure_event(cls,
                                scenario_name: str,
                                pool_id: int) -> Optional[L1ScenarioEvent]:
        """生成失控事件"""
        scenario = cls.get_scenario(scenario_name)
        if not scenario:
            return None

        return L1ScenarioEvent(
            event_id=f"FAIL_{scenario_name}_{pool_id}_{int(time.time())}",
            scenario_type=scenario['l1_type'],
            pool_id=pool_id,
            severity=scenario['severity'],
            measured_value=1.5,  # 超阈值
            threshold_value=1.0,
            timestamp=time.time(),
        )

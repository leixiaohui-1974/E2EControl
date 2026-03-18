# Phase 5.10: Autonomous Decision Engine
# 自主决策引擎 - L5完全自主控制的核心

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
import numpy as np

from .online_learning import OnlineLearningEngine, LearningConfig
from .experience_memory import ExperienceMemory, Experience, ExperienceType, MemoryPriority
from .knowledge_transfer import KnowledgeTransferEngine, Pattern, PatternType

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """决策置信度等级"""
    VERY_LOW = 1  # 需要人工确认
    LOW = 2  # 建议人工审核
    MEDIUM = 3  # 可以执行但需监控
    HIGH = 4  # 可以安全执行
    VERY_HIGH = 5  # 完全自主执行


class DecisionType(Enum):
    """决策类型"""
    CONTROL_ACTION = "control"  # 控制动作
    PARAMETER_ADJUSTMENT = "parameter"  # 参数调整
    MODE_SWITCH = "mode"  # 模式切换
    EMERGENCY_RESPONSE = "emergency"  # 应急响应
    OPTIMIZATION = "optimization"  # 优化决策
    EXPLORATION = "exploration"  # 探索决策
    NO_ACTION = "no_action"  # 不采取行动


class DecisionOutcome(Enum):
    """决策结果"""
    PENDING = "pending"
    EXECUTED = "executed"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    FAILURE = "failure"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class DecisionContext:
    """决策上下文"""
    context_id: str
    timestamp: datetime

    # 系统状态
    system_state: Dict[str, Any]
    pool_states: List[Dict[str, Any]]

    # 环境信息
    weather_conditions: Optional[Dict[str, Any]] = None
    time_of_day: Optional[str] = None
    season: Optional[str] = None

    # 历史信息
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    active_faults: List[Dict[str, Any]] = field(default_factory=list)
    pending_commands: List[Dict[str, Any]] = field(default_factory=list)

    # 约束
    constraints: Dict[str, Any] = field(default_factory=dict)
    objectives: Dict[str, float] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """决策"""
    decision_id: str
    decision_type: DecisionType
    timestamp: datetime

    # 决策内容
    action: Dict[str, Any]
    parameters: Dict[str, float] = field(default_factory=dict)
    target_pools: List[int] = field(default_factory=list)

    # 置信度
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.5

    # 来源
    reasoning: str = ""
    source_patterns: List[str] = field(default_factory=list)
    source_experiences: List[str] = field(default_factory=list)

    # 预期效果
    expected_outcome: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, float] = field(default_factory=dict)

    # 执行状态
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    actual_result: Optional[Dict[str, Any]] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 反馈
    reward: Optional[float] = None
    feedback: Optional[str] = None


@dataclass
class L5AutonomyConfig:
    """L5自主配置"""
    # 自主级别阈值
    min_confidence_for_auto_execute: ConfidenceLevel = ConfidenceLevel.HIGH
    min_confidence_score: float = 0.7

    # 探索设置
    exploration_rate: float = 0.1
    exploration_decay: float = 0.99
    min_exploration_rate: float = 0.01

    # 安全设置
    safety_margin: float = 0.2
    max_action_magnitude: float = 10.0
    require_approval_for_emergency: bool = False

    # 学习设置
    learning_from_outcomes: bool = True
    update_frequency: float = 60.0  # seconds
    min_experiences_for_learning: int = 10

    # 决策设置
    decision_timeout: float = 5.0  # seconds
    max_pending_decisions: int = 100
    use_ensemble_decisions: bool = True


class AutonomousDecisionEngine:
    """自主决策引擎"""

    def __init__(
        self,
        config: Optional[L5AutonomyConfig] = None,
        learning_engine: Optional[OnlineLearningEngine] = None,
        experience_memory: Optional[ExperienceMemory] = None,
        knowledge_engine: Optional[KnowledgeTransferEngine] = None,
    ):
        self.config = config or L5AutonomyConfig()

        # 核心组件
        self.learning_engine = learning_engine or OnlineLearningEngine()
        self.experience_memory = experience_memory or ExperienceMemory()
        self.knowledge_engine = knowledge_engine or KnowledgeTransferEngine()

        # 决策历史
        self._decision_history: deque = deque(maxlen=10000)
        self._pending_decisions: Dict[str, Decision] = {}

        # 状态
        self._lock = threading.RLock()
        self._running = False
        self._decision_thread: Optional[threading.Thread] = None
        self._next_decision_id = 1
        self._current_exploration_rate = self.config.exploration_rate

        # 回调
        self._decision_callbacks: List[Callable[[Decision], None]] = []
        self._approval_callbacks: List[Callable[[Decision], bool]] = []

        # 统计
        self.stats = {
            'decisions_made': 0,
            'decisions_executed': 0,
            'decisions_successful': 0,
            'decisions_failed': 0,
            'decisions_rejected': 0,
            'average_confidence': 0.0,
            'average_reward': 0.0,
            'exploration_decisions': 0,
        }

        logger.info("Autonomous Decision Engine initialized")

    def make_decision(
        self,
        context: DecisionContext,
        force_type: Optional[DecisionType] = None,
    ) -> Decision:
        """做出决策"""
        with self._lock:
            decision_id = f"dec-{self._next_decision_id:08d}"
            self._next_decision_id += 1

            # 分析当前状态
            state_analysis = self._analyze_state(context)

            # 确定决策类型
            if force_type:
                decision_type = force_type
            else:
                decision_type = self._determine_decision_type(context, state_analysis)

            # 探索 vs 利用
            should_explore = np.random.random() < self._current_exploration_rate

            if should_explore and decision_type not in [DecisionType.EMERGENCY_RESPONSE]:
                decision = self._make_exploration_decision(decision_id, context, decision_type)
                self.stats['exploration_decisions'] += 1
            else:
                decision = self._make_exploitation_decision(decision_id, context, decision_type, state_analysis)

            # 评估决策
            self._evaluate_decision(decision, context)

            # 记录决策
            self._decision_history.append(decision)
            self._pending_decisions[decision_id] = decision
            self.stats['decisions_made'] += 1

            # 更新统计
            self._update_statistics()

            # 通知回调
            for callback in self._decision_callbacks:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"Decision callback error: {e}")

            logger.info(f"Decision made: {decision_id}, type={decision_type.value}, confidence={decision.confidence.value}")

            return decision

    def _analyze_state(self, context: DecisionContext) -> Dict[str, Any]:
        """分析当前状态"""
        analysis = {
            'urgency': 'normal',
            'stability': 'stable',
            'trend': 'steady',
            'anomalies': [],
            'opportunities': [],
        }

        # 检查紧急情况
        for pool_state in context.pool_states:
            water_level = pool_state.get('water_level', 3.0)
            if water_level > 4.5 or water_level < 1.0:
                analysis['urgency'] = 'high'
                analysis['anomalies'].append({
                    'pool_id': pool_state.get('pool_id'),
                    'type': 'water_level_extreme',
                    'value': water_level,
                })

        # 检查活动故障
        if context.active_faults:
            analysis['urgency'] = 'high'
            analysis['stability'] = 'unstable'

        # 检查优化机会
        system_efficiency = context.system_state.get('efficiency', 1.0)
        if system_efficiency < 0.8 and analysis['urgency'] == 'normal':
            analysis['opportunities'].append({
                'type': 'efficiency_improvement',
                'potential': 1.0 - system_efficiency,
            })

        return analysis

    def _determine_decision_type(
        self,
        context: DecisionContext,
        analysis: Dict[str, Any],
    ) -> DecisionType:
        """确定决策类型"""
        # 紧急情况优先
        if analysis['urgency'] == 'high':
            if context.active_faults:
                return DecisionType.EMERGENCY_RESPONSE
            return DecisionType.CONTROL_ACTION

        # 不稳定状态
        if analysis['stability'] == 'unstable':
            return DecisionType.CONTROL_ACTION

        # 有优化机会
        if analysis['opportunities']:
            return DecisionType.OPTIMIZATION

        # 正常控制
        return DecisionType.CONTROL_ACTION

    def _make_exploration_decision(
        self,
        decision_id: str,
        context: DecisionContext,
        decision_type: DecisionType,
    ) -> Decision:
        """做出探索性决策"""
        # 随机选择一个动作变体
        base_action = self._get_baseline_action(context, decision_type)

        # 添加随机扰动
        explored_action = {}
        for key, value in base_action.items():
            if isinstance(value, (int, float)):
                # 添加随机扰动 (±20%)
                perturbation = np.random.uniform(-0.2, 0.2)
                explored_action[key] = value * (1 + perturbation)
            else:
                explored_action[key] = value

        return Decision(
            decision_id=decision_id,
            decision_type=decision_type,
            timestamp=datetime.now(),
            action=explored_action,
            target_pools=[ps.get('pool_id', i) for i, ps in enumerate(context.pool_states)],
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.3,
            reasoning="Exploration decision for learning",
        )

    def _make_exploitation_decision(
        self,
        decision_id: str,
        context: DecisionContext,
        decision_type: DecisionType,
        analysis: Dict[str, Any],
    ) -> Decision:
        """做出利用性决策 (基于已有知识)"""
        # 1. 从知识库获取匹配的模式
        applicable_patterns = self.knowledge_engine.get_applicable_patterns(
            context.system_state,
            context='water_network',
            pattern_type=self._map_decision_to_pattern_type(decision_type),
        )

        # 2. 从经验中获取相似情况的最佳动作
        best_action_from_experience = self.experience_memory.get_best_action_for_state(
            context.system_state,
            experience_type=self._map_decision_to_experience_type(decision_type),
        )

        # 3. 综合决策
        if applicable_patterns:
            best_pattern, match_score = applicable_patterns[0]
            action = best_pattern.actions.copy()
            parameters = best_pattern.parameters.copy()
            source_patterns = [best_pattern.pattern_id]
            confidence_score = match_score * best_pattern.success_rate
            reasoning = f"Based on pattern: {best_pattern.name}"
        elif best_action_from_experience:
            action = best_action_from_experience
            parameters = {}
            source_patterns = []
            confidence_score = 0.6
            reasoning = "Based on similar experience"
        else:
            # 使用基线动作
            action = self._get_baseline_action(context, decision_type)
            parameters = {}
            source_patterns = []
            confidence_score = 0.4
            reasoning = "Baseline action (no matching patterns or experiences)"

        # 4. 应用学习到的参数调整
        learned_params = self.learning_engine.get_all_parameters()
        for key, value in learned_params.items():
            if key in parameters or key in action:
                if key in parameters:
                    parameters[key] = value
                if key in action and isinstance(action[key], (int, float)):
                    action[key] = value

        # 5. 确定置信度等级
        if confidence_score >= 0.8:
            confidence = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            confidence = ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.VERY_LOW

        return Decision(
            decision_id=decision_id,
            decision_type=decision_type,
            timestamp=datetime.now(),
            action=action,
            parameters=parameters,
            target_pools=[ps.get('pool_id', i) for i, ps in enumerate(context.pool_states)],
            confidence=confidence,
            confidence_score=confidence_score,
            reasoning=reasoning,
            source_patterns=source_patterns,
        )

    def _get_baseline_action(
        self,
        context: DecisionContext,
        decision_type: DecisionType,
    ) -> Dict[str, Any]:
        """获取基线动作"""
        if decision_type == DecisionType.EMERGENCY_RESPONSE:
            return {
                'mode': 'emergency',
                'gate_action': 'safety_position',
                'alert': True,
            }
        elif decision_type == DecisionType.CONTROL_ACTION:
            return {
                'mode': 'normal',
                'gate_adjustment': 0.0,
                'setpoint_tracking': True,
            }
        elif decision_type == DecisionType.OPTIMIZATION:
            return {
                'mode': 'optimize',
                'objective': 'efficiency',
                'constraint': 'safety',
            }
        elif decision_type == DecisionType.PARAMETER_ADJUSTMENT:
            return {
                'action': 'tune',
                'target': 'controller_gains',
            }
        else:
            return {'action': 'monitor'}

    def _map_decision_to_pattern_type(self, decision_type: DecisionType) -> Optional[PatternType]:
        """映射决策类型到模式类型"""
        mapping = {
            DecisionType.CONTROL_ACTION: PatternType.CONTROL_STRATEGY,
            DecisionType.EMERGENCY_RESPONSE: PatternType.FAULT_RESPONSE,
            DecisionType.OPTIMIZATION: PatternType.OPTIMIZATION,
            DecisionType.PARAMETER_ADJUSTMENT: PatternType.OPTIMIZATION,
        }
        return mapping.get(decision_type)

    def _map_decision_to_experience_type(self, decision_type: DecisionType) -> Optional[ExperienceType]:
        """映射决策类型到经验类型"""
        mapping = {
            DecisionType.CONTROL_ACTION: ExperienceType.NORMAL_OPERATION,
            DecisionType.EMERGENCY_RESPONSE: ExperienceType.EMERGENCY_RESPONSE,
            DecisionType.OPTIMIZATION: ExperienceType.OPTIMIZATION,
        }
        return mapping.get(decision_type)

    def _evaluate_decision(self, decision: Decision, context: DecisionContext):
        """评估决策"""
        # 计算风险
        risk = {}

        # 检查动作幅度
        for key, value in decision.action.items():
            if isinstance(value, (int, float)):
                if abs(value) > self.config.max_action_magnitude:
                    risk['magnitude_risk'] = abs(value) / self.config.max_action_magnitude

        # 检查安全约束
        for constraint_name, constraint_value in context.constraints.items():
            if constraint_name in decision.action:
                if decision.action[constraint_name] > constraint_value:
                    risk['constraint_violation'] = constraint_name

        decision.risk_assessment = risk

        # 预期结果
        if decision.source_patterns:
            pattern = self.knowledge_engine.knowledge_base.get_pattern(decision.source_patterns[0])
            if pattern:
                decision.expected_outcome = {
                    'success_rate': pattern.success_rate,
                    'confidence': pattern.confidence.value,
                }

    def execute_decision(
        self,
        decision_id: str,
        executor: Optional[Callable[[Decision], bool]] = None,
    ) -> bool:
        """执行决策"""
        with self._lock:
            decision = self._pending_decisions.get(decision_id)
            if not decision:
                logger.warning(f"Decision not found: {decision_id}")
                return False

            # 检查是否需要审批
            if decision.confidence.value < self.config.min_confidence_for_auto_execute.value:
                # 需要审批
                approved = self._request_approval(decision)
                if not approved:
                    decision.outcome = DecisionOutcome.REJECTED
                    self.stats['decisions_rejected'] += 1
                    logger.info(f"Decision rejected: {decision_id}")
                    return False

            # 执行决策
            decision.executed_at = datetime.now()

            try:
                if executor:
                    success = executor(decision)
                else:
                    # 模拟执行
                    success = True

                if success:
                    decision.outcome = DecisionOutcome.EXECUTED
                    self.stats['decisions_executed'] += 1
                    logger.info(f"Decision executed: {decision_id}")
                else:
                    decision.outcome = DecisionOutcome.FAILURE
                    self.stats['decisions_failed'] += 1
                    logger.warning(f"Decision execution failed: {decision_id}")

                return success

            except Exception as e:
                logger.error(f"Decision execution error: {e}")
                decision.outcome = DecisionOutcome.FAILURE
                self.stats['decisions_failed'] += 1
                return False

    def _request_approval(self, decision: Decision) -> bool:
        """请求审批"""
        for callback in self._approval_callbacks:
            try:
                if callback(decision):
                    return True
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        # 默认: 高置信度自动批准
        return decision.confidence.value >= ConfidenceLevel.MEDIUM.value

    def provide_feedback(
        self,
        decision_id: str,
        success: bool,
        reward: float,
        actual_result: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
    ):
        """提供决策反馈"""
        with self._lock:
            decision = self._pending_decisions.get(decision_id)
            if not decision:
                # 搜索历史
                for d in self._decision_history:
                    if d.decision_id == decision_id:
                        decision = d
                        break

            if not decision:
                logger.warning(f"Decision not found for feedback: {decision_id}")
                return

            # 更新决策结果
            decision.completed_at = datetime.now()
            decision.outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
            decision.reward = reward
            decision.actual_result = actual_result
            decision.feedback = feedback

            if success:
                self.stats['decisions_successful'] += 1
            else:
                self.stats['decisions_failed'] += 1

            # 存储经验
            if self.config.learning_from_outcomes:
                self._store_experience(decision)

            # 更新知识
            if decision.source_patterns:
                for pattern_id in decision.source_patterns:
                    self.knowledge_engine.update_pattern_feedback(
                        pattern_id,
                        success=success,
                        performance_delta=reward,
                    )

            # 添加学习样本
            if actual_result:
                self.learning_engine.add_sample(
                    inputs=actual_result.get('state', {}),
                    target=reward,
                    prediction=decision.confidence_score,
                )

            # 从决策中移除
            if decision_id in self._pending_decisions:
                del self._pending_decisions[decision_id]

            # 更新探索率
            self._current_exploration_rate = max(
                self.config.min_exploration_rate,
                self._current_exploration_rate * self.config.exploration_decay,
            )

            self._update_statistics()
            logger.info(f"Feedback received for {decision_id}: success={success}, reward={reward:.3f}")

    def _store_experience(self, decision: Decision):
        """存储决策经验"""
        experience_type = self._map_decision_to_experience_type(decision.decision_type)
        if not experience_type:
            experience_type = ExperienceType.NORMAL_OPERATION

        priority = MemoryPriority.NORMAL
        if decision.decision_type == DecisionType.EMERGENCY_RESPONSE:
            priority = MemoryPriority.HIGH
        if decision.reward is not None and abs(decision.reward) > 0.8:
            priority = MemoryPriority.HIGH

        self.experience_memory.store(
            state=decision.actual_result.get('state', {}) if decision.actual_result else {},
            action=decision.action,
            reward=decision.reward or 0.0,
            next_state=decision.actual_result.get('next_state') if decision.actual_result else None,
            experience_type=experience_type,
            priority=priority,
            context={
                'decision_id': decision.decision_id,
                'decision_type': decision.decision_type.value,
                'confidence': decision.confidence.value,
            },
        )

    def _update_statistics(self):
        """更新统计"""
        if self.stats['decisions_made'] > 0:
            # 计算平均置信度
            recent_decisions = list(self._decision_history)[-100:]
            if recent_decisions:
                self.stats['average_confidence'] = sum(
                    d.confidence_score for d in recent_decisions
                ) / len(recent_decisions)

            # 计算平均奖励
            rewarded_decisions = [d for d in recent_decisions if d.reward is not None]
            if rewarded_decisions:
                self.stats['average_reward'] = sum(
                    d.reward for d in rewarded_decisions
                ) / len(rewarded_decisions)

    def on_decision(self, callback: Callable[[Decision], None]):
        """注册决策回调"""
        self._decision_callbacks.append(callback)

    def on_approval_request(self, callback: Callable[[Decision], bool]):
        """注册审批回调"""
        self._approval_callbacks.append(callback)

    def start(self):
        """启动自主决策引擎"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.learning_engine.start()
            self.experience_memory.start()

            logger.info("Autonomous Decision Engine started")

    def stop(self):
        """停止自主决策引擎"""
        with self._lock:
            self._running = False
            self.learning_engine.stop()
            self.experience_memory.stop()

            logger.info("Autonomous Decision Engine stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'pending_decisions': len(self._pending_decisions),
            'decision_history_size': len(self._decision_history),
            'exploration_rate': self._current_exploration_rate,
            'running': self._running,
            'learning_stats': self.learning_engine.get_statistics(),
            'memory_stats': self.experience_memory.get_statistics(),
            'knowledge_stats': self.knowledge_engine.get_statistics(),
        }


class L5Controller:
    """L5级自主控制器 - 整合所有自主学习组件"""

    def __init__(
        self,
        num_pools: int = 5,
        config: Optional[L5AutonomyConfig] = None,
    ):
        self.num_pools = num_pools
        self.config = config or L5AutonomyConfig()

        # 初始化核心组件
        self.learning_engine = OnlineLearningEngine()
        self.experience_memory = ExperienceMemory()
        self.knowledge_engine = KnowledgeTransferEngine()
        self.decision_engine = AutonomousDecisionEngine(
            config=self.config,
            learning_engine=self.learning_engine,
            experience_memory=self.experience_memory,
            knowledge_engine=self.knowledge_engine,
        )

        # 初始化水网特定组件
        self._initialize_water_network()

        # 状态
        self._lock = threading.RLock()
        self._running = False
        self._control_thread: Optional[threading.Thread] = None
        self._control_interval = 1.0  # seconds

        # 当前状态
        self.current_context: Optional[DecisionContext] = None
        self.last_decision: Optional[Decision] = None

        # 统计
        self.stats = {
            'control_cycles': 0,
            'total_reward': 0.0,
            'start_time': None,
        }

        logger.info(f"L5 Controller initialized for {num_pools} pools")

    def _initialize_water_network(self):
        """初始化水网特定配置"""
        # 创建学习参数
        self.learning_engine.create_water_network_parameters(self.num_pools)

        # 创建知识模式
        self.knowledge_engine.create_water_network_patterns(self.num_pools)

    def start(self):
        """启动L5控制器"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.stats['start_time'] = datetime.now()
            self.decision_engine.start()

            self._control_thread = threading.Thread(
                target=self._control_loop,
                daemon=True,
            )
            self._control_thread.start()

            logger.info("L5 Controller started")

    def stop(self):
        """停止L5控制器"""
        with self._lock:
            self._running = False
            self.decision_engine.stop()

            if self._control_thread:
                self._control_thread.join(timeout=5.0)
                self._control_thread = None

            logger.info("L5 Controller stopped")

    def _control_loop(self):
        """控制主循环"""
        while self._running:
            try:
                if self.current_context:
                    # 做出决策
                    decision = self.decision_engine.make_decision(self.current_context)

                    # 自动执行高置信度决策
                    if decision.confidence.value >= self.config.min_confidence_for_auto_execute.value:
                        self.decision_engine.execute_decision(decision.decision_id)

                    self.last_decision = decision
                    self.stats['control_cycles'] += 1

                time.sleep(self._control_interval)

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(1.0)

    def update_state(
        self,
        system_state: Dict[str, Any],
        pool_states: List[Dict[str, Any]],
        weather: Optional[Dict[str, Any]] = None,
        faults: Optional[List[Dict[str, Any]]] = None,
    ):
        """更新系统状态"""
        with self._lock:
            self.current_context = DecisionContext(
                context_id=f"ctx-{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                system_state=system_state,
                pool_states=pool_states,
                weather_conditions=weather,
                active_faults=faults or [],
            )

    def provide_reward(self, reward: float, result: Optional[Dict[str, Any]] = None):
        """提供奖励反馈"""
        if self.last_decision:
            self.decision_engine.provide_feedback(
                decision_id=self.last_decision.decision_id,
                success=reward > 0.5,
                reward=reward,
                actual_result=result,
            )
            self.stats['total_reward'] += reward

    def get_current_decision(self) -> Optional[Decision]:
        """获取当前决策"""
        return self.last_decision

    def get_learned_parameters(self) -> Dict[str, float]:
        """获取学习到的参数"""
        return self.learning_engine.get_all_parameters()

    def get_statistics(self) -> Dict[str, Any]:
        """获取L5控制器统计"""
        runtime = None
        if self.stats['start_time']:
            runtime = (datetime.now() - self.stats['start_time']).total_seconds()

        return {
            **self.stats,
            'runtime_seconds': runtime,
            'average_reward': self.stats['total_reward'] / max(self.stats['control_cycles'], 1),
            'running': self._running,
            'decision_engine_stats': self.decision_engine.get_statistics(),
        }

    def export_knowledge(self) -> Dict[str, Any]:
        """导出所有学习到的知识"""
        return {
            'model_parameters': self.learning_engine.export_model(),
            'experiences': self.experience_memory.export_experiences(),
            'knowledge_base': self.knowledge_engine.knowledge_base.export(),
            'statistics': self.get_statistics(),
            'exported_at': datetime.now().isoformat(),
        }

    def import_knowledge(self, knowledge: Dict[str, Any]):
        """导入知识"""
        if 'model_parameters' in knowledge:
            self.learning_engine.import_model(knowledge['model_parameters'])

        if 'experiences' in knowledge:
            self.experience_memory.import_experiences(knowledge['experiences'])

        if 'knowledge_base' in knowledge:
            self.knowledge_engine.knowledge_base.import_patterns(knowledge['knowledge_base'])

        logger.info("Knowledge imported successfully")

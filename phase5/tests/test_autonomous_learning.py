# Phase 5.10: Autonomous Learning Tests
# L5自主学习模块测试

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np


class TestOnlineLearning:
    """在线学习引擎测试"""

    def test_learning_config_creation(self):
        """测试学习配置创建"""
        from phase5.autonomous_learning.online_learning import (
            LearningConfig, LearningAlgorithm, LearningRate
        )

        config = LearningConfig(
            algorithm=LearningAlgorithm.ADAM,
            learning_rate=0.001,
            learning_rate_strategy=LearningRate.ADAPTIVE,
        )

        assert config.algorithm == LearningAlgorithm.ADAM
        assert config.learning_rate == 0.001
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999

    def test_adaptive_optimizer(self):
        """测试自适应优化器"""
        from phase5.autonomous_learning.online_learning import (
            AdaptiveOptimizer, LearningConfig, LearningAlgorithm
        )

        config = LearningConfig(algorithm=LearningAlgorithm.SGD)
        optimizer = AdaptiveOptimizer(config)

        # 注册参数
        optimizer.register_parameter('test_param', 1.0, bounds=(0.0, 10.0))

        assert 'test_param' in optimizer.parameters
        assert optimizer.parameters['test_param'].value == 1.0

    def test_optimizer_step(self):
        """测试优化步骤"""
        from phase5.autonomous_learning.online_learning import (
            AdaptiveOptimizer, LearningConfig, LearningAlgorithm
        )

        config = LearningConfig(algorithm=LearningAlgorithm.SGD, learning_rate=0.1)
        optimizer = AdaptiveOptimizer(config)
        optimizer.register_parameter('param', 5.0)

        old_value, new_value = optimizer.step('param', gradient=1.0)

        assert old_value == 5.0
        assert new_value < old_value  # 应该减小 (梯度为正)

    def test_learning_engine_initialization(self):
        """测试学习引擎初始化"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine = OnlineLearningEngine()

        assert engine is not None
        assert engine.stats['samples_processed'] == 0
        assert engine.stats['updates_performed'] == 0

    def test_register_parameter(self):
        """测试参数注册"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine = OnlineLearningEngine()
        engine.register_parameter('gain', 1.5, bounds=(0.1, 10.0), importance=2.0)

        assert engine.get_parameter('gain') == 1.5
        assert 'gain' in engine.get_all_parameters()

    def test_add_sample(self):
        """测试添加样本"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine = OnlineLearningEngine()
        engine.add_sample(
            inputs={'x': 1.0, 'y': 2.0},
            target=3.0,
            prediction=2.8,
            weight=1.0,
        )

        assert engine.stats['samples_processed'] == 1

    def test_learn_step(self):
        """测试学习步骤"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine = OnlineLearningEngine()
        engine.register_parameter('param', 1.0)

        update = engine.learn_step('param', gradient=0.5)

        assert update is not None
        assert update.parameter_name == 'param'
        assert engine.stats['updates_performed'] == 1

    def test_create_water_network_parameters(self):
        """测试创建水网参数"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine = OnlineLearningEngine()
        params = engine.create_water_network_parameters(num_pools=3)

        assert len(params) > 0
        assert 'global_safety_margin' in params
        assert 'pool_0_level_kp' in params
        assert 'pool_2_level_ki' in params

    def test_export_import_model(self):
        """测试模型导出导入"""
        from phase5.autonomous_learning.online_learning import OnlineLearningEngine

        engine1 = OnlineLearningEngine()
        engine1.register_parameter('param1', 1.5)
        engine1.register_parameter('param2', 2.5)

        # 导出
        model_state = engine1.export_model()
        assert 'parameters' in model_state
        assert model_state['parameters']['param1'] == 1.5

        # 导入
        engine2 = OnlineLearningEngine()
        engine2.register_parameter('param1', 0.0)
        engine2.register_parameter('param2', 0.0)
        engine2.import_model(model_state)

        assert engine2.get_parameter('param1') == 1.5
        assert engine2.get_parameter('param2') == 2.5


class TestExperienceMemory:
    """经验记忆系统测试"""

    def test_experience_creation(self):
        """测试经验创建"""
        from phase5.autonomous_learning.experience_memory import (
            Experience, ExperienceType, MemoryPriority, OutcomeQuality
        )

        exp = Experience(
            experience_id="exp-001",
            timestamp=datetime.now(),
            experience_type=ExperienceType.NORMAL_OPERATION,
            priority=MemoryPriority.NORMAL,
            state={'water_level': 3.0},
            action={'gate_position': 50.0},
            reward=0.8,
        )

        assert exp.experience_id == "exp-001"
        assert exp.outcome == OutcomeQuality.GOOD  # reward 0.8 -> GOOD

    def test_memory_initialization(self):
        """测试记忆系统初始化"""
        from phase5.autonomous_learning.experience_memory import ExperienceMemory

        memory = ExperienceMemory(capacity=1000)

        assert memory.capacity == 1000
        assert len(memory.short_term) == 0
        assert len(memory.long_term) == 0

    def test_store_experience(self):
        """测试存储经验"""
        from phase5.autonomous_learning.experience_memory import (
            ExperienceMemory, ExperienceType
        )

        memory = ExperienceMemory()

        exp = memory.store(
            state={'level': 3.0},
            action={'gate': 50},
            reward=0.7,
            experience_type=ExperienceType.NORMAL_OPERATION,
        )

        assert exp is not None
        assert memory.stats['experiences_added'] == 1
        assert len(memory.short_term) == 1

    def test_recall_experiences(self):
        """测试回忆经验"""
        from phase5.autonomous_learning.experience_memory import (
            ExperienceMemory, ExperienceType
        )

        memory = ExperienceMemory()

        # 存储多个经验
        for i in range(5):
            memory.store(
                state={'level': 3.0 + i * 0.1},
                action={'gate': 50 + i},
                reward=0.5 + i * 0.1,
                experience_type=ExperienceType.NORMAL_OPERATION,
            )

        # 回忆
        results = memory.recall(min_reward=0.7, limit=3)

        assert len(results) <= 3
        assert all(e.reward >= 0.7 for e in results)

    def test_recall_similar(self):
        """测试相似经验回忆"""
        from phase5.autonomous_learning.experience_memory import ExperienceMemory

        memory = ExperienceMemory()

        # 存储经验
        memory.store(
            state={'level': 3.0, 'flow': 5.0},
            action={'gate': 50},
            reward=0.8,
        )
        memory.store(
            state={'level': 3.1, 'flow': 5.1},
            action={'gate': 52},
            reward=0.9,
        )
        memory.store(
            state={'level': 1.0, 'flow': 10.0},
            action={'gate': 100},
            reward=0.5,
        )

        # 查找相似
        results = memory.recall_similar({'level': 3.05, 'flow': 5.05}, top_k=2)

        assert len(results) >= 1
        # 最相似的应该是前两个经验

    def test_replay_buffer_sampling(self):
        """测试经验回放采样"""
        from phase5.autonomous_learning.experience_memory import (
            ExperienceMemory, ExperienceType
        )

        memory = ExperienceMemory()

        # 存储多个经验
        for i in range(20):
            memory.store(
                state={'i': i},
                action={'a': i},
                reward=i / 20.0,
            )

        # 采样
        batch = memory.sample_batch(5)

        assert len(batch) == 5

    def test_memory_consolidation(self):
        """测试记忆整合"""
        from phase5.autonomous_learning.experience_memory import (
            ExperienceMemory, MemoryPriority
        )

        memory = ExperienceMemory()

        # 存储高优先级经验
        memory.store(
            state={'critical': True},
            action={'emergency': True},
            reward=0.95,
            priority=MemoryPriority.CRITICAL,
        )

        # 关键经验应该直接进入长期记忆
        assert len(memory.long_term) == 1


class TestKnowledgeTransfer:
    """知识迁移引擎测试"""

    def test_pattern_creation(self):
        """测试模式创建"""
        from phase5.autonomous_learning.knowledge_transfer import (
            Pattern, PatternType, PatternConfidence
        )

        pattern = Pattern(
            pattern_id="pat-001",
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Test Pattern",
            conditions={'level': {'min': 2.0, 'max': 4.0}},
            actions={'gate': 'adjust'},
            parameters={'kp': 1.0},
        )

        assert pattern.pattern_id == "pat-001"
        assert pattern.pattern_type == PatternType.CONTROL_STRATEGY

    def test_pattern_statistics_update(self):
        """测试模式统计更新"""
        from phase5.autonomous_learning.knowledge_transfer import Pattern, PatternType

        pattern = Pattern(
            pattern_id="pat-001",
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Test",
            conditions={},
            actions={},
        )

        # 更新统计
        pattern.update_statistics(success=True)
        pattern.update_statistics(success=True)
        pattern.update_statistics(success=False)

        assert pattern.use_count == 3
        assert pattern.success_count == 2
        assert pattern.success_rate == 2/3

    def test_knowledge_base(self):
        """测试知识库"""
        from phase5.autonomous_learning.knowledge_transfer import (
            KnowledgeBase, Pattern, PatternType
        )

        kb = KnowledgeBase("test")

        pattern = Pattern(
            pattern_id="",
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Test Pattern",
            conditions={},
            actions={},
        )

        pattern_id = kb.add_pattern(pattern)

        assert pattern_id is not None
        assert kb.get_pattern(pattern_id) is not None

    def test_knowledge_transfer_engine_initialization(self):
        """测试知识迁移引擎初始化"""
        from phase5.autonomous_learning.knowledge_transfer import KnowledgeTransferEngine

        engine = KnowledgeTransferEngine()

        assert engine is not None
        assert engine.stats['patterns_learned'] == 0

    def test_learn_pattern(self):
        """测试学习模式"""
        from phase5.autonomous_learning.knowledge_transfer import (
            KnowledgeTransferEngine, PatternType
        )

        engine = KnowledgeTransferEngine()

        pattern = engine.learn_pattern(
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Learned Pattern",
            conditions={'level': {'min': 2.0}},
            actions={'adjust': True},
            source_context='test_context',
        )

        assert pattern is not None
        assert engine.stats['patterns_learned'] == 1

    def test_transfer_pattern(self):
        """测试模式迁移"""
        from phase5.autonomous_learning.knowledge_transfer import (
            KnowledgeTransferEngine, PatternType, TransferStrategy
        )

        engine = KnowledgeTransferEngine()

        # 创建源模式
        pattern = engine.learn_pattern(
            pattern_type=PatternType.CONTROL_STRATEGY,
            name="Source Pattern",
            conditions={'level': {'min': 2.0}},
            actions={'adjust': True},
            parameters={'kp': 1.0},
            source_context='source',
        )

        # 迁移到新上下文
        result = engine.transfer(
            pattern_id=pattern.pattern_id,
            target_context='target',
            strategy=TransferStrategy.DIRECT,
        )

        assert result.success
        assert result.adapted_pattern is not None

    def test_create_water_network_patterns(self):
        """测试创建水网模式"""
        from phase5.autonomous_learning.knowledge_transfer import KnowledgeTransferEngine

        engine = KnowledgeTransferEngine()
        patterns = engine.create_water_network_patterns(num_pools=3)

        assert len(patterns) >= 5
        assert any('Flood' in p.name for p in patterns)
        assert any('Drought' in p.name for p in patterns)


class TestAutonomousDecision:
    """自主决策引擎测试"""

    def test_decision_context_creation(self):
        """测试决策上下文创建"""
        from phase5.autonomous_learning.autonomous_decision import DecisionContext

        context = DecisionContext(
            context_id="ctx-001",
            timestamp=datetime.now(),
            system_state={'mode': 'normal'},
            pool_states=[{'pool_id': 0, 'water_level': 3.0}],
        )

        assert context.context_id == "ctx-001"
        assert len(context.pool_states) == 1

    def test_decision_creation(self):
        """测试决策创建"""
        from phase5.autonomous_learning.autonomous_decision import (
            Decision, DecisionType, ConfidenceLevel
        )

        decision = Decision(
            decision_id="dec-001",
            decision_type=DecisionType.CONTROL_ACTION,
            timestamp=datetime.now(),
            action={'gate_adjustment': 5.0},
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
        )

        assert decision.decision_id == "dec-001"
        assert decision.confidence == ConfidenceLevel.HIGH

    def test_decision_engine_initialization(self):
        """测试决策引擎初始化"""
        from phase5.autonomous_learning.autonomous_decision import AutonomousDecisionEngine

        engine = AutonomousDecisionEngine()

        assert engine is not None
        assert engine.stats['decisions_made'] == 0

    def test_make_decision(self):
        """测试做出决策"""
        from phase5.autonomous_learning.autonomous_decision import (
            AutonomousDecisionEngine, DecisionContext
        )

        engine = AutonomousDecisionEngine()

        context = DecisionContext(
            context_id="ctx-001",
            timestamp=datetime.now(),
            system_state={'mode': 'normal', 'efficiency': 0.9},
            pool_states=[
                {'pool_id': 0, 'water_level': 3.0},
                {'pool_id': 1, 'water_level': 3.5},
            ],
        )

        decision = engine.make_decision(context)

        assert decision is not None
        assert decision.decision_id is not None
        assert engine.stats['decisions_made'] == 1

    def test_execute_decision(self):
        """测试执行决策"""
        from phase5.autonomous_learning.autonomous_decision import (
            AutonomousDecisionEngine, DecisionContext, ConfidenceLevel
        )

        engine = AutonomousDecisionEngine()

        context = DecisionContext(
            context_id="ctx-001",
            timestamp=datetime.now(),
            system_state={'mode': 'normal'},
            pool_states=[{'pool_id': 0, 'water_level': 3.0}],
        )

        decision = engine.make_decision(context)

        # 强制设置高置信度
        decision.confidence = ConfidenceLevel.HIGH
        decision.confidence_score = 0.9

        success = engine.execute_decision(decision.decision_id)

        assert success
        assert engine.stats['decisions_executed'] == 1

    def test_provide_feedback(self):
        """测试提供反馈"""
        from phase5.autonomous_learning.autonomous_decision import (
            AutonomousDecisionEngine, DecisionContext, DecisionOutcome
        )

        engine = AutonomousDecisionEngine()

        context = DecisionContext(
            context_id="ctx-001",
            timestamp=datetime.now(),
            system_state={'mode': 'normal'},
            pool_states=[{'pool_id': 0, 'water_level': 3.0}],
        )

        decision = engine.make_decision(context)
        engine.execute_decision(decision.decision_id)

        # 提供反馈
        engine.provide_feedback(
            decision_id=decision.decision_id,
            success=True,
            reward=0.9,
            actual_result={'state': {'level': 3.2}},
        )

        assert decision.reward == 0.9
        assert decision.outcome == DecisionOutcome.SUCCESS


class TestL5Controller:
    """L5控制器测试"""

    def test_l5_controller_initialization(self):
        """测试L5控制器初始化"""
        from phase5.autonomous_learning.autonomous_decision import L5Controller

        controller = L5Controller(num_pools=3)

        assert controller is not None
        assert controller.num_pools == 3

    def test_update_state(self):
        """测试更新状态"""
        from phase5.autonomous_learning.autonomous_decision import L5Controller

        controller = L5Controller(num_pools=2)

        controller.update_state(
            system_state={'mode': 'normal'},
            pool_states=[
                {'pool_id': 0, 'water_level': 3.0},
                {'pool_id': 1, 'water_level': 3.5},
            ],
        )

        assert controller.current_context is not None
        assert len(controller.current_context.pool_states) == 2

    def test_get_learned_parameters(self):
        """测试获取学习参数"""
        from phase5.autonomous_learning.autonomous_decision import L5Controller

        controller = L5Controller(num_pools=2)
        params = controller.get_learned_parameters()

        assert len(params) > 0
        assert 'global_safety_margin' in params

    def test_export_import_knowledge(self):
        """测试知识导出导入"""
        from phase5.autonomous_learning.autonomous_decision import L5Controller

        controller1 = L5Controller(num_pools=2)

        # 导出知识
        knowledge = controller1.export_knowledge()

        assert 'model_parameters' in knowledge
        assert 'knowledge_base' in knowledge

        # 导入知识
        controller2 = L5Controller(num_pools=2)
        controller2.import_knowledge(knowledge)

    def test_statistics(self):
        """测试统计信息"""
        from phase5.autonomous_learning.autonomous_decision import L5Controller

        controller = L5Controller(num_pools=2)
        stats = controller.get_statistics()

        assert 'control_cycles' in stats
        assert 'total_reward' in stats
        assert 'decision_engine_stats' in stats


class TestModuleImports:
    """模块导入测试"""

    def test_import_all_components(self):
        """测试导入所有组件"""
        from phase5.autonomous_learning import (
            # Online Learning
            OnlineLearningEngine,
            LearningAlgorithm,
            ModelUpdate,
            LearningRate,
            AdaptiveOptimizer,
            # Experience Memory
            ExperienceMemory,
            Experience,
            ExperienceType,
            MemoryPriority,
            ReplayBuffer,
            # Knowledge Transfer
            KnowledgeTransferEngine,
            KnowledgeBase,
            Pattern,
            PatternType,
            TransferStrategy,
            # Autonomous Decision
            AutonomousDecisionEngine,
            DecisionContext,
            Decision,
            ConfidenceLevel,
            DecisionOutcome,
            L5Controller,
        )

        assert OnlineLearningEngine is not None
        assert ExperienceMemory is not None
        assert KnowledgeTransferEngine is not None
        assert AutonomousDecisionEngine is not None
        assert L5Controller is not None

    def test_module_all_exports(self):
        """测试模块__all__导出"""
        from phase5 import autonomous_learning

        expected_exports = [
            'OnlineLearningEngine',
            'ExperienceMemory',
            'KnowledgeTransferEngine',
            'AutonomousDecisionEngine',
            'L5Controller',
        ]

        for name in expected_exports:
            assert name in autonomous_learning.__all__


class TestIntegration:
    """集成测试"""

    def test_full_learning_cycle(self):
        """测试完整学习周期"""
        from phase5.autonomous_learning.autonomous_decision import (
            L5Controller, DecisionContext
        )

        controller = L5Controller(num_pools=2)

        # 更新状态
        controller.update_state(
            system_state={'mode': 'normal', 'efficiency': 0.85},
            pool_states=[
                {'pool_id': 0, 'water_level': 3.0, 'flow_rate': 5.0},
                {'pool_id': 1, 'water_level': 3.5, 'flow_rate': 4.5},
            ],
        )

        # 做出决策
        decision = controller.decision_engine.make_decision(controller.current_context)
        assert decision is not None

        # 执行决策
        controller.decision_engine.execute_decision(decision.decision_id)

        # 提供反馈
        controller.decision_engine.provide_feedback(
            decision_id=decision.decision_id,
            success=True,
            reward=0.85,
            actual_result={'state': {'efficiency': 0.9}},
        )

        # 检查学习效果
        stats = controller.get_statistics()
        assert stats['decision_engine_stats']['decisions_made'] >= 1

    def test_knowledge_accumulation(self):
        """测试知识积累"""
        from phase5.autonomous_learning.knowledge_transfer import (
            KnowledgeTransferEngine, PatternType
        )

        engine = KnowledgeTransferEngine()

        # 从多次经验中学习
        experiences = []
        for i in range(10):
            experiences.append({
                'state': {'level': 3.0 + i * 0.05, 'flow': 5.0},
                'action': {'gate': 50 + i},
                'reward': 0.8 if i < 7 else 0.3,
            })

        patterns = engine.extract_patterns_from_experiences(experiences, min_occurrences=3)

        # 应该至少提取出一个成功模式
        assert len(patterns) >= 1

    def test_emergency_response_decision(self):
        """测试应急响应决策"""
        from phase5.autonomous_learning.autonomous_decision import (
            AutonomousDecisionEngine, DecisionContext, DecisionType
        )

        engine = AutonomousDecisionEngine()

        # 创建紧急上下文 (高水位)
        context = DecisionContext(
            context_id="emergency-001",
            timestamp=datetime.now(),
            system_state={'mode': 'normal'},
            pool_states=[
                {'pool_id': 0, 'water_level': 5.0},  # 高水位!
            ],
            active_faults=[{'type': 'high_water_level'}],
        )

        decision = engine.make_decision(context)

        # 应该是紧急响应决策
        assert decision.decision_type == DecisionType.EMERGENCY_RESPONSE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

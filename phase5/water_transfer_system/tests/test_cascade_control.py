"""
级联控制系统测试
Tests for Cascade Control System
"""

import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from phase5.water_transfer_system.cascade_control import (
    CascadeControlSystem,
    ControlEffectiveness,
    ControlEffectEvaluator,
    ControlMetrics,
    EscalationEvent,
    EscalationReason,
    InterventionDecision,
    InterventionType,
    UpperLayerInterventionDecider,
    ExtendedL1Scenarios,
)
from phase5.water_transfer_system.local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
)
from phase5.water_transfer_system.core_types import ScenarioSeverity


class TestControlEffectiveness(unittest.TestCase):
    """控制有效性枚举测试"""

    def test_all_effectiveness_defined(self):
        """测试所有有效性状态已定义"""
        expected = [
            'EFFECTIVE', 'PARTIALLY_EFFECTIVE', 'INEFFECTIVE',
            'DETERIORATING', 'CRITICAL_FAILURE'
        ]
        for name in expected:
            self.assertTrue(hasattr(ControlEffectiveness, name))


class TestEscalationReason(unittest.TestCase):
    """上报原因枚举测试"""

    def test_all_reasons_defined(self):
        """测试所有上报原因已定义"""
        expected = [
            'CONTROL_TIMEOUT', 'THRESHOLD_EXCEEDED', 'RAPID_DETERIORATION',
            'RESOURCE_EXHAUSTED', 'CROSS_BOUNDARY_SPREAD', 'CASCADING_FAILURE',
            'MANUAL_ESCALATION'
        ]
        for name in expected:
            self.assertTrue(hasattr(EscalationReason, name))


class TestInterventionType(unittest.TestCase):
    """干预类型枚举测试"""

    def test_all_intervention_types_defined(self):
        """测试所有干预类型已定义"""
        expected = [
            'TAKEOVER', 'REINFORCE', 'COORDINATE',
            'ADJUST_TARGET', 'EMERGENCY_SHUTDOWN'
        ]
        for name in expected:
            self.assertTrue(hasattr(InterventionType, name))


class TestControlMetrics(unittest.TestCase):
    """控制指标测试"""

    def test_metrics_creation(self):
        """测试指标创建"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.3,
            quality_index=0.8,
        )
        self.assertEqual(metrics.pool_id, 5)
        self.assertEqual(metrics.level_error, 0.3)

    def test_effectiveness_score_perfect(self):
        """测试完美控制分数"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.0,
            quality_index=1.0,
            action_count=5,
            failed_actions=0,
        )
        score = metrics.get_effectiveness_score()
        self.assertGreaterEqual(score, 0.9)

    def test_effectiveness_score_poor(self):
        """测试差控制分数"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.8,
            level_trend="rising",
            quality_index=0.4,
            action_count=10,
            failed_actions=5,
        )
        score = metrics.get_effectiveness_score()
        self.assertLess(score, 0.5)


class TestControlEffectEvaluator(unittest.TestCase):
    """控制效果评估器测试"""

    def setUp(self):
        self.evaluator = ControlEffectEvaluator(pool_id=5)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.evaluator.pool_id, 5)
        self.assertEqual(self.evaluator.current_effectiveness, ControlEffectiveness.EFFECTIVE)

    def test_update_metrics(self):
        """测试更新指标"""
        metrics = ControlMetrics(pool_id=5, timestamp=time.time())
        self.evaluator.update_metrics(metrics)
        self.assertEqual(self.evaluator.current_metrics, metrics)

    def test_evaluate_effective(self):
        """测试评估-有效控制"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.1,
            quality_index=0.9,
        )
        self.evaluator.update_metrics(metrics)
        effectiveness, reason = self.evaluator.evaluate()
        self.assertEqual(effectiveness, ControlEffectiveness.EFFECTIVE)
        self.assertIsNone(reason)

    def test_evaluate_critical_failure(self):
        """测试评估-严重失控"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=1.5,  # 超临界阈值
            quality_index=0.2,  # 低于临界值
        )
        self.evaluator.update_metrics(metrics)
        effectiveness, reason = self.evaluator.evaluate()
        self.assertEqual(effectiveness, ControlEffectiveness.CRITICAL_FAILURE)
        self.assertEqual(reason, EscalationReason.THRESHOLD_EXCEEDED)

    def test_evaluate_timeout(self):
        """测试评估-控制超时"""
        self.evaluator.start_control()
        # 模拟超时
        self.evaluator.control_start_time = time.time() - 400  # 超过300秒

        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.6,  # 较大偏差
            level_trend="rising",  # 恶化趋势
            quality_index=0.5,
            action_count=10,
            failed_actions=5,  # 高失败率
        )
        self.evaluator.update_metrics(metrics)
        effectiveness, reason = self.evaluator.evaluate()
        self.assertEqual(effectiveness, ControlEffectiveness.INEFFECTIVE)
        self.assertEqual(reason, EscalationReason.CONTROL_TIMEOUT)

    def test_start_and_reset(self):
        """测试开始和重置"""
        self.evaluator.start_control()
        self.assertIsNotNone(self.evaluator.control_start_time)

        self.evaluator.reset()
        self.assertIsNone(self.evaluator.control_start_time)


class TestUpperLayerInterventionDecider(unittest.TestCase):
    """上层干预决策器测试"""

    def setUp(self):
        self.l2_decider = UpperLayerInterventionDecider(layer=2)
        self.l3_decider = UpperLayerInterventionDecider(layer=3)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.l2_decider.layer, 2)
        self.assertEqual(self.l3_decider.layer, 3)

    def test_decide_takeover(self):
        """测试决定接管"""
        escalation = EscalationEvent(
            event_id="ESC_001",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.THRESHOLD_EXCEEDED,
            effectiveness=ControlEffectiveness.CRITICAL_FAILURE,
            urgency=10,
        )
        decision = self.l2_decider.decide_intervention(escalation)
        self.assertEqual(decision.intervention_type, InterventionType.TAKEOVER)
        self.assertGreater(len(decision.commands), 0)

    def test_decide_reinforce(self):
        """测试决定增援"""
        escalation = EscalationEvent(
            event_id="ESC_002",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.CONTROL_TIMEOUT,
            effectiveness=ControlEffectiveness.INEFFECTIVE,
            urgency=6,
        )
        decision = self.l2_decider.decide_intervention(escalation)
        self.assertEqual(decision.intervention_type, InterventionType.REINFORCE)

    def test_decide_coordinate(self):
        """测试决定协调"""
        escalation = EscalationEvent(
            event_id="ESC_003",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.CROSS_BOUNDARY_SPREAD,
            effectiveness=ControlEffectiveness.INEFFECTIVE,
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            urgency=7,
        )
        decision = self.l2_decider.decide_intervention(escalation)
        self.assertEqual(decision.intervention_type, InterventionType.COORDINATE)
        self.assertGreater(len(decision.coordinations), 0)

    def test_decide_emergency_shutdown(self):
        """测试决定紧急停机"""
        escalation = EscalationEvent(
            event_id="ESC_004",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.CASCADING_FAILURE,
            effectiveness=ControlEffectiveness.CRITICAL_FAILURE,
            urgency=10,
        )
        decision = self.l2_decider.decide_intervention(escalation)
        self.assertEqual(decision.intervention_type, InterventionType.EMERGENCY_SHUTDOWN)

    def test_urgency_affects_decision(self):
        """测试紧急程度影响决策"""
        # 低紧急程度
        low_escalation = EscalationEvent(
            event_id="ESC_005",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.CONTROL_TIMEOUT,
            effectiveness=ControlEffectiveness.INEFFECTIVE,
            urgency=5,
        )
        low_decision = self.l2_decider.decide_intervention(low_escalation)

        # 高紧急程度
        high_escalation = EscalationEvent(
            event_id="ESC_006",
            source_layer=1,
            source_pool=5,
            reason=EscalationReason.CONTROL_TIMEOUT,
            effectiveness=ControlEffectiveness.INEFFECTIVE,
            urgency=9,
        )
        high_decision = self.l2_decider.decide_intervention(high_escalation)

        # 高紧急程度应该升级为TAKEOVER
        self.assertEqual(high_decision.intervention_type, InterventionType.TAKEOVER)


class TestCascadeControlSystem(unittest.TestCase):
    """级联控制系统测试"""

    def setUp(self):
        self.system = CascadeControlSystem(num_pools=20)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.system.num_pools, 20)
        self.assertEqual(len(self.system.evaluators), 20)
        self.assertIsNotNone(self.system.l2_decider)
        self.assertIsNotNone(self.system.l3_decider)

    def test_update_pool_metrics(self):
        """测试更新池指标"""
        metrics = ControlMetrics(pool_id=5, timestamp=time.time())
        self.system.update_pool_metrics(5, metrics)
        self.assertEqual(self.system.evaluators[5].current_metrics, metrics)

    def test_evaluate_and_escalate_no_issue(self):
        """测试评估-无问题不上报"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=0.1,
            quality_index=0.9,
        )
        self.system.update_pool_metrics(5, metrics)
        escalation = self.system.evaluate_and_escalate(5)
        self.assertIsNone(escalation)

    def test_evaluate_and_escalate_failure(self):
        """测试评估-失控上报"""
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=1.5,
            quality_index=0.2,
        )
        self.system.update_pool_metrics(5, metrics)
        escalation = self.system.evaluate_and_escalate(5)
        self.assertIsNotNone(escalation)
        self.assertEqual(escalation.source_pool, 5)

    def test_process_l1_escalations(self):
        """测试处理L1上报"""
        # 触发上报
        metrics = ControlMetrics(
            pool_id=5,
            timestamp=time.time(),
            level_error=1.5,
            quality_index=0.2,
        )
        self.system.update_pool_metrics(5, metrics)
        self.system.evaluate_and_escalate(5)

        # 处理上报
        decisions = self.system.process_l1_escalations()
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].is_executed)

    def test_inject_control_failure(self):
        """测试注入控制失败"""
        self.system.inject_control_failure(
            pool_id=5,
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.CRITICAL
        )

        # 应该产生上报
        self.assertGreater(len(self.system.l1_to_l2_escalations), 0)

    def test_control_step(self):
        """测试控制步"""
        result = self.system.control_step(dt=60.0)

        self.assertIn('control_time', result)
        self.assertIn('escalations', result)
        self.assertIn('l2_interventions', result)
        self.assertIn('stats', result)

    def test_escalate_l2_to_l3(self):
        """测试L2上报L3"""
        self.system.escalate_l2_to_l3(
            region_id=0,
            reason=EscalationReason.CROSS_BOUNDARY_SPREAD,
            affected_pools=[5, 6, 7]
        )
        self.assertEqual(len(self.system.l2_to_l3_escalations), 1)
        self.assertEqual(self.system.stats['l2_escalations'], 1)

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.system.get_system_status()
        self.assertEqual(status['total_pools'], 20)
        self.assertIn('ineffective_pools', status)
        self.assertIn('stats', status)


class TestExtendedL1Scenarios(unittest.TestCase):
    """扩展L1场景测试"""

    def test_scenarios_defined(self):
        """测试场景已定义"""
        expected_scenarios = [
            'GATE_JAMMED_OPEN',
            'GATE_JAMMED_CLOSED',
            'SENSOR_FAILURE_BLIND',
            'POLLUTION_RAPID_SPREAD',
            'LEVEL_UNCONTROLLABLE_RISE',
            'LEVEL_UNCONTROLLABLE_DROP',
            'CASCADING_GATE_FAILURES',
            'MULTI_POOL_FLOOD',
            'ICE_JAM_BLOCKING',
            'LEAKAGE_UNCONTAINED',
        ]
        for name in expected_scenarios:
            self.assertIn(name, ExtendedL1Scenarios.CONTROL_FAILURE_SCENARIOS)

    def test_get_scenario(self):
        """测试获取场景"""
        scenario = ExtendedL1Scenarios.get_scenario('GATE_JAMMED_OPEN')
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario['l1_type'], L1ScenarioType.L1_GATE_STUCK)
        self.assertEqual(scenario['severity'], ScenarioSeverity.CRITICAL)

    def test_generate_failure_event(self):
        """测试生成失控事件"""
        event = ExtendedL1Scenarios.generate_failure_event('POLLUTION_RAPID_SPREAD', pool_id=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.pool_id, 5)
        self.assertEqual(event.scenario_type, L1ScenarioType.L1_POLLUTION_TRACKING)


class TestCascadeIntegration(unittest.TestCase):
    """级联控制集成测试"""

    def test_l1_failure_escalation_workflow(self):
        """测试L1失控上报工作流"""
        system = CascadeControlSystem(num_pools=20)

        # 1. 注入失控场景
        system.inject_control_failure(
            pool_id=5,
            scenario_type=L1ScenarioType.L1_GATE_STUCK,
            severity=ScenarioSeverity.CRITICAL
        )

        # 2. 处理上报
        decisions = system.process_l1_escalations()

        # 3. 验证干预
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].is_executed)

    def test_multi_pool_failure_escalation(self):
        """测试多池失控上报"""
        system = CascadeControlSystem(num_pools=20)

        # 多池同时失控
        for pool_id in [5, 6, 7]:
            metrics = ControlMetrics(
                pool_id=pool_id,
                timestamp=time.time(),
                level_error=1.5,
                quality_index=0.2,
            )
            system.update_pool_metrics(pool_id, metrics)
            system.evaluate_and_escalate(pool_id)

        # 处理所有上报
        decisions = system.process_l1_escalations()
        self.assertEqual(len(decisions), 3)

    def test_l2_to_l3_escalation_workflow(self):
        """测试L2到L3上报工作流"""
        system = CascadeControlSystem(num_pools=20)

        # 1. L2上报L3
        system.escalate_l2_to_l3(
            region_id=0,
            reason=EscalationReason.CASCADING_FAILURE,
            affected_pools=[0, 1, 2, 3, 4]
        )

        # 2. 处理上报
        decisions = system.process_l2_escalations()

        # 3. 验证L3干预
        self.assertEqual(len(decisions), 1)
        self.assertEqual(system.stats['l3_interventions'], 1)

    def test_full_cascade_control_loop(self):
        """测试完整级联控制循环"""
        system = CascadeControlSystem(num_pools=20)

        # 运行多个控制步
        for _ in range(3):
            result = system.control_step(dt=60.0)
            self.assertIn('control_time', result)

        # 验证统计
        status = system.get_system_status()
        self.assertEqual(status['total_pools'], 20)


if __name__ == '__main__':
    unittest.main()

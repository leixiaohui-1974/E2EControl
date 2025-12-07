"""
多层协同控制器测试
Tests for Multi-Layer Coordinator
"""

import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from phase5.water_transfer_system.multi_layer_coordinator import (
    MultiLayerCoordinator,
    MultiLayerEvent,
    MultiLayerEventType,
    MultiLayerDecision,
    DecisionPriority,
    L3GlobalScheduler,
    ScenarioCombinationGenerator,
    IntelligentDecisionEngine,
)
from phase5.water_transfer_system.l2_l1_coordinator import (
    CoordinationType,
    CoordinationRequest,
)
from phase5.water_transfer_system.local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
    L1ScenarioEvent,
)
from phase5.water_transfer_system.core_types import ScenarioSeverity


class TestMultiLayerEventType(unittest.TestCase):
    """多层事件类型测试"""

    def test_all_event_types_defined(self):
        """测试所有事件类型已定义"""
        expected_types = [
            'L1_LOCAL_ANOMALY', 'L1_SENSOR_ALERT', 'L1_GATE_ACTION',
            'L2_REGIONAL_COORDINATION', 'L2_CROSS_POOL_SPREAD', 'L2_EMERGENCY_RESPONSE',
            'L3_GLOBAL_OPTIMIZATION', 'L3_DEMAND_CHANGE', 'L3_MAINTENANCE_PLAN',
            'CROSS_LAYER_ESCALATION', 'CROSS_LAYER_DIRECTIVE', 'CROSS_REGION_COORDINATION',
        ]
        for type_name in expected_types:
            self.assertTrue(hasattr(MultiLayerEventType, type_name))


class TestDecisionPriority(unittest.TestCase):
    """决策优先级测试"""

    def test_priority_ordering(self):
        """测试优先级排序"""
        priorities = [
            DecisionPriority.OPTIMIZATION,
            DecisionPriority.LOW,
            DecisionPriority.NORMAL,
            DecisionPriority.HIGH,
            DecisionPriority.EMERGENCY,
            DecisionPriority.SAFETY_CRITICAL,
        ]
        for i in range(len(priorities) - 1):
            self.assertLess(priorities[i].value, priorities[i + 1].value)


class TestMultiLayerEvent(unittest.TestCase):
    """多层事件测试"""

    def test_event_creation(self):
        """测试事件创建"""
        event = MultiLayerEvent(
            event_id="TEST_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            l1_scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
        )
        self.assertEqual(event.event_id, "TEST_001")
        self.assertEqual(event.source_layer, 1)
        self.assertEqual(event.source_pool, 5)
        self.assertFalse(event.is_processed)


class TestL3GlobalScheduler(unittest.TestCase):
    """L3全局调度器测试"""

    def setUp(self):
        self.scheduler = L3GlobalScheduler(num_pools=60, num_regions=6)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.scheduler.num_pools, 60)
        self.assertEqual(self.scheduler.num_regions, 6)

    def test_receive_l2_report(self):
        """测试接收L2报告"""
        report = {
            'emergency_mode': True,
            'requires_global_coordination': True,
        }
        self.scheduler.receive_l2_report(region_id=0, report=report)
        self.assertEqual(self.scheduler.global_state['active_emergencies'], 1)
        # 全局协调为区域内每个池创建指令
        self.assertGreater(len(self.scheduler.pending_directives), 0)

    def test_optimize_global_schedule(self):
        """测试全局调度优化"""
        # 第一次应该优化
        result = self.scheduler.optimize_global_schedule()
        self.assertTrue(result['optimized'])

        # 立即再次调用不应该优化 (间隔不够)
        result = self.scheduler.optimize_global_schedule()
        self.assertFalse(result['optimized'])

    def test_process_maintenance_plan(self):
        """测试检修计划"""
        plan = self.scheduler.process_maintenance_plan(
            pool_id=5,
            start_time=time.time(),
            duration=3600.0
        )
        self.assertIn(5, self.scheduler.global_state['maintenance_pools'])
        self.assertIn(5, plan.directives)  # 检修池应该有指令


class TestMultiLayerCoordinator(unittest.TestCase):
    """多层协同控制器测试"""

    def setUp(self):
        self.coordinator = MultiLayerCoordinator(num_pools=60)

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.coordinator.l3_scheduler)
        self.assertIsNotNone(self.coordinator.l2_coordinator)
        self.assertEqual(self.coordinator.num_pools, 60)

    def test_receive_event(self):
        """测试接收事件"""
        event = MultiLayerEvent(
            event_id="TEST_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            priority=DecisionPriority.NORMAL,
        )
        self.coordinator.receive_event(event)
        self.assertEqual(len(self.coordinator.event_queue), 1)
        self.assertEqual(self.coordinator.stats['events_received'], 1)

    def test_receive_urgent_event(self):
        """测试接收紧急事件"""
        event = MultiLayerEvent(
            event_id="URGENT_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            l1_scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.CRITICAL,
            priority=DecisionPriority.EMERGENCY,
            affected_pools=[5, 6, 7],
        )
        self.coordinator.receive_event(event)
        # 紧急事件应该立即处理
        self.assertGreater(self.coordinator.stats['decisions_made'], 0)

    def test_inject_l1_event(self):
        """测试注入L1事件"""
        l1_event = L1ScenarioEvent(
            event_id="L1_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
        )
        self.coordinator.inject_l1_event(5, l1_event)
        self.assertEqual(self.coordinator.stats['events_received'], 1)

    def test_inject_l2_coordination(self):
        """测试注入L2协调"""
        self.coordinator.inject_l2_coordination(
            region_id=0,
            coord_type=CoordinationType.POLLUTION_SPREAD,
            affected_pools=[5, 6, 7]
        )
        self.assertEqual(self.coordinator.stats['events_received'], 1)

    def test_process_events(self):
        """测试处理事件"""
        # 添加多个事件
        for i in range(3):
            event = MultiLayerEvent(
                event_id=f"TEST_{i}",
                event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
                source_layer=1,
                source_pool=i,
                l1_scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
                severity=ScenarioSeverity.MEDIUM,
                priority=DecisionPriority.NORMAL,
            )
            self.coordinator.event_queue.append(event)

        decisions = self.coordinator.process_events()
        self.assertEqual(len(decisions), 3)
        self.assertEqual(self.coordinator.stats['events_processed'], 3)

    def test_execute_decisions(self):
        """测试执行决策"""
        # 创建决策
        decision = MultiLayerDecision(
            decision_id="DEC_001",
            source_event_id="EVENT_001",
            decision_layer=1,
            priority=DecisionPriority.HIGH,
        )
        decision.l1_commands[5] = []

        self.coordinator.decision_queue.append(decision)
        result = self.coordinator.execute_decisions()

        self.assertEqual(result['executed'], 1)

    def test_coordination_step(self):
        """测试协同控制步"""
        result = self.coordinator.coordination_step(dt=60.0)

        self.assertIn('control_time', result)
        self.assertIn('decisions_made', result)
        self.assertIn('l2_results', result)
        self.assertIn('stats', result)

    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.coordinator.get_system_status()

        self.assertIn('pending_events', status)
        self.assertIn('pending_decisions', status)
        self.assertIn('l2_status', status)
        self.assertIn('l3_status', status)

    def test_severity_to_priority(self):
        """测试严重程度到优先级转换"""
        self.assertEqual(
            self.coordinator._severity_to_priority(ScenarioSeverity.LOW),
            DecisionPriority.LOW
        )
        self.assertEqual(
            self.coordinator._severity_to_priority(ScenarioSeverity.CRITICAL),
            DecisionPriority.EMERGENCY
        )


class TestScenarioCombinationGenerator(unittest.TestCase):
    """场景组合生成器测试"""

    def setUp(self):
        self.generator = ScenarioCombinationGenerator(num_pools=60, num_regions=6)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.generator.num_pools, 60)
        self.assertEqual(self.generator.num_regions, 6)

    def test_templates_defined(self):
        """测试模板定义"""
        expected_templates = [
            'POLLUTION_CASCADE',
            'FLOOD_EMERGENCY',
            'ICE_CONTROL',
            'MAINTENANCE_DISCHARGE',
            'GATE_FAILURE_CASCADE',
            'SLOPE_EMERGENCY',
            'LEAKAGE_RESPONSE',
            'MULTI_REGION_COORDINATION',
        ]
        for template in expected_templates:
            self.assertIn(template, self.generator.COMBINATION_TEMPLATES)

    def test_generate_pollution_cascade(self):
        """测试生成污染级联场景"""
        events = self.generator.generate_combination(
            'POLLUTION_CASCADE',
            source_pool=5,
            severity=ScenarioSeverity.HIGH
        )
        self.assertGreater(len(events), 0)
        # 应该有L1, L2, L3三层事件
        layers = set(e.source_layer for e in events)
        self.assertTrue(len(layers) >= 2)

    def test_generate_flood_emergency(self):
        """测试生成防洪应急场景"""
        events = self.generator.generate_combination(
            'FLOOD_EMERGENCY',
            source_pool=10,
            severity=ScenarioSeverity.CRITICAL
        )
        self.assertGreater(len(events), 0)

    def test_generate_ice_control(self):
        """测试生成冰凌控制场景"""
        events = self.generator.generate_combination(
            'ICE_CONTROL',
            source_pool=30,
            severity=ScenarioSeverity.MEDIUM
        )
        self.assertGreater(len(events), 0)

    def test_generate_multi_region_combination(self):
        """测试生成多区域场景组合"""
        events = self.generator.generate_multi_region_combination(
            'POLLUTION_CASCADE',
            regions=[0, 1, 2],
            severity=ScenarioSeverity.HIGH
        )
        # 应该有3个区域的事件
        self.assertGreater(len(events), 3)

    def test_generate_random_combination(self):
        """测试生成随机场景组合"""
        events = self.generator.generate_random_combination(num_events=5)
        self.assertGreater(len(events), 0)

    def test_affected_pools_pollution(self):
        """测试污染影响池计算"""
        affected = self.generator._get_affected_pools(
            source_pool=5,
            l1_type=L1ScenarioType.L1_POLLUTION_DETECTED
        )
        # 污染影响下游
        self.assertIn(5, affected)
        self.assertIn(6, affected)
        self.assertNotIn(4, affected)  # 不影响上游

    def test_affected_pools_ice(self):
        """测试冰凌影响池计算"""
        affected = self.generator._get_affected_pools(
            source_pool=10,
            l1_type=L1ScenarioType.L1_ICE_BLOCKAGE
        )
        # 冰凌影响上下游
        self.assertIn(10, affected)
        self.assertIn(8, affected)  # 上游
        self.assertIn(12, affected)  # 下游


class TestIntelligentDecisionEngine(unittest.TestCase):
    """智能决策引擎测试"""

    def setUp(self):
        self.coordinator = MultiLayerCoordinator(num_pools=60)
        self.engine = IntelligentDecisionEngine(self.coordinator)

    def test_initialization(self):
        """测试初始化"""
        self.assertGreater(len(self.engine.rules), 0)

    def test_evaluate_pollution_event(self):
        """测试评估污染事件"""
        event = MultiLayerEvent(
            event_id="POLL_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            l1_scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
        )

        evaluation = self.engine.evaluate_event(event)
        self.assertIn('POLLUTION_ISOLATION', evaluation['matched_rules'])
        self.assertTrue(evaluation['escalation_needed'])

    def test_evaluate_flood_event(self):
        """测试评估防洪事件"""
        event = MultiLayerEvent(
            event_id="FLOOD_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=10,
            l1_scenario_type=L1ScenarioType.L1_LEVEL_HIGH,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
        )

        evaluation = self.engine.evaluate_event(event)
        self.assertIn('FLOOD_RESPONSE', evaluation['matched_rules'])

    def test_make_intelligent_decision(self):
        """测试智能决策"""
        event = MultiLayerEvent(
            event_id="INT_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            l1_scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
            affected_pools=[5, 6, 7],
        )

        decision = self.engine.make_intelligent_decision(event)
        self.assertIsNotNone(decision)
        self.assertIn(5, decision.l1_commands)

    def test_update_weights(self):
        """测试权重更新"""
        # 先做一个决策
        event = MultiLayerEvent(
            event_id="WEIGHT_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=5,
            l1_scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
        )

        decision = self.engine.make_intelligent_decision(event)
        if decision:
            # 更新权重
            self.engine.update_weights(decision.decision_id, success=True)
            # 验证权重存在
            self.assertGreater(len(self.engine.action_weights), 0)


class TestMultiLayerIntegration(unittest.TestCase):
    """多层集成测试"""

    def test_pollution_cascade_workflow(self):
        """测试污染级联工作流"""
        coordinator = MultiLayerCoordinator(num_pools=60)
        generator = ScenarioCombinationGenerator(num_pools=60)

        # 生成污染级联场景
        events = generator.generate_combination(
            'POLLUTION_CASCADE',
            source_pool=5,
            severity=ScenarioSeverity.HIGH
        )

        # 注入事件
        for event in events:
            coordinator.receive_event(event)

        # 执行协同控制步
        result = coordinator.coordination_step(dt=60.0)

        self.assertGreater(result['decisions_made'], 0)

    def test_multi_region_flood_response(self):
        """测试多区域防洪响应"""
        coordinator = MultiLayerCoordinator(num_pools=60)
        generator = ScenarioCombinationGenerator(num_pools=60)

        # 生成多区域防洪场景
        events = generator.generate_multi_region_combination(
            'FLOOD_EMERGENCY',
            regions=[0, 1],
            severity=ScenarioSeverity.CRITICAL
        )

        for event in events:
            coordinator.receive_event(event)

        result = coordinator.coordination_step(dt=60.0)
        self.assertIn('l2_results', result)

    def test_intelligent_decision_workflow(self):
        """测试智能决策工作流"""
        coordinator = MultiLayerCoordinator(num_pools=60)
        engine = IntelligentDecisionEngine(coordinator)

        # 创建事件
        event = MultiLayerEvent(
            event_id="INT_WORK_001",
            event_type=MultiLayerEventType.L1_LOCAL_ANOMALY,
            source_layer=1,
            source_pool=10,
            l1_scenario_type=L1ScenarioType.L1_GATE_STUCK,
            severity=ScenarioSeverity.HIGH,
            priority=DecisionPriority.HIGH,
            affected_pools=[9, 10, 11],
        )

        # 智能决策
        decision = engine.make_intelligent_decision(event)
        self.assertIsNotNone(decision)

        # 执行决策
        coordinator.decision_queue.append(decision)
        exec_result = coordinator.execute_decisions()
        self.assertEqual(exec_result['executed'], 1)

    def test_full_scenario_run(self):
        """测试完整场景运行"""
        coordinator = MultiLayerCoordinator(num_pools=60)
        generator = ScenarioCombinationGenerator(num_pools=60)

        # 生成随机场景
        events = generator.generate_random_combination(num_events=3)

        for event in events:
            coordinator.receive_event(event)

        # 多个控制步
        for _ in range(3):
            result = coordinator.coordination_step(dt=60.0)
            self.assertIn('stats', result)

        # 验证统计
        status = coordinator.get_system_status()
        self.assertGreater(status['stats']['events_received'], 0)


class TestCrossLayerCommunication(unittest.TestCase):
    """跨层通信测试"""

    def test_l1_to_l2_escalation(self):
        """测试L1到L2上报"""
        coordinator = MultiLayerCoordinator(num_pools=60)

        # 注入高严重程度L1事件
        l1_event = L1ScenarioEvent(
            event_id="ESC_L1_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )
        coordinator.inject_l1_event(5, l1_event)

        # 执行协同步
        result = coordinator.coordination_step(dt=60.0)

        # 应该有L2协调
        self.assertGreater(coordinator.stats['l2_coordinations'], 0)

    def test_l2_to_l3_escalation(self):
        """测试L2到L3上报"""
        coordinator = MultiLayerCoordinator(num_pools=60)

        # 注入需要全局协调的L2事件
        event = MultiLayerEvent(
            event_id="ESC_L2_001",
            event_type=MultiLayerEventType.L2_REGIONAL_COORDINATION,
            source_layer=2,
            source_region=0,
            priority=DecisionPriority.EMERGENCY,
            affected_pools=list(range(10)),
            affected_regions=[0, 1],  # 跨区域
            data={'coord_type': CoordinationType.EMERGENCY_DISCHARGE},
        )
        coordinator.receive_event(event)

        result = coordinator.coordination_step(dt=60.0)
        self.assertIn('l3_optimized', result)

    def test_l3_to_l2_directive(self):
        """测试L3到L2指令"""
        coordinator = MultiLayerCoordinator(num_pools=60)

        # 注入L3指令
        event = MultiLayerEvent(
            event_id="DIR_L3_001",
            event_type=MultiLayerEventType.L3_GLOBAL_OPTIMIZATION,
            source_layer=3,
            priority=DecisionPriority.NORMAL,
            affected_regions=[0, 1, 2],
            data={'action': 'FLOW_ADJUSTMENT'},
        )
        coordinator.receive_event(event)

        result = coordinator.coordination_step(dt=60.0)
        self.assertGreater(result['decisions_made'], 0)


if __name__ == '__main__':
    unittest.main()

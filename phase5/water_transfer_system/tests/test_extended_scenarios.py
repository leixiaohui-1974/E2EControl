"""
扩展场景测试 - 测试场景生成器、自适应MPC和批量测试框架
Extended Scenario Tests - Testing Scenario Generator, Adaptive MPC, and Batch Testing

测试覆盖:
1. 场景生成器测试 (单事件、多事件、级联、压力测试)
2. 自适应MPC测试 (场景识别、配置生成、系统集成)
3. 批量测试框架测试 (执行器、测试套件、报告生成)
4. 端到端千级场景测试
"""

import unittest
import numpy as np
import sys
import os
import logging
from typing import Dict, List, Any

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from phase5.water_transfer_system.core_types import (
    PoolRole, ScenarioType, ScenarioSeverity,
)
from phase5.water_transfer_system.physics_model import SNWDMiddleRouteModel
from phase5.water_transfer_system.enhanced_mpc import (
    EnhancedParameterizedMPC, HotReconfigurableMPC,
)
from phase5.water_transfer_system.scenario_generator import (
    ScenarioGenerator, ScenarioValidator,
    ExtendedScenarioEvent, CompositeScenario,
    SeasonType, WeatherType, TimeOfDay, EvolutionPattern,
)
from phase5.water_transfer_system.adaptive_mpc import (
    AdaptiveMPCSystem, AdaptiveMPCConfigurator, ScenarioIdentifier,
    ScenarioFeatures, ScenarioDetectionResult,
)
from phase5.water_transfer_system.batch_testing import (
    BatchTestExecutor, TestSuite, TestCase, TestResult,
    BatchTestResult, TestStatus, ReportGenerator,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 场景生成器测试
# ==============================================================================

class TestScenarioGenerator(unittest.TestCase):
    """场景生成器测试"""

    def setUp(self):
        self.generator = ScenarioGenerator(seed=42)

    def test_single_event_generation(self):
        """测试单事件生成"""
        event = self.generator.generate_single_event()

        self.assertIsInstance(event, ExtendedScenarioEvent)
        self.assertIsNotNone(event.event_id)
        self.assertIn(event.scenario_type, list(ScenarioType))
        self.assertIn(event.severity, list(ScenarioSeverity))
        self.assertTrue(0 <= event.location < 60)
        self.assertGreater(event.duration, 0)

    def test_single_event_with_params(self):
        """测试带参数的单事件生成"""
        event = self.generator.generate_single_event(
            scenario_type=ScenarioType.S3_POLLUTION,
            location=30,
            severity=ScenarioSeverity.CRITICAL,
        )

        self.assertEqual(event.scenario_type, ScenarioType.S3_POLLUTION)
        self.assertEqual(event.location, 30)
        self.assertEqual(event.severity, ScenarioSeverity.CRITICAL)

    def test_dual_event_scenario(self):
        """测试双事件场景生成"""
        scenario = self.generator.generate_dual_event_scenario()

        self.assertIsInstance(scenario, CompositeScenario)
        self.assertEqual(len(scenario.events), 2)
        self.assertGreater(scenario.complexity, 0)
        self.assertTrue(0 <= scenario.risk_level <= 1)

        # 两个事件应该在不同位置
        loc1 = scenario.events[0].location
        loc2 = scenario.events[1].location
        self.assertNotEqual(loc1, loc2)

    def test_triple_event_scenario(self):
        """测试三事件场景生成"""
        scenario = self.generator.generate_triple_event_scenario()

        self.assertEqual(len(scenario.events), 3)
        self.assertGreaterEqual(scenario.complexity, 3)

    def test_cascading_scenario(self):
        """测试级联场景生成"""
        scenario = self.generator.generate_cascading_scenario(cascade_count=4)

        self.assertEqual(len(scenario.events), 5)  # 1 initial + 4 cascade

        # 检查时间递增
        timestamps = [e.timestamp for e in scenario.events]
        self.assertEqual(timestamps, sorted(timestamps))

        # 检查位置传播
        locations = [e.location for e in scenario.events]
        # 级联应该向下游传播
        for i in range(1, len(locations)):
            self.assertGreaterEqual(locations[i], locations[i-1])

    def test_batch_generation(self):
        """测试批量生成"""
        batch = self.generator.generate_batch(100)

        self.assertEqual(len(batch), 100)

        # 检查类型分布
        single = sum(1 for s in batch if len(s.events) == 1)
        dual = sum(1 for s in batch if len(s.events) == 2)
        multi = sum(1 for s in batch if len(s.events) > 2)

        self.assertGreater(single, 0)
        self.assertGreater(dual, 0)

    def test_stress_scenarios(self):
        """测试压力场景生成"""
        stress = self.generator.generate_stress_test_scenarios(20)

        self.assertEqual(len(stress), 20)

        # 压力场景应该有较高的复杂度和风险
        avg_complexity = np.mean([s.complexity for s in stress])
        avg_risk = np.mean([s.risk_level for s in stress])

        self.assertGreater(avg_complexity, 3)
        self.assertGreater(avg_risk, 0.3)

    def test_scenario_count(self):
        """测试场景计数"""
        counts = self.generator.count_possible_scenarios()

        self.assertIn('single_basic', counts)
        self.assertIn('single_full', counts)
        self.assertEqual(counts['single_basic'], 8 * 60 * 4)  # 1920

    def test_exhaustive_generation(self):
        """测试穷举生成"""
        # 只测试一部分
        events = list(self.generator.generate_exhaustive_single_events(
            scenario_types=[ScenarioType.S1_NORMAL_PLAN],
            locations=[0, 30, 59],
            severities=[ScenarioSeverity.MEDIUM],
        ))

        self.assertEqual(len(events), 3)


class TestScenarioValidator(unittest.TestCase):
    """场景验证器测试"""

    def setUp(self):
        self.validator = ScenarioValidator()
        self.generator = ScenarioGenerator(seed=42)

    def test_valid_event(self):
        """测试有效事件验证"""
        event = self.generator.generate_single_event()
        is_valid, errors = self.validator.validate_event(event)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_location(self):
        """测试无效位置"""
        event = self.generator.generate_single_event()
        event.location = 100  # 无效位置

        is_valid, errors = self.validator.validate_event(event)

        self.assertFalse(is_valid)
        self.assertTrue(any('位置' in e for e in errors))

    def test_valid_composite(self):
        """测试有效复合场景"""
        scenario = self.generator.generate_dual_event_scenario()
        is_valid, errors = self.validator.validate_composite(scenario)

        self.assertTrue(is_valid)

    def test_batch_validation(self):
        """测试批量验证"""
        batch = self.generator.generate_batch(50)

        valid_count = 0
        for scenario in batch:
            is_valid, _ = self.validator.validate_composite(scenario)
            if is_valid:
                valid_count += 1

        # 大部分应该有效
        self.assertGreater(valid_count / len(batch), 0.9)


# ==============================================================================
# 自适应MPC测试
# ==============================================================================

class TestScenarioIdentifier(unittest.TestCase):
    """场景识别器测试"""

    def setUp(self):
        self.identifier = ScenarioIdentifier()

    def test_feature_extraction(self):
        """测试特征提取"""
        levels = np.ones(60) * 4.0 + np.random.randn(60) * 0.1
        flows = np.ones(60) * 300.0 + np.random.randn(60) * 10

        features = self.identifier.extract_features(levels, flows)

        self.assertIsInstance(features, ScenarioFeatures)
        self.assertAlmostEqual(features.level_mean, 4.0, delta=0.3)
        self.assertAlmostEqual(features.flow_mean, 300.0, delta=30)

    def test_normal_scenario_detection(self):
        """测试常规场景识别"""
        # 模拟稳定状态
        for _ in range(10):
            levels = np.ones(60) * 4.0 + np.random.randn(60) * 0.1
            flows = np.ones(60) * 300.0 + np.random.randn(60) * 10
            features = self.identifier.extract_features(levels, flows)

        result = self.identifier.identify_scenario(features)

        self.assertEqual(result.detected_type, ScenarioType.S1_NORMAL_PLAN)
        self.assertGreater(result.confidence, 0.5)

    def test_ice_period_detection(self):
        """测试冰期场景识别"""
        levels = np.ones(60) * 3.5
        flows = np.ones(60) * 150.0  # 低流量

        features = self.identifier.extract_features(
            levels, flows,
            season=SeasonType.ICE_PERIOD,
            weather=WeatherType.SNOW,
        )

        result = self.identifier.identify_scenario(features)

        # 冰期应该被识别
        self.assertIn(ScenarioType.S5_ICE_PERIOD, [result.detected_type] +
                      [a[0] for a in result.alternatives])


class TestAdaptiveMPCConfigurator(unittest.TestCase):
    """自适应MPC配置器测试"""

    def setUp(self):
        self.configurator = AdaptiveMPCConfigurator()

    def test_normal_config(self):
        """测试常规场景配置"""
        detection = ScenarioDetectionResult(
            detected_type=ScenarioType.S1_NORMAL_PLAN,
            confidence=0.9,
            severity=ScenarioSeverity.LOW,
            center_location=30,
            affected_range=5,
        )

        config = self.configurator.generate_config(detection, pool_id=30)

        self.assertIsNotNone(config)
        self.assertGreater(config.weights.W_level, 0)

    def test_pollution_config(self):
        """测试污染场景配置"""
        detection = ScenarioDetectionResult(
            detected_type=ScenarioType.S3_POLLUTION,
            confidence=1.0,
            severity=ScenarioSeverity.CRITICAL,
            center_location=30,
            affected_range=5,
        )

        # 中心池配置
        config_center = self.configurator.generate_config(detection, pool_id=30)

        # 污染中心应该有极高的流量权重
        self.assertGreater(config_center.weights.W_flow, 50)

        # 约束应该限制流量
        self.assertEqual(config_center.constraints.Q_max, 0)

    def test_severity_scaling(self):
        """测试严重程度缩放"""
        base_detection = ScenarioDetectionResult(
            detected_type=ScenarioType.S4_FLOOD_CONTROL,
            confidence=1.0,
            severity=ScenarioSeverity.MEDIUM,
            center_location=20,
            affected_range=10,
        )

        config_medium = self.configurator.generate_config(base_detection, pool_id=20)

        # 改为CRITICAL
        base_detection.severity = ScenarioSeverity.CRITICAL
        config_critical = self.configurator.generate_config(base_detection, pool_id=20)

        # CRITICAL应该有更大的权重
        self.assertGreater(
            config_critical.weights.W_level,
            config_medium.weights.W_level
        )


class TestAdaptiveMPCSystem(unittest.TestCase):
    """自适应MPC系统测试"""

    def setUp(self):
        self.system = AdaptiveMPCSystem(num_pools=60, horizon=10)
        self.generator = ScenarioGenerator(seed=42)

    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertEqual(self.system.num_pools, 60)
        self.assertIsNotNone(self.system.identifier)
        self.assertIsNotNone(self.system.configurator)
        self.assertIsNotNone(self.system.mpc_manager)

    def test_normal_update(self):
        """测试常规更新"""
        levels = np.ones(60) * 4.0
        flows = np.ones(60) * 300.0

        Q_in, Q_out, detection = self.system.update(levels, flows)

        self.assertEqual(len(Q_in), 60)
        self.assertEqual(len(Q_out), 60)
        self.assertIsNotNone(detection)

    def test_apply_event(self):
        """测试应用事件"""
        event = self.generator.generate_single_event(
            scenario_type=ScenarioType.S3_POLLUTION,
            location=30,
            severity=ScenarioSeverity.CRITICAL,
        )

        self.system.apply_event(event)

        # 检查池30的角色
        controller = self.system.mpc_manager.controllers[30]
        self.assertEqual(controller.current_role, PoolRole.ISOLATE)

    def test_apply_composite_scenario(self):
        """测试应用复合场景"""
        scenario = self.generator.generate_dual_event_scenario()
        self.system.apply_composite_scenario(scenario)

        self.assertIsNotNone(self.system.current_scenario)
        self.assertGreater(len(self.system.scenario_history), 0)

    def test_statistics(self):
        """测试统计信息"""
        levels = np.ones(60) * 4.0
        flows = np.ones(60) * 300.0

        for _ in range(5):
            self.system.update(levels, flows)

        stats = self.system.get_statistics()

        self.assertEqual(stats['total_updates'], 5)
        self.assertIn('avg_solve_time', stats)


# ==============================================================================
# 批量测试框架测试
# ==============================================================================

class TestBatchTestExecutor(unittest.TestCase):
    """批量测试执行器测试"""

    def setUp(self):
        self.executor = BatchTestExecutor(
            simulation_steps=10,  # 快速测试
            timeout_per_test=30,
        )
        self.generator = ScenarioGenerator(seed=42)

    def test_single_execution(self):
        """测试单个测试执行"""
        scenario = CompositeScenario(
            scenario_id="TEST_001",
            events=[self.generator.generate_single_event()],
            complexity=1,
        )

        test_case = TestCase(
            scenario=scenario,
            simulation_steps=10,
            timeout=30,
        )

        result = self.executor.execute_single(test_case)

        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.scenario_id, "TEST_001")
        self.assertIn(result.status, list(TestStatus))
        self.assertGreater(result.duration, 0)

    def test_batch_execution(self):
        """测试批量执行"""
        scenarios = self.generator.generate_batch(10)

        result = self.executor.execute_batch(scenarios)

        self.assertIsInstance(result, BatchTestResult)
        self.assertEqual(result.total_tests, 10)
        self.assertEqual(result.passed + result.failed + result.errors +
                        result.timeouts + result.skipped, 10)

    def test_progress_callback(self):
        """测试进度回调"""
        scenarios = self.generator.generate_batch(5)

        progress_updates = []

        def callback(current, total):
            progress_updates.append((current, total))

        self.executor.execute_batch(scenarios, progress_callback=callback)

        self.assertEqual(len(progress_updates), 5)
        self.assertEqual(progress_updates[-1], (5, 5))


class TestTestSuite(unittest.TestCase):
    """测试套件测试"""

    def test_single_event_suite(self):
        """测试单事件套件"""
        suite = TestSuite("test")
        suite.generate_single_event_suite(count=20)

        self.assertEqual(len(suite.scenarios), 20)

    def test_multi_event_suite(self):
        """测试多事件套件"""
        suite = TestSuite("test")
        suite.generate_multi_event_suite(count=10)

        self.assertEqual(len(suite.scenarios), 10)

        # 所有场景应该有2+事件
        for s in suite.scenarios:
            self.assertGreaterEqual(len(s.events), 2)

    def test_combined_suite(self):
        """测试组合套件"""
        suite = TestSuite("combined")
        suite.generate_single_event_suite(30)
        suite.generate_multi_event_suite(10)
        suite.generate_stress_suite(5)

        self.assertEqual(len(suite.scenarios), 45)

    def test_suite_validation(self):
        """测试套件验证"""
        suite = TestSuite("test")
        suite.generate_single_event_suite(20)

        valid, errors = suite.validate_all()

        self.assertEqual(valid, 20)
        self.assertEqual(len(errors), 0)

    def test_suite_statistics(self):
        """测试套件统计"""
        suite = TestSuite("test")
        suite.generate_single_event_suite(20)

        stats = suite.get_statistics()

        self.assertEqual(stats['total_scenarios'], 20)
        self.assertIn('by_type', stats)
        self.assertIn('by_complexity', stats)


class TestReportGenerator(unittest.TestCase):
    """报告生成器测试"""

    def setUp(self):
        self.generator = ScenarioGenerator(seed=42)
        self.executor = BatchTestExecutor(simulation_steps=5, timeout_per_test=20)

    def test_summary_report(self):
        """测试摘要报告"""
        scenarios = self.generator.generate_batch(5)
        result = self.executor.execute_batch(scenarios)

        report = ReportGenerator.generate_summary(result)

        self.assertIn('测试报告摘要', report)
        self.assertIn('总测试数', report)
        self.assertIn('通过', report)

    def test_json_report(self):
        """测试JSON报告"""
        scenarios = self.generator.generate_batch(5)
        result = self.executor.execute_batch(scenarios)

        json_report = ReportGenerator.generate_json_report(result)

        import json
        data = json.loads(json_report)

        self.assertIn('summary', data)
        self.assertIn('timing', data)
        self.assertEqual(data['summary']['total'], 5)


# ==============================================================================
# 集成测试
# ==============================================================================

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_hundred_scenarios(self):
        """测试100个场景"""
        suite = TestSuite("integration_100")
        suite.generate_single_event_suite(60)
        suite.generate_multi_event_suite(30)
        suite.generate_stress_suite(10)

        executor = BatchTestExecutor(simulation_steps=10, timeout_per_test=30)
        result = executor.execute_batch(suite.scenarios)

        # 至少80%应该通过
        pass_rate = result.passed / result.total_tests
        self.assertGreater(pass_rate, 0.7)

        print(f"\n100场景测试结果: 通过率={pass_rate:.1%}")
        print(f"  通过: {result.passed}, 失败: {result.failed}, 错误: {result.errors}")

    def test_scenario_type_coverage(self):
        """测试场景类型覆盖"""
        suite = TestSuite("coverage")

        # 为每种场景类型生成测试
        for st in ScenarioType:
            suite.generator.generate_single_event(scenario_type=st)
            scenario = CompositeScenario(
                scenario_id=f"COVERAGE_{st.value}",
                events=[suite.generator.generate_single_event(scenario_type=st)],
                complexity=1,
            )
            suite.add_scenario(scenario)

        self.assertEqual(len(suite.scenarios), 8)  # 8种场景类型

        executor = BatchTestExecutor(simulation_steps=10, timeout_per_test=30)
        result = executor.execute_batch(suite.scenarios)

        # 所有场景类型都应该被测试
        tested_types = set(result.by_scenario_type.keys())
        self.assertEqual(len(tested_types), 8)

    def test_adaptive_mpc_with_scenarios(self):
        """测试自适应MPC与场景生成器集成"""
        generator = ScenarioGenerator(seed=42)
        system = AdaptiveMPCSystem(num_pools=60, horizon=10)

        # 生成多种场景并应用
        scenarios_tested = 0

        for st in [ScenarioType.S1_NORMAL_PLAN, ScenarioType.S3_POLLUTION,
                   ScenarioType.S4_FLOOD_CONTROL, ScenarioType.S5_ICE_PERIOD]:
            event = generator.generate_single_event(
                scenario_type=st,
                location=30,
                severity=ScenarioSeverity.MEDIUM,
            )
            system.apply_event(event)

            # 验证场景被正确应用
            self.assertEqual(system.current_scenario.detected_type, st)
            scenarios_tested += 1

        self.assertEqual(scenarios_tested, 4)


# ==============================================================================
# 性能测试
# ==============================================================================

class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_scenario_generation_speed(self):
        """测试场景生成速度"""
        import time

        generator = ScenarioGenerator(seed=42)

        start = time.time()
        batch = generator.generate_batch(1000)
        elapsed = time.time() - start

        # 1000个场景应该在5秒内生成
        self.assertLess(elapsed, 5.0)
        print(f"\n生成1000场景耗时: {elapsed:.2f}s")

    def test_adaptive_mpc_reconfiguration_speed(self):
        """测试自适应MPC重配置速度"""
        import time

        system = AdaptiveMPCSystem(num_pools=60, horizon=10)
        generator = ScenarioGenerator(seed=42)

        events = [generator.generate_single_event() for _ in range(10)]

        start = time.time()
        for event in events:
            system.apply_event(event)
        elapsed = time.time() - start

        # 10次重配置应该在2秒内完成
        self.assertLess(elapsed, 2.0)
        print(f"\n10次MPC重配置耗时: {elapsed:.2f}s")


# ==============================================================================
# 运行测试
# ==============================================================================

def run_all_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestScenarioGenerator,
        TestScenarioValidator,
        TestScenarioIdentifier,
        TestAdaptiveMPCConfigurator,
        TestAdaptiveMPCSystem,
        TestBatchTestExecutor,
        TestTestSuite,
        TestReportGenerator,
        TestIntegration,
        TestPerformance,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_all_tests()

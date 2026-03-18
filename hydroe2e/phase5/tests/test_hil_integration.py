"""
Phase 5 HIL Testing Framework - 集成测试

测试场景生成、工况注入、评估引擎、测试执行和报告生成的完整流程
"""

import unittest
import tempfile
import shutil
import json

from hydroe2e.phase5.hil_testing.scenario_generator import (
    ScenarioGenerator, Scenario, AutonomousLevel,
    ScenarioCategory, DifficultyLevel
)
from hydroe2e.phase5.hil_testing.condition_injector import (
    ConditionInjector, InjectionType, FlowInjector
)
from hydroe2e.phase5.hil_testing.evaluation_engine import (
    EvaluationEngine, TestResult, SafetyMetrics, ControlMetrics, IntelligenceMetrics
)
from hydroe2e.phase5.hil_testing.test_runner import (
    HILTestRunner, TestSuite, TestStatus, SimulationEnvironment, SimulationState
)
from hydroe2e.phase5.hil_testing.report_generator import (
    ReportGenerator, ReportFormat, ReportConfig
)


class TestScenarioGenerator(unittest.TestCase):
    """场景生成器测试"""

    def setUp(self):
        self.generator = ScenarioGenerator()

    def test_create_scenario_basic(self):
        """测试基本场景创建"""
        scenario = self.generator.create_scenario(
            id="TEST_001",
            name="测试场景",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=300.0
        )

        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.id, "TEST_001")
        self.assertEqual(scenario.name, "测试场景")
        self.assertEqual(scenario.category, ScenarioCategory.S1_NORMAL)
        self.assertEqual(scenario.duration, 300.0)

    def test_create_scenario_with_description(self):
        """测试带描述的场景创建"""
        scenario = self.generator.create_scenario(
            id="TEST_002",
            name="带描述场景",
            category="S2_FLOOD",
            difficulty=3,
            autonomous_level="L3",
            duration=600.0,
            description="入流突增测试"
        )

        self.assertEqual(scenario.description, "入流突增测试")
        self.assertEqual(scenario.category, ScenarioCategory.S2_FLOOD)

    def test_scenario_category_enum(self):
        """测试场景类别枚举"""
        # 验证场景类别枚举存在
        self.assertTrue(hasattr(ScenarioCategory, 'S1_NORMAL'))
        self.assertTrue(hasattr(ScenarioCategory, 'S2_FLOOD'))
        self.assertTrue(hasattr(ScenarioCategory, 'S3_DROUGHT'))
        self.assertTrue(hasattr(ScenarioCategory, 'S4_ICE'))
        self.assertTrue(hasattr(ScenarioCategory, 'S5_POLLUTION'))
        self.assertTrue(hasattr(ScenarioCategory, 'S6_EQUIPMENT'))
        self.assertTrue(hasattr(ScenarioCategory, 'S7_SECURITY'))

    def test_autonomous_level_enum(self):
        """测试自主等级枚举"""
        for i in range(6):
            level = AutonomousLevel(i)
            self.assertIsNotNone(level)

    def test_difficulty_level_enum(self):
        """测试难度等级枚举"""
        for i in range(1, 6):
            level = DifficultyLevel(i)
            self.assertIsNotNone(level)


class TestConditionInjector(unittest.TestCase):
    """工况注入器测试"""

    def setUp(self):
        self.injector = ConditionInjector()

    def test_add_flow_injection(self):
        """测试流量注入添加"""
        self.injector.add_flow_injection(
            target='inflow_1',
            injection_type='step',
            start_time=10.0,
            magnitude=5.0
        )

        self.assertEqual(len(self.injector.flow_injector.injections), 1)

    def test_add_sensor_injection(self):
        """测试传感器注入添加"""
        self.injector.add_sensor_injection(
            target='sensor_level_1',
            injection_type='drift',
            start_time=100.0,
            magnitude=0.3,
            parameters={'drift_rate': 0.001}
        )

        self.assertEqual(len(self.injector.sensor_injector.injections), 1)

    def test_add_actuator_injection(self):
        """测试执行器注入添加"""
        self.injector.add_actuator_injection(
            target='gate_1',
            injection_type='stuck',
            start_time=200.0,
            parameters={'stuck_position': 0.5}
        )

        self.assertEqual(len(self.injector.actuator_injector.injections), 1)

    def test_clear_all(self):
        """测试清除所有注入"""
        self.injector.add_flow_injection(
            target='inflow_1',
            injection_type='step',
            start_time=10.0,
            magnitude=5.0
        )

        self.injector.clear_all()

        self.assertEqual(len(self.injector.flow_injector.injections), 0)

    def test_injection_types(self):
        """测试注入类型枚举"""
        self.assertTrue(hasattr(InjectionType, 'STEP'))
        self.assertTrue(hasattr(InjectionType, 'RAMP'))
        self.assertTrue(hasattr(InjectionType, 'PULSE'))


class TestFlowInjector(unittest.TestCase):
    """流量注入器详细测试"""

    def setUp(self):
        self.injector = ConditionInjector()

    def test_step_injection(self):
        """测试阶跃注入"""
        self.injector.add_flow_injection(
            target='inflow_1',
            injection_type='step',
            start_time=10.0,
            magnitude=5.0
        )

        # 测试注入被添加
        self.assertEqual(len(self.injector.flow_injector.injections), 1)

        # 注入前
        value = self.injector.flow_injector.get_value('inflow_1', 5.0, 10.0)
        self.assertEqual(value, 10.0)

        # 注入后
        value = self.injector.flow_injector.get_value('inflow_1', 15.0, 10.0)
        self.assertEqual(value, 15.0)

    def test_pulse_injection(self):
        """测试脉冲注入"""
        self.injector.add_flow_injection(
            target='inflow_1',
            injection_type='pulse',
            start_time=10.0,
            end_time=20.0,
            magnitude=5.0
        )

        # 脉冲期间
        value = self.injector.flow_injector.get_value('inflow_1', 15.0, 10.0)
        self.assertEqual(value, 15.0)

        # 脉冲结束后
        value = self.injector.flow_injector.get_value('inflow_1', 25.0, 10.0)
        self.assertEqual(value, 10.0)


class TestEvaluationEngine(unittest.TestCase):
    """评估引擎测试"""

    def setUp(self):
        self.engine = EvaluationEngine()
        self.generator = ScenarioGenerator()

    def test_evaluate_basic(self):
        """测试基本评估"""
        scenario = self.generator.create_scenario(
            id="EVAL_001",
            name="评估测试场景",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=100.0
        )

        # 创建模拟数据
        sim_data = {
            'times': list(range(100)),
            'water_levels': {'pool_1': [2.0 + 0.01 * i for i in range(100)]},
            'gate_positions': {'gate_1': [0.5] * 100},
            'inflows': {'inflow_1': [10.0] * 100},
            'outflows': {'outflow_1': [10.0] * 100},
            'alarms': [],
            'control_actions': []
        }

        result = self.engine.evaluate(scenario, sim_data)

        self.assertIsInstance(result, TestResult)
        self.assertIsNotNone(result.safety_metrics)
        self.assertIsNotNone(result.control_metrics)

    def test_safety_metrics_calculation(self):
        """测试安全指标计算"""
        # 正常水位数据
        water_levels = [2.0 + 0.05 * (i % 10 - 5) for i in range(100)]

        metrics = self.engine.calculate_safety_metrics(
            water_levels=water_levels,
            setpoint=2.0,
            overflow_limit=3.0,
            dry_out_limit=0.5
        )

        self.assertIsInstance(metrics, SafetyMetrics)
        self.assertLessEqual(metrics.max_deviation, 0.3)
        self.assertTrue(metrics.no_overflow)
        self.assertTrue(metrics.no_dry_out)


class TestSimulationEnvironment(unittest.TestCase):
    """仿真环境测试"""

    def setUp(self):
        self.sim = SimulationEnvironment()

    def test_reset(self):
        """测试环境重置"""
        self.sim.reset()
        self.assertEqual(self.sim.current_time, 0.0)
        self.assertEqual(len(self.sim.state_history), 1)

    def test_reset_with_initial_state(self):
        """测试带初始状态的重置"""
        initial = {
            'water_levels': {'pool_1': 2.5},
            'gate_positions': {'gate_1': 0.7},
            'inflows': {'inflow_1': 12.0}
        }

        self.sim.reset(initial)

        self.assertEqual(self.sim.water_levels['pool_1'], 2.5)
        self.assertEqual(self.sim.gate_positions['gate_1'], 0.7)
        self.assertEqual(self.sim.inflows['inflow_1'], 12.0)

    def test_step(self):
        """测试仿真步进"""
        self.sim.reset()
        initial_time = self.sim.current_time

        state = self.sim.step()

        self.assertIsInstance(state, SimulationState)
        self.assertGreater(self.sim.current_time, initial_time)

    def test_get_history(self):
        """测试获取历史记录"""
        self.sim.reset()

        for _ in range(10):
            self.sim.step()

        history = self.sim.get_history()

        self.assertEqual(len(history), 11)  # 初始状态 + 10步


class TestHILTestRunner(unittest.TestCase):
    """HIL测试执行器测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.runner = HILTestRunner(
            scenarios_dir=os.path.join(self.temp_dir, 'scenarios'),
            output_dir=os.path.join(self.temp_dir, 'results')
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quick_validation(self):
        """测试快速验证"""
        result = self.runner.run_quick_validation()

        self.assertIsInstance(result, dict)
        self.assertIn('scenario_generator', result)
        self.assertIn('simulation_env', result)
        self.assertIn('overall', result)

    def test_create_test_suite(self):
        """测试创建测试套件"""
        scenarios = []
        for i in range(3):
            s = self.runner.scenario_generator.create_scenario(
                id=f"TEST_{i:03d}",
                name=f"测试场景{i+1}",
                category="S1_NORMAL",
                difficulty=1,
                autonomous_level="L1",
                duration=60.0
            )
            scenarios.append(s)

        suite = self.runner.create_test_suite(
            name="测试套件",
            scenarios=scenarios,
            description="单元测试套件"
        )

        self.assertIsInstance(suite, TestSuite)
        self.assertEqual(suite.total_count, 3)
        self.assertEqual(suite.name, "测试套件")

    def test_run_suite(self):
        """测试运行测试套件"""
        scenario = self.runner.scenario_generator.create_scenario(
            id="RUN_001",
            name="执行测试场景",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=30.0
        )

        suite = self.runner.create_test_suite(
            name="执行测试",
            scenarios=[scenario],
            description="测试执行功能"
        )

        self.runner.run_suite(suite)

        self.assertGreater(suite.passed_count + suite.failed_count, 0)

    def test_get_summary(self):
        """测试获取汇总"""
        scenario = self.runner.scenario_generator.create_scenario(
            id="SUM_001",
            name="汇总测试",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=30.0
        )

        suite = self.runner.create_test_suite(
            name="汇总测试套件",
            scenarios=[scenario]
        )

        self.runner.run_suite(suite)

        summary = self.runner.get_summary()

        self.assertIn('total_suites', summary)
        self.assertIn('total_cases', summary)
        self.assertIn('passed', summary)
        self.assertIn('pass_rate', summary)

    def test_export_results(self):
        """测试导出结果"""
        scenario = self.runner.scenario_generator.create_scenario(
            id="EXP_001",
            name="导出测试",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=30.0
        )

        suite = self.runner.create_test_suite(
            name="导出测试套件",
            scenarios=[scenario]
        )

        self.runner.run_suite(suite)

        filepath = self.runner.export_results()

        self.assertTrue(os.path.exists(filepath))

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn('summary', data)
        self.assertIn('suites', data)


class TestReportGenerator(unittest.TestCase):
    """报告生成器测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_dir=self.temp_dir)
        self.runner = HILTestRunner()

        # 创建测试数据
        scenario = self.runner.scenario_generator.create_scenario(
            id="RPT_001",
            name="报告测试场景",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=30.0
        )

        self.suite = self.runner.create_test_suite(
            name="报告测试套件",
            scenarios=[scenario],
            description="用于测试报告生成"
        )

        self.runner.run_suite(self.suite)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_html_report(self):
        """测试生成HTML报告"""
        filepath = self.generator.generate([self.suite], ReportFormat.HTML)

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.html'))

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('<html', content)

    def test_generate_markdown_report(self):
        """测试生成Markdown报告"""
        filepath = self.generator.generate([self.suite], ReportFormat.MARKDOWN)

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.md'))

    def test_generate_json_report(self):
        """测试生成JSON报告"""
        filepath = self.generator.generate([self.suite], ReportFormat.JSON)

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.json'))

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIn('metadata', data)
        self.assertIn('summary', data)
        self.assertIn('suites', data)

    def test_generate_text_report(self):
        """测试生成文本报告"""
        filepath = self.generator.generate([self.suite], ReportFormat.TEXT)

        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.txt'))


class TestEndToEndWorkflow(unittest.TestCase):
    """端到端工作流测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow(self):
        """测试完整工作流程"""
        # 1. 创建测试执行器
        runner = HILTestRunner(
            output_dir=os.path.join(self.temp_dir, 'results')
        )

        # 2. 创建多种场景
        scenarios = []

        # L1正常场景
        s1 = runner.scenario_generator.create_scenario(
            id="E2E_S1_001",
            name="稳态运行",
            category="S1_NORMAL",
            difficulty=1,
            autonomous_level="L1",
            duration=60.0
        )
        scenarios.append(s1)

        # L2洪水场景
        s2 = runner.scenario_generator.create_scenario(
            id="E2E_S2_001",
            name="入流突增",
            category="S2_FLOOD",
            difficulty=2,
            autonomous_level="L2",
            duration=60.0
        )
        scenarios.append(s2)

        # 3. 创建测试套件
        suite = runner.create_test_suite(
            name="端到端测试套件",
            scenarios=scenarios,
            description="测试完整工作流程"
        )

        # 4. 运行测试
        runner.run_suite(suite)

        # 5. 验证结果
        self.assertEqual(suite.total_count, 2)

        # 6. 导出结果
        result_file = runner.export_results()
        self.assertTrue(os.path.exists(result_file))

        # 7. 生成报告
        report_gen = ReportGenerator(
            output_dir=os.path.join(self.temp_dir, 'reports')
        )

        html_report = report_gen.generate([suite], ReportFormat.HTML)
        self.assertTrue(os.path.exists(html_report))

        # 8. 验证汇总
        summary = runner.get_summary()
        self.assertEqual(summary['total_cases'], 2)


class TestScenarioYAMLLoading(unittest.TestCase):
    """场景YAML文件加载测试"""

    def setUp(self):
        self.scenarios_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'hil_testing', 'scenarios'
        )
        self.generator = ScenarioGenerator(self.scenarios_dir)

    def test_scenarios_directory_exists(self):
        """测试场景目录存在"""
        self.assertTrue(os.path.exists(self.scenarios_dir))

    def test_scenario_files_exist(self):
        """测试场景文件存在"""
        expected_files = [
            's1_normal.yaml', 's2_flood.yaml', 's3_drought.yaml',
            's4_ice.yaml', 's5_pollution.yaml', 's6_equipment.yaml',
            's7_security.yaml'
        ]

        for filename in expected_files:
            filepath = os.path.join(self.scenarios_dir, filename)
            self.assertTrue(
                os.path.exists(filepath),
                f"场景文件不存在: {filename}"
            )

    def test_load_s1_normal_scenarios(self):
        """测试加载S1正常场景"""
        filepath = os.path.join(self.scenarios_dir, 's1_normal.yaml')
        if os.path.exists(filepath):
            scenarios = self.generator.load_scenario(filepath)
            self.assertIsInstance(scenarios, list)
            self.assertGreater(len(scenarios), 0)

            for s in scenarios:
                self.assertEqual(s.category, ScenarioCategory.S1_NORMAL)

    def test_load_s2_flood_scenarios(self):
        """测试加载S2洪水场景"""
        filepath = os.path.join(self.scenarios_dir, 's2_flood.yaml')
        if os.path.exists(filepath):
            scenarios = self.generator.load_scenario(filepath)
            self.assertIsInstance(scenarios, list)
            self.assertGreater(len(scenarios), 0)

            for s in scenarios:
                self.assertEqual(s.category, ScenarioCategory.S2_FLOOD)

    def test_load_all_scenarios(self):
        """测试加载所有场景"""
        all_scenarios = self.generator.load_all_scenarios()
        self.assertIsInstance(all_scenarios, dict)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestScenarioGenerator,
        TestConditionInjector,
        TestFlowInjector,
        TestEvaluationEngine,
        TestSimulationEnvironment,
        TestHILTestRunner,
        TestReportGenerator,
        TestEndToEndWorkflow,
        TestScenarioYAMLLoading,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # 打印汇总
    print("\n" + "=" * 60)
    print("HIL测试框架集成测试汇总")
    print("=" * 60)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n所有测试通过!")
    else:
        print("\n存在失败的测试，请检查上方详情。")

    sys.exit(0 if result.wasSuccessful() else 1)

"""
控制器-仿真器集成模块测试
Tests for Controller-Simulator Integration Module
"""

import pytest
import numpy as np

from ..integrated_simulation import (
    SimulationConfig,
    ScenarioInjectionPlan,
    ControlInterface,
    StateSynchronizer,
    ScenarioInjector,
    RealTimeMetrics,
    RealTimeMonitor,
    ClosedLoopSimulation,
    ScenarioTestCase,
    ScenarioTestRunner,
    BatchScenarioEvaluator,
)
from ..hydraulic_simulator import FullLineHydraulicSimulator
from ..cascade_control import CascadeControlSystem, InterventionDecision, InterventionType
from ..local_pool_scenarios import L1ScenarioType, L1ScenarioEvent, L1ActionType, L1ActionCommand
from ..core_types import ScenarioSeverity


# ==============================================================================
# 仿真配置测试
# ==============================================================================

class TestSimulationConfig:
    """仿真配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SimulationConfig()
        assert config.num_pools == 60
        assert config.dt == 60.0
        assert config.total_duration == 86400.0
        assert config.enable_cascade is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = SimulationConfig(
            num_pools=20,
            dt=30.0,
            total_duration=3600.0,
            enable_cascade=False,
        )
        assert config.num_pools == 20
        assert config.dt == 30.0
        assert config.enable_cascade is False


# ==============================================================================
# 场景注入计划测试
# ==============================================================================

class TestScenarioInjectionPlan:
    """场景注入计划测试"""

    def test_create_plan(self):
        """测试创建计划"""
        plan = ScenarioInjectionPlan(plan_id="TEST_PLAN")
        assert plan.plan_id == "TEST_PLAN"
        assert len(plan.scenarios) == 0

    def test_add_scenario(self):
        """测试添加场景"""
        plan = ScenarioInjectionPlan(plan_id="TEST_PLAN")
        scenario = L1ScenarioEvent(
            event_id="SC001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )
        plan.add_scenario(100.0, 5, scenario)
        assert len(plan.scenarios) == 1
        assert plan.scenarios[0][0] == 100.0

    def test_scenarios_sorted_by_time(self):
        """测试场景按时间排序"""
        plan = ScenarioInjectionPlan(plan_id="TEST_PLAN")

        # 逆序添加
        for i, t in enumerate([300.0, 100.0, 200.0]):
            scenario = L1ScenarioEvent(
                event_id=f"SC{i}",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=i,
                severity=ScenarioSeverity.MEDIUM,
            )
            plan.add_scenario(t, i, scenario)

        # 验证排序
        times = [s[0] for s in plan.scenarios]
        assert times == [100.0, 200.0, 300.0]


# ==============================================================================
# 控制接口测试
# ==============================================================================

class TestControlInterface:
    """控制接口测试"""

    def test_apply_gate_command(self):
        """测试闸门控制指令"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        interface = ControlInterface(sim)

        interface.apply_gate_command(5, 2.0)

        assert sim.gate_models["GATE_5"].target_opening == 2.0
        assert len(interface.executed_commands) == 1

    def test_apply_inflow_adjustment(self):
        """测试入流调整"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        initial_inflow = sim.upstream_inflow
        interface = ControlInterface(sim)

        interface.apply_inflow_adjustment(0, 10.0)

        assert sim.upstream_inflow == initial_inflow + 10.0

    def test_apply_l1_action_gate(self):
        """测试L1闸门动作"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        interface = ControlInterface(sim)

        action = L1ActionCommand(
            command_id="CMD001",
            pool_id=3,
            action_type=L1ActionType.GATE_ADJUST,
            gate_position=2.5,
        )
        interface.apply_l1_action(action)

        assert sim.gate_models["GATE_3"].target_opening == 2.5

    def test_apply_intervention_takeover(self):
        """测试接管干预"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        interface = ControlInterface(sim)

        intervention = InterventionDecision(
            decision_id="INT001",
            escalation_event_id="ESC001",
            target_layer=1,
            intervention_type=InterventionType.TAKEOVER,
            additional_resources={'affected_pools': [4, 5, 6]},
        )
        interface.apply_intervention(intervention)

        # 验证闸门被重置
        for pool_id in [4, 5, 6]:
            assert sim.gate_models[f"GATE_{pool_id}"].target_opening == 1.5

    def test_apply_intervention_emergency(self):
        """测试紧急停机干预"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        interface = ControlInterface(sim)

        intervention = InterventionDecision(
            decision_id="INT001",
            escalation_event_id="ESC001",
            target_layer=1,
            intervention_type=InterventionType.EMERGENCY_SHUTDOWN,
            additional_resources={'affected_pools': [5]},
        )
        interface.apply_intervention(intervention)

        assert sim.gate_models["GATE_5"].target_opening == 0.0


# ==============================================================================
# 状态同步器测试
# ==============================================================================

class TestStateSynchronizer:
    """状态同步器测试"""

    def test_sync_pool_states(self):
        """测试同步渠池状态"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        cascade = CascadeControlSystem(num_pools=10)
        sync = StateSynchronizer(sim, cascade)

        # 运行几步仿真
        for _ in range(5):
            sim.step()

        sync.sync_pool_states()

        # 验证评估器存在并已更新
        for pool_id in range(10):
            assert pool_id in cascade.evaluators
            evaluator = cascade.evaluators[pool_id]
            assert evaluator.pool_id == pool_id

    def test_sync_scenario_states(self):
        """测试同步场景状态"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        cascade = CascadeControlSystem(num_pools=10)
        sync = StateSynchronizer(sim, cascade)

        # 注入污染场景
        scenario = L1ScenarioEvent(
            event_id="SC001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=5,
            severity=ScenarioSeverity.HIGH,
        )
        sim.inject_scenario(5, scenario)
        sim.step()  # 执行一步使污染生效

        # 同步状态
        sync.sync_pool_states()
        sync.sync_scenario_states()

        # 验证场景已被注入
        assert 5 in sim.active_scenarios


# ==============================================================================
# 场景注入器测试
# ==============================================================================

class TestScenarioInjector:
    """场景注入器测试"""

    def test_set_plan(self):
        """测试设置注入计划"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        injector = ScenarioInjector(sim)

        plan = ScenarioInjectionPlan(plan_id="TEST")
        injector.set_plan(plan)

        assert injector.injection_plan == plan
        assert injector.next_injection_index == 0

    def test_check_and_inject(self):
        """测试检查并注入"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        injector = ScenarioInjector(sim)

        # 创建计划
        plan = ScenarioInjectionPlan(plan_id="TEST")
        scenario = L1ScenarioEvent(
            event_id="SC001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )
        plan.add_scenario(100.0, 5, scenario)
        injector.set_plan(plan)

        # 时间未到，不注入
        injector.check_and_inject(50.0)
        assert 5 not in sim.active_scenarios

        # 时间到达，注入
        injector.check_and_inject(100.0)
        assert 5 in sim.active_scenarios
        assert len(injector.injected_scenarios) == 1

    def test_generate_random_plan(self):
        """测试生成随机计划"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        injector = ScenarioInjector(sim)

        plan = injector.generate_random_plan(num_scenarios=5, duration=1000.0)

        assert len(plan.scenarios) == 5
        # 验证时间排序
        times = [s[0] for s in plan.scenarios]
        assert times == sorted(times)


# ==============================================================================
# 实时监控器测试
# ==============================================================================

class TestRealTimeMonitor:
    """实时监控器测试"""

    def test_collect_metrics(self):
        """测试收集指标"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        cascade = CascadeControlSystem(num_pools=10)
        monitor = RealTimeMonitor(sim, cascade)

        # 运行几步
        for _ in range(5):
            sim.step()

        metrics = monitor.collect_metrics()

        assert metrics.timestamp == sim.state.current_time
        assert metrics.avg_level_error >= 0
        assert len(monitor.metrics_history) == 1

    def test_get_summary(self):
        """测试获取摘要"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        cascade = CascadeControlSystem(num_pools=10)
        monitor = RealTimeMonitor(sim, cascade)

        # 收集多次指标
        for _ in range(10):
            sim.step()
            monitor.collect_metrics()

        summary = monitor.get_summary()

        assert 'duration' in summary
        assert 'total_samples' in summary
        assert summary['total_samples'] == 10

    def test_empty_summary(self):
        """测试空摘要"""
        sim = FullLineHydraulicSimulator(num_pools=10)
        cascade = CascadeControlSystem(num_pools=10)
        monitor = RealTimeMonitor(sim, cascade)

        summary = monitor.get_summary()
        assert summary == {}


# ==============================================================================
# 闭环仿真测试
# ==============================================================================

class TestClosedLoopSimulation:
    """闭环仿真测试"""

    def test_initialization(self):
        """测试初始化"""
        config = SimulationConfig(num_pools=10, dt=60.0)
        sim = ClosedLoopSimulation(config)

        assert sim.simulator.num_pools == 10
        assert sim.cascade_system is not None
        assert sim.control_interface is not None

    def test_single_step(self):
        """测试单步仿真"""
        config = SimulationConfig(num_pools=10, dt=60.0)
        sim = ClosedLoopSimulation(config)

        result = sim.step()

        assert 'time' in result
        assert 'step' in result
        assert 'pool_states' in result

    def test_run_simulation(self):
        """测试运行仿真"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=600.0,  # 10分钟
        )
        sim = ClosedLoopSimulation(config)

        result = sim.run()

        assert result['execution']['total_steps'] == 10
        assert 'performance' in result
        assert 'monitoring' in result

    def test_run_with_scenarios(self):
        """测试带场景的仿真"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=600.0,
        )
        sim = ClosedLoopSimulation(config)

        # 创建场景计划
        plan = ScenarioInjectionPlan(plan_id="TEST")
        scenario = L1ScenarioEvent(
            event_id="SC001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )
        plan.add_scenario(100.0, 5, scenario)
        sim.set_scenario_plan(plan)

        result = sim.run()

        assert result['scenarios']['injected'] == 1

    def test_run_with_callback(self):
        """测试带回调的仿真"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=300.0,
        )
        sim = ClosedLoopSimulation(config)

        callback_count = [0]

        def callback(step_result):
            callback_count[0] += 1

        sim.run(callback=callback)

        assert callback_count[0] == 5

    def test_get_pool_state(self):
        """测试获取池状态"""
        config = SimulationConfig(num_pools=10)
        sim = ClosedLoopSimulation(config)
        sim.step()

        state = sim.get_pool_state(5)
        assert state is not None
        assert state.pool_id == 5

    def test_get_all_levels(self):
        """测试获取所有水位"""
        config = SimulationConfig(num_pools=10)
        sim = ClosedLoopSimulation(config)
        sim.step()

        levels = sim.get_all_levels()
        assert len(levels) == 10


# ==============================================================================
# 场景测试用例测试
# ==============================================================================

class TestScenarioTestCase:
    """场景测试用例测试"""

    def test_create_test_case(self):
        """测试创建测试用例"""
        tc = ScenarioTestCase(
            test_id="TC001",
            name="测试用例1",
            description="测试描述",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )

        assert tc.test_id == "TC001"
        assert tc.passed is False
        assert tc.actual_response_time is None


# ==============================================================================
# 场景测试运行器测试
# ==============================================================================

class TestScenarioTestRunner:
    """场景测试运行器测试"""

    def test_initialization(self):
        """测试初始化"""
        runner = ScenarioTestRunner()
        assert runner.config is not None
        assert len(runner.test_results) == 0

    def test_run_single_test(self):
        """测试运行单个测试"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=600.0,
        )
        runner = ScenarioTestRunner(config)

        tc = ScenarioTestCase(
            test_id="TC001",
            name="水位上涨测试",
            description="测试",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
            max_level_deviation=1.0,
        )

        result = runner.run_test(tc)

        assert result.actual_max_deviation is not None
        assert len(runner.test_results) == 1

    def test_generate_standard_suite(self):
        """测试生成标准套件"""
        runner = ScenarioTestRunner()
        suite = runner.generate_standard_suite()

        assert len(suite) == 5
        assert all(isinstance(tc, ScenarioTestCase) for tc in suite)

    def test_run_suite(self):
        """测试运行套件"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=300.0,
        )
        runner = ScenarioTestRunner(config)

        # 简化套件
        suite = [
            ScenarioTestCase(
                test_id="TC001",
                name="测试1",
                description="测试",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=5,
                severity=ScenarioSeverity.MEDIUM,
                max_level_deviation=1.0,
            ),
            ScenarioTestCase(
                test_id="TC002",
                name="测试2",
                description="测试",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_DROP,
                pool_id=5,
                severity=ScenarioSeverity.MEDIUM,
                max_level_deviation=1.0,
            ),
        ]

        result = runner.run_suite(suite)

        assert result['total'] == 2
        assert 'pass_rate' in result


# ==============================================================================
# 批量场景评估测试
# ==============================================================================

class TestBatchScenarioEvaluator:
    """批量场景评估测试"""

    def test_initialization(self):
        """测试初始化"""
        evaluator = BatchScenarioEvaluator()
        assert evaluator.base_config is not None
        assert len(evaluator.evaluation_results) == 0

    def test_evaluate_scenario_combination(self):
        """测试评估场景组合"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=300.0,
        )
        evaluator = BatchScenarioEvaluator(config)

        scenarios = [
            L1ScenarioEvent(
                event_id="SC001",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=3,
                severity=ScenarioSeverity.MEDIUM,
            ),
            L1ScenarioEvent(
                event_id="SC002",
                scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
                pool_id=7,
                severity=ScenarioSeverity.HIGH,
            ),
        ]
        inject_times = [60.0, 120.0]

        result = evaluator.evaluate_scenario_combination(scenarios, inject_times)

        assert 'scenarios' in result
        assert 'result' in result
        assert len(evaluator.evaluation_results) == 1

    def test_run_stress_test(self):
        """测试压力测试"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=300.0,
        )
        evaluator = BatchScenarioEvaluator(config)

        result = evaluator.run_stress_test(num_scenarios=3)

        assert result['type'] == 'stress_test'
        assert result['num_scenarios'] == 3
        assert 'result' in result

    def test_run_cascade_test(self):
        """测试级联测试"""
        config = SimulationConfig(
            num_pools=20,
            dt=60.0,
            total_duration=600.0,
        )
        evaluator = BatchScenarioEvaluator(config)

        result = evaluator.run_cascade_test()

        assert result['type'] == 'cascade_test'
        assert 'result' in result


# ==============================================================================
# 集成测试
# ==============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_closed_loop_simulation(self):
        """测试完整闭环仿真"""
        config = SimulationConfig(
            num_pools=15,
            dt=60.0,
            total_duration=600.0,
            enable_cascade=True,
            performance_logging=True,
        )
        sim = ClosedLoopSimulation(config)

        # 添加多个场景
        plan = ScenarioInjectionPlan(plan_id="INTEG_TEST")
        scenarios = [
            (60.0, 5, L1ScenarioEvent(
                event_id="SC001",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=5,
                severity=ScenarioSeverity.MEDIUM,
            )),
            (120.0, 10, L1ScenarioEvent(
                event_id="SC002",
                scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
                pool_id=10,
                severity=ScenarioSeverity.HIGH,
            )),
        ]
        for t, pool_id, sc in scenarios:
            plan.add_scenario(t, pool_id, sc)

        sim.set_scenario_plan(plan)
        result = sim.run()

        # 验证结果
        assert result['execution']['total_steps'] == 10
        assert result['scenarios']['injected'] == 2
        assert 'avg_rmse' in result['performance']

    def test_multi_scenario_stress(self):
        """测试多场景压力"""
        config = SimulationConfig(
            num_pools=20,
            dt=60.0,
            total_duration=600.0,
        )
        sim = ClosedLoopSimulation(config)

        # 生成随机场景
        plan = sim.scenario_injector.generate_random_plan(
            num_scenarios=5,
            duration=500.0,
        )
        sim.set_scenario_plan(plan)

        result = sim.run()

        assert result['scenarios']['injected'] <= 5
        assert result['execution']['total_steps'] == 10

    def test_control_effectiveness(self):
        """测试控制效果"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=1200.0,  # 20分钟
            enable_cascade=True,
        )
        sim = ClosedLoopSimulation(config)

        # 注入一个需要上报的严重场景
        plan = ScenarioInjectionPlan(plan_id="CONTROL_TEST")
        scenario = L1ScenarioEvent(
            event_id="CRITICAL_001",
            scenario_type=L1ScenarioType.L1_GATE_STUCK,
            pool_id=5,
            severity=ScenarioSeverity.CRITICAL,
        )
        plan.add_scenario(60.0, 5, scenario)
        sim.set_scenario_plan(plan)

        result = sim.run()

        # 严重场景应该触发上报或干预
        assert result['scenarios']['injected'] == 1

    def test_performance_metrics_collection(self):
        """测试性能指标收集"""
        config = SimulationConfig(
            num_pools=10,
            dt=60.0,
            total_duration=600.0,
            performance_logging=True,
            log_interval=1,  # 每步记录
        )
        sim = ClosedLoopSimulation(config)
        result = sim.run()

        # 验证监控数据
        assert len(sim.monitor.metrics_history) == 10
        assert result['monitoring']['total_samples'] == 10

    def test_long_simulation(self):
        """测试长时间仿真"""
        config = SimulationConfig(
            num_pools=30,
            dt=60.0,
            total_duration=3600.0,  # 1小时
        )
        sim = ClosedLoopSimulation(config)

        # 随机场景
        plan = sim.scenario_injector.generate_random_plan(
            num_scenarios=10,
            duration=3500.0,
        )
        sim.set_scenario_plan(plan)

        result = sim.run()

        assert result['execution']['total_steps'] == 60
        assert result['scenarios']['injected'] <= 10

        # 验证水位在合理范围
        for pool_id in range(30):
            state = sim.get_pool_state(pool_id)
            assert 0.5 <= state.water_level <= 6.0

"""
控制器-仿真器集成模块
Controller-Simulator Integration Module

核心功能:
1. 闭环控制仿真环境
2. 多层控制器与水力学仿真器集成
3. 场景自动注入与评估
4. 实时性能监控
5. 全场景自动化测试

架构:
┌─────────────────────────────────────────────────────────────┐
│                    IntegratedSimulation                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐│
│  │              FullLineHydraulicSimulator                 ││
│  │  (60渠池物理仿真: 水位、流量、闸门)                      ││
│  └──────────────────────┬──────────────────────────────────┘│
│                         │ 状态反馈                          │
│  ┌──────────────────────▼──────────────────────────────────┐│
│  │              CascadeControlSystem                       ││
│  │  (级联控制: L1→L2→L3上报, L3→L2→L1干预)                 ││
│  └──────────────────────┬──────────────────────────────────┘│
│                         │ 控制指令                          │
│  ┌──────────────────────▼──────────────────────────────────┐│
│  │              ScenarioInjector                           ││
│  │  (场景注入: 污染、边坡、水位异常等)                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import json
from collections import deque

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity,
    CanalPoolConfig, ControlDirective, ControlPlan,
)
from .local_pool_scenarios import (
    L1ScenarioType, L1ScenarioEvent, L1ActionType, L1ActionCommand,
)
from .hydraulic_simulator import (
    FullLineHydraulicSimulator, PoolState, SimulationState,
    PoolPhysicalParams, GateParams,
    SimulationRecorder, SimulationReplayer, SimulationRecord,
    PerformanceAnalyzer, PerformanceMetrics,
)
from .cascade_control import (
    CascadeControlSystem, ControlEffectiveness, ControlMetrics,
    EscalationEvent, InterventionDecision, InterventionType,
    ExtendedL1Scenarios,
)
from .l1_controller import L1Controller, L1ControllerManager, L1ControlResult

logger = logging.getLogger(__name__)


# ==============================================================================
# 仿真配置
# ==============================================================================

@dataclass
class SimulationConfig:
    """仿真配置"""
    num_pools: int = 60
    dt: float = 60.0                    # 仿真步长 [s]
    total_duration: float = 86400.0     # 总仿真时长 [s] (默认1天)

    # 控制配置
    control_interval: float = 60.0      # 控制间隔 [s]
    enable_cascade: bool = True         # 启用级联控制
    enable_l1_autonomy: bool = True     # 启用L1自主控制

    # 场景配置
    scenario_injection_enabled: bool = True
    max_concurrent_scenarios: int = 10

    # 性能监控
    performance_logging: bool = True
    log_interval: int = 100             # 每N步记录一次

    # 边界条件
    upstream_inflow: float = 50.0       # 上游来水 [m³/s]
    downstream_level: float = 2.5       # 下游水位 [m]


@dataclass
class ScenarioInjectionPlan:
    """场景注入计划"""
    plan_id: str
    scenarios: List[Tuple[float, int, L1ScenarioEvent]] = field(default_factory=list)
    # (注入时间, 池ID, 场景事件)

    def add_scenario(self, time: float, pool_id: int, scenario: L1ScenarioEvent):
        """添加场景"""
        self.scenarios.append((time, pool_id, scenario))
        self.scenarios.sort(key=lambda x: x[0])


# ==============================================================================
# 控制接口适配器
# ==============================================================================

class ControlInterface:
    """
    控制接口

    将控制指令转换为仿真器可执行的动作
    """

    def __init__(self, simulator: FullLineHydraulicSimulator):
        self.simulator = simulator
        self.pending_commands: List[L1ActionCommand] = []
        self.executed_commands: List[Dict] = []

    def apply_gate_command(self, pool_id: int, target_opening: float):
        """应用闸门控制指令"""
        self.simulator.set_gate_opening(pool_id, target_opening)
        self.executed_commands.append({
            'time': self.simulator.state.current_time,
            'pool_id': pool_id,
            'action': 'gate_opening',
            'value': target_opening,
        })

    def apply_inflow_adjustment(self, pool_id: int, adjustment: float):
        """应用入流调整"""
        if pool_id == 0:
            # 上游入流调整
            self.simulator.set_upstream_inflow(
                self.simulator.upstream_inflow + adjustment
            )

    def apply_l1_action(self, action: L1ActionCommand):
        """应用L1控制动作"""
        if action.action_type == L1ActionType.GATE_ADJUST:
            # 闸门调整
            if 'target_opening' in action.parameters:
                self.apply_gate_command(
                    action.pool_id,
                    action.parameters['target_opening']
                )
        elif action.action_type == L1ActionType.EMERGENCY_DISCHARGE:
            # 紧急退水：增大出流闸门开度
            current_opening = self.simulator.state.gate_states.get(
                f"GATE_{action.pool_id}", 1.5
            )
            self.apply_gate_command(action.pool_id, current_opening + 0.5)

    def apply_intervention(self, intervention: InterventionDecision):
        """应用上层干预决策"""
        if intervention.intervention_type == InterventionType.TAKEOVER:
            # 接管控制：重置闸门到安全位置
            for pool_id in intervention.affected_pools:
                self.apply_gate_command(pool_id, 1.5)  # 默认安全开度

        elif intervention.intervention_type == InterventionType.COORDINATE:
            # 协调邻域
            for pool_id in intervention.affected_pools:
                # 调整邻近池的闸门
                current = self.simulator.state.gate_states.get(f"GATE_{pool_id}", 1.5)
                self.apply_gate_command(pool_id, current * 0.9)

        elif intervention.intervention_type == InterventionType.EMERGENCY_SHUTDOWN:
            # 紧急停机：关闭相关闸门
            for pool_id in intervention.affected_pools:
                self.apply_gate_command(pool_id, 0.0)


# ==============================================================================
# 状态同步器
# ==============================================================================

class StateSynchronizer:
    """
    状态同步器

    将仿真器状态同步到控制系统
    """

    def __init__(self,
                 simulator: FullLineHydraulicSimulator,
                 cascade_system: CascadeControlSystem):
        self.simulator = simulator
        self.cascade_system = cascade_system

    def sync_pool_states(self):
        """同步渠池状态到级联控制系统"""
        from .cascade_control import ControlMetrics

        for pool_id, pool_state in self.simulator.state.pool_states.items():
            # 构造控制指标
            target_level = self.simulator.pool_params[pool_id].target_level
            level_error = abs(pool_state.water_level - target_level)

            # 判断趋势
            history = list(self.simulator.state.level_history.get(pool_id, []))
            if len(history) >= 3:
                recent = history[-3:]
                if recent[-1] > recent[0] + 0.05:
                    trend = "rising"
                elif recent[-1] < recent[0] - 0.05:
                    trend = "falling"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # 更新级联控制系统的评估器
            if pool_id in self.cascade_system.evaluators:
                evaluator = self.cascade_system.evaluators[pool_id]
                # 创建新的控制指标并提交评估
                metrics = ControlMetrics(
                    pool_id=pool_id,
                    timestamp=self.simulator.state.current_time,
                    level_error=level_error,
                    level_trend=trend,
                    quality_index=pool_state.quality_index,
                    gate_response=1.0,  # 假设闸门正常响应
                )
                evaluator.update_metrics(metrics)

    def sync_scenario_states(self):
        """同步场景状态"""
        # 场景状态通过 sync_pool_states 中的 quality_index 已经同步
        # 这里只记录场景类型供后续分析
        pass


# ==============================================================================
# 场景注入器
# ==============================================================================

class ScenarioInjector:
    """
    场景注入器

    按计划向仿真器注入各类场景
    """

    def __init__(self, simulator: FullLineHydraulicSimulator):
        self.simulator = simulator
        self.injection_plan: Optional[ScenarioInjectionPlan] = None
        self.injected_scenarios: List[Dict] = []
        self.next_injection_index: int = 0

    def set_plan(self, plan: ScenarioInjectionPlan):
        """设置注入计划"""
        self.injection_plan = plan
        self.next_injection_index = 0

    def check_and_inject(self, current_time: float):
        """检查并注入场景"""
        if not self.injection_plan:
            return

        while self.next_injection_index < len(self.injection_plan.scenarios):
            inject_time, pool_id, scenario = self.injection_plan.scenarios[
                self.next_injection_index
            ]

            if inject_time <= current_time:
                # 注入场景
                self.simulator.inject_scenario(pool_id, scenario)
                self.injected_scenarios.append({
                    'time': current_time,
                    'pool_id': pool_id,
                    'scenario': scenario,
                })
                logger.info(f"注入场景: t={current_time:.0f}s, 池{pool_id}, "
                           f"{scenario.scenario_type.value}")
                self.next_injection_index += 1
            else:
                break

    def generate_random_plan(self,
                             num_scenarios: int = 5,
                             duration: float = 3600.0) -> ScenarioInjectionPlan:
        """生成随机注入计划"""
        plan = ScenarioInjectionPlan(plan_id=f"RANDOM_{int(time.time())}")

        scenario_types = [
            L1ScenarioType.L1_LEVEL_RAPID_RISE,
            L1ScenarioType.L1_LEVEL_RAPID_DROP,
            L1ScenarioType.L1_POLLUTION_DETECTED,
            L1ScenarioType.L1_GATE_STUCK,
            L1ScenarioType.L1_DISCHARGE_EMERGENCY,
        ]

        for i in range(num_scenarios):
            inject_time = np.random.uniform(60, duration - 300)
            pool_id = np.random.randint(0, self.simulator.num_pools)
            scenario_type = np.random.choice(scenario_types)
            severity = np.random.choice([
                ScenarioSeverity.LOW,
                ScenarioSeverity.MEDIUM,
                ScenarioSeverity.HIGH,
            ])

            scenario = L1ScenarioEvent(
                event_id=f"RAND_{i:03d}",
                scenario_type=scenario_type,
                pool_id=pool_id,
                severity=severity,
            )

            plan.add_scenario(inject_time, pool_id, scenario)

        return plan


# ==============================================================================
# 实时性能监控
# ==============================================================================

@dataclass
class RealTimeMetrics:
    """实时性能指标"""
    timestamp: float = 0.0

    # 水位指标
    avg_level_error: float = 0.0
    max_level_error: float = 0.0
    pools_in_alarm: int = 0

    # 控制指标
    escalation_count: int = 0
    intervention_count: int = 0
    control_effectiveness: float = 1.0

    # 场景指标
    active_scenarios: int = 0
    resolved_scenarios: int = 0


class RealTimeMonitor:
    """
    实时性能监控器
    """

    def __init__(self,
                 simulator: FullLineHydraulicSimulator,
                 cascade_system: CascadeControlSystem):
        self.simulator = simulator
        self.cascade_system = cascade_system

        self.metrics_history: List[RealTimeMetrics] = []
        self.alarm_threshold: float = 0.3  # 水位偏差告警阈值

    def collect_metrics(self) -> RealTimeMetrics:
        """收集实时指标"""
        metrics = RealTimeMetrics(
            timestamp=self.simulator.state.current_time
        )

        # 水位指标
        level_errors = []
        alarm_count = 0

        for pool_id, state in self.simulator.state.pool_states.items():
            target = self.simulator.pool_params[pool_id].target_level
            error = abs(state.water_level - target)
            level_errors.append(error)

            if error > self.alarm_threshold:
                alarm_count += 1

        if level_errors:
            metrics.avg_level_error = np.mean(level_errors)
            metrics.max_level_error = np.max(level_errors)
        metrics.pools_in_alarm = alarm_count

        # 场景指标
        metrics.active_scenarios = len(self.simulator.active_scenarios)

        self.metrics_history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.metrics_history:
            return {}

        return {
            'duration': self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            'total_samples': len(self.metrics_history),
            'avg_level_error': np.mean([m.avg_level_error for m in self.metrics_history]),
            'max_level_error': max(m.max_level_error for m in self.metrics_history),
            'max_pools_in_alarm': max(m.pools_in_alarm for m in self.metrics_history),
            'max_active_scenarios': max(m.active_scenarios for m in self.metrics_history),
        }


# ==============================================================================
# 闭环仿真环境
# ==============================================================================

class ClosedLoopSimulation:
    """
    闭环控制仿真环境

    集成仿真器、控制器、场景注入、性能监控
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # 核心组件
        self.simulator = FullLineHydraulicSimulator(
            num_pools=config.num_pools,
            dt=config.dt,
        )
        self.simulator.set_upstream_inflow(config.upstream_inflow)

        self.cascade_system = CascadeControlSystem(
            num_pools=config.num_pools
        )

        # 辅助组件
        self.control_interface = ControlInterface(self.simulator)
        self.state_sync = StateSynchronizer(self.simulator, self.cascade_system)
        self.scenario_injector = ScenarioInjector(self.simulator)
        self.monitor = RealTimeMonitor(self.simulator, self.cascade_system)

        # 记录器
        self.recorder = SimulationRecorder(self.simulator)

        # 统计
        self.total_escalations: int = 0
        self.total_interventions: int = 0
        self.run_history: List[Dict] = []

    def set_scenario_plan(self, plan: ScenarioInjectionPlan):
        """设置场景注入计划"""
        self.scenario_injector.set_plan(plan)

    def step(self) -> Dict[str, Any]:
        """
        执行一步闭环仿真

        流程:
        1. 场景注入检查
        2. 状态同步
        3. 级联控制决策
        4. 应用控制指令
        5. 物理仿真步进
        6. 性能监控
        """
        current_time = self.simulator.state.current_time
        step_result = {
            'time': current_time,
            'step': self.simulator.state.step_count,
        }

        # 1. 场景注入
        if self.config.scenario_injection_enabled:
            self.scenario_injector.check_and_inject(current_time)

        # 2. 状态同步
        self.state_sync.sync_pool_states()
        self.state_sync.sync_scenario_states()

        # 3. 级联控制
        if self.config.enable_cascade:
            cascade_result = self.cascade_system.control_step(self.config.dt)

            # 处理上报和干预 (cascade_result已返回计数而非列表)
            escalation_count = cascade_result.get('escalations', 0)
            l2_count = cascade_result.get('l2_interventions', 0)
            l3_count = cascade_result.get('l3_interventions', 0)

            step_result['escalations'] = escalation_count
            step_result['interventions'] = l2_count + l3_count

            self.total_escalations += escalation_count
            self.total_interventions += l2_count + l3_count

        # 5. 物理仿真
        pool_states = self.simulator.step()
        step_result['pool_states'] = len(pool_states)

        # 6. 性能监控
        if self.config.performance_logging:
            metrics = self.monitor.collect_metrics()
            step_result['metrics'] = {
                'avg_error': metrics.avg_level_error,
                'max_error': metrics.max_level_error,
                'alarms': metrics.pools_in_alarm,
            }

        return step_result

    def run(self,
            duration: Optional[float] = None,
            callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        运行闭环仿真

        Args:
            duration: 仿真时长，None则使用配置值
            callback: 每步回调函数

        Returns:
            仿真结果摘要
        """
        if duration is None:
            duration = self.config.total_duration

        num_steps = int(duration / self.config.dt)
        start_time = time.time()

        logger.info(f"开始闭环仿真: {num_steps}步, 时长{duration/3600:.1f}小时")

        # 开始记录
        self.recorder.start_recording(f"RUN_{int(time.time())}")

        for i in range(num_steps):
            step_result = self.step()

            if callback:
                callback(step_result)

            # 日志
            if i > 0 and i % self.config.log_interval == 0:
                logger.info(f"仿真进度: {i}/{num_steps} "
                           f"({100*i/num_steps:.1f}%), "
                           f"上报:{self.total_escalations}, "
                           f"干预:{self.total_interventions}")

        # 停止记录
        record = self.recorder.stop_recording()

        # 性能分析
        analyzer = PerformanceAnalyzer(self.simulator)
        performance = analyzer.generate_report()

        # 监控摘要
        monitor_summary = self.monitor.get_summary()

        elapsed = time.time() - start_time

        result = {
            'config': {
                'num_pools': self.config.num_pools,
                'duration': duration,
                'dt': self.config.dt,
            },
            'execution': {
                'total_steps': num_steps,
                'elapsed_seconds': elapsed,
                'steps_per_second': num_steps / elapsed,
            },
            'control': {
                'total_escalations': self.total_escalations,
                'total_interventions': self.total_interventions,
            },
            'scenarios': {
                'injected': len(self.scenario_injector.injected_scenarios),
                'active_at_end': len(self.simulator.active_scenarios),
            },
            'performance': performance['summary'],
            'monitoring': monitor_summary,
            'record_id': record.record_id if record else None,
        }

        self.run_history.append(result)
        return result

    def get_pool_state(self, pool_id: int) -> Optional[PoolState]:
        """获取池状态"""
        return self.simulator.get_pool_state(pool_id)

    def get_all_levels(self) -> Dict[int, float]:
        """获取所有水位"""
        return self.simulator.get_all_levels()


# ==============================================================================
# 全场景自动化测试
# ==============================================================================

@dataclass
class ScenarioTestCase:
    """场景测试用例"""
    test_id: str
    name: str
    description: str

    # 场景配置
    scenario_type: L1ScenarioType
    pool_id: int
    severity: ScenarioSeverity
    inject_time: float = 60.0

    # 期望结果
    expected_response_time: float = 300.0  # 期望响应时间 [s]
    max_level_deviation: float = 0.5       # 最大水位偏差 [m]
    should_escalate: bool = False

    # 实际结果
    actual_response_time: Optional[float] = None
    actual_max_deviation: Optional[float] = None
    escalated: bool = False
    passed: bool = False


class ScenarioTestRunner:
    """
    场景测试运行器

    自动执行场景测试用例并评估结果
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig(
            num_pools=20,
            dt=60.0,
            total_duration=1800.0,  # 30分钟
        )
        self.test_results: List[ScenarioTestCase] = []

    def run_test(self, test_case: ScenarioTestCase) -> ScenarioTestCase:
        """运行单个测试用例"""
        logger.info(f"运行测试: {test_case.test_id} - {test_case.name}")

        # 创建仿真环境
        sim = ClosedLoopSimulation(self.config)

        # 创建场景注入计划
        plan = ScenarioInjectionPlan(plan_id=f"TEST_{test_case.test_id}")
        scenario = L1ScenarioEvent(
            event_id=test_case.test_id,
            scenario_type=test_case.scenario_type,
            pool_id=test_case.pool_id,
            severity=test_case.severity,
        )
        plan.add_scenario(test_case.inject_time, test_case.pool_id, scenario)
        sim.set_scenario_plan(plan)

        # 运行仿真
        result = sim.run(duration=self.config.total_duration)

        # 分析结果
        test_case.actual_max_deviation = result['performance']['max_rmse']
        test_case.escalated = result['control']['total_escalations'] > 0

        # 计算响应时间（简化）
        test_case.actual_response_time = 180.0  # TODO: 从监控数据分析

        # 判断是否通过
        test_case.passed = (
            test_case.actual_max_deviation <= test_case.max_level_deviation and
            test_case.escalated == test_case.should_escalate
        )

        self.test_results.append(test_case)
        return test_case

    def run_suite(self, test_cases: List[ScenarioTestCase]) -> Dict[str, Any]:
        """运行测试套件"""
        logger.info(f"运行测试套件: {len(test_cases)} 个用例")

        for tc in test_cases:
            self.run_test(tc)

        # 统计
        passed = sum(1 for tc in self.test_results if tc.passed)
        failed = len(self.test_results) - passed

        return {
            'total': len(self.test_results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.test_results) if self.test_results else 0,
            'results': [
                {
                    'test_id': tc.test_id,
                    'name': tc.name,
                    'passed': tc.passed,
                    'max_deviation': tc.actual_max_deviation,
                    'escalated': tc.escalated,
                }
                for tc in self.test_results
            ],
        }

    def generate_standard_suite(self) -> List[ScenarioTestCase]:
        """生成标准测试套件"""
        return [
            ScenarioTestCase(
                test_id="TC001",
                name="水位快速上涨响应测试",
                description="测试L1控制器对水位快速上涨的响应能力",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=10,
                severity=ScenarioSeverity.MEDIUM,
                max_level_deviation=0.5,
            ),
            ScenarioTestCase(
                test_id="TC002",
                name="水位快速下降响应测试",
                description="测试L1控制器对水位快速下降的响应能力",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_DROP,
                pool_id=10,
                severity=ScenarioSeverity.MEDIUM,
                max_level_deviation=0.5,
            ),
            ScenarioTestCase(
                test_id="TC003",
                name="污染事件响应测试",
                description="测试系统对污染检测的响应",
                scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
                pool_id=15,
                severity=ScenarioSeverity.HIGH,
                should_escalate=True,
                max_level_deviation=0.8,
            ),
            ScenarioTestCase(
                test_id="TC004",
                name="闸门卡住响应测试",
                description="测试系统对闸门故障的处理",
                scenario_type=L1ScenarioType.L1_GATE_STUCK,
                pool_id=8,
                severity=ScenarioSeverity.HIGH,
                should_escalate=True,
                max_level_deviation=1.0,
            ),
            ScenarioTestCase(
                test_id="TC005",
                name="紧急退水响应测试",
                description="测试紧急退水场景的处理",
                scenario_type=L1ScenarioType.L1_DISCHARGE_EMERGENCY,
                pool_id=5,
                severity=ScenarioSeverity.CRITICAL,
                should_escalate=True,
                max_level_deviation=1.0,
            ),
        ]


# ==============================================================================
# 批量场景评估
# ==============================================================================

class BatchScenarioEvaluator:
    """
    批量场景评估器

    对多种场景组合进行批量评估
    """

    def __init__(self, base_config: Optional[SimulationConfig] = None):
        self.base_config = base_config or SimulationConfig(
            num_pools=30,
            dt=60.0,
            total_duration=3600.0,
        )
        self.evaluation_results: List[Dict] = []

    def evaluate_scenario_combination(self,
                                       scenarios: List[L1ScenarioEvent],
                                       inject_times: List[float]) -> Dict[str, Any]:
        """评估场景组合"""
        sim = ClosedLoopSimulation(self.base_config)

        # 创建注入计划
        plan = ScenarioInjectionPlan(plan_id=f"BATCH_{int(time.time())}")
        for i, (sc, t) in enumerate(zip(scenarios, inject_times)):
            plan.add_scenario(t, sc.pool_id, sc)

        sim.set_scenario_plan(plan)

        # 运行仿真
        result = sim.run()

        evaluation = {
            'scenarios': [
                {
                    'type': sc.scenario_type.value,
                    'pool_id': sc.pool_id,
                    'severity': sc.severity.value,
                }
                for sc in scenarios
            ],
            'result': result,
        }

        self.evaluation_results.append(evaluation)
        return evaluation

    def run_stress_test(self, num_scenarios: int = 10) -> Dict[str, Any]:
        """运行压力测试"""
        logger.info(f"运行压力测试: {num_scenarios} 个并发场景")

        sim = ClosedLoopSimulation(self.base_config)

        # 生成随机场景计划
        plan = sim.scenario_injector.generate_random_plan(
            num_scenarios=num_scenarios,
            duration=self.base_config.total_duration,
        )
        sim.set_scenario_plan(plan)

        # 运行
        result = sim.run()

        return {
            'type': 'stress_test',
            'num_scenarios': num_scenarios,
            'result': result,
        }

    def run_cascade_test(self) -> Dict[str, Any]:
        """运行级联场景测试"""
        logger.info("运行级联场景测试")

        sim = ClosedLoopSimulation(self.base_config)

        # 创建级联场景：污染从上游向下游传播
        plan = ScenarioInjectionPlan(plan_id="CASCADE_TEST")

        for i in range(5):
            pool_id = 5 + i * 3  # 池5, 8, 11, 14, 17
            inject_time = 60.0 + i * 120.0  # 间隔2分钟

            scenario = L1ScenarioEvent(
                event_id=f"CASCADE_{i}",
                scenario_type=L1ScenarioType.L1_POLLUTION_TRACKING,
                pool_id=pool_id,
                severity=ScenarioSeverity.MEDIUM,
            )
            plan.add_scenario(inject_time, pool_id, scenario)

        sim.set_scenario_plan(plan)
        result = sim.run()

        return {
            'type': 'cascade_test',
            'result': result,
        }


# ==============================================================================
# 导出类和函数
# ==============================================================================

__all__ = [
    'SimulationConfig',
    'ScenarioInjectionPlan',
    'ControlInterface',
    'StateSynchronizer',
    'ScenarioInjector',
    'RealTimeMetrics',
    'RealTimeMonitor',
    'ClosedLoopSimulation',
    'ScenarioTestCase',
    'ScenarioTestRunner',
    'BatchScenarioEvaluator',
]

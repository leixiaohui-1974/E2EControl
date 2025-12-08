"""
批量测试框架 - 支持千级场景测试和统计分析
Batch Testing Framework - Supporting Thousands of Scenarios

功能:
1. 批量场景测试执行
2. 并行/串行测试模式
3. 测试结果统计分析
4. 性能基准测试
5. 回归测试套件
6. 报告生成
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity,
    ControlDirective, ControlPlan, ScenarioEvent,
)
from .scenario_generator import (
    ScenarioGenerator, ScenarioValidator,
    ExtendedScenarioEvent, CompositeScenario,
    SeasonType, WeatherType, TimeOfDay, EvolutionPattern,
)
from .adaptive_mpc import AdaptiveMPCSystem, ScenarioDetectionResult
from .physics_model import SNWDMiddleRouteModel
from .orchestrator import GlobalOrchestrator

logger = logging.getLogger(__name__)


# ==============================================================================
# 测试结果类
# ==============================================================================

class TestStatus(Enum):
    """测试状态"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """
    单个测试结果
    """
    scenario_id: str
    status: TestStatus
    duration: float                         # 执行时间 [s]

    # 场景信息
    scenario_type: ScenarioType = ScenarioType.S1_NORMAL_PLAN
    complexity: int = 1
    event_count: int = 1

    # 检测结果
    detected_correctly: bool = True
    detection_confidence: float = 1.0

    # 控制结果
    final_level_mean: float = 4.0
    final_level_std: float = 0.1
    max_level_violation: float = 0.0
    max_flow_violation: float = 0.0

    # 性能指标
    mpc_solve_time: float = 0.0
    scenario_identification_time: float = 0.0

    # 错误信息
    error_message: str = ""
    error_traceback: str = ""

    # 详细数据
    level_history: List[float] = field(default_factory=list)
    flow_history: List[float] = field(default_factory=list)
    role_assignments: Dict[int, str] = field(default_factory=dict)


@dataclass
class BatchTestResult:
    """
    批量测试结果
    """
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    timeouts: int = 0

    total_duration: float = 0.0
    avg_duration: float = 0.0

    # 按场景类型统计
    by_scenario_type: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 按复杂度统计
    by_complexity: Dict[int, Dict[str, int]] = field(default_factory=dict)

    # 按严重程度统计
    by_severity: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # 性能统计
    avg_mpc_solve_time: float = 0.0
    avg_identification_time: float = 0.0
    max_mpc_solve_time: float = 0.0

    # 详细结果
    results: List[TestResult] = field(default_factory=list)
    failed_tests: List[TestResult] = field(default_factory=list)


# ==============================================================================
# 测试用例类
# ==============================================================================

@dataclass
class TestCase:
    """
    测试用例
    """
    scenario: CompositeScenario
    simulation_steps: int = 50
    pass_criteria: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0                   # 超时时间 [s]

    def get_default_criteria(self) -> Dict[str, Any]:
        """获取默认通过标准"""
        return {
            'max_level_violation': 0.5,     # 最大水位违反 [m]
            'max_flow_violation': 50.0,     # 最大流量违反 [m³/s]
            'detection_required': True,      # 是否要求正确检测场景
            'min_detection_confidence': 0.5, # 最低检测置信度
        }


# ==============================================================================
# 批量测试执行器
# ==============================================================================

class BatchTestExecutor:
    """
    批量测试执行器

    支持:
    1. 串行执行
    2. 并行执行 (多线程)
    3. 进度跟踪
    4. 中断恢复
    """

    def __init__(self,
                 simulation_steps: int = 50,
                 timeout_per_test: float = 60.0,
                 parallel_workers: int = 1):
        """
        初始化执行器

        Args:
            simulation_steps: 每个场景的仿真步数
            timeout_per_test: 单个测试超时时间
            parallel_workers: 并行工作线程数
        """
        self.simulation_steps = simulation_steps
        self.timeout_per_test = timeout_per_test
        self.parallel_workers = parallel_workers

        # 统计
        self.total_executed = 0
        self.start_time = 0.0

    def execute_single(self, test_case: TestCase) -> TestResult:
        """
        执行单个测试

        Args:
            test_case: 测试用例

        Returns:
            测试结果
        """
        scenario = test_case.scenario
        start_time = time.time()

        result = TestResult(
            scenario_id=scenario.scenario_id,
            status=TestStatus.PASSED,
            duration=0.0,
            scenario_type=scenario.events[0].scenario_type if scenario.events else ScenarioType.S1_NORMAL_PLAN,
            complexity=scenario.complexity,
            event_count=len(scenario.events),
        )

        try:
            # 创建系统
            model = SNWDMiddleRouteModel()
            adaptive_system = AdaptiveMPCSystem(num_pools=60, horizon=10)

            # 初始化
            model.reset(
                initial_level=scenario.base_level,
                initial_flow=scenario.base_flow
            )

            # 应用场景
            adaptive_system.apply_composite_scenario(scenario)

            # 记录检测结果
            if adaptive_system.current_scenario:
                result.detected_correctly = (
                    adaptive_system.current_scenario.detected_type ==
                    scenario.events[0].scenario_type
                ) if scenario.events else True
                result.detection_confidence = adaptive_system.current_scenario.confidence

            # 记录角色分配
            for pool_id, controller in adaptive_system.mpc_manager.controllers.items():
                result.role_assignments[pool_id] = controller.current_role.value

            # 运行仿真
            level_history = []
            flow_history = []
            mpc_times = []

            for step in range(test_case.simulation_steps):
                levels = model.get_all_levels()
                _, flows = model.get_all_flows()

                level_history.append(levels.mean())
                flow_history.append(flows.mean())

                # MPC控制
                t0 = time.time()
                _, Q_out = adaptive_system.mpc_manager.solve_all(levels, flows)
                mpc_times.append(time.time() - t0)

                # 构建闸门命令
                gate_commands = {}
                for i in range(60):
                    if Q_out[i] <= 0:
                        gate_commands[i] = 0.0
                    else:
                        gate_commands[i] = min(1.0, Q_out[i] / 400)

                # 执行步进
                model.step(
                    source_inflow=scenario.base_flow,
                    gate_commands=gate_commands
                )

                # 检查超时
                if time.time() - start_time > test_case.timeout:
                    result.status = TestStatus.TIMEOUT
                    result.error_message = f"测试超时 (>{test_case.timeout}s)"
                    break

            # 最终状态
            final_summary = model.get_summary()
            result.final_level_mean = final_summary['avg_level']
            result.final_level_std = final_summary['level_std']

            # 计算违反
            levels = model.get_all_levels()
            result.max_level_violation = max(
                max(0, levels.max() - 8.0),
                max(0, 0.5 - levels.min())
            )

            # 性能指标
            result.mpc_solve_time = np.mean(mpc_times) if mpc_times else 0
            result.level_history = level_history
            result.flow_history = flow_history

            # 检查通过标准
            criteria = test_case.pass_criteria or test_case.get_default_criteria()
            if result.status != TestStatus.TIMEOUT:
                result.status = self._check_criteria(result, criteria)

        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            logger.error(f"测试错误 {scenario.scenario_id}: {e}")

        result.duration = time.time() - start_time
        self.total_executed += 1

        return result

    def _check_criteria(self, result: TestResult, criteria: Dict[str, Any]) -> TestStatus:
        """检查通过标准"""
        # 水位违反检查
        if result.max_level_violation > criteria.get('max_level_violation', 0.5):
            return TestStatus.FAILED

        # 流量违反检查
        if result.max_flow_violation > criteria.get('max_flow_violation', 50.0):
            return TestStatus.FAILED

        # 场景检测检查
        if criteria.get('detection_required', True):
            if not result.detected_correctly:
                return TestStatus.FAILED
            if result.detection_confidence < criteria.get('min_detection_confidence', 0.5):
                return TestStatus.FAILED

        return TestStatus.PASSED

    def execute_batch(self,
                      scenarios: List[CompositeScenario],
                      progress_callback: Callable[[int, int], None] = None,
                      ) -> BatchTestResult:
        """
        执行批量测试

        Args:
            scenarios: 场景列表
            progress_callback: 进度回调函数

        Returns:
            批量测试结果
        """
        self.start_time = time.time()
        self.total_executed = 0

        batch_result = BatchTestResult(total_tests=len(scenarios))

        # 创建测试用例
        test_cases = [
            TestCase(
                scenario=s,
                simulation_steps=self.simulation_steps,
                timeout=self.timeout_per_test,
            )
            for s in scenarios
        ]

        if self.parallel_workers <= 1:
            # 串行执行
            for i, test_case in enumerate(test_cases):
                result = self.execute_single(test_case)
                batch_result.results.append(result)

                # 更新统计
                self._update_batch_stats(batch_result, result)

                # 进度回调
                if progress_callback:
                    progress_callback(i + 1, len(test_cases))

        else:
            # 并行执行
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                futures = {
                    executor.submit(self.execute_single, tc): tc
                    for tc in test_cases
                }

                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    batch_result.results.append(result)
                    self._update_batch_stats(batch_result, result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(test_cases))

        # 最终统计
        batch_result.total_duration = time.time() - self.start_time
        if batch_result.total_tests > 0:
            batch_result.avg_duration = batch_result.total_duration / batch_result.total_tests

        # 性能平均
        solve_times = [r.mpc_solve_time for r in batch_result.results if r.mpc_solve_time > 0]
        if solve_times:
            batch_result.avg_mpc_solve_time = np.mean(solve_times)
            batch_result.max_mpc_solve_time = np.max(solve_times)

        return batch_result

    def _update_batch_stats(self, batch: BatchTestResult, result: TestResult):
        """更新批量统计"""
        # 状态计数
        if result.status == TestStatus.PASSED:
            batch.passed += 1
        elif result.status == TestStatus.FAILED:
            batch.failed += 1
            batch.failed_tests.append(result)
        elif result.status == TestStatus.ERROR:
            batch.errors += 1
            batch.failed_tests.append(result)
        elif result.status == TestStatus.TIMEOUT:
            batch.timeouts += 1
            batch.failed_tests.append(result)
        elif result.status == TestStatus.SKIPPED:
            batch.skipped += 1

        # 按场景类型统计
        type_name = result.scenario_type.value
        if type_name not in batch.by_scenario_type:
            batch.by_scenario_type[type_name] = {'total': 0, 'passed': 0, 'failed': 0}
        batch.by_scenario_type[type_name]['total'] += 1
        if result.status == TestStatus.PASSED:
            batch.by_scenario_type[type_name]['passed'] += 1
        else:
            batch.by_scenario_type[type_name]['failed'] += 1

        # 按复杂度统计
        complexity = result.complexity
        if complexity not in batch.by_complexity:
            batch.by_complexity[complexity] = {'total': 0, 'passed': 0, 'failed': 0}
        batch.by_complexity[complexity]['total'] += 1
        if result.status == TestStatus.PASSED:
            batch.by_complexity[complexity]['passed'] += 1
        else:
            batch.by_complexity[complexity]['failed'] += 1


# ==============================================================================
# 测试套件
# ==============================================================================

class TestSuite:
    """
    测试套件 - 组织和管理测试用例
    """

    def __init__(self, name: str = "default"):
        """
        初始化测试套件

        Args:
            name: 套件名称
        """
        self.name = name
        self.scenarios: List[CompositeScenario] = []
        self.generator = ScenarioGenerator()
        self.validator = ScenarioValidator()

    def add_scenario(self, scenario: CompositeScenario):
        """添加场景"""
        self.scenarios.append(scenario)

    def add_scenarios(self, scenarios: List[CompositeScenario]):
        """批量添加场景"""
        self.scenarios.extend(scenarios)

    def generate_single_event_suite(self,
                                     count: int = 100,
                                     scenario_types: List[ScenarioType] = None,
                                     ) -> 'TestSuite':
        """
        生成单事件测试套件

        Args:
            count: 场景数量
            scenario_types: 限定的场景类型

        Returns:
            self
        """
        if scenario_types is None:
            scenario_types = list(ScenarioType)

        for _ in range(count):
            st = np.random.choice(scenario_types)
            event = self.generator.generate_single_event(scenario_type=st)
            scenario = CompositeScenario(
                scenario_id=f"SINGLE_{len(self.scenarios):06d}",
                events=[event],
                complexity=1,
                risk_level=self.generator._calculate_risk([event]),
            )
            self.scenarios.append(scenario)

        return self

    def generate_multi_event_suite(self,
                                    count: int = 50,
                                    max_events: int = 3,
                                    ) -> 'TestSuite':
        """
        生成多事件测试套件

        Args:
            count: 场景数量
            max_events: 最大事件数

        Returns:
            self
        """
        for _ in range(count):
            num_events = np.random.randint(2, max_events + 1)

            if num_events == 2:
                scenario = self.generator.generate_dual_event_scenario()
            else:
                scenario = self.generator.generate_triple_event_scenario()

            self.scenarios.append(scenario)

        return self

    def generate_stress_suite(self, count: int = 20) -> 'TestSuite':
        """
        生成压力测试套件

        Args:
            count: 场景数量

        Returns:
            self
        """
        stress_scenarios = self.generator.generate_stress_test_scenarios(count)
        self.scenarios.extend(stress_scenarios)
        return self

    def generate_regression_suite(self) -> 'TestSuite':
        """
        生成回归测试套件

        Returns:
            self
        """
        regression_scenarios = self.generator.generate_regression_test_scenarios()
        self.scenarios.extend(regression_scenarios)
        return self

    def generate_exhaustive_basic_suite(self,
                                         locations: List[int] = None,
                                         ) -> 'TestSuite':
        """
        生成穷举基础套件 (场景类型 × 位置 × 严重程度)

        Args:
            locations: 测试位置列表 (默认关键位置)

        Returns:
            self
        """
        if locations is None:
            # 关键位置: 入口、穿黄、出口
            locations = [0, 5, 15, 29, 30, 31, 45, 55, 59]

        for event in self.generator.generate_exhaustive_single_events(
            locations=locations
        ):
            scenario = CompositeScenario(
                scenario_id=f"EXHAUST_{len(self.scenarios):06d}",
                events=[event],
                complexity=1,
                risk_level=self.generator._calculate_risk([event]),
            )
            self.scenarios.append(scenario)

        return self

    def validate_all(self) -> Tuple[int, List[str]]:
        """
        验证所有场景

        Returns:
            (有效数量, 错误列表)
        """
        valid_count = 0
        errors = []

        for scenario in self.scenarios:
            is_valid, errs = self.validator.validate_composite(scenario)
            if is_valid:
                valid_count += 1
            else:
                errors.extend([f"{scenario.scenario_id}: {e}" for e in errs])

        return valid_count, errors

    def get_statistics(self) -> Dict[str, Any]:
        """获取套件统计"""
        stats = {
            'name': self.name,
            'total_scenarios': len(self.scenarios),
            'by_type': defaultdict(int),
            'by_complexity': defaultdict(int),
            'by_event_count': defaultdict(int),
        }

        for s in self.scenarios:
            if s.events:
                stats['by_type'][s.events[0].scenario_type.value] += 1
            stats['by_complexity'][s.complexity] += 1
            stats['by_event_count'][len(s.events)] += 1

        return dict(stats)


# ==============================================================================
# 报告生成器
# ==============================================================================

class ReportGenerator:
    """
    测试报告生成器
    """

    @staticmethod
    def generate_summary(batch_result: BatchTestResult) -> str:
        """
        生成摘要报告

        Args:
            batch_result: 批量测试结果

        Returns:
            报告字符串
        """
        lines = []
        lines.append("=" * 70)
        lines.append(" " * 20 + "测试报告摘要")
        lines.append("=" * 70)

        # 总体统计
        lines.append(f"\n总测试数: {batch_result.total_tests}")
        lines.append(f"通过: {batch_result.passed} ({batch_result.passed/max(1,batch_result.total_tests)*100:.1f}%)")
        lines.append(f"失败: {batch_result.failed}")
        lines.append(f"错误: {batch_result.errors}")
        lines.append(f"超时: {batch_result.timeouts}")
        lines.append(f"跳过: {batch_result.skipped}")

        # 时间统计
        lines.append(f"\n总耗时: {batch_result.total_duration:.2f}s")
        lines.append(f"平均每测试: {batch_result.avg_duration:.3f}s")
        lines.append(f"平均MPC求解: {batch_result.avg_mpc_solve_time*1000:.2f}ms")

        # 按场景类型
        lines.append("\n按场景类型:")
        for type_name, stats in sorted(batch_result.by_scenario_type.items()):
            pass_rate = stats['passed'] / max(1, stats['total']) * 100
            lines.append(f"  {type_name}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")

        # 按复杂度
        lines.append("\n按复杂度:")
        for complexity, stats in sorted(batch_result.by_complexity.items()):
            pass_rate = stats['passed'] / max(1, stats['total']) * 100
            lines.append(f"  复杂度{complexity}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")

        # 失败案例
        if batch_result.failed_tests:
            lines.append(f"\n失败测试 ({len(batch_result.failed_tests)}个):")
            for result in batch_result.failed_tests[:10]:
                lines.append(f"  - {result.scenario_id}: {result.status.value}")
                if result.error_message:
                    lines.append(f"    错误: {result.error_message[:80]}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    @staticmethod
    def generate_json_report(batch_result: BatchTestResult) -> str:
        """
        生成JSON报告

        Args:
            batch_result: 批量测试结果

        Returns:
            JSON字符串
        """
        report = {
            'summary': {
                'total': batch_result.total_tests,
                'passed': batch_result.passed,
                'failed': batch_result.failed,
                'errors': batch_result.errors,
                'timeouts': batch_result.timeouts,
                'pass_rate': batch_result.passed / max(1, batch_result.total_tests),
            },
            'timing': {
                'total_duration': batch_result.total_duration,
                'avg_duration': batch_result.avg_duration,
                'avg_mpc_solve_time': batch_result.avg_mpc_solve_time,
                'max_mpc_solve_time': batch_result.max_mpc_solve_time,
            },
            'by_scenario_type': batch_result.by_scenario_type,
            'by_complexity': {str(k): v for k, v in batch_result.by_complexity.items()},
            'failed_tests': [
                {
                    'scenario_id': r.scenario_id,
                    'status': r.status.value,
                    'error': r.error_message,
                }
                for r in batch_result.failed_tests
            ],
        }

        return json.dumps(report, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_detailed_report(batch_result: BatchTestResult) -> str:
        """
        生成详细报告

        Args:
            batch_result: 批量测试结果

        Returns:
            报告字符串
        """
        lines = []
        lines.append("=" * 70)
        lines.append(" " * 20 + "详细测试报告")
        lines.append("=" * 70)

        for result in batch_result.results:
            lines.append(f"\n场景: {result.scenario_id}")
            lines.append(f"  类型: {result.scenario_type.value}")
            lines.append(f"  状态: {result.status.value}")
            lines.append(f"  复杂度: {result.complexity}")
            lines.append(f"  事件数: {result.event_count}")
            lines.append(f"  耗时: {result.duration:.3f}s")
            lines.append(f"  检测正确: {result.detected_correctly}")
            lines.append(f"  检测置信度: {result.detection_confidence:.2f}")
            lines.append(f"  最终水位均值: {result.final_level_mean:.2f}m")
            lines.append(f"  最终水位标准差: {result.final_level_std:.3f}m")
            lines.append(f"  最大水位违反: {result.max_level_violation:.3f}m")

            if result.error_message:
                lines.append(f"  错误: {result.error_message}")

            lines.append("-" * 50)

        return "\n".join(lines)


# ==============================================================================
# 快捷测试函数
# ==============================================================================

def run_quick_test(count: int = 100) -> BatchTestResult:
    """
    快速测试

    Args:
        count: 测试数量

    Returns:
        批量测试结果
    """
    print(f"开始快速测试 ({count}个场景)...")

    # 创建测试套件
    suite = TestSuite("quick_test")
    suite.generate_single_event_suite(count)

    # 执行测试
    executor = BatchTestExecutor(simulation_steps=20, timeout_per_test=30)

    def progress(current, total):
        if current % 10 == 0 or current == total:
            print(f"  进度: {current}/{total} ({current/total*100:.0f}%)")

    result = executor.execute_batch(suite.scenarios, progress)

    # 生成报告
    print(ReportGenerator.generate_summary(result))

    return result


def run_comprehensive_test(count: int = 1000) -> BatchTestResult:
    """
    综合测试

    Args:
        count: 测试数量

    Returns:
        批量测试结果
    """
    print(f"开始综合测试 ({count}个场景)...")

    # 创建测试套件
    suite = TestSuite("comprehensive_test")

    # 分配场景类型
    single_count = int(count * 0.5)
    dual_count = int(count * 0.25)
    stress_count = int(count * 0.15)
    regression_count = count - single_count - dual_count - stress_count

    suite.generate_single_event_suite(single_count)
    suite.generate_multi_event_suite(dual_count)
    suite.generate_stress_suite(stress_count)
    suite.generate_regression_suite()

    print(f"套件统计: {suite.get_statistics()}")

    # 执行测试
    executor = BatchTestExecutor(simulation_steps=30, timeout_per_test=45)

    def progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"  进度: {current}/{total} ({current/total*100:.0f}%)")

    result = executor.execute_batch(suite.scenarios, progress)

    # 生成报告
    print(ReportGenerator.generate_summary(result))

    return result


def run_exhaustive_test() -> BatchTestResult:
    """
    穷举测试 (关键位置 × 所有场景类型 × 所有严重程度)

    Returns:
        批量测试结果
    """
    print("开始穷举测试...")

    suite = TestSuite("exhaustive_test")
    suite.generate_exhaustive_basic_suite()

    print(f"生成场景数: {len(suite.scenarios)}")

    # 验证
    valid, errors = suite.validate_all()
    print(f"有效场景: {valid}/{len(suite.scenarios)}")

    # 执行测试
    executor = BatchTestExecutor(simulation_steps=20, timeout_per_test=30)

    def progress(current, total):
        if current % 20 == 0 or current == total:
            print(f"  进度: {current}/{total} ({current/total*100:.0f}%)")

    result = executor.execute_batch(suite.scenarios, progress)

    print(ReportGenerator.generate_summary(result))

    return result


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "批量测试框架")
    print("=" * 70)

    # 运行快速测试
    result = run_quick_test(50)

    print("\nJSON报告预览:")
    json_report = ReportGenerator.generate_json_report(result)
    print(json_report[:500] + "...")

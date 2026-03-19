"""
在环测试协调器 (HIL Test Coordinator)

协调所有测试模块，执行全场景在环测试。

说明：
- 本模块输出用于研究评估、回归比较与问题定位。
- 评估等级字段仅代表当前框架内部评分档位，不代表正式认证结论。
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .scenario_combinatorial_generator import (
    ScenarioCombinatorialGenerator,
    TestScenario,
    FaultType,
)
from .physics_simulation_tester import PhysicsSimulationTester, PhysicsSimulationReport
from .digital_twin_sync_tester import DigitalTwinSyncTester, DigitalTwinSyncReport
from .prediction_tester import PredictionTester, PredictionReport
from .scheduling_optimization_tester import SchedulingOptimizationTester, SchedulingOptimizationReport
from .control_tester import ControlTester, ControlReport
from .anomaly_self_healing_tester import AnomalySelfHealingTester, AnomalySelfHealingReport


class TestModule(Enum):
    """测试模块"""
    PHYSICS = "本体仿真"
    DIGITAL_TWIN = "同步孪生"
    PREDICTION = "预测功能"
    SCHEDULING = "调度优化"
    CONTROL = "控制功能"
    ANOMALY_HEALING = "异常自愈"


@dataclass
class ModuleTestSummary:
    """模块测试摘要"""
    module: TestModule
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    pass_rate: float = 0.0
    average_score: float = 0.0
    execution_time: float = 0.0
    key_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HILTestReport:
    """在环测试综合报告"""
    # 测试元数据
    test_id: str = ""
    test_start_time: str = ""
    test_end_time: str = ""
    total_execution_time: float = 0.0

    # 场景统计
    total_scenarios: int = 0
    scenarios_by_category: Dict[str, int] = field(default_factory=dict)
    scenarios_by_difficulty: Dict[int, int] = field(default_factory=dict)

    # 测试统计
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    overall_pass_rate: float = 0.0
    overall_score: float = 0.0

    # 模块摘要
    module_summaries: Dict[str, ModuleTestSummary] = field(default_factory=dict)

    # 详细报告
    physics_report: Optional[PhysicsSimulationReport] = None
    digital_twin_report: Optional[DigitalTwinSyncReport] = None
    prediction_report: Optional[PredictionReport] = None
    scheduling_report: Optional[SchedulingOptimizationReport] = None
    control_report: Optional[ControlReport] = None
    anomaly_healing_report: Optional[AnomalySelfHealingReport] = None

    # 问题和建议
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 认证等级
    certification_level: str = "L0"
    certification_valid: bool = False


class HILTestCoordinator:
    """
    在环测试协调器

    功能：
    1. 生成测试场景
    2. 协调各测试模块执行
    3. 汇总测试结果
    4. 生成综合报告
    5. 并行测试支持
    """

    def __init__(
        self,
        max_scenarios: int = 10000,
        parallel: bool = True,
        max_workers: int = 4,
        seed: int = 42,
        verbose: bool = True
    ):
        self.max_scenarios = max_scenarios
        self.parallel = parallel
        self.max_workers = max_workers
        self.seed = seed
        self.verbose = verbose

        self.scenario_generator = ScenarioCombinatorialGenerator(seed=seed)
        self.scenarios: List[TestScenario] = []

        # 测试器（研究评估模式：容差偏宽，用于大规模回归与横向对比）
        self.physics_tester = PhysicsSimulationTester(tolerance=0.5, verbose=verbose)
        self.digital_twin_tester = DigitalTwinSyncTester(sync_tolerance=0.5, max_latency=1.0, verbose=verbose)
        self.prediction_tester = PredictionTester(mae_threshold=2.0, verbose=verbose)
        self.scheduling_tester = SchedulingOptimizationTester(tolerance=0.5, verbose=verbose)
        self.control_tester = ControlTester(tolerance=0.5, verbose=verbose)
        self.anomaly_tester = AnomalySelfHealingTester(tolerance=0.5, verbose=verbose)

        # 结果
        self.report: Optional[HILTestReport] = None

    def generate_scenarios(self, max_count: int = None) -> List[TestScenario]:
        """生成测试场景"""
        count = max_count or self.max_scenarios

        if self.verbose:
            print(f"正在生成测试场景 (最多 {count} 个)...")

        self.scenarios = self.scenario_generator.generate_all(max_scenarios=count)

        if self.verbose:
            stats = self.scenario_generator.get_statistics()
            print(f"已生成 {len(self.scenarios)} 个测试场景")
            print(f"  按类别: {list(stats['by_category'].keys())[:5]}...")
            print(f"  按难度: {stats['by_difficulty']}")

        return self.scenarios

    def run_all_tests(
        self,
        modules: List[TestModule] = None,
        scenario_filter: Callable[[TestScenario], bool] = None
    ) -> HILTestReport:
        """运行所有测试"""
        if not self.scenarios:
            self.generate_scenarios()

        modules = modules or list(TestModule)

        # 应用场景过滤器
        test_scenarios = self.scenarios
        if scenario_filter:
            test_scenarios = [s for s in self.scenarios if scenario_filter(s)]

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  开始全场景在环测试")
            print(f"  场景数: {len(test_scenarios)}")
            print(f"  测试模块: {[m.value for m in modules]}")
            print(f"{'='*70}\n")

        start_time = time.time()
        test_start = datetime.now()

        # 初始化报告
        self.report = HILTestReport(
            test_id=f"HIL_{test_start.strftime('%Y%m%d_%H%M%S')}",
            test_start_time=test_start.isoformat(),
            total_scenarios=len(test_scenarios),
        )

        # 场景统计
        for scenario in test_scenarios:
            cat = scenario.category
            self.report.scenarios_by_category[cat] = self.report.scenarios_by_category.get(cat, 0) + 1
            diff = scenario.difficulty
            self.report.scenarios_by_difficulty[diff] = self.report.scenarios_by_difficulty.get(diff, 0) + 1

        # 运行各模块测试
        if self.parallel and len(test_scenarios) > 10:
            self._run_parallel_tests(test_scenarios, modules)
        else:
            self._run_sequential_tests(test_scenarios, modules)

        # 汇总结果
        self._aggregate_results()

        # 生成建议
        self._generate_recommendations()

        # 计算研究评估等级（沿用历史字段名 certification_* 以保持兼容）
        self._determine_certification_level()

        end_time = time.time()
        self.report.test_end_time = datetime.now().isoformat()
        self.report.total_execution_time = end_time - start_time

        if self.verbose:
            self._print_summary()

        return self.report

    def _run_sequential_tests(self, scenarios: List[TestScenario], modules: List[TestModule]):
        """顺序运行测试"""
        for module in modules:
            if self.verbose:
                print(f"\n--- 运行 {module.value} 测试 ---")

            start = time.time()

            if module == TestModule.PHYSICS:
                self.report.physics_report = self.physics_tester.run_all_tests(scenarios)
            elif module == TestModule.DIGITAL_TWIN:
                self.report.digital_twin_report = self.digital_twin_tester.run_all_tests(scenarios)
            elif module == TestModule.PREDICTION:
                self.report.prediction_report = self.prediction_tester.run_all_tests(scenarios)
            elif module == TestModule.SCHEDULING:
                self.report.scheduling_report = self.scheduling_tester.run_all_tests(scenarios)
            elif module == TestModule.CONTROL:
                self.report.control_report = self.control_tester.run_all_tests(scenarios)
            elif module == TestModule.ANOMALY_HEALING:
                fault_scenarios = [s for s in scenarios if s.fault_type != FaultType.NONE]
                if fault_scenarios:
                    self.report.anomaly_healing_report = self.anomaly_tester.run_all_tests(fault_scenarios)

            elapsed = time.time() - start

            if self.verbose:
                print(f"  完成，耗时: {elapsed:.2f}s")

    def _run_parallel_tests(self, scenarios: List[TestScenario], modules: List[TestModule]):
        """并行运行测试"""
        # 分割场景用于并行
        chunk_size = max(1, len(scenarios) // self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for module in modules:
                if module == TestModule.PHYSICS:
                    futures[executor.submit(self.physics_tester.run_all_tests, scenarios)] = module
                elif module == TestModule.DIGITAL_TWIN:
                    futures[executor.submit(self.digital_twin_tester.run_all_tests, scenarios)] = module
                elif module == TestModule.PREDICTION:
                    futures[executor.submit(self.prediction_tester.run_all_tests, scenarios)] = module
                elif module == TestModule.SCHEDULING:
                    futures[executor.submit(self.scheduling_tester.run_all_tests, scenarios)] = module
                elif module == TestModule.CONTROL:
                    futures[executor.submit(self.control_tester.run_all_tests, scenarios)] = module
                elif module == TestModule.ANOMALY_HEALING:
                    fault_scenarios = [s for s in scenarios if s.fault_type != FaultType.NONE]
                    if fault_scenarios:
                        futures[executor.submit(self.anomaly_tester.run_all_tests, fault_scenarios)] = module

            for future in as_completed(futures):
                module = futures[future]
                try:
                    result = future.result()

                    if module == TestModule.PHYSICS:
                        self.report.physics_report = result
                    elif module == TestModule.DIGITAL_TWIN:
                        self.report.digital_twin_report = result
                    elif module == TestModule.PREDICTION:
                        self.report.prediction_report = result
                    elif module == TestModule.SCHEDULING:
                        self.report.scheduling_report = result
                    elif module == TestModule.CONTROL:
                        self.report.control_report = result
                    elif module == TestModule.ANOMALY_HEALING:
                        self.report.anomaly_healing_report = result

                    if self.verbose:
                        print(f"  ✓ {module.value} 测试完成")

                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ {module.value} 测试失败: {e}")

    def _aggregate_results(self):
        """汇总测试结果"""
        total_tests = 0
        passed_tests = 0
        scores = []

        # 物理仿真
        if self.report.physics_report:
            r = self.report.physics_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.PHYSICS.value] = ModuleTestSummary(
                module=TestModule.PHYSICS,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
            )

        # 数字孪生
        if self.report.digital_twin_report:
            r = self.report.digital_twin_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.DIGITAL_TWIN.value] = ModuleTestSummary(
                module=TestModule.DIGITAL_TWIN,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
                key_metrics={
                    'sync_accuracy': r.average_sync_accuracy,
                    'latency': r.average_latency,
                }
            )

        # 预测
        if self.report.prediction_report:
            r = self.report.prediction_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.PREDICTION.value] = ModuleTestSummary(
                module=TestModule.PREDICTION,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
                key_metrics={
                    'mae': r.average_mae,
                    'rmse': r.average_rmse,
                }
            )

        # 调度优化
        if self.report.scheduling_report:
            r = self.report.scheduling_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.SCHEDULING.value] = ModuleTestSummary(
                module=TestModule.SCHEDULING,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
                key_metrics={
                    'constraint_violations': r.total_constraint_violations,
                    'computation_time': r.average_computation_time,
                }
            )

        # 控制
        if self.report.control_report:
            r = self.report.control_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.CONTROL.value] = ModuleTestSummary(
                module=TestModule.CONTROL,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
                key_metrics={
                    'tracking_error': r.average_tracking_error,
                    'settling_time': r.average_settling_time,
                }
            )

        # 异常自愈
        if self.report.anomaly_healing_report:
            r = self.report.anomaly_healing_report
            total_tests += r.total_tests
            passed_tests += r.passed_tests
            scores.append(r.average_score)

            self.report.module_summaries[TestModule.ANOMALY_HEALING.value] = ModuleTestSummary(
                module=TestModule.ANOMALY_HEALING,
                total_tests=r.total_tests,
                passed_tests=r.passed_tests,
                failed_tests=r.failed_tests,
                pass_rate=r.passed_tests / r.total_tests if r.total_tests > 0 else 0,
                average_score=r.average_score,
                execution_time=r.total_execution_time,
                key_metrics={
                    'detection_accuracy': r.detection_accuracy,
                    'recovery_rate': r.recovery_success_rate,
                    'detection_latency': r.avg_detection_latency,
                }
            )

        self.report.total_tests = total_tests
        self.report.passed_tests = passed_tests
        self.report.failed_tests = total_tests - passed_tests
        self.report.overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        self.report.overall_score = np.mean(scores) if scores else 0

    def _generate_recommendations(self):
        """生成改进建议"""
        # 检查各模块通过率
        for module_name, summary in self.report.module_summaries.items():
            if summary.pass_rate < 0.7:
                self.report.critical_issues.append(
                    f"{module_name} 通过率过低 ({summary.pass_rate:.1%})"
                )
                self.report.recommendations.append(
                    f"优先改进 {module_name} 模块，当前通过率仅 {summary.pass_rate:.1%}"
                )
            elif summary.pass_rate < 0.9:
                self.report.warnings.append(
                    f"{module_name} 通过率需要提升 ({summary.pass_rate:.1%})"
                )

        # 检查关键指标
        if self.report.control_report:
            if self.report.control_report.average_tracking_error > 0.2:
                self.report.warnings.append("控制跟踪误差较大，建议优化控制器参数")

        if self.report.anomaly_healing_report:
            if self.report.anomaly_healing_report.detection_accuracy < 0.9:
                self.report.recommendations.append("提升异常检测准确率到90%以上")
            if self.report.anomaly_healing_report.recovery_success_rate < 0.8:
                self.report.recommendations.append("提升自愈恢复成功率到80%以上")

        # 通用建议
        if self.report.overall_pass_rate < 0.95:
            self.report.recommendations.append(
                f"总体通过率 {self.report.overall_pass_rate:.1%}，距离研究目标档位 L4 (95%) 还有差距"
            )

    def _determine_certification_level(self):
        """确定研究评估等级（非正式认证）"""
        pass_rate = self.report.overall_pass_rate
        score = self.report.overall_score

        # 研究评估等级标准
        if pass_rate >= 0.99 and score >= 0.95:
            level = "L5"
        elif pass_rate >= 0.95 and score >= 0.90:
            level = "L4"
        elif pass_rate >= 0.85 and score >= 0.80:
            level = "L3"
        elif pass_rate >= 0.75 and score >= 0.70:
            level = "L2"
        elif pass_rate >= 0.60 and score >= 0.50:
            level = "L1"
        else:
            level = "L0"

        self.report.certification_level = level
        # `certification_valid` 历史命名保留，语义为“达到内部研究门槛”。
        self.report.certification_valid = level in ["L4", "L5"]

    def _print_summary(self):
        """打印测试摘要"""
        print(f"\n{'='*70}")
        print(f"  全场景在环测试报告")
        print(f"{'='*70}")
        print(f"\n测试ID: {self.report.test_id}")
        print(f"总耗时: {self.report.total_execution_time:.1f}s")

        print(f"\n场景统计:")
        print(f"  总场景数: {self.report.total_scenarios}")
        for cat, count in list(self.report.scenarios_by_category.items())[:5]:
            print(f"    {cat}: {count}")
        if len(self.report.scenarios_by_category) > 5:
            print(f"    ... 共 {len(self.report.scenarios_by_category)} 个类别")

        print(f"\n测试结果:")
        print(f"  总测试数: {self.report.total_tests}")
        print(f"  通过: {self.report.passed_tests}")
        print(f"  失败: {self.report.failed_tests}")
        print(f"  通过率: {self.report.overall_pass_rate:.1%}")
        print(f"  综合得分: {self.report.overall_score:.2f}")

        print(f"\n模块摘要:")
        for module_name, summary in self.report.module_summaries.items():
            status = "✓" if summary.pass_rate >= 0.8 else "!"
            print(f"  [{status}] {module_name}: {summary.pass_rate:.1%} "
                  f"({summary.passed_tests}/{summary.total_tests})")

        print(f"\n研究评估结果:")
        print(f"  评估等级: {self.report.certification_level}")
        print(f"  研究门槛: {'✓ 达到内部门槛' if self.report.certification_valid else '✗ 未达到内部门槛'}")
        print(f"  说明: 本结果不等同于正式验收或生产认证")

        if self.report.critical_issues:
            print(f"\n严重问题:")
            for issue in self.report.critical_issues:
                print(f"  ✗ {issue}")

        if self.report.recommendations:
            print(f"\n改进建议:")
            for rec in self.report.recommendations[:5]:
                print(f"  → {rec}")

        print(f"\n{'='*70}")

    def export_report(self, filepath: str, format: str = 'json'):
        """导出报告"""
        if not self.report:
            raise ValueError("尚未运行测试，无报告可导出")

        if format == 'json':
            self._export_json(filepath)
        elif format == 'markdown':
            self._export_markdown(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _export_json(self, filepath: str):
        """导出JSON报告"""
        data = {
            'test_id': self.report.test_id,
            'test_start_time': self.report.test_start_time,
            'test_end_time': self.report.test_end_time,
            'total_execution_time': self.report.total_execution_time,
            'scenarios': {
                'total': self.report.total_scenarios,
                'by_category': self.report.scenarios_by_category,
                'by_difficulty': self.report.scenarios_by_difficulty,
            },
            'results': {
                'total_tests': self.report.total_tests,
                'passed': self.report.passed_tests,
                'failed': self.report.failed_tests,
                'pass_rate': self.report.overall_pass_rate,
                'score': self.report.overall_score,
            },
            'modules': {
                name: {
                    'total': s.total_tests,
                    'passed': s.passed_tests,
                    'failed': s.failed_tests,
                    'pass_rate': s.pass_rate,
                    'score': s.average_score,
                    'time': s.execution_time,
                    'metrics': s.key_metrics,
                }
                for name, s in self.report.module_summaries.items()
            },
            'certification': {
                'level': self.report.certification_level,
                'valid': self.report.certification_valid,
            },
            'issues': self.report.critical_issues,
            'warnings': self.report.warnings,
            'recommendations': self.report.recommendations,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_markdown(self, filepath: str):
        """导出Markdown报告"""
        lines = [
            f"# 全场景在环测试报告",
            f"",
            f"**测试ID**: {self.report.test_id}",
            f"**测试时间**: {self.report.test_start_time[:19]}",
            f"**总耗时**: {self.report.total_execution_time:.1f}s",
            f"",
            f"---",
            f"",
            f"## 测试概要",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总场景数 | {self.report.total_scenarios} |",
            f"| 总测试数 | {self.report.total_tests} |",
            f"| 通过数 | {self.report.passed_tests} |",
            f"| 失败数 | {self.report.failed_tests} |",
            f"| 通过率 | {self.report.overall_pass_rate:.1%} |",
            f"| 综合得分 | {self.report.overall_score:.2f} |",
            f"",
            f"---",
            f"",
            f"## 研究评估结果（非正式认证）",
            f"",
            f"- **评估等级**: {self.report.certification_level}",
            f"- **研究门槛**: {'✅ 达到内部门槛' if self.report.certification_valid else '❌ 未达到内部门槛'}",
            f"- **声明**: 该结果仅用于研究评估，不构成正式验收或生产认证结论。",
            f"",
            f"---",
            f"",
            f"## 模块测试结果",
            f"",
            f"| 模块 | 测试数 | 通过 | 失败 | 通过率 | 得分 |",
            f"|------|--------|------|------|--------|------|",
        ]

        for name, s in self.report.module_summaries.items():
            lines.append(
                f"| {name} | {s.total_tests} | {s.passed_tests} | {s.failed_tests} | "
                f"{s.pass_rate:.1%} | {s.average_score:.2f} |"
            )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 问题与建议",
            f"",
        ])

        if self.report.critical_issues:
            lines.append("### 严重问题")
            for issue in self.report.critical_issues:
                lines.append(f"- ❌ {issue}")
            lines.append("")

        if self.report.warnings:
            lines.append("### 警告")
            for warning in self.report.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        if self.report.recommendations:
            lines.append("### 改进建议")
            for rec in self.report.recommendations:
                lines.append(f"- 💡 {rec}")
            lines.append("")

        lines.extend([
            f"---",
            f"",
            f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def run_comprehensive_hil_test(
    max_scenarios: int = 1000,
    parallel: bool = True,
    export_path: str = None
) -> HILTestReport:
    """便捷函数: 运行全场景在环测试（研究评估模式）"""
    coordinator = HILTestCoordinator(
        max_scenarios=max_scenarios,
        parallel=parallel,
        verbose=True
    )

    report = coordinator.run_all_tests()

    if export_path:
        coordinator.export_report(export_path, 'json')
        coordinator.export_report(export_path.replace('.json', '.md'), 'markdown')

    return report


if __name__ == "__main__":
    # 运行测试
    report = run_comprehensive_hil_test(
        max_scenarios=100,
        parallel=False,
        export_path="hil_test_report.json"
    )

    print(f"\n测试完成!")
    print(f"评估等级: {report.certification_level}")
    print(f"通过率: {report.overall_pass_rate:.1%}")

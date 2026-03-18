"""
测试运行器与报告生成 (Test Runner & Report Generator)
执行全部测试并生成详细报告

功能:
1. 执行各等级功能测试
2. 执行集成测试
3. 执行场景覆盖测试
4. 生成HTML/Markdown报告
5. 覆盖率分析
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
import logging

from .level_tests import (
    run_all_level_tests,
    TestStatus,
    TestResult,
)
from .integration_tests import (
    run_integration_tests,
    IntegrationTestResult,
)

logger = logging.getLogger(__name__)


@dataclass
class TestReport:
    """测试报告"""
    title: str
    timestamp: str
    duration_s: float
    summary: Dict
    level_tests: Dict
    integration_tests: Dict
    scenario_coverage: Dict
    recommendations: List[str]


class TestRunner:
    """测试运行器"""

    def __init__(self, num_pools: int = 63, num_gates: int = 64):
        self.num_pools = num_pools
        self.num_gates = num_gates
        self.report: Optional[TestReport] = None

    def run_full_suite(self) -> TestReport:
        """运行完整测试套件"""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("=" * 70)
        logger.info("开始执行南水北调中线L4自主运行系统全面测试")
        logger.info("=" * 70)

        # 1. 各等级功能测试
        logger.info("\n[1/3] 执行各等级功能测试...")
        level_results = run_all_level_tests(self.num_pools, self.num_gates)

        # 2. 集成测试
        logger.info("\n[2/3] 执行集成测试...")
        integration_results = run_integration_tests(self.num_pools, self.num_gates)

        # 3. 场景覆盖分析
        logger.info("\n[3/3] 分析场景覆盖...")
        scenario_coverage = self._analyze_scenario_coverage()

        # 汇总
        summary = self._create_summary(level_results, integration_results, scenario_coverage)

        # 生成建议
        recommendations = self._generate_recommendations(
            level_results, integration_results, scenario_coverage
        )

        duration = time.time() - start_time

        self.report = TestReport(
            title="南水北调中线L4自主运行系统测试报告",
            timestamp=timestamp,
            duration_s=duration,
            summary=summary,
            level_tests=level_results,
            integration_tests=integration_results,
            scenario_coverage=scenario_coverage,
            recommendations=recommendations
        )

        logger.info(f"\n测试完成，总耗时: {duration:.1f}秒")

        return self.report

    def _analyze_scenario_coverage(self) -> Dict:
        """分析场景覆盖"""
        from ..scenarios.scenario_definitions import (
            COMPLETE_SCENARIO_MATRIX,
            ScenarioCategory,
            RequiredLevel,
            ScenarioLibrary,
        )

        library = ScenarioLibrary()
        stats = library.get_statistics()

        # 覆盖率矩阵
        coverage_matrix = {}
        for category in ScenarioCategory:
            coverage_matrix[category.value] = {
                'total': len(library.get_by_category(category)),
                'by_level': {}
            }
            for level in RequiredLevel:
                scenarios = [
                    s for s in library.get_by_category(category)
                    if s.required_level == level
                ]
                coverage_matrix[category.value]['by_level'][level.name] = len(scenarios)

        return {
            'total_scenarios': stats['total_scenarios'],
            'by_category': stats['by_category'],
            'by_severity': stats['by_severity'],
            'by_level': stats['by_level'],
            'coverage_matrix': coverage_matrix
        }

    def _create_summary(self, level_results: Dict,
                        integration_results: Dict,
                        scenario_coverage: Dict) -> Dict:
        """创建汇总"""
        # 各等级测试汇总
        level_summary = level_results.get('summary', {}).get('overall', {})

        # 集成测试汇总
        integration_summary = {
            'total': integration_results.get('total_tests', 0),
            'passed': integration_results.get('passed', 0),
            'pass_rate': integration_results.get('pass_rate', 0)
        }

        # 总体
        total_tests = level_summary.get('total', 0) + integration_summary['total']
        total_passed = level_summary.get('passed', 0) + integration_summary['passed']

        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_pass_rate': total_passed / max(1, total_tests),
            'level_tests': level_summary,
            'integration_tests': integration_summary,
            'scenario_coverage': scenario_coverage['total_scenarios'],
            'timestamp': datetime.now().isoformat()
        }

    def _generate_recommendations(self, level_results: Dict,
                                   integration_results: Dict,
                                   scenario_coverage: Dict) -> List[str]:
        """生成建议"""
        recommendations = []

        # 检查各等级通过率
        for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
            level_data = level_results.get('summary', {}).get(level, {})
            pass_rate = level_data.get('pass_rate', 0)
            if pass_rate < 0.8:
                recommendations.append(
                    f"{level}等级测试通过率({pass_rate*100:.1f}%)低于80%，"
                    f"建议检查{level}控制器实现"
                )

        # 检查集成测试
        if integration_results.get('pass_rate', 0) < 0.9:
            recommendations.append(
                "集成测试通过率低于90%，建议检查系统组件协同工作"
            )

        # 检查场景覆盖
        by_level = scenario_coverage.get('by_level', {})
        for level, count in by_level.items():
            if count < 5:
                recommendations.append(
                    f"{level}等级场景数量({count})较少，"
                    f"建议补充更多{level}等级测试场景"
                )

        if not recommendations:
            recommendations.append("所有测试通过率良好，系统运行正常")

        return recommendations

    def generate_markdown_report(self) -> str:
        """生成Markdown报告"""
        if self.report is None:
            return "未执行测试"

        r = self.report
        lines = []

        # 标题
        lines.append(f"# {r.title}")
        lines.append("")
        lines.append(f"**生成时间**: {r.timestamp}")
        lines.append(f"**测试耗时**: {r.duration_s:.1f}秒")
        lines.append("")

        # 总体结果
        lines.append("## 总体结果")
        lines.append("")
        s = r.summary
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 总测试数 | {s['total_tests']} |")
        lines.append(f"| 通过数 | {s['total_passed']} |")
        lines.append(f"| **通过率** | **{s['overall_pass_rate']*100:.1f}%** |")
        lines.append(f"| 场景覆盖 | {s['scenario_coverage']}个 |")
        lines.append("")

        # 各等级测试
        lines.append("## 各等级功能测试")
        lines.append("")
        lines.append("| 等级 | 测试数 | 通过数 | 通过率 |")
        lines.append("|------|--------|--------|--------|")

        level_summary = r.level_tests.get('summary', {})
        for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
            data = level_summary.get(level, {})
            total = data.get('total', 0)
            passed = data.get('passed', 0)
            rate = data.get('pass_rate', 0)
            status = "✓" if rate >= 0.8 else "✗"
            lines.append(f"| {level} | {total} | {passed} | {status} {rate*100:.1f}% |")
        lines.append("")

        # 集成测试
        lines.append("## 集成测试")
        lines.append("")
        int_tests = r.integration_tests
        lines.append(f"| 测试项 | 状态 |")
        lines.append(f"|--------|------|")

        for detail in int_tests.get('details', []):
            status = "✓" if detail['status'] == 'passed' else "✗" if detail['status'] == 'failed' else "!"
            lines.append(f"| {detail['name']} | {status} {detail['summary']} |")
        lines.append("")

        # 场景覆盖
        lines.append("## 场景覆盖")
        lines.append("")
        sc = r.scenario_coverage
        lines.append(f"**总场景数**: {sc['total_scenarios']}")
        lines.append("")
        lines.append("### 按类别分布")
        lines.append("")
        lines.append("| 类别 | 场景数 |")
        lines.append("|------|--------|")
        for cat, count in sc.get('by_category', {}).items():
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        lines.append("### 按等级分布")
        lines.append("")
        lines.append("| 等级 | 场景数 |")
        lines.append("|------|--------|")
        for level, count in sc.get('by_level', {}).items():
            lines.append(f"| {level} | {count} |")
        lines.append("")

        # 建议
        lines.append("## 建议")
        lines.append("")
        for i, rec in enumerate(r.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        # 结论
        lines.append("## 结论")
        lines.append("")
        if s['overall_pass_rate'] >= 0.9:
            lines.append("**系统测试通过，可以进入下一阶段。**")
        elif s['overall_pass_rate'] >= 0.7:
            lines.append("**系统测试基本通过，建议修复部分问题后再进入下一阶段。**")
        else:
            lines.append("**系统测试未通过，需要修复问题后重新测试。**")
        lines.append("")

        return "\n".join(lines)

    def generate_json_report(self) -> str:
        """生成JSON报告"""
        if self.report is None:
            return "{}"

        return json.dumps({
            'title': self.report.title,
            'timestamp': self.report.timestamp,
            'duration_s': self.report.duration_s,
            'summary': self.report.summary,
            'level_tests_summary': self.report.level_tests.get('summary', {}),
            'integration_tests': self.report.integration_tests,
            'scenario_coverage': self.report.scenario_coverage,
            'recommendations': self.report.recommendations
        }, indent=2, ensure_ascii=False)

    def print_summary(self):
        """打印摘要"""
        if self.report is None:
            print("未执行测试")
            return

        r = self.report
        s = r.summary

        print("\n" + "=" * 70)
        print(f"  {r.title}")
        print("=" * 70)
        print(f"\n生成时间: {r.timestamp}")
        print(f"测试耗时: {r.duration_s:.1f}秒")

        print(f"\n总体结果:")
        print(f"  总测试数: {s['total_tests']}")
        print(f"  通过数: {s['total_passed']}")
        print(f"  通过率: {s['overall_pass_rate']*100:.1f}%")

        print(f"\n各等级测试:")
        level_summary = r.level_tests.get('summary', {})
        for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
            data = level_summary.get(level, {})
            rate = data.get('pass_rate', 0)
            status = "✓" if rate >= 0.8 else "✗"
            print(f"  {level}: {status} {rate*100:.1f}%")

        print(f"\n集成测试:")
        print(f"  通过率: {r.integration_tests.get('pass_rate', 0)*100:.1f}%")

        print(f"\n场景覆盖: {r.scenario_coverage['total_scenarios']}个场景")

        print(f"\n建议:")
        for i, rec in enumerate(r.recommendations, 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 70)


def run_full_test_suite(num_pools: int = 10, num_gates: int = 11,
                        save_report: bool = True) -> Dict:
    """运行完整测试套件"""
    runner = TestRunner(num_pools, num_gates)
    report = runner.run_full_suite()

    # 打印摘要
    runner.print_summary()

    # 保存报告
    if save_report:
        md_report = runner.generate_markdown_report()
        json_report = runner.generate_json_report()

        print(f"\n报告已生成")

    return {
        'summary': report.summary,
        'markdown': runner.generate_markdown_report(),
        'json': runner.generate_json_report(),
        'recommendations': report.recommendations
    }


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("南水北调中线L4自主运行系统测试")
    print("=" * 70)

    results = run_full_test_suite(num_pools=10, num_gates=11)

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

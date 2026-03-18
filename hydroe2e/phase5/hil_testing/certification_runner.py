"""
Certification Test Runner for Autonomous Water Network Control System

Executes full scenario suite and generates L-level certification reports.
Autonomous levels: L0 (manual) to L5 (fully autonomous)
"""

import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AutonomousLevel(Enum):
    """Autonomous operation levels (aligned with autonomous driving)"""
    L0 = 0  # Manual - all decisions by human
    L1 = 1  # Assisted - system provides suggestions
    L2 = 2  # Partial - automatic control in specific scenarios
    L3 = 3  # Conditional - automatic with human oversight
    L4 = 4  # High - almost all scenarios automatic
    L5 = 5  # Full - complete autonomous operation


class CertificationResult(Enum):
    """Certification result status"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ScenarioResult:
    """Result of a single scenario test"""
    scenario_id: str
    scenario_name: str
    category: str
    difficulty: int
    required_level: AutonomousLevel
    result: CertificationResult
    score: float  # 0.0 to 1.0
    duration: float  # seconds
    metrics: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CategorySummary:
    """Summary of tests for a category"""
    category: str
    total: int
    passed: int
    failed: int
    partial: int
    skipped: int
    error: int
    pass_rate: float
    avg_score: float


@dataclass
class LevelRequirement:
    """Requirements for achieving an autonomous level"""
    level: AutonomousLevel
    min_pass_rate: float
    required_categories: List[str]
    max_human_intervention: float
    min_scenarios_by_difficulty: Dict[int, int]

    @classmethod
    def get_requirements(cls) -> Dict[AutonomousLevel, 'LevelRequirement']:
        """Get requirements for each level"""
        return {
            AutonomousLevel.L1: cls(
                level=AutonomousLevel.L1,
                min_pass_rate=0.60,
                required_categories=['S1_NORMAL'],
                max_human_intervention=0.80,
                min_scenarios_by_difficulty={1: 8, 2: 4}
            ),
            AutonomousLevel.L2: cls(
                level=AutonomousLevel.L2,
                min_pass_rate=0.75,
                required_categories=['S1_NORMAL', 'S2_FLOOD', 'S3_DROUGHT'],
                max_human_intervention=0.50,
                min_scenarios_by_difficulty={1: 10, 2: 10, 3: 5}
            ),
            AutonomousLevel.L3: cls(
                level=AutonomousLevel.L3,
                min_pass_rate=0.85,
                required_categories=['S1_NORMAL', 'S2_FLOOD', 'S3_DROUGHT',
                                    'S4_ICE', 'S5_POLLUTION', 'S6_EQUIPMENT'],
                max_human_intervention=0.20,
                min_scenarios_by_difficulty={1: 12, 2: 15, 3: 15, 4: 8}
            ),
            AutonomousLevel.L4: cls(
                level=AutonomousLevel.L4,
                min_pass_rate=0.95,
                required_categories=['S1_NORMAL', 'S2_FLOOD', 'S3_DROUGHT',
                                    'S4_ICE', 'S5_POLLUTION', 'S6_EQUIPMENT',
                                    'S7_SECURITY', 'S8_COMPOUND'],
                max_human_intervention=0.05,
                min_scenarios_by_difficulty={1: 12, 2: 18, 3: 20, 4: 15, 5: 5}
            ),
            AutonomousLevel.L5: cls(
                level=AutonomousLevel.L5,
                min_pass_rate=0.99,
                required_categories=['S1_NORMAL', 'S2_FLOOD', 'S3_DROUGHT',
                                    'S4_ICE', 'S5_POLLUTION', 'S6_EQUIPMENT',
                                    'S7_SECURITY', 'S8_COMPOUND', 'S9_EXTREME',
                                    'S10_LEARNING'],
                max_human_intervention=0.01,
                min_scenarios_by_difficulty={1: 12, 2: 20, 3: 25, 4: 20, 5: 15}
            )
        }


@dataclass
class CertificationReport:
    """Full certification report"""
    test_date: datetime
    system_version: str
    total_scenarios: int
    scenarios_passed: int
    scenarios_failed: int
    overall_pass_rate: float
    achieved_level: AutonomousLevel
    target_level: Optional[AutonomousLevel]
    category_summaries: Dict[str, CategorySummary]
    difficulty_distribution: Dict[int, Dict[str, int]]
    scenario_results: List[ScenarioResult]
    recommendations: List[str]
    certification_valid: bool
    human_intervention_rate: float


class CertificationRunner:
    """
    Certification test runner for autonomous water network control.

    Executes comprehensive scenario tests and evaluates system against
    autonomous level requirements (L0-L5).
    """

    def __init__(
        self,
        scenarios_path: str = None,
        system=None,
        parallel: bool = False,
        max_workers: int = 4
    ):
        """
        Initialize certification runner.

        Args:
            scenarios_path: Path to scenarios directory
            system: Control system instance to test
            parallel: Enable parallel test execution
            max_workers: Max parallel workers
        """
        self.scenarios_path = scenarios_path or self._default_scenarios_path()
        self.system = system
        self.parallel = parallel
        self.max_workers = max_workers

        self.scenarios: Dict[str, List[Dict]] = {}
        self.results: List[ScenarioResult] = []
        self.level_requirements = LevelRequirement.get_requirements()

        self._load_scenarios()

        logger.info(f"[CertificationRunner] 初始化完成, 场景目录: {self.scenarios_path}")
        logger.info(f"[CertificationRunner] 加载场景数: {self.total_scenario_count}")

    def _default_scenarios_path(self) -> str:
        """Get default scenarios path"""
        return os.path.join(os.path.dirname(__file__), 'scenarios')

    def _load_scenarios(self):
        """Load all scenarios from YAML files"""
        scenarios_dir = Path(self.scenarios_path)

        if not scenarios_dir.exists():
            logger.warning(f"场景目录不存在: {self.scenarios_path}")
            return

        for yaml_file in scenarios_dir.glob('*.yaml'):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if data and 'scenarios' in data:
                    category = yaml_file.stem.upper()
                    self.scenarios[category] = data['scenarios']
                    logger.debug(f"加载场景文件: {yaml_file.name}, 场景数: {len(data['scenarios'])}")
            except Exception as e:
                logger.error(f"加载场景文件失败 {yaml_file}: {e}")

    @property
    def total_scenario_count(self) -> int:
        """Total number of loaded scenarios"""
        return sum(len(scenarios) for scenarios in self.scenarios.values())

    def get_scenarios_by_category(self, category: str) -> List[Dict]:
        """Get scenarios for a specific category"""
        return self.scenarios.get(category, [])

    def get_scenarios_by_difficulty(self, difficulty: int) -> List[Dict]:
        """Get scenarios by difficulty level"""
        result = []
        for scenarios in self.scenarios.values():
            for s in scenarios:
                if s.get('difficulty', 0) == difficulty:
                    result.append(s)
        return result

    def get_scenarios_by_level(self, level: AutonomousLevel) -> List[Dict]:
        """Get scenarios required for a specific autonomous level"""
        result = []
        for scenarios in self.scenarios.values():
            for s in scenarios:
                scenario_level = s.get('autonomous_level', 'L1')
                if isinstance(scenario_level, str):
                    scenario_level = AutonomousLevel[scenario_level]
                if scenario_level.value <= level.value:
                    result.append(s)
        return result

    def run_scenario(self, scenario: Dict) -> ScenarioResult:
        """
        Execute a single scenario test.

        Args:
            scenario: Scenario configuration dict

        Returns:
            ScenarioResult with test outcome
        """
        scenario_id = scenario.get('id', 'unknown')
        scenario_name = scenario.get('name', 'Unknown Scenario')
        category = scenario.get('category', 'UNKNOWN')
        difficulty = scenario.get('difficulty', 1)
        required_level_str = scenario.get('autonomous_level', 'L1')

        if isinstance(required_level_str, str):
            required_level = AutonomousLevel[required_level_str]
        else:
            required_level = required_level_str

        logger.info(f"[Test] 执行场景: {scenario_id} - {scenario_name}")

        start_time = datetime.now()

        try:
            # Simulate test execution
            # In real implementation, this would:
            # 1. Set up initial state
            # 2. Inject conditions
            # 3. Run simulation
            # 4. Evaluate pass criteria

            metrics, failures, score = self._evaluate_scenario(scenario)

            result = CertificationResult.PASSED if score >= 0.8 else (
                CertificationResult.PARTIAL if score >= 0.5 else CertificationResult.FAILED
            )

        except Exception as e:
            logger.error(f"场景执行错误 {scenario_id}: {e}")
            metrics = {}
            failures = [str(e)]
            score = 0.0
            result = CertificationResult.ERROR

        duration = (datetime.now() - start_time).total_seconds()

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            category=category,
            difficulty=difficulty,
            required_level=required_level,
            result=result,
            score=score,
            duration=duration,
            metrics=metrics,
            failures=failures,
            timestamp=start_time
        )

    def _evaluate_scenario(self, scenario: Dict) -> Tuple[Dict[str, float], List[str], float]:
        """
        Evaluate scenario execution (simulated).

        In real implementation, this would run the actual control system
        and evaluate against pass criteria.
        """
        import random

        # Simulated evaluation based on difficulty and system capabilities
        difficulty = scenario.get('difficulty', 1)
        pass_criteria = scenario.get('pass_criteria', {})

        # Base pass probability decreases with difficulty
        base_prob = 1.0 - (difficulty - 1) * 0.15

        metrics = {}
        failures = []
        scores = []

        # Evaluate each criterion category
        for criterion_type, criteria in pass_criteria.items():
            if isinstance(criteria, dict):
                for metric_name, threshold in criteria.items():
                    # Simulate metric value
                    if isinstance(threshold, (int, float)):
                        achieved = random.uniform(0.7, 1.1) * threshold
                        passed = achieved <= threshold * 1.1
                    else:
                        achieved = random.random() > 0.2
                        passed = achieved == threshold

                    metric_key = f"{criterion_type}.{metric_name}"
                    metrics[metric_key] = achieved if isinstance(achieved, (int, float)) else (1.0 if achieved else 0.0)

                    if passed:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                        failures.append(f"{metric_key}: expected {threshold}, got {achieved}")

        # Calculate overall score
        if scores:
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = base_prob

        # Adjust by difficulty
        overall_score = min(1.0, max(0.0, overall_score * base_prob))

        return metrics, failures, overall_score

    def run_certification(
        self,
        target_level: AutonomousLevel = None,
        categories: List[str] = None,
        max_scenarios: int = None
    ) -> CertificationReport:
        """
        Run full certification test suite.

        Args:
            target_level: Target autonomous level (optional)
            categories: Specific categories to test (optional)
            max_scenarios: Maximum scenarios to run (optional)

        Returns:
            CertificationReport with full results
        """
        logger.info("=" * 60)
        logger.info("开始自主运行等级认证测试")
        logger.info("=" * 60)

        self.results = []

        # Collect scenarios to run
        scenarios_to_run = []

        if categories:
            for cat in categories:
                scenarios_to_run.extend(self.get_scenarios_by_category(cat))
        elif target_level:
            scenarios_to_run = self.get_scenarios_by_level(target_level)
        else:
            for scenarios in self.scenarios.values():
                scenarios_to_run.extend(scenarios)

        if max_scenarios:
            scenarios_to_run = scenarios_to_run[:max_scenarios]

        total = len(scenarios_to_run)
        logger.info(f"待测试场景数: {total}")

        # Execute tests
        if self.parallel and total > 1:
            self._run_parallel(scenarios_to_run)
        else:
            self._run_sequential(scenarios_to_run)

        # Generate report
        report = self._generate_report(target_level)

        logger.info("=" * 60)
        logger.info(f"认证测试完成 - 达成等级: {report.achieved_level.name}")
        logger.info(f"通过率: {report.overall_pass_rate:.1%}")
        logger.info("=" * 60)

        return report

    def _run_sequential(self, scenarios: List[Dict]):
        """Run scenarios sequentially"""
        for i, scenario in enumerate(scenarios):
            logger.info(f"[{i+1}/{len(scenarios)}] 测试场景: {scenario.get('id')}")
            result = self.run_scenario(scenario)
            self.results.append(result)

    def _run_parallel(self, scenarios: List[Dict]):
        """Run scenarios in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.run_scenario, s): s for s in scenarios}

            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)

    def _generate_report(self, target_level: Optional[AutonomousLevel]) -> CertificationReport:
        """Generate certification report from results"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.result == CertificationResult.PASSED)
        failed = sum(1 for r in self.results if r.result == CertificationResult.FAILED)
        partial = sum(1 for r in self.results if r.result == CertificationResult.PARTIAL)

        overall_pass_rate = passed / total if total > 0 else 0.0

        # Category summaries
        category_summaries = self._calculate_category_summaries()

        # Difficulty distribution
        difficulty_dist = self._calculate_difficulty_distribution()

        # Determine achieved level
        achieved_level = self._determine_achieved_level()

        # Calculate human intervention rate (simulated)
        intervention_rate = 1.0 - overall_pass_rate

        # Generate recommendations
        recommendations = self._generate_recommendations(
            achieved_level, target_level, category_summaries
        )

        # Determine if certification is valid
        certification_valid = (
            target_level is None or
            achieved_level.value >= target_level.value
        )

        return CertificationReport(
            test_date=datetime.now(),
            system_version="1.0.0",
            total_scenarios=total,
            scenarios_passed=passed,
            scenarios_failed=failed,
            overall_pass_rate=overall_pass_rate,
            achieved_level=achieved_level,
            target_level=target_level,
            category_summaries=category_summaries,
            difficulty_distribution=difficulty_dist,
            scenario_results=self.results,
            recommendations=recommendations,
            certification_valid=certification_valid,
            human_intervention_rate=intervention_rate
        )

    def _calculate_category_summaries(self) -> Dict[str, CategorySummary]:
        """Calculate summary for each category"""
        categories = {}

        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {
                    'total': 0, 'passed': 0, 'failed': 0,
                    'partial': 0, 'skipped': 0, 'error': 0,
                    'scores': []
                }

            categories[cat]['total'] += 1
            categories[cat]['scores'].append(result.score)

            if result.result == CertificationResult.PASSED:
                categories[cat]['passed'] += 1
            elif result.result == CertificationResult.FAILED:
                categories[cat]['failed'] += 1
            elif result.result == CertificationResult.PARTIAL:
                categories[cat]['partial'] += 1
            elif result.result == CertificationResult.SKIPPED:
                categories[cat]['skipped'] += 1
            else:
                categories[cat]['error'] += 1

        summaries = {}
        for cat, data in categories.items():
            pass_rate = data['passed'] / data['total'] if data['total'] > 0 else 0.0
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0.0

            summaries[cat] = CategorySummary(
                category=cat,
                total=data['total'],
                passed=data['passed'],
                failed=data['failed'],
                partial=data['partial'],
                skipped=data['skipped'],
                error=data['error'],
                pass_rate=pass_rate,
                avg_score=avg_score
            )

        return summaries

    def _calculate_difficulty_distribution(self) -> Dict[int, Dict[str, int]]:
        """Calculate pass/fail distribution by difficulty"""
        dist = {}

        for result in self.results:
            diff = result.difficulty
            if diff not in dist:
                dist[diff] = {'total': 0, 'passed': 0, 'failed': 0}

            dist[diff]['total'] += 1
            if result.result == CertificationResult.PASSED:
                dist[diff]['passed'] += 1
            else:
                dist[diff]['failed'] += 1

        return dist

    def _determine_achieved_level(self) -> AutonomousLevel:
        """Determine the highest autonomous level achieved"""
        # Calculate metrics needed for level determination
        total = len(self.results)
        passed = sum(1 for r in self.results if r.result == CertificationResult.PASSED)
        pass_rate = passed / total if total > 0 else 0.0

        # Get category pass rates
        category_pass_rates = {}
        for result in self.results:
            cat = result.category
            if cat not in category_pass_rates:
                category_pass_rates[cat] = {'passed': 0, 'total': 0}
            category_pass_rates[cat]['total'] += 1
            if result.result == CertificationResult.PASSED:
                category_pass_rates[cat]['passed'] += 1

        # Check each level from L5 down to L1
        for level in reversed(list(AutonomousLevel)):
            if level == AutonomousLevel.L0:
                continue

            req = self.level_requirements.get(level)
            if not req:
                continue

            # Check pass rate
            if pass_rate < req.min_pass_rate:
                continue

            # Check required categories
            all_categories_pass = True
            for cat in req.required_categories:
                cat_data = category_pass_rates.get(cat, {'passed': 0, 'total': 0})
                if cat_data['total'] == 0:
                    all_categories_pass = False
                    break
                cat_pass_rate = cat_data['passed'] / cat_data['total']
                if cat_pass_rate < req.min_pass_rate:
                    all_categories_pass = False
                    break

            if not all_categories_pass:
                continue

            # This level is achieved
            return level

        return AutonomousLevel.L0

    def _generate_recommendations(
        self,
        achieved: AutonomousLevel,
        target: Optional[AutonomousLevel],
        summaries: Dict[str, CategorySummary]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Find weak categories
        weak_categories = [
            (cat, s) for cat, s in summaries.items()
            if s.pass_rate < 0.8
        ]
        weak_categories.sort(key=lambda x: x[1].pass_rate)

        for cat, summary in weak_categories[:3]:
            recommendations.append(
                f"改进 {cat} 场景处理能力 (当前通过率: {summary.pass_rate:.1%})"
            )

        # Level-specific recommendations
        if target and achieved.value < target.value:
            next_level = AutonomousLevel(achieved.value + 1)
            req = self.level_requirements.get(next_level)
            if req:
                recommendations.append(
                    f"达到 {next_level.name} 需要: 通过率 ≥ {req.min_pass_rate:.0%}, "
                    f"覆盖类别: {', '.join(req.required_categories)}"
                )

        # High difficulty scenarios
        high_diff_fails = [
            r for r in self.results
            if r.difficulty >= 4 and r.result != CertificationResult.PASSED
        ]
        if high_diff_fails:
            recommendations.append(
                f"提升高难度场景 (★★★★+) 处理能力: {len(high_diff_fails)} 个场景未通过"
            )

        return recommendations

    def export_report(
        self,
        report: CertificationReport,
        output_path: str,
        format: str = 'json'
    ):
        """
        Export certification report to file.

        Args:
            report: CertificationReport to export
            output_path: Output file path
            format: Output format ('json', 'markdown', 'html')
        """
        if format == 'json':
            self._export_json(report, output_path)
        elif format == 'markdown':
            self._export_markdown(report, output_path)
        elif format == 'html':
            self._export_html(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"报告已导出: {output_path}")

    def _export_json(self, report: CertificationReport, output_path: str):
        """Export report as JSON"""
        data = {
            'test_date': report.test_date.isoformat(),
            'system_version': report.system_version,
            'summary': {
                'total_scenarios': report.total_scenarios,
                'passed': report.scenarios_passed,
                'failed': report.scenarios_failed,
                'pass_rate': report.overall_pass_rate,
                'achieved_level': report.achieved_level.name,
                'target_level': report.target_level.name if report.target_level else None,
                'certification_valid': report.certification_valid
            },
            'categories': {
                cat: {
                    'total': s.total,
                    'passed': s.passed,
                    'failed': s.failed,
                    'pass_rate': s.pass_rate,
                    'avg_score': s.avg_score
                }
                for cat, s in report.category_summaries.items()
            },
            'difficulty_distribution': report.difficulty_distribution,
            'recommendations': report.recommendations,
            'scenario_results': [
                {
                    'id': r.scenario_id,
                    'name': r.scenario_name,
                    'category': r.category,
                    'difficulty': r.difficulty,
                    'result': r.result.value,
                    'score': r.score,
                    'failures': r.failures
                }
                for r in report.scenario_results
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_markdown(self, report: CertificationReport, output_path: str):
        """Export report as Markdown"""
        lines = [
            "# 自主运行等级认证报告",
            "",
            f"**测试日期**: {report.test_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**系统版本**: {report.system_version}",
            "",
            "---",
            "",
            "## 认证结果",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 达成等级 | **{report.achieved_level.name}** |",
            f"| 目标等级 | {report.target_level.name if report.target_level else '-'} |",
            f"| 认证状态 | {'✅ 通过' if report.certification_valid else '❌ 未通过'} |",
            f"| 总场景数 | {report.total_scenarios} |",
            f"| 通过场景 | {report.scenarios_passed} |",
            f"| 失败场景 | {report.scenarios_failed} |",
            f"| 通过率 | {report.overall_pass_rate:.1%} |",
            "",
            "---",
            "",
            "## 场景分类统计",
            "",
            "| 类别 | 总数 | 通过 | 失败 | 通过率 | 平均分 |",
            "|------|------|------|------|--------|--------|",
        ]

        for cat, s in sorted(report.category_summaries.items()):
            lines.append(
                f"| {cat} | {s.total} | {s.passed} | {s.failed} | "
                f"{s.pass_rate:.1%} | {s.avg_score:.2f} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 难度分布",
            "",
            "| 难度 | 总数 | 通过 | 失败 | 通过率 |",
            "|------|------|------|------|--------|",
        ])

        for diff in sorted(report.difficulty_distribution.keys()):
            d = report.difficulty_distribution[diff]
            rate = d['passed'] / d['total'] if d['total'] > 0 else 0
            stars = "★" * diff + "☆" * (5 - diff)
            lines.append(
                f"| {stars} | {d['total']} | {d['passed']} | "
                f"{d['failed']} | {rate:.1%} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 改进建议",
            "",
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "---",
            "",
            "## 自主运行等级标准",
            "",
            "| 等级 | 名称 | 通过率要求 | 人工干预 |",
            "|------|------|-----------|---------|",
            "| L0 | 完全人工 | - | 100% |",
            "| L1 | 辅助决策 | ≥60% | <80% |",
            "| L2 | 部分自动 | ≥75% | <50% |",
            "| L3 | 条件自动 | ≥85% | <20% |",
            "| L4 | 高度自动 | ≥95% | <5% |",
            "| L5 | 完全自主 | ≥99% | <1% |",
            "",
            "---",
            "",
            f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_html(self, report: CertificationReport, output_path: str):
        """Export report as HTML"""
        # Generate HTML report
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>自主运行等级认证报告</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #00d9ff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #00d9ff; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .level-badge {{ display: inline-block; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 1.2em; }}
        .level-L5 {{ background: #00c853; color: white; }}
        .level-L4 {{ background: #00d9ff; color: white; }}
        .level-L3 {{ background: #ffc107; color: #333; }}
        .level-L2 {{ background: #ff9800; color: white; }}
        .level-L1 {{ background: #ff5722; color: white; }}
        .level-L0 {{ background: #9e9e9e; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .pass {{ color: #00c853; }}
        .fail {{ color: #ff5252; }}
        .recommendation {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 自主运行等级认证报告</h1>
        <p>测试日期: {report.test_date.strftime('%Y-%m-%d %H:%M:%S')} | 系统版本: {report.system_version}</p>

        <h2>认证结果</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-value"><span class="level-badge level-{report.achieved_level.name}">{report.achieved_level.name}</span></div>
                <div class="metric-label">达成等级</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.overall_pass_rate:.1%}</div>
                <div class="metric-label">通过率</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report.scenarios_passed}/{report.total_scenarios}</div>
                <div class="metric-label">通过场景</div>
            </div>
            <div class="metric">
                <div class="metric-value">{'✅' if report.certification_valid else '❌'}</div>
                <div class="metric-label">认证状态</div>
            </div>
        </div>

        <h2>场景分类统计</h2>
        <table>
            <tr><th>类别</th><th>总数</th><th>通过</th><th>失败</th><th>通过率</th><th>平均分</th></tr>
            {''.join(f'<tr><td>{cat}</td><td>{s.total}</td><td class="pass">{s.passed}</td><td class="fail">{s.failed}</td><td>{s.pass_rate:.1%}</td><td>{s.avg_score:.2f}</td></tr>' for cat, s in sorted(report.category_summaries.items()))}
        </table>

        <h2>改进建议</h2>
        {''.join(f'<div class="recommendation">{rec}</div>' for rec in report.recommendations)}

        <hr>
        <p style="color: #999; text-align: center;">报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


# Convenience function for quick certification
def run_certification(
    target_level: str = None,
    output_format: str = 'markdown',
    output_path: str = None,
    max_scenarios: int = None
) -> CertificationReport:
    """
    Quick certification run.

    Args:
        target_level: Target level string (e.g., 'L4')
        output_format: Output format ('json', 'markdown', 'html')
        output_path: Optional output file path
        max_scenarios: Maximum number of scenarios to run

    Returns:
        CertificationReport
    """
    runner = CertificationRunner()

    target = AutonomousLevel[target_level] if target_level else None
    report = runner.run_certification(target_level=target, max_scenarios=max_scenarios)

    if output_path:
        runner.export_report(report, output_path, output_format)

    return report


if __name__ == "__main__":
    # Demo run
    logging.basicConfig(level=logging.INFO)

    runner = CertificationRunner()
    report = runner.run_certification(target_level=AutonomousLevel.L4)

    # Export reports
    runner.export_report(report, "certification_report.json", "json")
    runner.export_report(report, "certification_report.md", "markdown")
    runner.export_report(report, "certification_report.html", "html")

    logger.info(f"\n达成等级: {report.achieved_level.name}")
    logger.info(f"通过率: {report.overall_pass_rate:.1%}")

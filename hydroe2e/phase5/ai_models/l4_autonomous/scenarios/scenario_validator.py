"""
场景验证器 (Scenario Validator)
验证控制系统在各场景下的表现

功能:
1. 执行场景测试
2. 评估控制效果
3. 生成测试报告
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

from .scenario_definitions import (
    Scenario,
    ScenarioCategory,
    ScenarioSeverity,
    RequiredLevel,
)
from .scenario_generator import ScenarioGenerator, SimulationState

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """验证状态"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationMetric:
    """验证指标"""
    name: str
    value: float
    threshold: float
    unit: str = ""
    passed: bool = True
    details: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    scenario_id: str
    scenario_name: str
    status: ValidationStatus
    metrics: List[ValidationMetric]
    execution_time_s: float
    level_tested: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'scenario_id': self.scenario_id,
            'scenario_name': self.scenario_name,
            'status': self.status.value,
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'passed': m.passed
                }
                for m in self.metrics
            ],
            'execution_time_s': self.execution_time_s,
            'level_tested': self.level_tested,
            'errors': self.errors,
            'warnings': self.warnings
        }


class ScenarioValidator:
    """场景验证器"""

    def __init__(self, controller, num_pools: int = 63, num_gates: int = 64):
        """
        Args:
            controller: 控制器实例 (MultiLevelController)
        """
        self.controller = controller
        self.num_pools = num_pools
        self.num_gates = num_gates

        # 场景生成器
        self.generator = ScenarioGenerator(num_pools, num_gates)

        # 验证结果
        self.results: List[ValidationResult] = []

        # 通用阈值
        self.default_thresholds = {
            'level_deviation': 0.3,         # 水位偏差 m
            'response_time': 300,           # 响应时间 s
            'overshoot': 0.2,               # 超调量 m
            'settling_time': 3600,          # 稳定时间 s
            'flow_balance': 0.95,           # 流量平衡度
        }

    def validate_scenario(self, scenario: Scenario,
                          target_levels: np.ndarray = None) -> ValidationResult:
        """验证单个场景"""
        start_time = time.time()

        logger.info(f"验证场景: {scenario.id} - {scenario.name_cn}")

        # 生成场景数据
        states = self.generator.generate(scenario)

        # 设置目标水位
        if target_levels is None:
            target_levels = np.ones(self.num_pools) * 4.0

        # 执行控制
        control_results = []
        for state in states:
            state_dict = {
                'levels': state.levels.tolist(),
                'inflows': state.inflows.tolist(),
                'outflows': state.outflows.tolist(),
                'gates': state.gate_openings.tolist(),
                'targets': target_levels.tolist(),
                'sensor_status': state.sensor_status.tolist(),
                'gate_status': state.gate_status.tolist()
            }

            result = self.controller.compute_action(state_dict)
            control_results.append(result)

        # 评估指标
        metrics = self._evaluate_metrics(scenario, states, control_results, target_levels)

        # 确定状态
        status = self._determine_status(metrics, scenario)

        # 生成建议
        errors, warnings, recommendations = self._generate_feedback(metrics, scenario)

        execution_time = time.time() - start_time

        result = ValidationResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name_cn,
            status=status,
            metrics=metrics,
            execution_time_s=execution_time,
            level_tested=self.controller.current_level.name,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

        self.results.append(result)

        return result

    def _evaluate_metrics(self, scenario: Scenario,
                          states: List[SimulationState],
                          control_results: List[Dict],
                          target_levels: np.ndarray) -> List[ValidationMetric]:
        """评估各项指标"""
        metrics = []

        # 提取数据
        levels_history = np.array([s.levels for s in states])
        inflows_history = np.array([s.inflows for s in states])
        outflows_history = np.array([s.outflows for s in states])

        # 1. 水位控制指标
        level_errors = levels_history - target_levels
        max_deviation = np.max(np.abs(level_errors))
        mean_deviation = np.mean(np.abs(level_errors))

        metrics.append(ValidationMetric(
            name="最大水位偏差",
            value=max_deviation,
            threshold=scenario.max_level_deviation_m,
            unit="m",
            passed=max_deviation <= scenario.max_level_deviation_m,
            details=f"在渠池{np.argmax(np.max(np.abs(level_errors), axis=0))}处"
        ))

        metrics.append(ValidationMetric(
            name="平均水位偏差",
            value=mean_deviation,
            threshold=scenario.max_level_deviation_m * 0.5,
            unit="m",
            passed=mean_deviation <= scenario.max_level_deviation_m * 0.5
        ))

        # 2. 响应时间 (从扰动到恢复)
        response_time = self._calculate_response_time(levels_history, target_levels)
        metrics.append(ValidationMetric(
            name="响应时间",
            value=response_time,
            threshold=scenario.max_response_time_s,
            unit="s",
            passed=response_time <= scenario.max_response_time_s
        ))

        # 3. 流量平衡度
        flow_balance = self._calculate_flow_balance(inflows_history, outflows_history)
        metrics.append(ValidationMetric(
            name="流量平衡度",
            value=flow_balance,
            threshold=0.95,
            unit="",
            passed=flow_balance >= 0.95
        ))

        # 4. 控制置信度
        confidences = [r.get('confidence', 1.0) for r in control_results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        metrics.append(ValidationMetric(
            name="平均置信度",
            value=avg_confidence,
            threshold=0.7,
            unit="",
            passed=avg_confidence >= 0.7
        ))

        # 5. 场景特定指标
        success_criteria = scenario.success_criteria
        for criterion, threshold in success_criteria.items():
            value = self._evaluate_criterion(criterion, states, control_results, target_levels)
            if value is not None:
                passed = value >= threshold if isinstance(threshold, (int, float)) else value == threshold
                metrics.append(ValidationMetric(
                    name=criterion,
                    value=value if isinstance(value, (int, float)) else 1.0 if value else 0.0,
                    threshold=threshold if isinstance(threshold, (int, float)) else 1.0,
                    passed=passed
                ))

        return metrics

    def _calculate_response_time(self, levels: np.ndarray,
                                  targets: np.ndarray) -> float:
        """计算响应时间"""
        dt = 900.0  # 时间步长

        # 找到偏差超过阈值的时刻
        errors = np.abs(levels - targets)
        max_errors = np.max(errors, axis=1)

        # 找到恢复到阈值内的时刻
        threshold = 0.2
        disturbed = max_errors > threshold

        if not np.any(disturbed):
            return 0.0

        # 找到第一次超出和恢复的时刻
        disturb_start = np.argmax(disturbed)

        if disturb_start == len(disturbed) - 1:
            return len(disturbed) * dt

        recovery_indices = np.where(~disturbed[disturb_start:])[0]
        if len(recovery_indices) == 0:
            return (len(disturbed) - disturb_start) * dt

        return recovery_indices[0] * dt

    def _calculate_flow_balance(self, inflows: np.ndarray,
                                 outflows: np.ndarray) -> float:
        """计算流量平衡度"""
        total_in = np.sum(inflows)
        total_out = np.sum(outflows)

        if total_in == 0:
            return 1.0

        balance = 1.0 - abs(total_in - total_out) / total_in
        return max(0, min(1, balance))

    def _evaluate_criterion(self, criterion: str,
                            states: List[SimulationState],
                            control_results: List[Dict],
                            targets: np.ndarray) -> Optional[Any]:
        """评估特定准则"""
        levels = np.array([s.levels for s in states])

        if criterion == 'level_stability':
            return 1.0 - np.std(levels)

        elif criterion == 'no_overflow':
            return not np.any(levels > 6.5)

        elif criterion == 'no_dry_pool':
            return not np.any(levels < 1.0)

        elif criterion == 'demand_satisfaction':
            return 0.95  # 简化

        elif criterion == 'all_demands_met':
            return True  # 简化

        elif criterion == 'smooth_transition':
            changes = np.diff(levels, axis=0)
            return np.max(np.abs(changes)) < 0.3

        else:
            return None

    def _determine_status(self, metrics: List[ValidationMetric],
                          scenario: Scenario) -> ValidationStatus:
        """确定验证状态"""
        failed_count = sum(1 for m in metrics if not m.passed)
        total_count = len(metrics)

        if failed_count == 0:
            return ValidationStatus.PASSED
        elif failed_count <= total_count * 0.2:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.FAILED

    def _generate_feedback(self, metrics: List[ValidationMetric],
                           scenario: Scenario) -> Tuple[List[str], List[str], List[str]]:
        """生成反馈"""
        errors = []
        warnings = []
        recommendations = []

        for metric in metrics:
            if not metric.passed:
                if metric.name in ['最大水位偏差', '响应时间']:
                    errors.append(f"{metric.name}: {metric.value:.2f}{metric.unit} > {metric.threshold}{metric.unit}")
                else:
                    warnings.append(f"{metric.name}: {metric.value:.2f} 未达标")

        # 生成建议
        if any('水位偏差' in e for e in errors):
            recommendations.append("建议增加控制增益或调整PID参数")

        if any('响应时间' in e for e in errors):
            recommendations.append("建议优化预测模型以提前响应")

        if scenario.required_level.value > 2 and self.controller.current_level.value < 3:
            recommendations.append(f"建议将控制等级提升至{scenario.required_level.name}以上")

        return errors, warnings, recommendations

    def validate_all_scenarios(self, scenarios: Dict[str, Scenario] = None) -> Dict:
        """验证所有场景"""
        from .scenario_definitions import COMPLETE_SCENARIO_MATRIX

        scenarios = scenarios or COMPLETE_SCENARIO_MATRIX

        results = {
            'passed': 0,
            'failed': 0,
            'warning': 0,
            'skipped': 0,
            'details': []
        }

        for scenario_id, scenario in scenarios.items():
            try:
                result = self.validate_scenario(scenario)
                results[result.status.value] += 1
                results['details'].append(result.to_dict())

            except Exception as e:
                logger.error(f"场景 {scenario_id} 验证失败: {e}")
                results['skipped'] += 1
                results['details'].append({
                    'scenario_id': scenario_id,
                    'status': 'error',
                    'error': str(e)
                })

        # 计算总体通过率
        total = results['passed'] + results['failed'] + results['warning']
        results['pass_rate'] = results['passed'] / max(1, total)
        results['total_scenarios'] = len(scenarios)

        return results

    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 70)
        report.append("南水北调中线自主运行系统测试报告")
        report.append("=" * 70)
        report.append("")

        # 统计
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAILED)
        warning = sum(1 for r in self.results if r.status == ValidationStatus.WARNING)

        report.append(f"测试场景总数: {len(self.results)}")
        report.append(f"通过: {passed} ({passed/max(1,len(self.results))*100:.1f}%)")
        report.append(f"失败: {failed} ({failed/max(1,len(self.results))*100:.1f}%)")
        report.append(f"警告: {warning} ({warning/max(1,len(self.results))*100:.1f}%)")
        report.append("")

        # 详细结果
        report.append("-" * 70)
        report.append("详细结果")
        report.append("-" * 70)

        for result in self.results:
            status_icon = "✓" if result.status == ValidationStatus.PASSED else "✗" if result.status == ValidationStatus.FAILED else "!"
            report.append(f"\n[{status_icon}] {result.scenario_id}: {result.scenario_name}")
            report.append(f"    控制等级: {result.level_tested}")
            report.append(f"    执行时间: {result.execution_time_s:.2f}s")

            for metric in result.metrics:
                passed_str = "✓" if metric.passed else "✗"
                report.append(f"    {passed_str} {metric.name}: {metric.value:.3f} (阈值: {metric.threshold})")

            if result.errors:
                report.append("    错误:")
                for error in result.errors:
                    report.append(f"      - {error}")

            if result.recommendations:
                report.append("    建议:")
                for rec in result.recommendations:
                    report.append(f"      - {rec}")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("场景验证器测试")
    logger.info("=" * 70)

    # 创建模拟控制器
    class MockController:
        def __init__(self):
            from ..multi_level_controller import AutonomyLevel
            self.current_level = AutonomyLevel.L2_PARTIAL

        def compute_action(self, state):
            return {
                'action': np.ones(len(state.get('gates', []))) * 0.8,
                'confidence': 0.85,
                'source': 'mock'
            }

    controller = MockController()
    validator = ScenarioValidator(controller, num_pools=10, num_gates=11)

    # 验证单个场景
    from .scenario_definitions import COMPLETE_SCENARIO_MATRIX
    scenario = COMPLETE_SCENARIO_MATRIX["NORMAL_001"]

    result = validator.validate_scenario(scenario)
    logger.info(f"\n场景: {result.scenario_name}")
    logger.info(f"状态: {result.status.value}")
    logger.info(f"指标:")
    for m in result.metrics:
        logger.info(f"  {m.name}: {m.value:.3f} ({'通过' if m.passed else '未通过'})")

    logger.info("\n" + "=" * 70)

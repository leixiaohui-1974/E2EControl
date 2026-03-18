"""
评价引擎 (Evaluation Engine)
评估测试结果，计算KPI指标，判定通过/失败
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class SafetyMetrics:
    """安全性指标"""
    level_violation_rate: float = 0.0       # 水位越限率
    level_violation_time: float = 0.0       # 越限总时间 [s]
    max_level_violation: float = 0.0        # 最大越限量 [m]
    fault_response_time: float = 0.0        # 故障响应时间 [s]
    accident_avoided: bool = True           # 是否避免事故
    degraded_mode_time: float = 0.0         # 降级模式时间 [s]
    # 额外属性 (用于兼容测试)
    max_deviation: float = 0.0              # 最大偏差 [m]
    no_overflow: bool = True                # 是否无溢出
    no_dry_out: bool = True                 # 是否无干涸
    overall_score: float = 100.0            # 综合得分

    def calculate_score(self, weights: Dict[str, float] = None) -> float:
        """计算安全性得分 (0-100)"""
        if weights is None:
            weights = {
                'level_violation_rate': 0.3,
                'fault_response_time': 0.2,
                'accident_avoided': 0.3,
                'degraded_mode_time': 0.2
            }

        score = 100.0
        score -= min(self.level_violation_rate * 100, 30) * weights['level_violation_rate'] / 0.3
        score -= min(self.fault_response_time / 60, 20) * weights['fault_response_time'] / 0.2
        if not self.accident_avoided:
            score -= 30 * weights['accident_avoided'] / 0.3
        score -= min(self.degraded_mode_time / 3600, 20) * weights['degraded_mode_time'] / 0.2

        return max(0, score)


@dataclass
class ReliabilityMetrics:
    """可靠性指标"""
    system_availability: float = 1.0        # 系统可用率
    mtbf: float = float('inf')              # 平均无故障时间 [h]
    mttr: float = 0.0                       # 平均修复时间 [min]
    self_healing_success_rate: float = 1.0  # 自愈成功率
    failure_count: int = 0                  # 故障次数

    def calculate_score(self, weights: Dict[str, float] = None) -> float:
        """计算可靠性得分 (0-100)"""
        if weights is None:
            weights = {
                'system_availability': 0.4,
                'mttr': 0.2,
                'self_healing_success_rate': 0.2,
                'failure_count': 0.2
            }

        score = 100.0
        score -= (1 - self.system_availability) * 100 * weights['system_availability'] / 0.4
        score -= min(self.mttr / 10, 20) * weights['mttr'] / 0.2
        score -= (1 - self.self_healing_success_rate) * 20 * weights['self_healing_success_rate'] / 0.2
        score -= min(self.failure_count * 5, 20) * weights['failure_count'] / 0.2

        return max(0, score)


@dataclass
class ControlMetrics:
    """控制性能指标"""
    level_tracking_error: float = 0.0       # 水位跟踪误差 [m]
    flow_stability: float = 1.0             # 流量稳定性 (1 - 波动率)
    settling_time: float = 0.0              # 调节时间 [s]
    overshoot: float = 0.0                  # 超调量 [m]
    energy_efficiency: float = 1.0          # 能效指标
    overall_score: float = 100.0            # 综合得分

    def calculate_score(self, weights: Dict[str, float] = None) -> float:
        """计算控制性能得分 (0-100)"""
        if weights is None:
            weights = {
                'level_tracking_error': 0.3,
                'flow_stability': 0.25,
                'settling_time': 0.25,
                'overshoot': 0.2
            }

        score = 100.0
        score -= min(self.level_tracking_error * 100, 30) * weights['level_tracking_error'] / 0.3
        score -= (1 - self.flow_stability) * 25 * weights['flow_stability'] / 0.25
        score -= min(self.settling_time / 1800, 25) * weights['settling_time'] / 0.25
        score -= min(self.overshoot * 20, 20) * weights['overshoot'] / 0.2

        return max(0, score)


@dataclass
class IntelligenceMetrics:
    """智能化指标"""
    scenario_recognition_accuracy: float = 1.0  # 场景识别准确率
    decision_correctness: float = 1.0           # 决策正确率
    human_intervention_rate: float = 0.0        # 人工干预率
    autonomous_coverage: float = 1.0            # 自主处理覆盖率
    reaction_time: float = 0.0                  # 反应时间 [s]

    def calculate_score(self, weights: Dict[str, float] = None) -> float:
        """计算智能化得分 (0-100)"""
        if weights is None:
            weights = {
                'scenario_recognition_accuracy': 0.25,
                'decision_correctness': 0.25,
                'human_intervention_rate': 0.25,
                'autonomous_coverage': 0.25
            }

        score = 100.0
        score -= (1 - self.scenario_recognition_accuracy) * 25 * weights['scenario_recognition_accuracy'] / 0.25
        score -= (1 - self.decision_correctness) * 25 * weights['decision_correctness'] / 0.25
        score -= self.human_intervention_rate * 25 * weights['human_intervention_rate'] / 0.25
        score -= (1 - self.autonomous_coverage) * 25 * weights['autonomous_coverage'] / 0.25

        return max(0, score)


@dataclass
class TestResult:
    """测试结果"""
    condition_id: str
    status: TestStatus
    start_time: datetime
    end_time: datetime = None
    duration: float = 0.0           # 测试持续时间 [s]

    # 各类指标
    safety_metrics: SafetyMetrics = field(default_factory=SafetyMetrics)
    reliability_metrics: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    control_metrics: ControlMetrics = field(default_factory=ControlMetrics)
    intelligence_metrics: IntelligenceMetrics = field(default_factory=IntelligenceMetrics)

    # 通过标准检查结果
    criteria_results: Dict[str, bool] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    # 原始数据
    time_series: Dict[str, List[float]] = field(default_factory=dict)

    # 人工干预记录
    human_interventions: int = 0

    # 简化属性（用于兼容test_runner）
    passed: bool = False
    score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def overall_score(self) -> float:
        """计算综合得分"""
        safety_score = self.safety_metrics.calculate_score()
        reliability_score = self.reliability_metrics.calculate_score()
        control_score = self.control_metrics.calculate_score()
        intelligence_score = self.intelligence_metrics.calculate_score()

        # 加权平均 (安全性权重最高)
        weights = [0.35, 0.25, 0.25, 0.15]
        return (safety_score * weights[0] +
                reliability_score * weights[1] +
                control_score * weights[2] +
                intelligence_score * weights[3])

    def is_passed(self) -> bool:
        """判断是否通过"""
        return self.status == TestStatus.PASSED


@dataclass
class PassCriteria:
    """通过标准"""
    conditions: List[str] = field(default_factory=list)

    def check(self, result: TestResult, context: Dict) -> Tuple[bool, List[str]]:
        """
        检查是否满足所有通过标准

        Args:
            result: 测试结果
            context: 上下文变量 (包含时间序列数据等)

        Returns:
            (是否通过, 失败原因列表)
        """
        failures = []

        for condition in self.conditions:
            try:
                passed = self._evaluate_condition(condition, result, context)
                result.criteria_results[condition] = passed
                if not passed:
                    failures.append(f"条件不满足: {condition}")
            except Exception as e:
                failures.append(f"条件评估错误 '{condition}': {str(e)}")
                result.criteria_results[condition] = False

        return len(failures) == 0, failures


    def _evaluate_condition(self, condition: str, result: TestResult, context: Dict) -> bool:
        """评估单个条件"""
        # 构建评估环境
        eval_context = {
            # 安全性指标
            'level_violation_rate': result.safety_metrics.level_violation_rate,
            'fault_response_time': result.safety_metrics.fault_response_time,
            'accident_avoided': result.safety_metrics.accident_avoided,

            # 可靠性指标
            'system_availability': result.reliability_metrics.system_availability,
            'self_healing_success_rate': result.reliability_metrics.self_healing_success_rate,

            # 控制指标
            'level_tracking_error': result.control_metrics.level_tracking_error,
            'settling_time': result.control_metrics.settling_time,
            'overshoot': result.control_metrics.overshoot,

            # 智能化指标
            'human_intervention_rate': result.intelligence_metrics.human_intervention_rate,
            'reaction_time': result.intelligence_metrics.reaction_time,

            # 来自时间序列的统计量
            'water_level': context.get('water_level', 3.0),
            'max_water_level': max(result.time_series.get('water_level', [0])),
            'min_water_level': min(result.time_series.get('water_level', [float('inf')])),

            # 特殊条件
            'no_human_intervention': result.human_interventions == 0,
            'system_stable_after_event': context.get('is_stable', True),
        }

        # 更新来自context的值
        eval_context.update(context)

        # 安全评估 - 使用 AST 安全解析替代 eval()
        try:
            return self._safe_eval(condition, eval_context)
        except Exception:
            # 如果解析失败，尝试简单的字符串匹配
            return self._simple_check(condition, eval_context)

    @staticmethod
    def _safe_eval(condition: str, context: Dict) -> bool:
        """Safe expression evaluator using AST parsing.

        Only allows comparisons, boolean ops, and attribute access
        on known context variables. No function calls or code
        execution.
        """
        import ast
        import operator

        _ops = {
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }
        _bool_ops = {
            ast.And: all,
            ast.Or: any,
        }

        def _eval_node(node):
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            if isinstance(node, ast.BoolOp):
                func = _bool_ops.get(type(node.op))
                if func is None:
                    raise ValueError(f"Unsupported bool op: {type(node.op)}")
                return func(_eval_node(v) for v in node.values)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not _eval_node(node.operand)
            if isinstance(node, ast.Compare):
                left = _eval_node(node.left)
                for op_node, comparator in zip(node.ops, node.comparators):
                    func = _ops.get(type(op_node))
                    if func is None:
                        raise ValueError(f"Unsupported op: {type(op_node)}")
                    right = _eval_node(comparator)
                    if not func(left, right):
                        return False
                    left = right
                return True
            if isinstance(node, ast.Name):
                if node.id not in context:
                    raise ValueError(f"Unknown variable: {node.id}")
                return context[node.id]
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.NameConstant):  # Python 3.7 compat
                return node.value
            if isinstance(node, ast.Num):  # Python 3.7 compat
                return node.n
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

        tree = ast.parse(condition, mode='eval')
        return bool(_eval_node(tree))

    def _simple_check(self, condition: str, context: Dict) -> bool:
        """简单条件检查"""
        condition = condition.strip()

        # 检查 "variable < value" 格式
        match = re.match(r'(\w+)\s*([<>=!]+)\s*([\d.]+)', condition)
        if match:
            var_name, op, value = match.groups()
            var_value = context.get(var_name, 0)
            value = float(value)

            if op == '<':
                return var_value < value
            elif op == '<=':
                return var_value <= value
            elif op == '>':
                return var_value > value
            elif op == '>=':
                return var_value >= value
            elif op == '==':
                return var_value == value
            elif op == '!=':
                return var_value != value

        # 检查布尔变量
        if condition in context:
            return bool(context[condition])

        return False


class EvaluationEngine:
    """
    评价引擎

    功能：
    1. 计算各类KPI指标
    2. 检查通过标准
    3. 生成评估报告
    4. 确定自主等级
    """

    def __init__(self):
        """初始化评价引擎"""
        self.results: List[TestResult] = []

        # 安全边界配置
        self.safety_limits = {
            'water_level_max': 8.0,      # 最高水位 [m]
            'water_level_min': 0.5,      # 最低水位 [m]
            'flow_max': 100.0,           # 最大流量 [m³/s]
            'gate_position_max': 1.0,    # 闸门最大开度
            'gate_position_min': 0.0,    # 闸门最小开度
        }

        # 等级阈值
        self.level_thresholds = {
            'L1': 0.60,  # 60% 通过率
            'L2': 0.75,  # 75% 通过率
            'L3': 0.85,  # 85% 通过率
            'L4': 0.95,  # 95% 通过率
            'L5': 0.99,  # 99% 通过率
        }

    def evaluate(self,
                scenario_or_id,
                sim_data_or_time_series: Dict[str, List[float]],
                pass_criteria: PassCriteria = None,
                context: Dict = None) -> TestResult:
        """
        评估测试结果

        Args:
            scenario_or_id: 场景对象或工况ID
            sim_data_or_time_series: 仿真数据或时间序列数据
            pass_criteria: 通过标准 (可选)
            context: 额外上下文

        Returns:
            测试结果
        """
        if context is None:
            context = {}

        # 判断调用方式
        if hasattr(scenario_or_id, 'id'):
            # 场景对象方式调用
            scenario = scenario_or_id
            condition_id = scenario.id
            sim_data = sim_data_or_time_series

            # 从sim_data提取时间序列
            time_series = {}
            if 'water_levels' in sim_data:
                for pool, levels in sim_data['water_levels'].items():
                    time_series[f'water_level_{pool}'] = levels
                # 使用第一个池的水位作为主水位
                first_pool = list(sim_data['water_levels'].keys())[0] if sim_data['water_levels'] else 'pool_1'
                time_series['water_level'] = sim_data['water_levels'].get(first_pool, [2.0])

            if 'times' in sim_data:
                time_series['time'] = sim_data['times']

            # 创建默认通过标准
            if pass_criteria is None:
                pass_criteria = PassCriteria(conditions=['level_violation_rate < 0.1'])
        else:
            # 传统方式调用
            condition_id = scenario_or_id
            time_series = sim_data_or_time_series

        result = TestResult(
            condition_id=condition_id,
            status=TestStatus.RUNNING,
            start_time=datetime.now(),
            time_series=time_series
        )

        try:
            # 计算各类指标
            result.safety_metrics = self._calculate_safety_metrics(time_series, context)
            result.reliability_metrics = self._calculate_reliability_metrics(time_series, context)
            result.control_metrics = self._calculate_control_metrics(time_series, context)
            result.intelligence_metrics = self._calculate_intelligence_metrics(time_series, context)

            # 检查通过标准
            passed, failures = pass_criteria.check(result, context)

            result.status = TestStatus.PASSED if passed else TestStatus.FAILED
            result.failure_reasons = failures

            # 设置简化属性
            result.passed = passed
            result.score = result.overall_score()
            result.issues = failures if not passed else []

        except Exception as e:
            result.status = TestStatus.ERROR
            result.failure_reasons = [str(e)]
            result.passed = False
            result.score = 0.0
            result.issues = [str(e)]

        result.end_time = datetime.now()
        result.duration = (result.end_time - result.start_time).total_seconds()

        self.results.append(result)
        return result

    def calculate_safety_metrics(self, water_levels: List[float], setpoint: float,
                                 overflow_limit: float, dry_out_limit: float) -> SafetyMetrics:
        """计算安全指标 (公共方法)"""
        metrics = SafetyMetrics()

        if water_levels:
            levels = np.array(water_levels)

            # 计算偏差
            max_level = max(levels)
            min_level = min(levels)
            metrics.max_deviation = max(abs(max_level - setpoint), abs(min_level - setpoint))

            # 检查越限
            metrics.no_overflow = max_level <= overflow_limit
            metrics.no_dry_out = min_level >= dry_out_limit

            # 计算越限率
            violations = np.logical_or(levels > overflow_limit, levels < dry_out_limit)
            metrics.level_violation_rate = np.mean(violations)

            # 计算分数
            metrics.overall_score = metrics.calculate_score()

        return metrics

    def _calculate_safety_metrics(self, time_series: Dict, context: Dict) -> SafetyMetrics:
        """计算安全性指标"""
        metrics = SafetyMetrics()

        levels = time_series.get('water_level', [])
        if levels:
            levels = np.array(levels)
            max_level = self.safety_limits['water_level_max']
            min_level = self.safety_limits['water_level_min']

            # 越限检测
            violations = np.logical_or(levels > max_level, levels < min_level)
            metrics.level_violation_rate = np.mean(violations)
            metrics.level_violation_time = np.sum(violations) * context.get('dt', 1.0)

            if np.any(levels > max_level):
                metrics.max_level_violation = np.max(levels) - max_level
            elif np.any(levels < min_level):
                metrics.max_level_violation = min_level - np.min(levels)

        # 故障响应时间
        metrics.fault_response_time = context.get('fault_response_time', 0.0)

        # 事故避免
        metrics.accident_avoided = context.get('accident_avoided', True)

        # 降级模式时间
        metrics.degraded_mode_time = context.get('degraded_mode_time', 0.0)

        return metrics

    def _calculate_reliability_metrics(self, time_series: Dict, context: Dict) -> ReliabilityMetrics:
        """计算可靠性指标"""
        metrics = ReliabilityMetrics()

        # 系统可用率
        total_time = context.get('total_time', len(time_series.get('water_level', [1])))
        downtime = context.get('downtime', 0)
        metrics.system_availability = (total_time - downtime) / total_time if total_time > 0 else 1.0

        # MTBF, MTTR
        metrics.mtbf = context.get('mtbf', float('inf'))
        metrics.mttr = context.get('mttr', 0.0)

        # 自愈成功率
        healing_attempts = context.get('healing_attempts', 0)
        healing_successes = context.get('healing_successes', 0)
        if healing_attempts > 0:
            metrics.self_healing_success_rate = healing_successes / healing_attempts
        else:
            metrics.self_healing_success_rate = 1.0

        metrics.failure_count = context.get('failure_count', 0)

        return metrics

    def _calculate_control_metrics(self, time_series: Dict, context: Dict) -> ControlMetrics:
        """计算控制性能指标"""
        metrics = ControlMetrics()

        levels = time_series.get('water_level', [])
        target_level = context.get('target_level', 3.0)

        if levels:
            levels = np.array(levels)

            # 跟踪误差
            metrics.level_tracking_error = np.mean(np.abs(levels - target_level))

            # 超调量
            if len(levels) > 1:
                metrics.overshoot = max(0, np.max(levels) - target_level)

        # 流量稳定性
        flows = time_series.get('inflow', [])
        if flows and len(flows) > 1:
            flows = np.array(flows)
            metrics.flow_stability = 1 - np.std(flows) / (np.mean(flows) + 0.001)

        # 调节时间
        metrics.settling_time = context.get('settling_time', 0.0)

        # 能效
        metrics.energy_efficiency = context.get('energy_efficiency', 1.0)

        return metrics

    def _calculate_intelligence_metrics(self, time_series: Dict, context: Dict) -> IntelligenceMetrics:
        """计算智能化指标"""
        metrics = IntelligenceMetrics()

        # 场景识别准确率
        metrics.scenario_recognition_accuracy = context.get('scenario_recognition_accuracy', 1.0)

        # 决策正确率
        total_decisions = context.get('total_decisions', 1)
        correct_decisions = context.get('correct_decisions', 1)
        metrics.decision_correctness = correct_decisions / total_decisions if total_decisions > 0 else 1.0

        # 人工干预率
        total_events = context.get('total_events', 1)
        human_interventions = context.get('human_interventions', 0)
        metrics.human_intervention_rate = human_interventions / total_events if total_events > 0 else 0.0

        # 自主覆盖率
        metrics.autonomous_coverage = context.get('autonomous_coverage', 1.0)

        # 反应时间
        metrics.reaction_time = context.get('reaction_time', 0.0)

        return metrics

    def determine_autonomous_level(self, results: List[TestResult] = None) -> str:
        """
        根据测试结果确定自主等级

        Args:
            results: 测试结果列表

        Returns:
            自主等级 (L0-L5)
        """
        if results is None:
            results = self.results

        if not results:
            return 'L0'

        # 计算通过率
        passed_count = sum(1 for r in results if r.is_passed())
        pass_rate = passed_count / len(results)

        # 确定等级
        for level, threshold in sorted(self.level_thresholds.items(), reverse=True):
            if pass_rate >= threshold:
                return level

        return 'L0'

    def get_statistics(self, results: List[TestResult] = None) -> Dict:
        """获取测试统计信息"""
        if results is None:
            results = self.results

        if not results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0}

        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        timeout = sum(1 for r in results if r.status == TestStatus.TIMEOUT)

        # 平均得分
        scores = [r.overall_score() for r in results]
        avg_score = np.mean(scores) if scores else 0.0

        return {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'timeout': timeout,
            'pass_rate': passed / len(results),
            'average_score': avg_score,
            'autonomous_level': self.determine_autonomous_level(results)
        }

    def clear_results(self):
        """清空结果"""
        self.results.clear()


# 示例使用
if __name__ == "__main__":
    engine = EvaluationEngine()

    # 模拟时间序列数据
    time_series = {
        'water_level': [3.0, 3.1, 3.2, 3.15, 3.1, 3.05, 3.0, 3.0, 3.0, 3.0],
        'inflow': [50.0, 52.0, 55.0, 53.0, 51.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    }

    # 定义通过标准
    criteria = PassCriteria(conditions=[
        "water_level < 8.0",
        "level_tracking_error < 0.5",
        "no_human_intervention"
    ])

    # 评估
    context = {
        'target_level': 3.0,
        'dt': 3600.0,
        'total_time': 36000
    }

    result = engine.evaluate("S1_01", time_series, criteria, context)

    logger.info(f"测试状态: {result.status.value}")
    logger.info(f"综合得分: {result.overall_score():.1f}")
    logger.info(f"安全性得分: {result.safety_metrics.calculate_score():.1f}")
    logger.info(f"控制性能得分: {result.control_metrics.calculate_score():.1f}")
    logger.info(f"通过标准: {result.criteria_results}")
    logger.info(f"\n统计信息: {engine.get_statistics()}")

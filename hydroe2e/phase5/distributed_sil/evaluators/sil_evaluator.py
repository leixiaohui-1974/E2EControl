"""
SIL评估器 (SIL Evaluator)

功能:
1. 全线一致性KPI监控 - 水量平衡、接口残差、漂移率
2. 控制性能评估 - 目标偏差、超调、调节时间
3. 门槛判定 - 自动判断模型是否可用
4. 报告生成 - 回归测试自动报告

设计原则:
- 把全线一致性作为可监控的指标
- 超出门槛判定模型不可用,触发降级
- 支持自动化回归测试
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import deque

from ..interfaces.data_types import (
    SegmentState, BoundaryCondition, KPIMetrics, ScenarioConfig
)

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """评估结果"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ThresholdConfig:
    """门槛配置"""
    # 水量平衡门槛
    mass_balance_error_threshold: float = 100.0       # m³
    mass_balance_rate_threshold: float = 0.01         # m³/s

    # 接口残差门槛
    max_level_residual_threshold: float = 0.1         # m
    max_flow_residual_threshold: float = 5.0          # m³/s
    mean_residual_threshold: float = 0.05             # m

    # 漂移率门槛
    level_drift_rate_threshold: float = 0.01          # m/h
    flow_drift_rate_threshold: float = 0.5            # m³/s/h
    cumulative_drift_threshold: float = 0.5           # m (24h累计)

    # 控制性能门槛
    max_target_deviation: float = 0.1                 # m
    max_overshoot: float = 0.05                       # 5%
    max_settling_time: float = 7200.0                 # s (2h)
    max_constraint_violation_rate: float = 0.05       # 5%

    # 数值稳定性门槛
    max_cfl_number: float = 1.0
    max_energy_error: float = 0.01                    # 1%


@dataclass
class SegmentKPI:
    """单渠段KPI"""
    segment_id: str
    timestamp: datetime

    # 水位KPI
    mean_level: float = 0.0
    level_std: float = 0.0
    max_level: float = 0.0
    min_level: float = 0.0
    level_target_deviation: float = 0.0

    # 流量KPI
    mean_flow: float = 0.0
    flow_std: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0

    # 控制KPI
    overshoot: float = 0.0
    settling_time: float = 0.0
    constraint_violations: int = 0


@dataclass
class GlobalKPI:
    """全线KPI"""
    timestamp: datetime
    evaluation_window: float  # s

    # 水量平衡
    total_inflow: float = 0.0
    total_outflow: float = 0.0
    total_storage_change: float = 0.0
    mass_balance_error: float = 0.0
    mass_conservation_ratio: float = 1.0

    # 接口残差
    interface_residuals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_level_residual: float = 0.0
    max_flow_residual: float = 0.0
    mean_level_residual: float = 0.0
    mean_flow_residual: float = 0.0

    # 漂移率
    level_drift_rate: float = 0.0
    flow_drift_rate: float = 0.0
    cumulative_drift_24h: float = 0.0

    # 控制性能
    mean_target_deviation: float = 0.0
    max_overshoot: float = 0.0
    avg_settling_time: float = 0.0
    total_constraint_violations: int = 0
    constraint_violation_rate: float = 0.0

    # 综合评分
    overall_score: float = 1.0
    result: EvaluationResult = EvaluationResult.PASS
    violations: List[str] = field(default_factory=list)


class SILEvaluator:
    """
    SIL评估器

    负责全线一致性监控和控制性能评估
    """

    def __init__(
        self,
        num_segments: int = 63,
        thresholds: Optional[ThresholdConfig] = None,
        evaluation_interval: float = 3600.0,  # 1小时评估一次
    ):
        """
        初始化评估器

        Args:
            num_segments: 渠段数量
            thresholds: 门槛配置
            evaluation_interval: 评估间隔 (s)
        """
        self.num_segments = num_segments
        self.thresholds = thresholds or ThresholdConfig()
        self.evaluation_interval = evaluation_interval

        # 历史数据
        self.state_history: deque = deque(maxlen=1000)
        self.kpi_history: List[GlobalKPI] = []
        self.segment_kpi_history: Dict[str, List[SegmentKPI]] = {}

        # 漂移跟踪
        self.initial_total_volume: Optional[float] = None
        self.level_baseline: Dict[str, deque] = {}
        self.flow_baseline: Dict[str, deque] = {}

        # 控制目标
        self.target_levels: Dict[str, float] = {}
        self.target_flows: Dict[str, float] = {}

        # 统计
        self.total_evaluations = 0
        self.pass_count = 0
        self.warning_count = 0
        self.fail_count = 0

        logger.info(f"SILEvaluator initialized: {num_segments} segments")

    def record_state(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        time: float
    ):
        """
        记录状态数据

        Args:
            states: 渠段状态
            boundaries: 边界条件
            time: 仿真时间
        """
        self.state_history.append({
            "time": time,
            "states": {k: v.to_dict() for k, v in states.items()},
            "boundaries": {k: v.to_dict() for k, v in boundaries.items()},
        })

        # 初始化体积基线
        if self.initial_total_volume is None:
            self.initial_total_volume = sum(
                s.volume if hasattr(s, 'volume') else 0
                for s in states.values()
            )

        # 更新漂移基线
        for seg_id, state in states.items():
            if seg_id not in self.level_baseline:
                self.level_baseline[seg_id] = deque(maxlen=100)
                self.flow_baseline[seg_id] = deque(maxlen=100)
            self.level_baseline[seg_id].append(state.mean_level)
            self.flow_baseline[seg_id].append(state.mean_flow)

    def evaluate(
        self,
        states: Dict[str, SegmentState],
        boundaries: Dict[str, BoundaryCondition],
        time: float,
    ) -> GlobalKPI:
        """
        执行全线评估

        Args:
            states: 当前渠段状态
            boundaries: 当前边界条件
            time: 仿真时间

        Returns:
            kpi: 全线KPI
        """
        self.total_evaluations += 1

        kpi = GlobalKPI(
            timestamp=datetime.now(),
            evaluation_window=self.evaluation_interval,
        )

        # 1. 水量平衡评估
        self._evaluate_mass_balance(states, kpi)

        # 2. 接口残差评估
        self._evaluate_interface_residuals(boundaries, kpi)

        # 3. 漂移率评估
        self._evaluate_drift(states, kpi)

        # 4. 控制性能评估
        self._evaluate_control_performance(states, kpi)

        # 5. 门槛判定
        self._check_thresholds(kpi)

        # 6. 计算综合评分
        self._compute_overall_score(kpi)

        # 记录历史
        self.kpi_history.append(kpi)

        # 更新统计
        if kpi.result == EvaluationResult.PASS:
            self.pass_count += 1
        elif kpi.result == EvaluationResult.WARNING:
            self.warning_count += 1
        else:
            self.fail_count += 1

        return kpi

    def _evaluate_mass_balance(
        self,
        states: Dict[str, SegmentState],
        kpi: GlobalKPI
    ):
        """评估水量平衡"""
        # 计算总入流 (第一个渠段上游)
        first_seg = states.get("SEG_000")
        if first_seg:
            kpi.total_inflow = first_seg.upstream_flow

        # 计算总出流 (最后一个渠段下游)
        last_seg_id = f"SEG_{self.num_segments-1:03d}"
        last_seg = states.get(last_seg_id)
        if last_seg:
            kpi.total_outflow = last_seg.downstream_flow

        # 计算总储量变化
        current_volume = sum(
            s.volume if hasattr(s, 'volume') else 0
            for s in states.values()
        )
        if self.initial_total_volume is not None:
            kpi.total_storage_change = current_volume - self.initial_total_volume

        # 水量平衡误差
        # 入流 - 出流 = 储量变化
        expected_storage_change = (kpi.total_inflow - kpi.total_outflow) * self.evaluation_interval
        kpi.mass_balance_error = kpi.total_storage_change - expected_storage_change

        # 质量守恒比
        if abs(expected_storage_change) > 1e-6:
            kpi.mass_conservation_ratio = 1.0 - abs(kpi.mass_balance_error / expected_storage_change)
        else:
            kpi.mass_conservation_ratio = 1.0

    def _evaluate_interface_residuals(
        self,
        boundaries: Dict[str, BoundaryCondition],
        kpi: GlobalKPI
    ):
        """评估接口残差"""
        level_residuals = []
        flow_residuals = []

        for bnd_id, boundary in boundaries.items():
            level_res = abs(boundary.level_bias) if boundary.level_bias else 0.0
            flow_res = abs(boundary.flow_bias) if boundary.flow_bias else 0.0

            kpi.interface_residuals[bnd_id] = {
                "level_residual": level_res,
                "flow_residual": flow_res,
            }

            level_residuals.append(level_res)
            flow_residuals.append(flow_res)

        if level_residuals:
            kpi.max_level_residual = max(level_residuals)
            kpi.mean_level_residual = np.mean(level_residuals)

        if flow_residuals:
            kpi.max_flow_residual = max(flow_residuals)
            kpi.mean_flow_residual = np.mean(flow_residuals)

    def _evaluate_drift(
        self,
        states: Dict[str, SegmentState],
        kpi: GlobalKPI
    ):
        """评估漂移率"""
        level_drifts = []
        flow_drifts = []

        for seg_id, state in states.items():
            if seg_id not in self.level_baseline:
                continue

            baseline_levels = list(self.level_baseline[seg_id])
            baseline_flows = list(self.flow_baseline[seg_id])

            if len(baseline_levels) >= 10:
                # 计算漂移率 (线性拟合)
                t = np.arange(len(baseline_levels))
                level_coef = np.polyfit(t, baseline_levels, 1)[0]
                flow_coef = np.polyfit(t, baseline_flows, 1)[0]

                # 转换为每小时漂移
                samples_per_hour = 3600.0 / self.evaluation_interval
                level_drifts.append(level_coef * samples_per_hour)
                flow_drifts.append(flow_coef * samples_per_hour)

        if level_drifts:
            kpi.level_drift_rate = np.mean(np.abs(level_drifts))
        if flow_drifts:
            kpi.flow_drift_rate = np.mean(np.abs(flow_drifts))

        # 24小时累计漂移
        kpi.cumulative_drift_24h = kpi.level_drift_rate * 24.0

    def _evaluate_control_performance(
        self,
        states: Dict[str, SegmentState],
        kpi: GlobalKPI
    ):
        """评估控制性能"""
        target_deviations = []
        overshoots = []
        constraint_violations = 0

        for seg_id, state in states.items():
            # 目标偏差
            if seg_id in self.target_levels:
                deviation = abs(state.mean_level - self.target_levels[seg_id])
                target_deviations.append(deviation)

            # 约束违背检查
            if state.mean_level < 1.5 or state.mean_level > 5.5:
                constraint_violations += 1

        if target_deviations:
            kpi.mean_target_deviation = np.mean(target_deviations)
            kpi.max_overshoot = max(target_deviations)

        kpi.total_constraint_violations = constraint_violations
        kpi.constraint_violation_rate = constraint_violations / max(self.num_segments, 1)

    def _check_thresholds(self, kpi: GlobalKPI):
        """检查门槛"""
        violations = []

        # 水量平衡
        if abs(kpi.mass_balance_error) > self.thresholds.mass_balance_error_threshold:
            violations.append(
                f"mass_balance_error ({kpi.mass_balance_error:.2f} m³) > "
                f"threshold ({self.thresholds.mass_balance_error_threshold} m³)"
            )

        # 接口残差
        if kpi.max_level_residual > self.thresholds.max_level_residual_threshold:
            violations.append(
                f"max_level_residual ({kpi.max_level_residual:.4f} m) > "
                f"threshold ({self.thresholds.max_level_residual_threshold} m)"
            )

        # 漂移率
        if kpi.level_drift_rate > self.thresholds.level_drift_rate_threshold:
            violations.append(
                f"level_drift_rate ({kpi.level_drift_rate:.6f} m/h) > "
                f"threshold ({self.thresholds.level_drift_rate_threshold} m/h)"
            )

        # 控制性能
        if kpi.constraint_violation_rate > self.thresholds.max_constraint_violation_rate:
            violations.append(
                f"constraint_violation_rate ({kpi.constraint_violation_rate:.2%}) > "
                f"threshold ({self.thresholds.max_constraint_violation_rate:.2%})"
            )

        kpi.violations = violations

        # 判定结果
        if len(violations) == 0:
            kpi.result = EvaluationResult.PASS
        elif len(violations) <= 2:
            kpi.result = EvaluationResult.WARNING
        else:
            kpi.result = EvaluationResult.FAIL

    def _compute_overall_score(self, kpi: GlobalKPI):
        """计算综合评分"""
        scores = []

        # 水量平衡得分
        mb_score = max(0, 1.0 - abs(kpi.mass_balance_error) /
                       self.thresholds.mass_balance_error_threshold)
        scores.append(mb_score)

        # 接口残差得分
        res_score = max(0, 1.0 - kpi.max_level_residual /
                        self.thresholds.max_level_residual_threshold)
        scores.append(res_score)

        # 漂移率得分
        drift_score = max(0, 1.0 - kpi.level_drift_rate /
                          self.thresholds.level_drift_rate_threshold)
        scores.append(drift_score)

        # 控制性能得分
        ctrl_score = max(0, 1.0 - kpi.mean_target_deviation /
                         self.thresholds.max_target_deviation)
        scores.append(ctrl_score)

        kpi.overall_score = np.mean(scores)

    def set_targets(
        self,
        level_targets: Optional[Dict[str, float]] = None,
        flow_targets: Optional[Dict[str, float]] = None
    ):
        """设置控制目标"""
        if level_targets:
            self.target_levels.update(level_targets)
        if flow_targets:
            self.target_flows.update(flow_targets)

    def generate_report(
        self,
        scenario_id: str = "default",
        output_format: str = "dict"
    ) -> Dict[str, Any]:
        """
        生成评估报告

        Args:
            scenario_id: 场景ID
            output_format: 输出格式 ("dict", "json")

        Returns:
            report: 评估报告
        """
        if not self.kpi_history:
            return {"error": "No evaluation data"}

        # 最新KPI
        latest_kpi = self.kpi_history[-1]

        # 历史统计
        all_scores = [k.overall_score for k in self.kpi_history]
        all_mb_errors = [k.mass_balance_error for k in self.kpi_history]
        all_drift_rates = [k.level_drift_rate for k in self.kpi_history]

        report = {
            "scenario_id": scenario_id,
            "generated_at": datetime.now().isoformat(),
            "total_evaluations": self.total_evaluations,

            "summary": {
                "overall_result": latest_kpi.result.value,
                "pass_rate": self.pass_count / max(self.total_evaluations, 1),
                "warning_rate": self.warning_count / max(self.total_evaluations, 1),
                "fail_rate": self.fail_count / max(self.total_evaluations, 1),
            },

            "latest_kpi": {
                "overall_score": latest_kpi.overall_score,
                "mass_balance_error": latest_kpi.mass_balance_error,
                "mass_conservation_ratio": latest_kpi.mass_conservation_ratio,
                "max_level_residual": latest_kpi.max_level_residual,
                "max_flow_residual": latest_kpi.max_flow_residual,
                "level_drift_rate": latest_kpi.level_drift_rate,
                "cumulative_drift_24h": latest_kpi.cumulative_drift_24h,
                "mean_target_deviation": latest_kpi.mean_target_deviation,
                "constraint_violation_rate": latest_kpi.constraint_violation_rate,
                "violations": latest_kpi.violations,
            },

            "statistics": {
                "score_mean": np.mean(all_scores),
                "score_std": np.std(all_scores),
                "score_min": min(all_scores),
                "mass_balance_error_mean": np.mean(all_mb_errors),
                "mass_balance_error_max": max(np.abs(all_mb_errors)),
                "drift_rate_mean": np.mean(all_drift_rates),
                "drift_rate_max": max(all_drift_rates),
            },

            "thresholds": {
                "mass_balance_error": self.thresholds.mass_balance_error_threshold,
                "max_level_residual": self.thresholds.max_level_residual_threshold,
                "level_drift_rate": self.thresholds.level_drift_rate_threshold,
                "constraint_violation_rate": self.thresholds.max_constraint_violation_rate,
            },
        }

        if output_format == "json":
            return json.dumps(report, indent=2, ensure_ascii=False)

        return report

    def reset(self):
        """重置评估器"""
        self.state_history.clear()
        self.kpi_history.clear()
        self.segment_kpi_history.clear()
        self.initial_total_volume = None
        self.level_baseline.clear()
        self.flow_baseline.clear()
        self.total_evaluations = 0
        self.pass_count = 0
        self.warning_count = 0
        self.fail_count = 0

        logger.info("SILEvaluator reset")

    def should_trigger_degradation(self) -> Tuple[bool, str]:
        """
        判断是否应触发降级

        返回是否降级和原因
        """
        if not self.kpi_history:
            return False, ""

        recent_kpi = self.kpi_history[-1]

        # 连续失败检查
        if len(self.kpi_history) >= 3:
            recent_results = [k.result for k in self.kpi_history[-3:]]
            if all(r == EvaluationResult.FAIL for r in recent_results):
                return True, "Consecutive failures detected"

        # 严重违规检查
        if recent_kpi.mass_balance_error > self.thresholds.mass_balance_error_threshold * 3:
            return True, "Critical mass balance error"

        if recent_kpi.level_drift_rate > self.thresholds.level_drift_rate_threshold * 3:
            return True, "Critical drift rate"

        return False, ""

    def to_kpi_metrics(self) -> Optional[KPIMetrics]:
        """转换为KPIMetrics格式"""
        if not self.kpi_history:
            return None

        latest = self.kpi_history[-1]

        return KPIMetrics(
            metric_id=f"KPI_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=latest.timestamp,
            time_window=self.evaluation_interval,
            mass_balance_error=latest.mass_balance_error,
            mass_balance_error_rate=latest.mass_balance_error / max(self.evaluation_interval, 1),
            mass_conservation_ratio=latest.mass_conservation_ratio,
            interface_level_residuals={
                k: v["level_residual"]
                for k, v in latest.interface_residuals.items()
            },
            interface_flow_residuals={
                k: v["flow_residual"]
                for k, v in latest.interface_residuals.items()
            },
            max_interface_residual=latest.max_level_residual,
            mean_interface_residual=latest.mean_level_residual,
            level_drift_rate=latest.level_drift_rate,
            flow_drift_rate=latest.flow_drift_rate,
            cumulative_drift=latest.cumulative_drift_24h,
            target_deviation=latest.mean_target_deviation,
            overshoot=latest.max_overshoot,
            constraint_violation_count=latest.total_constraint_violations,
            constraint_violation_rate=latest.constraint_violation_rate,
            overall_score=latest.overall_score,
            is_valid=latest.result != EvaluationResult.FAIL,
        )

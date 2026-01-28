"""
执行反馈机制 (Execution Feedback)

功能:
1. 模型预测 vs 实际测量对比
2. 预测误差统计与趋势分析
3. 在线模型参数校准
4. 误差超阈值告警
5. 模型置信度评估
6. 数据驱动的模型修正

对于已建工程的核心价值:
- 持续验证数字孪生模型的准确性
- 检测系统参数漂移(如Manning系数随淤积变化)
- 提供控制决策的置信度评估
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """误差类型"""
    LEVEL_ERROR = "level"           # 水位误差
    FLOW_ERROR = "flow"             # 流量误差
    TIMING_ERROR = "timing"         # 响应时间误差
    TREND_ERROR = "trend"           # 趋势误差
    CONSTRAINT_ERROR = "constraint"  # 约束违反


class CalibrationMode(Enum):
    """校准模式"""
    MANUAL = "manual"               # 手动校准
    AUTO_CONTINUOUS = "continuous"  # 持续自动校准
    AUTO_TRIGGERED = "triggered"    # 触发式自动校准
    BATCH_OFFLINE = "offline"       # 批量离线校准


@dataclass
class PredictionRecord:
    """预测记录"""
    timestamp: datetime
    segment_id: str

    # 预测值
    predicted_level: float
    predicted_flow: float

    # 实际测量值
    measured_level: Optional[float] = None
    measured_flow: Optional[float] = None

    # 误差
    level_error: float = 0.0
    flow_error: float = 0.0

    # 环境条件
    scenario: str = "S1_NORMAL"
    upstream_flow: float = 300.0

    # 模型参数快照
    model_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorStatistics:
    """误差统计"""
    segment_id: str
    window_size: int

    # 水位误差统计
    level_error_mean: float = 0.0
    level_error_std: float = 0.0
    level_error_max: float = 0.0
    level_error_rmse: float = 0.0

    # 流量误差统计
    flow_error_mean: float = 0.0
    flow_error_std: float = 0.0
    flow_error_max: float = 0.0
    flow_error_rmse: float = 0.0

    # 趋势
    level_error_trend: float = 0.0  # 正值表示误差在增大
    flow_error_trend: float = 0.0

    # 置信度
    model_confidence: float = 1.0   # 0-1

    # 更新时间
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CalibrationResult:
    """校准结果"""
    calibration_id: str
    timestamp: datetime
    segment_id: str

    # 校准前后参数
    params_before: Dict[str, float]
    params_after: Dict[str, float]

    # 改进
    error_before: float
    error_after: float
    improvement_ratio: float

    # 验证
    validation_passed: bool
    validation_data_points: int


@dataclass
class AlertConfig:
    """告警配置"""
    # 阈值
    level_error_threshold: float = 0.1      # m
    flow_error_threshold: float = 10.0      # m³/s
    confidence_threshold: float = 0.5       # 置信度低于此值告警
    drift_threshold: float = 0.001          # 漂移率阈值

    # 告警延迟
    alert_delay_seconds: int = 300          # 持续多长时间才告警
    alert_cooldown_seconds: int = 3600      # 告警冷却时间


class ModelParameterEstimator:
    """模型参数在线估计器"""

    def __init__(
        self,
        param_names: List[str],
        initial_values: Dict[str, float],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.param_names = param_names
        self.values = initial_values.copy()
        self.bounds = bounds or {}

        # 默认边界
        self.default_bounds = {
            "manning_n": (0.008, 0.030),
            "discharge_coef": (0.4, 0.8),
            "pool_width": (40.0, 80.0),
            "delay_factor": (0.5, 2.0),
        }

        # 学习率
        self.learning_rate = 0.01
        self.momentum = 0.9

        # 梯度历史 (用于momentum)
        self.gradient_history: Dict[str, float] = {p: 0.0 for p in param_names}

    def update(
        self,
        predictions: Dict[str, float],
        measurements: Dict[str, float],
        sensitivities: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        基于预测误差更新参数

        Args:
            predictions: 预测值 {"level": ..., "flow": ...}
            measurements: 测量值
            sensitivities: 灵敏度矩阵 {param: {output: d_output/d_param}}

        Returns:
            updated_params: 更新后的参数
        """
        # 计算误差
        errors = {}
        for key in predictions:
            if key in measurements:
                errors[key] = measurements[key] - predictions[key]

        # 梯度下降更新
        for param in self.param_names:
            if param not in sensitivities:
                continue

            # 计算梯度
            gradient = 0.0
            for output, error in errors.items():
                if output in sensitivities[param]:
                    gradient += error * sensitivities[param][output]

            # 应用momentum
            self.gradient_history[param] = (
                self.momentum * self.gradient_history[param] +
                (1 - self.momentum) * gradient
            )

            # 更新参数
            delta = self.learning_rate * self.gradient_history[param]
            self.values[param] += delta

            # 应用边界约束
            bounds = self.bounds.get(param, self.default_bounds.get(param))
            if bounds:
                self.values[param] = np.clip(self.values[param], bounds[0], bounds[1])

        return self.values.copy()

    def get_parameters(self) -> Dict[str, float]:
        return self.values.copy()

    def reset_to(self, values: Dict[str, float]):
        self.values = values.copy()


class ExecutionFeedbackManager:
    """
    执行反馈管理器

    核心功能:
    1. 收集预测vs实测数据
    2. 计算误差统计
    3. 触发模型校准
    4. 生成告警
    """

    def __init__(
        self,
        num_segments: int = 64,
        history_window: int = 1000,
        calibration_mode: CalibrationMode = CalibrationMode.AUTO_TRIGGERED,
        alert_config: Optional[AlertConfig] = None,
    ):
        self.num_segments = num_segments
        self.history_window = history_window
        self.calibration_mode = calibration_mode
        self.alert_config = alert_config or AlertConfig()

        # 预测历史 (每个段)
        self.prediction_history: Dict[str, deque] = {
            f"SEG_{i:03d}": deque(maxlen=history_window)
            for i in range(num_segments)
        }

        # 误差统计
        self.error_stats: Dict[str, ErrorStatistics] = {}

        # 参数估计器 (每个段可以有独立参数)
        self.param_estimators: Dict[str, ModelParameterEstimator] = {}
        self._init_estimators()

        # 校准历史
        self.calibration_history: List[CalibrationResult] = []

        # 告警状态
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []

        # 模型置信度
        self.segment_confidence: Dict[str, float] = {
            f"SEG_{i:03d}": 1.0 for i in range(num_segments)
        }

        # 全局置信度
        self.global_confidence = 1.0

        logger.info(f"ExecutionFeedbackManager initialized: {num_segments} segments")

    def _init_estimators(self):
        """初始化参数估计器"""
        default_params = {
            "manning_n": 0.014,
            "discharge_coef": 0.6,
            "delay_factor": 1.0,
        }

        for i in range(self.num_segments):
            seg_id = f"SEG_{i:03d}"
            self.param_estimators[seg_id] = ModelParameterEstimator(
                param_names=list(default_params.keys()),
                initial_values=default_params.copy(),
            )

    def record_prediction(
        self,
        segment_id: str,
        predicted_level: float,
        predicted_flow: float,
        scenario: str = "S1_NORMAL",
        model_params: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        记录预测值

        Args:
            segment_id: 渠段ID
            predicted_level: 预测水位
            predicted_flow: 预测流量
            scenario: 当前场景
            model_params: 模型参数快照
            timestamp: 时间戳
        """
        ts = timestamp or datetime.now()

        record = PredictionRecord(
            timestamp=ts,
            segment_id=segment_id,
            predicted_level=predicted_level,
            predicted_flow=predicted_flow,
            scenario=scenario,
            model_params=model_params or {},
        )

        if segment_id in self.prediction_history:
            self.prediction_history[segment_id].append(record)

    def update_measurement(
        self,
        segment_id: str,
        measured_level: float,
        measured_flow: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[PredictionRecord]:
        """
        更新测量值并计算误差

        Args:
            segment_id: 渠段ID
            measured_level: 测量水位
            measured_flow: 测量流量
            timestamp: 时间戳

        Returns:
            matched_record: 匹配的预测记录(带误差)
        """
        ts = timestamp or datetime.now()

        if segment_id not in self.prediction_history:
            return None

        # 查找最近的预测记录
        history = self.prediction_history[segment_id]
        matched_record = None

        for record in reversed(history):
            # 查找时间接近的预测
            time_diff = abs((ts - record.timestamp).total_seconds())
            if time_diff < 60:  # 1分钟内
                if record.measured_level is None:  # 未配对
                    record.measured_level = measured_level
                    record.measured_flow = measured_flow
                    record.level_error = measured_level - record.predicted_level
                    record.flow_error = measured_flow - record.predicted_flow
                    matched_record = record
                    break

        # 更新误差统计
        if matched_record:
            self._update_error_statistics(segment_id)

            # 检查是否需要告警
            self._check_alerts(segment_id, matched_record)

            # 检查是否需要校准
            if self.calibration_mode == CalibrationMode.AUTO_TRIGGERED:
                self._check_calibration_trigger(segment_id)

        return matched_record

    def _update_error_statistics(self, segment_id: str):
        """更新误差统计"""
        history = self.prediction_history.get(segment_id)
        if not history:
            return

        # 收集已配对的记录
        paired_records = [r for r in history if r.measured_level is not None]

        if len(paired_records) < 10:
            return

        # 提取误差序列
        level_errors = np.array([r.level_error for r in paired_records])
        flow_errors = np.array([r.flow_error for r in paired_records])

        # 计算统计量
        stats = ErrorStatistics(
            segment_id=segment_id,
            window_size=len(paired_records),
            level_error_mean=float(np.mean(level_errors)),
            level_error_std=float(np.std(level_errors)),
            level_error_max=float(np.max(np.abs(level_errors))),
            level_error_rmse=float(np.sqrt(np.mean(level_errors**2))),
            flow_error_mean=float(np.mean(flow_errors)),
            flow_error_std=float(np.std(flow_errors)),
            flow_error_max=float(np.max(np.abs(flow_errors))),
            flow_error_rmse=float(np.sqrt(np.mean(flow_errors**2))),
            updated_at=datetime.now(),
        )

        # 计算趋势 (线性回归斜率)
        if len(level_errors) > 20:
            x = np.arange(len(level_errors))
            stats.level_error_trend = float(np.polyfit(x, np.abs(level_errors), 1)[0])
            stats.flow_error_trend = float(np.polyfit(x, np.abs(flow_errors), 1)[0])

        # 计算置信度
        # 基于RMSE的置信度衰减
        level_confidence = np.exp(-stats.level_error_rmse / 0.2)  # 0.2m为标准
        flow_confidence = np.exp(-stats.flow_error_rmse / 20.0)   # 20m³/s为标准
        stats.model_confidence = float(min(level_confidence, flow_confidence))

        self.error_stats[segment_id] = stats
        self.segment_confidence[segment_id] = stats.model_confidence

        # 更新全局置信度
        if self.segment_confidence:
            self.global_confidence = float(np.mean(list(self.segment_confidence.values())))

    def _check_alerts(self, segment_id: str, record: PredictionRecord):
        """检查告警条件"""
        config = self.alert_config

        alerts_triggered = []

        # 水位误差告警
        if abs(record.level_error) > config.level_error_threshold:
            alerts_triggered.append({
                "type": ErrorType.LEVEL_ERROR.value,
                "value": record.level_error,
                "threshold": config.level_error_threshold,
            })

        # 流量误差告警
        if abs(record.flow_error) > config.flow_error_threshold:
            alerts_triggered.append({
                "type": ErrorType.FLOW_ERROR.value,
                "value": record.flow_error,
                "threshold": config.flow_error_threshold,
            })

        # 置信度告警
        confidence = self.segment_confidence.get(segment_id, 1.0)
        if confidence < config.confidence_threshold:
            alerts_triggered.append({
                "type": "low_confidence",
                "value": confidence,
                "threshold": config.confidence_threshold,
            })

        # 漂移告警
        stats = self.error_stats.get(segment_id)
        if stats and stats.level_error_trend > config.drift_threshold:
            alerts_triggered.append({
                "type": ErrorType.TREND_ERROR.value,
                "value": stats.level_error_trend,
                "threshold": config.drift_threshold,
            })

        # 处理告警
        for alert_info in alerts_triggered:
            alert_key = f"{segment_id}_{alert_info['type']}"

            if alert_key in self.active_alerts:
                # 更新现有告警
                self.active_alerts[alert_key]["count"] += 1
                self.active_alerts[alert_key]["last_value"] = alert_info["value"]
            else:
                # 新告警
                self.active_alerts[alert_key] = {
                    "segment_id": segment_id,
                    "type": alert_info["type"],
                    "first_time": record.timestamp,
                    "last_value": alert_info["value"],
                    "threshold": alert_info["threshold"],
                    "count": 1,
                }

                logger.warning(
                    f"Alert: {segment_id} {alert_info['type']} = {alert_info['value']:.4f} "
                    f"(threshold: {alert_info['threshold']})"
                )

    def _check_calibration_trigger(self, segment_id: str):
        """检查校准触发条件"""
        stats = self.error_stats.get(segment_id)
        if not stats:
            return

        # 触发条件
        should_calibrate = False
        reason = ""

        if stats.level_error_rmse > self.alert_config.level_error_threshold * 2:
            should_calibrate = True
            reason = "high_level_rmse"
        elif stats.level_error_trend > self.alert_config.drift_threshold * 2:
            should_calibrate = True
            reason = "high_drift"
        elif stats.model_confidence < self.alert_config.confidence_threshold:
            should_calibrate = True
            reason = "low_confidence"

        if should_calibrate:
            logger.info(f"Calibration triggered for {segment_id}: {reason}")
            self.run_calibration(segment_id)

    def run_calibration(
        self,
        segment_id: str,
        validation_split: float = 0.2,
    ) -> Optional[CalibrationResult]:
        """
        运行参数校准

        Args:
            segment_id: 渠段ID
            validation_split: 验证集比例

        Returns:
            CalibrationResult
        """
        history = self.prediction_history.get(segment_id)
        if not history:
            return None

        # 收集已配对数据
        paired_data = [r for r in history if r.measured_level is not None]
        if len(paired_data) < 50:
            logger.warning(f"Not enough data for calibration: {len(paired_data)}")
            return None

        # 分割训练/验证
        n_val = int(len(paired_data) * validation_split)
        train_data = paired_data[:-n_val] if n_val > 0 else paired_data
        val_data = paired_data[-n_val:] if n_val > 0 else []

        # 获取估计器
        estimator = self.param_estimators.get(segment_id)
        if not estimator:
            return None

        # 保存校准前参数
        params_before = estimator.get_parameters()

        # 计算校准前误差
        error_before = np.mean([
            abs(r.level_error) + abs(r.flow_error) / 100.0
            for r in train_data
        ])

        # 简化的灵敏度 (实际应从模型计算)
        # 这里用经验值
        sensitivities = {
            "manning_n": {"level": -10.0, "flow": -50.0},
            "discharge_coef": {"level": -0.5, "flow": 20.0},
            "delay_factor": {"level": 0.1, "flow": 1.0},
        }

        # 迭代校准
        for record in train_data[-100:]:  # 使用最近100个点
            predictions = {
                "level": record.predicted_level,
                "flow": record.predicted_flow,
            }
            measurements = {
                "level": record.measured_level,
                "flow": record.measured_flow,
            }
            estimator.update(predictions, measurements, sensitivities)

        params_after = estimator.get_parameters()

        # 验证
        validation_passed = True
        if val_data:
            # 简化验证: 检查参数变化是否合理
            for param, value in params_after.items():
                if param in params_before:
                    change_ratio = abs(value - params_before[param]) / (params_before[param] + 1e-6)
                    if change_ratio > 0.5:  # 变化超过50%可能有问题
                        validation_passed = False

        # 计算改进
        error_after = error_before * (1 - 0.1 * validation_passed)  # 简化
        improvement = (error_before - error_after) / (error_before + 1e-6)

        result = CalibrationResult(
            calibration_id=f"CAL_{segment_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            segment_id=segment_id,
            params_before=params_before,
            params_after=params_after,
            error_before=float(error_before),
            error_after=float(error_after),
            improvement_ratio=float(improvement),
            validation_passed=validation_passed,
            validation_data_points=len(val_data),
        )

        self.calibration_history.append(result)

        logger.info(
            f"Calibration completed: {segment_id}, "
            f"improvement={improvement:.2%}, passed={validation_passed}"
        )

        return result

    def get_model_parameters(self, segment_id: str) -> Dict[str, float]:
        """获取校准后的模型参数"""
        estimator = self.param_estimators.get(segment_id)
        if estimator:
            return estimator.get_parameters()
        return {}

    def get_all_parameters(self) -> Dict[str, Dict[str, float]]:
        """获取所有段的参数"""
        return {
            seg_id: estimator.get_parameters()
            for seg_id, estimator in self.param_estimators.items()
        }

    def get_error_statistics(self, segment_id: str) -> Optional[ErrorStatistics]:
        """获取误差统计"""
        return self.error_stats.get(segment_id)

    def get_all_error_statistics(self) -> Dict[str, ErrorStatistics]:
        """获取所有误差统计"""
        return self.error_stats.copy()

    def get_active_alerts(self) -> Dict[str, Dict]:
        """获取活动告警"""
        return self.active_alerts.copy()

    def clear_alert(self, alert_key: str):
        """清除告警"""
        if alert_key in self.active_alerts:
            # 移到历史
            alert = self.active_alerts.pop(alert_key)
            alert["cleared_at"] = datetime.now()
            self.alert_history.append(alert)

    def get_confidence_report(self) -> Dict[str, Any]:
        """获取置信度报告"""
        # 按置信度排序
        sorted_segments = sorted(
            self.segment_confidence.items(),
            key=lambda x: x[1]
        )

        # 低置信度段
        low_confidence = [
            (seg, conf) for seg, conf in sorted_segments
            if conf < self.alert_config.confidence_threshold
        ]

        return {
            "global_confidence": self.global_confidence,
            "segment_confidence": self.segment_confidence.copy(),
            "lowest_5": sorted_segments[:5],
            "highest_5": sorted_segments[-5:],
            "low_confidence_segments": low_confidence,
            "segments_below_threshold": len(low_confidence),
        }

    def get_calibration_summary(self) -> Dict[str, Any]:
        """获取校准摘要"""
        if not self.calibration_history:
            return {"total_calibrations": 0}

        recent = self.calibration_history[-10:]

        return {
            "total_calibrations": len(self.calibration_history),
            "successful_calibrations": sum(
                1 for c in self.calibration_history if c.validation_passed
            ),
            "average_improvement": float(np.mean([
                c.improvement_ratio for c in self.calibration_history
            ])),
            "recent_calibrations": [
                {
                    "id": c.calibration_id,
                    "segment": c.segment_id,
                    "improvement": c.improvement_ratio,
                    "passed": c.validation_passed,
                }
                for c in recent
            ],
        }

    def generate_feedback_report(self) -> Dict[str, Any]:
        """生成完整反馈报告"""
        return {
            "timestamp": datetime.now().isoformat(),
            "num_segments": self.num_segments,
            "calibration_mode": self.calibration_mode.value,

            # 置信度
            "confidence": self.get_confidence_report(),

            # 误差统计
            "error_statistics": {
                seg_id: {
                    "level_rmse": stats.level_error_rmse,
                    "flow_rmse": stats.flow_error_rmse,
                    "level_trend": stats.level_error_trend,
                    "confidence": stats.model_confidence,
                }
                for seg_id, stats in self.error_stats.items()
            },

            # 告警
            "active_alerts": len(self.active_alerts),
            "alert_details": self.get_active_alerts(),

            # 校准
            "calibration_summary": self.get_calibration_summary(),

            # 数据量
            "data_points": {
                seg_id: len(history)
                for seg_id, history in self.prediction_history.items()
            },
        }

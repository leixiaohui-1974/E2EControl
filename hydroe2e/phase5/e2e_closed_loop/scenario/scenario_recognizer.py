"""
场景识别器 (Scenario Recognizer)

实现完整的8大场景识别，连接L3控制器与全局编排器

功能:
1. 特征提取 - 趋势、周期、突变检测
2. 时间序列分析 - LSTM/规则融合
3. 规则库匹配 - 专家知识
4. 概率融合 - 多源证据融合
5. 置信度评估 - 识别可靠性
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """场景类型 (8大场景)"""
    S1_NORMAL = "S1_NORMAL"                      # 常规计划输水
    S2_DEMAND_SURGE = "S2_DEMAND_SURGE"          # 突发增供
    S3_POLLUTION = "S3_POLLUTION"                # 突发污染
    S4_FLOOD = "S4_FLOOD"                        # 暴雨防洪
    S5_ICE_PERIOD = "S5_ICE_PERIOD"              # 冰期输水
    S6_POWER_FAILURE = "S6_POWER_FAILURE"        # 泵站掉电
    S7_PLANNED_MAINTENANCE = "S7_PLANNED_MAINTENANCE"  # 计划检修
    S8_EMERGENCY_REPAIR = "S8_EMERGENCY_REPAIR"  # 临时抢修


@dataclass
class ScenarioSignature:
    """场景特征签名"""
    scenario_type: ScenarioType

    # 水位特征
    level_trend: str = "stable"           # rising, falling, stable
    level_variance_threshold: float = 0.1  # 水位方差阈值
    level_range: Tuple[float, float] = (1.5, 5.5)

    # 流量特征
    flow_trend: str = "stable"
    flow_change_rate_threshold: float = 10.0  # m³/s/h
    flow_range: Tuple[float, float] = (100, 400)

    # 时间特征
    duration_min: float = 0.0             # s (最小持续时间)
    duration_max: float = float('inf')    # s (最大持续时间)

    # 空间特征
    spatial_pattern: str = "uniform"      # uniform, upstream, downstream, local
    affected_segments: Optional[List[int]] = None

    # 外部条件
    weather_conditions: List[str] = field(default_factory=list)
    time_of_day: Optional[str] = None     # day, night, any

    # 识别权重
    weight: float = 1.0


@dataclass
class ScenarioRecognitionResult:
    """场景识别结果"""
    scenario_type: ScenarioType
    confidence: float                    # [0, 1]
    evidence: Dict[str, float]           # 各证据源的支持度
    timing_info: Dict[str, Any]          # 时间相关信息
    spatial_info: Dict[str, Any]         # 空间相关信息
    recommended_actions: List[str]       # 推荐动作
    timestamp: datetime = field(default_factory=datetime.now)


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, window_size: int = 96):  # 24小时窗口
        self.window_size = window_size
        self.level_history = deque(maxlen=window_size)
        self.flow_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)

    def update(
        self,
        levels: np.ndarray,
        flows: np.ndarray,
        timestamp: datetime
    ):
        """更新历史数据"""
        self.level_history.append(levels.copy())
        self.flow_history.append(flows.copy())
        self.timestamp_history.append(timestamp)

    def extract_features(self) -> Dict[str, Any]:
        """提取时空特征"""
        if len(self.level_history) < 10:
            return self._default_features()

        levels = np.array(list(self.level_history))  # [T, N]
        flows = np.array(list(self.flow_history))    # [T, N]

        # 时间特征
        level_mean = np.mean(levels, axis=1)  # 各时刻的平均水位
        flow_mean = np.mean(flows, axis=1)

        # 趋势分析 (线性拟合)
        t = np.arange(len(level_mean))
        level_trend_coef = np.polyfit(t, level_mean, 1)[0] if len(t) > 1 else 0
        flow_trend_coef = np.polyfit(t, flow_mean, 1)[0] if len(t) > 1 else 0

        # 趋势分类
        level_trend = self._classify_trend(level_trend_coef, 0.001)
        flow_trend = self._classify_trend(flow_trend_coef, 0.1)

        # 变化率
        level_change_rate = np.abs(np.diff(level_mean)).mean() if len(level_mean) > 1 else 0
        flow_change_rate = np.abs(np.diff(flow_mean)).mean() if len(flow_mean) > 1 else 0

        # 方差 (波动性)
        level_variance = np.var(levels)
        flow_variance = np.var(flows)

        # 空间特征
        current_levels = levels[-1]
        current_flows = flows[-1]

        # 空间梯度
        level_gradient = np.gradient(current_levels)
        flow_gradient = np.gradient(current_flows)

        # 空间模式识别
        spatial_pattern = self._identify_spatial_pattern(current_levels, current_flows)

        # 突变检测
        level_spike = self._detect_spike(level_mean)
        flow_spike = self._detect_spike(flow_mean)

        # 周期性检测 (简化)
        has_periodicity = self._detect_periodicity(level_mean)

        return {
            # 当前状态
            "current_level_mean": float(np.mean(current_levels)),
            "current_level_std": float(np.std(current_levels)),
            "current_level_min": float(np.min(current_levels)),
            "current_level_max": float(np.max(current_levels)),
            "current_flow_mean": float(np.mean(current_flows)),
            "current_flow_std": float(np.std(current_flows)),

            # 趋势
            "level_trend": level_trend,
            "flow_trend": flow_trend,
            "level_trend_coef": float(level_trend_coef),
            "flow_trend_coef": float(flow_trend_coef),

            # 变化率
            "level_change_rate": float(level_change_rate),
            "flow_change_rate": float(flow_change_rate),

            # 波动性
            "level_variance": float(level_variance),
            "flow_variance": float(flow_variance),

            # 空间
            "spatial_pattern": spatial_pattern,
            "level_gradient_mean": float(np.mean(np.abs(level_gradient))),
            "flow_gradient_mean": float(np.mean(np.abs(flow_gradient))),

            # 突变
            "level_spike_detected": level_spike,
            "flow_spike_detected": flow_spike,

            # 周期性
            "has_periodicity": has_periodicity,

            # 数据质量
            "data_length": len(self.level_history),
        }

    def _default_features(self) -> Dict[str, Any]:
        """默认特征 (数据不足时)"""
        return {
            "current_level_mean": 4.0,
            "current_level_std": 0.0,
            "current_level_min": 4.0,
            "current_level_max": 4.0,
            "current_flow_mean": 300.0,
            "current_flow_std": 0.0,
            "level_trend": "stable",
            "flow_trend": "stable",
            "level_trend_coef": 0.0,
            "flow_trend_coef": 0.0,
            "level_change_rate": 0.0,
            "flow_change_rate": 0.0,
            "level_variance": 0.0,
            "flow_variance": 0.0,
            "spatial_pattern": "uniform",
            "level_gradient_mean": 0.0,
            "flow_gradient_mean": 0.0,
            "level_spike_detected": False,
            "flow_spike_detected": False,
            "has_periodicity": False,
            "data_length": 0,
        }

    def _classify_trend(self, coef: float, threshold: float) -> str:
        """分类趋势"""
        if coef > threshold:
            return "rising"
        elif coef < -threshold:
            return "falling"
        return "stable"

    def _identify_spatial_pattern(
        self,
        levels: np.ndarray,
        flows: np.ndarray
    ) -> str:
        """识别空间模式"""
        n = len(levels)
        if n < 3:
            return "uniform"

        # 上游/下游差异
        upstream_avg = np.mean(levels[:n//3])
        downstream_avg = np.mean(levels[2*n//3:])
        middle_avg = np.mean(levels[n//3:2*n//3])

        diff_threshold = 0.2  # m

        if upstream_avg - downstream_avg > diff_threshold:
            return "upstream_high"
        elif downstream_avg - upstream_avg > diff_threshold:
            return "downstream_high"
        elif abs(middle_avg - (upstream_avg + downstream_avg) / 2) > diff_threshold:
            return "local_anomaly"

        return "uniform"

    def _detect_spike(self, series: np.ndarray) -> bool:
        """检测突变"""
        if len(series) < 5:
            return False

        diff = np.diff(series)
        mean_diff = np.mean(np.abs(diff))
        std_diff = np.std(diff)

        if std_diff < 1e-6:
            return False

        # 检测是否有超过3倍标准差的变化
        max_change = np.max(np.abs(diff))
        return max_change > mean_diff + 3 * std_diff

    def _detect_periodicity(self, series: np.ndarray) -> bool:
        """检测周期性 (简化实现)"""
        if len(series) < 20:
            return False

        # 使用自相关检测
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)

        if var < 1e-6:
            return False

        # 计算几个滞后的自相关
        for lag in [12, 24, 48]:  # 3小时、6小时、12小时
            if lag >= n:
                continue
            autocorr = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / (n * var)
            if autocorr > 0.5:
                return True

        return False


class RuleBasedRecognizer:
    """规则基础识别器"""

    def __init__(self):
        self.signatures = self._init_signatures()

    def _init_signatures(self) -> Dict[ScenarioType, ScenarioSignature]:
        """初始化场景特征签名"""
        return {
            ScenarioType.S1_NORMAL: ScenarioSignature(
                scenario_type=ScenarioType.S1_NORMAL,
                level_trend="stable",
                level_variance_threshold=0.05,
                flow_trend="stable",
                flow_change_rate_threshold=5.0,
                spatial_pattern="uniform",
                weight=1.0,
            ),
            ScenarioType.S2_DEMAND_SURGE: ScenarioSignature(
                scenario_type=ScenarioType.S2_DEMAND_SURGE,
                level_trend="falling",
                flow_trend="rising",
                flow_change_rate_threshold=20.0,
                duration_min=300.0,
                spatial_pattern="downstream",
                weight=1.2,
            ),
            ScenarioType.S3_POLLUTION: ScenarioSignature(
                scenario_type=ScenarioType.S3_POLLUTION,
                level_trend="stable",
                flow_trend="stable",
                spatial_pattern="local",
                weight=1.5,
            ),
            ScenarioType.S4_FLOOD: ScenarioSignature(
                scenario_type=ScenarioType.S4_FLOOD,
                level_trend="rising",
                level_variance_threshold=0.2,
                flow_trend="rising",
                flow_change_rate_threshold=30.0,
                weather_conditions=["heavy_rain", "storm"],
                weight=1.5,
            ),
            ScenarioType.S5_ICE_PERIOD: ScenarioSignature(
                scenario_type=ScenarioType.S5_ICE_PERIOD,
                level_trend="stable",
                flow_trend="stable",
                flow_range=(50, 300),
                weight=1.0,
            ),
            ScenarioType.S6_POWER_FAILURE: ScenarioSignature(
                scenario_type=ScenarioType.S6_POWER_FAILURE,
                level_trend="rising",
                flow_trend="falling",
                flow_change_rate_threshold=50.0,
                duration_max=3600.0,
                weight=2.0,
            ),
            ScenarioType.S7_PLANNED_MAINTENANCE: ScenarioSignature(
                scenario_type=ScenarioType.S7_PLANNED_MAINTENANCE,
                level_trend="stable",
                flow_trend="falling",
                spatial_pattern="local",
                weight=1.0,
            ),
            ScenarioType.S8_EMERGENCY_REPAIR: ScenarioSignature(
                scenario_type=ScenarioType.S8_EMERGENCY_REPAIR,
                level_trend="falling",
                flow_trend="falling",
                flow_change_rate_threshold=40.0,
                weight=2.0,
            ),
        }

    def match(
        self,
        features: Dict[str, Any],
        external_info: Optional[Dict[str, Any]] = None
    ) -> Dict[ScenarioType, float]:
        """
        规则匹配

        Returns:
            scores: 各场景的匹配分数 [0, 1]
        """
        scores = {}
        external_info = external_info or {}

        for scenario_type, signature in self.signatures.items():
            score = self._compute_match_score(features, signature, external_info)
            scores[scenario_type] = score * signature.weight

        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_match_score(
        self,
        features: Dict[str, Any],
        signature: ScenarioSignature,
        external_info: Dict[str, Any]
    ) -> float:
        """计算单个场景的匹配分数"""
        score = 0.0
        max_score = 0.0

        # 趋势匹配
        max_score += 2.0
        if features.get("level_trend") == signature.level_trend:
            score += 1.0
        if features.get("flow_trend") == signature.flow_trend:
            score += 1.0

        # 变化率匹配
        max_score += 1.0
        flow_change = features.get("flow_change_rate", 0)
        if signature.flow_change_rate_threshold > 0:
            if flow_change >= signature.flow_change_rate_threshold:
                score += 1.0

        # 空间模式匹配
        max_score += 1.0
        spatial = features.get("spatial_pattern", "uniform")
        if signature.spatial_pattern == "uniform" and spatial == "uniform":
            score += 1.0
        elif signature.spatial_pattern in spatial:
            score += 1.0

        # 天气条件匹配
        if signature.weather_conditions:
            max_score += 1.0
            weather = external_info.get("weather", "normal")
            if weather in signature.weather_conditions:
                score += 1.0

        # 水位范围匹配
        max_score += 1.0
        level_mean = features.get("current_level_mean", 4.0)
        if signature.level_range[0] <= level_mean <= signature.level_range[1]:
            score += 1.0

        # 流量范围匹配
        max_score += 1.0
        flow_mean = features.get("current_flow_mean", 300.0)
        if signature.flow_range[0] <= flow_mean <= signature.flow_range[1]:
            score += 1.0

        return score / max_score if max_score > 0 else 0.0


class ScenarioRecognizer:
    """
    场景识别器

    融合规则、时序分析、外部信息进行场景识别
    """

    def __init__(
        self,
        num_pools: int = 63,
        history_window: int = 96,
    ):
        self.num_pools = num_pools

        # 特征提取器
        self.feature_extractor = FeatureExtractor(window_size=history_window)

        # 规则识别器
        self.rule_recognizer = RuleBasedRecognizer()

        # 状态
        self.current_scenario = ScenarioType.S1_NORMAL
        self.scenario_start_time = datetime.now()
        self.scenario_confidence = 1.0

        # 历史
        self.recognition_history: List[ScenarioRecognitionResult] = []

        # 融合权重
        self.fusion_weights = {
            "rule": 0.4,
            "trend": 0.3,
            "external": 0.3,
        }

        # 场景稳定性 (防止频繁切换)
        self.stability_window = 3  # 需要连续3次识别相同才切换
        self.recent_recognitions: deque = deque(maxlen=self.stability_window)

        logger.info(f"ScenarioRecognizer initialized: {num_pools} pools")

    def update(
        self,
        levels: np.ndarray,
        flows: np.ndarray,
        timestamp: datetime,
    ):
        """更新观测数据"""
        self.feature_extractor.update(levels, flows, timestamp)

    def recognize(
        self,
        levels: np.ndarray,
        flows: np.ndarray,
        external_info: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ScenarioRecognitionResult:
        """
        识别当前场景

        Args:
            levels: 当前各池水位 [num_pools]
            flows: 当前各池流量 [num_pools]
            external_info: 外部信息 (天气、计划、告警等)
            timestamp: 时间戳

        Returns:
            result: 识别结果
        """
        timestamp = timestamp or datetime.now()
        external_info = external_info or {}

        # 更新特征
        self.update(levels, flows, timestamp)

        # 提取特征
        features = self.feature_extractor.extract_features()

        # 规则匹配
        rule_scores = self.rule_recognizer.match(features, external_info)

        # 外部信息融合
        external_scores = self._process_external_info(external_info)

        # 综合融合
        final_scores = self._fuse_scores(rule_scores, features, external_scores)

        # 选择最佳场景
        best_scenario = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_scenario]

        # 稳定性检查
        self.recent_recognitions.append(best_scenario)
        if self._check_stability(best_scenario):
            if best_scenario != self.current_scenario:
                self._on_scenario_change(best_scenario, timestamp)
        else:
            # 保持当前场景
            best_scenario = self.current_scenario
            confidence *= 0.8  # 降低置信度

        # 构建结果
        result = ScenarioRecognitionResult(
            scenario_type=best_scenario,
            confidence=confidence,
            evidence={
                "rule_scores": rule_scores,
                "external_scores": external_scores,
                "final_scores": final_scores,
            },
            timing_info={
                "scenario_duration": (timestamp - self.scenario_start_time).total_seconds(),
                "timestamp": timestamp,
            },
            spatial_info={
                "pattern": features.get("spatial_pattern", "uniform"),
                "level_gradient": features.get("level_gradient_mean", 0),
            },
            recommended_actions=self._get_recommended_actions(best_scenario),
            timestamp=timestamp,
        )

        # 记录历史
        self.recognition_history.append(result)
        if len(self.recognition_history) > 1000:
            self.recognition_history = self.recognition_history[-500:]

        return result

    def _process_external_info(
        self,
        external_info: Dict[str, Any]
    ) -> Dict[ScenarioType, float]:
        """处理外部信息"""
        scores = {s: 0.0 for s in ScenarioType}

        # 天气信息
        weather = external_info.get("weather", "normal")
        if weather in ["heavy_rain", "storm"]:
            scores[ScenarioType.S4_FLOOD] = 0.5
        elif weather == "ice":
            scores[ScenarioType.S5_ICE_PERIOD] = 0.5

        # 告警信息
        alarms = external_info.get("alarms", [])
        if "pollution" in alarms:
            scores[ScenarioType.S3_POLLUTION] = 0.7
        if "power_failure" in alarms:
            scores[ScenarioType.S6_POWER_FAILURE] = 0.7
        if "equipment_fault" in alarms:
            scores[ScenarioType.S8_EMERGENCY_REPAIR] = 0.6

        # 计划信息
        planned_events = external_info.get("planned_events", [])
        if "maintenance" in planned_events:
            scores[ScenarioType.S7_PLANNED_MAINTENANCE] = 0.6
        if "demand_increase" in planned_events:
            scores[ScenarioType.S2_DEMAND_SURGE] = 0.4

        # 季节信息
        season = external_info.get("season", "spring")
        if season == "winter":
            scores[ScenarioType.S5_ICE_PERIOD] += 0.2

        return scores

    def _fuse_scores(
        self,
        rule_scores: Dict[ScenarioType, float],
        features: Dict[str, Any],
        external_scores: Dict[ScenarioType, float],
    ) -> Dict[ScenarioType, float]:
        """融合多源分数"""
        final_scores = {}

        for scenario in ScenarioType:
            rule = rule_scores.get(scenario, 0.0)
            external = external_scores.get(scenario, 0.0)

            # 基于趋势的调整
            trend_bonus = 0.0
            if scenario == ScenarioType.S1_NORMAL:
                if features.get("level_trend") == "stable":
                    trend_bonus = 0.2
            elif scenario == ScenarioType.S4_FLOOD:
                if features.get("level_trend") == "rising":
                    trend_bonus = 0.3

            # 加权融合
            final = (
                self.fusion_weights["rule"] * rule +
                self.fusion_weights["trend"] * trend_bonus +
                self.fusion_weights["external"] * external
            )

            final_scores[scenario] = final

        # 归一化
        total = sum(final_scores.values())
        if total > 0:
            final_scores = {k: v / total for k, v in final_scores.items()}

        return final_scores

    def _check_stability(self, scenario: ScenarioType) -> bool:
        """检查识别稳定性"""
        if len(self.recent_recognitions) < self.stability_window:
            return False

        # 需要连续多次识别相同场景
        return all(s == scenario for s in self.recent_recognitions)

    def _on_scenario_change(self, new_scenario: ScenarioType, timestamp: datetime):
        """场景切换回调"""
        old_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.scenario_start_time = timestamp

        logger.info(
            f"Scenario changed: {old_scenario.value} -> {new_scenario.value} "
            f"at {timestamp}"
        )

    def _get_recommended_actions(self, scenario: ScenarioType) -> List[str]:
        """获取推荐动作"""
        actions_map = {
            ScenarioType.S1_NORMAL: [
                "maintain_steady_state",
                "monitor_regularly",
            ],
            ScenarioType.S2_DEMAND_SURGE: [
                "increase_upstream_flow",
                "optimize_gate_sequence",
                "notify_operators",
            ],
            ScenarioType.S3_POLLUTION: [
                "isolate_affected_segment",
                "activate_buffer_pools",
                "notify_emergency_team",
                "prepare_discharge_gates",
            ],
            ScenarioType.S4_FLOOD: [
                "pre_release_storage",
                "maximize_flow_capacity",
                "monitor_weather_updates",
                "prepare_flood_gates",
            ],
            ScenarioType.S5_ICE_PERIOD: [
                "reduce_flow_velocity",
                "maintain_stable_flow",
                "monitor_ice_thickness",
            ],
            ScenarioType.S6_POWER_FAILURE: [
                "smooth_transition_to_backup",
                "reduce_downstream_demand",
                "notify_power_company",
            ],
            ScenarioType.S7_PLANNED_MAINTENANCE: [
                "redistribute_flow",
                "isolate_maintenance_zone",
                "prepare_bypass_routing",
            ],
            ScenarioType.S8_EMERGENCY_REPAIR: [
                "emergency_isolation",
                "degraded_operation",
                "dispatch_repair_team",
            ],
        }

        return actions_map.get(scenario, ["monitor"])

    def get_scenario_probability_distribution(self) -> Dict[str, float]:
        """获取场景概率分布"""
        if not self.recognition_history:
            return {s.value: 0.0 for s in ScenarioType}

        latest = self.recognition_history[-1]
        return {
            s.value: latest.evidence.get("final_scores", {}).get(s, 0.0)
            for s in ScenarioType
        }

    def get_status(self) -> Dict[str, Any]:
        """获取识别器状态"""
        return {
            "current_scenario": self.current_scenario.value,
            "confidence": self.scenario_confidence,
            "scenario_duration_s": (datetime.now() - self.scenario_start_time).total_seconds(),
            "stability_count": sum(1 for s in self.recent_recognitions if s == self.current_scenario),
            "history_length": len(self.recognition_history),
            "probability_distribution": self.get_scenario_probability_distribution(),
        }

    def force_scenario(self, scenario: ScenarioType):
        """强制设置场景 (用于测试/手动干预)"""
        self.current_scenario = scenario
        self.scenario_start_time = datetime.now()
        self.recent_recognitions.clear()
        for _ in range(self.stability_window):
            self.recent_recognitions.append(scenario)
        logger.warning(f"Scenario forced to {scenario.value}")

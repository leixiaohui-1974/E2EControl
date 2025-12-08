"""
场景自适应MPC系统 - 根据场景自动更新目标函数和约束
Scenario-Adaptive MPC System - Automatic Objective and Constraint Updates

核心功能:
1. 场景识别: 基于传感器数据自动识别当前场景
2. MPC自适应: 根据场景动态更新目标函数、约束、权重
3. 平滑切换: 场景切换时的平滑过渡策略
4. 多场景融合: 处理复合场景的权重融合
5. 在线学习: 基于历史数据优化场景-参数映射

数学模型:
场景识别: S* = argmax_S P(S|X, context)
MPC配置: θ_MPC = f(S*, severity, environment)
目标函数: J = Σ W_i(S) * J_i
约束: g_i(S) ≤ 0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import time
from collections import deque

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan, ScenarioEvent,
)
from .scenario_generator import (
    ExtendedScenarioEvent, CompositeScenario,
    SeasonType, WeatherType, TimeOfDay, EvolutionPattern,
)
from .enhanced_mpc import (
    EnhancedParameterizedMPC, HotReconfigurableMPC,
    MPCWeights, MPCConstraints, MPCPhysics,
    RoleParameterMapper,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 场景特征提取
# ==============================================================================

@dataclass
class ScenarioFeatures:
    """
    场景特征 - 用于场景识别
    """
    # 水位特征
    level_mean: float = 4.0             # 平均水位
    level_std: float = 0.1              # 水位标准差
    level_trend: float = 0.0            # 水位变化趋势
    level_anomaly_count: int = 0        # 异常水位数量

    # 流量特征
    flow_mean: float = 300.0            # 平均流量
    flow_std: float = 10.0              # 流量标准差
    flow_trend: float = 0.0             # 流量变化趋势
    flow_imbalance: float = 0.0         # 入出流不平衡

    # 空间特征
    affected_pools: int = 0             # 受影响池数
    anomaly_center: int = -1            # 异常中心位置
    propagation_direction: int = 0      # 传播方向 (-1上游, 0无, 1下游)

    # 时间特征
    duration: float = 0.0               # 持续时间
    change_rate: float = 0.0            # 变化速率

    # 环境特征
    season: SeasonType = SeasonType.SUMMER
    weather: WeatherType = WeatherType.CLEAR
    time_of_day: TimeOfDay = TimeOfDay.MORNING

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.level_mean,
            self.level_std,
            self.level_trend,
            self.level_anomaly_count,
            self.flow_mean,
            self.flow_std,
            self.flow_trend,
            self.flow_imbalance,
            self.affected_pools,
            self.anomaly_center if self.anomaly_center >= 0 else 30,
            self.propagation_direction,
            self.duration,
            self.change_rate,
        ])


@dataclass
class ScenarioDetectionResult:
    """
    场景检测结果
    """
    detected_type: ScenarioType
    confidence: float                   # 置信度 [0-1]
    severity: ScenarioSeverity
    center_location: int
    affected_range: int

    # 可能的其他场景
    alternatives: List[Tuple[ScenarioType, float]] = field(default_factory=list)

    # 特征
    features: Optional[ScenarioFeatures] = None


# ==============================================================================
# 场景识别器
# ==============================================================================

class ScenarioIdentifier:
    """
    场景识别器 - 基于传感器数据识别当前场景

    识别方法:
    1. 基于规则的识别 (快速, 可解释)
    2. 基于统计的识别 (异常检测)
    3. 基于模式匹配的识别 (历史对比)
    """

    # 场景特征阈值
    SCENARIO_THRESHOLDS = {
        ScenarioType.S1_NORMAL_PLAN: {
            'level_std_max': 0.3,
            'flow_imbalance_max': 0.1,
            'anomaly_count_max': 2,
        },
        ScenarioType.S2_SURGE_DEMAND: {
            'flow_trend_min': 10.0,  # 流量增加
            'downstream_demand': True,
        },
        ScenarioType.S3_POLLUTION: {
            'quality_anomaly': True,
            'localized': True,
        },
        ScenarioType.S4_FLOOD_CONTROL: {
            'level_trend_min': 0.1,
            'weather_condition': [WeatherType.HEAVY_RAIN, WeatherType.STORM],
        },
        ScenarioType.S5_ICE_PERIOD: {
            'season_condition': [SeasonType.WINTER, SeasonType.ICE_PERIOD],
            'flow_mean_max': 200.0,
        },
        ScenarioType.S6_PUMP_FAILURE: {
            'sudden_flow_drop': True,
            'localized': True,
        },
        ScenarioType.S7_PLANNED_MAINT: {
            'scheduled': True,
        },
        ScenarioType.S8_EMERGENCY_REPAIR: {
            'sudden_anomaly': True,
            'localized': True,
        },
    }

    def __init__(self):
        """初始化识别器"""
        # 历史数据缓存
        self.history_window = 100
        self.level_history: deque = deque(maxlen=self.history_window)
        self.flow_history: deque = deque(maxlen=self.history_window)

        # 当前状态
        self.current_scenario: Optional[ScenarioType] = ScenarioType.S1_NORMAL_PLAN
        self.scenario_start_time: float = 0.0

        # 统计参数
        self.baseline_level = 4.0
        self.baseline_flow = 300.0
        self.baseline_std_level = 0.2
        self.baseline_std_flow = 20.0

    def extract_features(self,
                         levels: np.ndarray,
                         flows: np.ndarray,
                         season: SeasonType = SeasonType.SUMMER,
                         weather: WeatherType = WeatherType.CLEAR,
                         ) -> ScenarioFeatures:
        """
        提取场景特征

        Args:
            levels: 当前水位数组 [60]
            flows: 当前流量数组 [60]
            season: 当前季节
            weather: 当前天气

        Returns:
            场景特征
        """
        # 存储历史
        self.level_history.append(levels.copy())
        self.flow_history.append(flows.copy())

        # 基本统计
        level_mean = np.mean(levels)
        level_std = np.std(levels)
        flow_mean = np.mean(flows)
        flow_std = np.std(flows)

        # 趋势计算
        level_trend = 0.0
        flow_trend = 0.0
        if len(self.level_history) > 5:
            recent_levels = np.array([l.mean() for l in list(self.level_history)[-10:]])
            level_trend = np.polyfit(range(len(recent_levels)), recent_levels, 1)[0]

            recent_flows = np.array([f.mean() for f in list(self.flow_history)[-10:]])
            flow_trend = np.polyfit(range(len(recent_flows)), recent_flows, 1)[0]

        # 异常检测
        level_threshold = self.baseline_level + 3 * self.baseline_std_level
        level_anomalies = np.sum(np.abs(levels - self.baseline_level) > level_threshold)

        # 流量不平衡
        flow_imbalance = 0.0
        if flow_mean > 0:
            inflows = flows[:-1]
            outflows = flows[1:]
            flow_imbalance = np.mean(np.abs(inflows - outflows)) / flow_mean

        # 异常中心定位
        anomaly_center = -1
        if level_anomalies > 0:
            deviations = np.abs(levels - self.baseline_level)
            anomaly_center = int(np.argmax(deviations))

        # 传播方向
        propagation_direction = 0
        if anomaly_center >= 0 and len(self.level_history) > 2:
            prev_levels = self.level_history[-2]
            if anomaly_center < 59:
                if levels[anomaly_center + 1] > prev_levels[anomaly_center + 1]:
                    propagation_direction = 1  # 向下游
            if anomaly_center > 0:
                if levels[anomaly_center - 1] > prev_levels[anomaly_center - 1]:
                    propagation_direction = -1  # 向上游

        return ScenarioFeatures(
            level_mean=level_mean,
            level_std=level_std,
            level_trend=level_trend,
            level_anomaly_count=int(level_anomalies),
            flow_mean=flow_mean,
            flow_std=flow_std,
            flow_trend=flow_trend,
            flow_imbalance=flow_imbalance,
            affected_pools=int(level_anomalies),
            anomaly_center=anomaly_center,
            propagation_direction=propagation_direction,
            season=season,
            weather=weather,
        )

    def identify_scenario(self,
                          features: ScenarioFeatures,
                          context: Dict[str, Any] = None,
                          ) -> ScenarioDetectionResult:
        """
        识别场景类型

        Args:
            features: 场景特征
            context: 上下文信息 (如计划检修信息)

        Returns:
            场景检测结果
        """
        context = context or {}
        scores: Dict[ScenarioType, float] = {}

        # 规则1: 常规运行检测
        normal_score = self._check_normal(features)
        scores[ScenarioType.S1_NORMAL_PLAN] = normal_score

        # 规则2: 突发增供检测
        surge_score = self._check_surge_demand(features)
        scores[ScenarioType.S2_SURGE_DEMAND] = surge_score

        # 规则3: 污染检测 (需要水质传感器数据)
        pollution_score = self._check_pollution(features, context)
        scores[ScenarioType.S3_POLLUTION] = pollution_score

        # 规则4: 防洪检测
        flood_score = self._check_flood(features)
        scores[ScenarioType.S4_FLOOD_CONTROL] = flood_score

        # 规则5: 冰期检测
        ice_score = self._check_ice_period(features)
        scores[ScenarioType.S5_ICE_PERIOD] = ice_score

        # 规则6: 泵站故障检测
        pump_failure_score = self._check_pump_failure(features)
        scores[ScenarioType.S6_PUMP_FAILURE] = pump_failure_score

        # 规则7: 计划检修 (需要调度计划)
        maint_score = self._check_planned_maintenance(features, context)
        scores[ScenarioType.S7_PLANNED_MAINT] = maint_score

        # 规则8: 紧急抢修检测
        emergency_score = self._check_emergency(features)
        scores[ScenarioType.S8_EMERGENCY_REPAIR] = emergency_score

        # 选择最高分场景
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_scores[0]

        # 确定严重程度
        severity = self._determine_severity(features, best_type)

        # 计算影响范围
        affected_range = self._calculate_affected_range(features)

        # 获取备选场景
        alternatives = [(t, s) for t, s in sorted_scores[1:4] if s > 0.3]

        return ScenarioDetectionResult(
            detected_type=best_type,
            confidence=best_score,
            severity=severity,
            center_location=max(0, features.anomaly_center),
            affected_range=affected_range,
            alternatives=alternatives,
            features=features,
        )

    def _check_normal(self, features: ScenarioFeatures) -> float:
        """检测常规运行"""
        score = 1.0

        # 水位稳定
        if features.level_std > 0.5:
            score -= 0.3
        if abs(features.level_mean - self.baseline_level) > 0.5:
            score -= 0.2

        # 流量稳定
        if features.flow_imbalance > 0.2:
            score -= 0.3

        # 无异常
        if features.level_anomaly_count > 3:
            score -= 0.4

        return max(0.0, score)

    def _check_surge_demand(self, features: ScenarioFeatures) -> float:
        """检测突发增供"""
        score = 0.0

        # 流量增加趋势
        if features.flow_trend > 5.0:
            score += 0.5
        if features.flow_trend > 10.0:
            score += 0.3

        # 下游水位下降
        if features.level_trend < -0.05:
            score += 0.2

        return min(1.0, score)

    def _check_pollution(self, features: ScenarioFeatures,
                          context: Dict[str, Any]) -> float:
        """检测污染"""
        score = 0.0

        # 检查水质传感器数据
        if context.get('water_quality_alert'):
            score += 0.8

        # 局部异常 (污染通常是局部的)
        if 0 < features.affected_pools < 10:
            if features.anomaly_center >= 0:
                score += 0.2

        return min(1.0, score)

    def _check_flood(self, features: ScenarioFeatures) -> float:
        """检测防洪"""
        score = 0.0

        # 天气条件
        if features.weather in [WeatherType.HEAVY_RAIN, WeatherType.STORM]:
            score += 0.4

        # 水位上升趋势
        if features.level_trend > 0.1:
            score += 0.3

        # 高水位
        if features.level_mean > self.baseline_level + 0.5:
            score += 0.3

        return min(1.0, score)

    def _check_ice_period(self, features: ScenarioFeatures) -> float:
        """检测冰期"""
        score = 0.0

        # 季节条件
        if features.season in [SeasonType.WINTER, SeasonType.ICE_PERIOD]:
            score += 0.5

        # 低流量
        if features.flow_mean < 200:
            score += 0.3

        # 天气条件
        if features.weather == WeatherType.SNOW:
            score += 0.2

        return min(1.0, score)

    def _check_pump_failure(self, features: ScenarioFeatures) -> float:
        """检测泵站故障"""
        score = 0.0

        # 突然的流量下降
        if features.flow_trend < -20:
            score += 0.5

        # 局部异常
        if 0 < features.affected_pools < 5:
            score += 0.3

        # 变化快速
        if features.change_rate > 0.5:
            score += 0.2

        return min(1.0, score)

    def _check_planned_maintenance(self, features: ScenarioFeatures,
                                    context: Dict[str, Any]) -> float:
        """检测计划检修"""
        # 主要依赖调度计划
        if context.get('maintenance_scheduled'):
            return 0.9
        return 0.0

    def _check_emergency(self, features: ScenarioFeatures) -> float:
        """检测紧急抢修"""
        score = 0.0

        # 突发异常
        if features.level_anomaly_count > 0:
            score += 0.3

        # 快速变化
        if abs(features.level_trend) > 0.2:
            score += 0.3

        # 局部问题
        if 0 < features.affected_pools < 5:
            score += 0.2

        return min(1.0, score)

    def _determine_severity(self,
                            features: ScenarioFeatures,
                            scenario_type: ScenarioType) -> ScenarioSeverity:
        """确定严重程度"""
        # 基于特征计算严重程度分数
        severity_score = 0.0

        # 异常数量
        severity_score += min(1.0, features.level_anomaly_count / 10) * 0.3

        # 偏离程度
        level_deviation = abs(features.level_mean - self.baseline_level) / self.baseline_level
        severity_score += min(1.0, level_deviation * 2) * 0.3

        # 变化速率
        severity_score += min(1.0, abs(features.level_trend) * 5) * 0.2

        # 影响范围
        severity_score += min(1.0, features.affected_pools / 20) * 0.2

        # 映射到严重程度
        if severity_score < 0.25:
            return ScenarioSeverity.LOW
        elif severity_score < 0.5:
            return ScenarioSeverity.MEDIUM
        elif severity_score < 0.75:
            return ScenarioSeverity.HIGH
        else:
            return ScenarioSeverity.CRITICAL

    def _calculate_affected_range(self, features: ScenarioFeatures) -> int:
        """计算影响范围"""
        return max(3, min(30, features.affected_pools + 2))


# ==============================================================================
# 自适应MPC配置器
# ==============================================================================

@dataclass
class AdaptiveMPCConfig:
    """
    自适应MPC配置
    """
    # 目标函数权重
    weights: MPCWeights

    # 约束配置
    constraints: MPCConstraints

    # 角色
    role: PoolRole = PoolRole.TRANSMIT

    # 物理参数调整
    physics_adjustment: Dict[str, float] = field(default_factory=dict)

    # 参考轨迹调整
    reference_bias: float = 0.0
    reference_trajectory: Optional[np.ndarray] = None

    # 切换参数
    transition_steps: int = 5           # 过渡步数
    transition_alpha: float = 0.8       # 过渡平滑因子


class AdaptiveMPCConfigurator:
    """
    自适应MPC配置器

    根据场景自动生成MPC配置:
    1. 目标函数权重 (水位跟踪、流量平滑、能耗等)
    2. 约束条件 (水位上下限、流量限制、变化率限制)
    3. 物理参数调整 (效率系数、延迟补偿)
    4. 参考轨迹 (目标水位偏移、预测轨迹)
    """

    # 场景-权重配置表
    SCENARIO_WEIGHTS = {
        ScenarioType.S1_NORMAL_PLAN: {
            'W_level': 10.0,
            'W_flow': 5.0,
            'W_smooth': 2.0,
            'W_coupling': 1.0,
            'W_trajectory': 15.0,
            'W_energy': 0.5,
        },
        ScenarioType.S2_SURGE_DEMAND: {
            'W_level': 5.0,
            'W_flow': 15.0,       # 高权重保证流量
            'W_smooth': 1.0,      # 降低平滑要求
            'W_coupling': 2.0,
            'W_trajectory': 10.0,
            'W_energy': 0.1,
        },
        ScenarioType.S3_POLLUTION: {
            'W_level': 2.0,
            'W_flow': 100.0,      # 极高权重控制流量
            'W_smooth': 0.5,
            'W_coupling': 5.0,
            'W_trajectory': 5.0,
            'W_energy': 0.0,
        },
        ScenarioType.S4_FLOOD_CONTROL: {
            'W_level': 20.0,      # 高权重控制水位
            'W_flow': 10.0,
            'W_smooth': 1.0,
            'W_coupling': 3.0,
            'W_trajectory': 25.0,
            'W_energy': 0.0,
        },
        ScenarioType.S5_ICE_PERIOD: {
            'W_level': 15.0,
            'W_flow': 8.0,
            'W_smooth': 10.0,     # 高平滑权重
            'W_coupling': 2.0,
            'W_trajectory': 20.0,
            'W_energy': 1.0,
        },
        ScenarioType.S6_PUMP_FAILURE: {
            'W_level': 8.0,
            'W_flow': 12.0,
            'W_smooth': 3.0,
            'W_coupling': 4.0,
            'W_trajectory': 15.0,
            'W_energy': 0.2,
        },
        ScenarioType.S7_PLANNED_MAINT: {
            'W_level': 12.0,
            'W_flow': 8.0,
            'W_smooth': 5.0,
            'W_coupling': 2.0,
            'W_trajectory': 18.0,
            'W_energy': 0.3,
        },
        ScenarioType.S8_EMERGENCY_REPAIR: {
            'W_level': 10.0,
            'W_flow': 15.0,
            'W_smooth': 2.0,
            'W_coupling': 5.0,
            'W_trajectory': 12.0,
            'W_energy': 0.0,
        },
    }

    # 场景-约束配置表
    SCENARIO_CONSTRAINTS = {
        ScenarioType.S1_NORMAL_PLAN: {
            'Z_min': 0.5,
            'Z_max': 8.0,
            'Q_min': 0.0,
            'Q_max': 400.0,
            'delta_Q_max': 50.0,
        },
        ScenarioType.S2_SURGE_DEMAND: {
            'Z_min': 0.3,         # 允许更低水位
            'Z_max': 8.0,
            'Q_min': 0.0,
            'Q_max': 450.0,       # 提高流量上限
            'delta_Q_max': 80.0,  # 允许更快变化
        },
        ScenarioType.S3_POLLUTION: {
            'Z_min': 0.5,
            'Z_max': 8.5,         # 允许更高水位(蓄水)
            'Q_min': 0.0,
            'Q_max': 0.0,         # 隔离区零流量
            'delta_Q_max': 100.0, # 快速关闭
        },
        ScenarioType.S4_FLOOD_CONTROL: {
            'Z_min': 0.3,
            'Z_max': 7.0,         # 降低目标水位
            'Q_min': 0.0,
            'Q_max': 500.0,       # 提高泄流能力
            'delta_Q_max': 100.0,
        },
        ScenarioType.S5_ICE_PERIOD: {
            'Z_min': 1.0,         # 保持最小水深
            'Z_max': 6.0,
            'Q_min': 50.0,        # 保持最小流量
            'Q_max': 250.0,       # 限制最大流量
            'delta_Q_max': 20.0,  # 严格限制变化率
        },
        ScenarioType.S6_PUMP_FAILURE: {
            'Z_min': 0.5,
            'Z_max': 8.0,
            'Q_min': 0.0,
            'Q_max': 350.0,
            'delta_Q_max': 60.0,
        },
        ScenarioType.S7_PLANNED_MAINT: {
            'Z_min': 0.5,
            'Z_max': 8.0,
            'Q_min': 0.0,
            'Q_max': 400.0,
            'delta_Q_max': 40.0,
        },
        ScenarioType.S8_EMERGENCY_REPAIR: {
            'Z_min': 0.5,
            'Z_max': 8.0,
            'Q_min': 0.0,
            'Q_max': 400.0,
            'delta_Q_max': 80.0,
        },
    }

    # 严重程度权重乘数
    SEVERITY_MULTIPLIERS = {
        ScenarioSeverity.LOW: {
            'weight_scale': 0.8,
            'constraint_scale': 0.9,
        },
        ScenarioSeverity.MEDIUM: {
            'weight_scale': 1.0,
            'constraint_scale': 1.0,
        },
        ScenarioSeverity.HIGH: {
            'weight_scale': 1.3,
            'constraint_scale': 1.1,
        },
        ScenarioSeverity.CRITICAL: {
            'weight_scale': 1.5,
            'constraint_scale': 1.2,
        },
    }

    def __init__(self):
        """初始化配置器"""
        self.current_config: Optional[AdaptiveMPCConfig] = None
        self.config_history: List[AdaptiveMPCConfig] = []

    def generate_config(self,
                        detection_result: ScenarioDetectionResult,
                        pool_id: int,
                        pool_role: PoolRole = None,
                        ) -> AdaptiveMPCConfig:
        """
        生成自适应MPC配置

        Args:
            detection_result: 场景检测结果
            pool_id: 渠池ID
            pool_role: 指定角色 (可选, 否则自动确定)

        Returns:
            MPC配置
        """
        scenario_type = detection_result.detected_type
        severity = detection_result.severity
        center = detection_result.center_location
        affected_range = detection_result.affected_range

        # 1. 获取基础权重
        base_weights = self.SCENARIO_WEIGHTS.get(
            scenario_type,
            self.SCENARIO_WEIGHTS[ScenarioType.S1_NORMAL_PLAN]
        )

        # 2. 获取基础约束
        base_constraints = self.SCENARIO_CONSTRAINTS.get(
            scenario_type,
            self.SCENARIO_CONSTRAINTS[ScenarioType.S1_NORMAL_PLAN]
        )

        # 3. 应用严重程度调整
        severity_mult = self.SEVERITY_MULTIPLIERS.get(
            severity,
            self.SEVERITY_MULTIPLIERS[ScenarioSeverity.MEDIUM]
        )

        # 调整权重
        adjusted_weights = MPCWeights(
            W_level=base_weights['W_level'] * severity_mult['weight_scale'],
            W_flow=base_weights['W_flow'] * severity_mult['weight_scale'],
            W_smooth=base_weights['W_smooth'],
            W_coupling=base_weights['W_coupling'],
            W_trajectory=base_weights['W_trajectory'],
            W_energy=base_weights['W_energy'],
        )

        # 调整约束
        adjusted_constraints = MPCConstraints(
            Z_min=base_constraints['Z_min'],
            Z_max=base_constraints['Z_max'],
            Q_min=base_constraints['Q_min'],
            Q_max=base_constraints['Q_max'] * severity_mult['constraint_scale'],
            delta_Q_max=base_constraints['delta_Q_max'] * severity_mult['constraint_scale'],
        )

        # 4. 根据位置关系调整 (相对于事故中心)
        if pool_role is None:
            pool_role = self._determine_role(
                pool_id, center, affected_range, scenario_type
            )

        # 应用角色调整
        role_weights = RoleParameterMapper.get_weights_for_role(
            pool_role, adjusted_weights
        )
        role_constraints = RoleParameterMapper.get_constraints_for_role(
            pool_role, adjusted_constraints
        )

        # 5. 计算参考偏移
        reference_bias = self._calculate_reference_bias(
            pool_id, center, scenario_type, severity
        )

        # 6. 确定过渡参数
        transition_steps = self._determine_transition_steps(scenario_type, severity)

        config = AdaptiveMPCConfig(
            weights=role_weights,
            constraints=role_constraints,
            role=pool_role,
            reference_bias=reference_bias,
            transition_steps=transition_steps,
        )

        self.current_config = config
        self.config_history.append(config)

        return config

    def _determine_role(self,
                        pool_id: int,
                        center: int,
                        affected_range: int,
                        scenario_type: ScenarioType) -> PoolRole:
        """确定渠池角色"""
        distance = abs(pool_id - center)

        # 特殊场景处理
        if scenario_type == ScenarioType.S3_POLLUTION:
            if distance == 0:
                return PoolRole.ISOLATE
            elif distance <= 3 and pool_id < center:
                return PoolRole.BUFFER
            elif distance <= 3 and pool_id > center:
                return PoolRole.DRAIN
            else:
                return PoolRole.TRANSMIT

        elif scenario_type == ScenarioType.S4_FLOOD_CONTROL:
            if distance <= affected_range:
                return PoolRole.DRAIN
            else:
                return PoolRole.TRANSMIT

        elif scenario_type == ScenarioType.S5_ICE_PERIOD:
            return PoolRole.STABLE

        elif scenario_type == ScenarioType.S6_PUMP_FAILURE:
            if distance == 0:
                return PoolRole.FIX
            elif distance <= 3 and pool_id < center:
                return PoolRole.BUFFER
            elif distance <= 3 and pool_id > center:
                return PoolRole.PASS
            else:
                return PoolRole.TRANSMIT

        elif scenario_type == ScenarioType.S7_PLANNED_MAINT:
            if distance == 0:
                return PoolRole.MAINT
            elif distance <= 2:
                return PoolRole.WAIT
            else:
                return PoolRole.TRANSMIT

        elif scenario_type == ScenarioType.S8_EMERGENCY_REPAIR:
            if distance == 0:
                return PoolRole.FIX
            elif distance <= 3:
                return PoolRole.ISLAND
            else:
                return PoolRole.BYPASS

        else:
            return PoolRole.TRANSMIT

    def _calculate_reference_bias(self,
                                   pool_id: int,
                                   center: int,
                                   scenario_type: ScenarioType,
                                   severity: ScenarioSeverity) -> float:
        """计算参考值偏移"""
        distance = abs(pool_id - center)

        # 基础偏移
        base_bias = {
            ScenarioType.S1_NORMAL_PLAN: 0.0,
            ScenarioType.S2_SURGE_DEMAND: -0.2,    # 降低水位以增加流量
            ScenarioType.S3_POLLUTION: 0.5,        # 提高水位蓄水
            ScenarioType.S4_FLOOD_CONTROL: -0.5,   # 降低水位预泄
            ScenarioType.S5_ICE_PERIOD: 0.0,
            ScenarioType.S6_PUMP_FAILURE: 0.2,
            ScenarioType.S7_PLANNED_MAINT: 0.3,
            ScenarioType.S8_EMERGENCY_REPAIR: 0.0,
        }.get(scenario_type, 0.0)

        # 距离衰减
        decay = np.exp(-distance / 10)

        # 严重程度调整
        severity_factor = {
            ScenarioSeverity.LOW: 0.5,
            ScenarioSeverity.MEDIUM: 1.0,
            ScenarioSeverity.HIGH: 1.5,
            ScenarioSeverity.CRITICAL: 2.0,
        }.get(severity, 1.0)

        return base_bias * decay * severity_factor

    def _determine_transition_steps(self,
                                     scenario_type: ScenarioType,
                                     severity: ScenarioSeverity) -> int:
        """确定过渡步数"""
        # 紧急场景快速切换
        if severity == ScenarioSeverity.CRITICAL:
            return 2

        # 某些场景需要快速响应
        fast_scenarios = [
            ScenarioType.S3_POLLUTION,
            ScenarioType.S4_FLOOD_CONTROL,
            ScenarioType.S8_EMERGENCY_REPAIR,
        ]
        if scenario_type in fast_scenarios:
            return 3

        # 默认平滑过渡
        return 5


# ==============================================================================
# 自适应MPC系统
# ==============================================================================

class AdaptiveMPCSystem:
    """
    自适应MPC系统 - 集成场景识别和MPC配置

    工作流程:
    1. 传感器数据 -> 特征提取 -> 场景识别
    2. 场景识别结果 -> MPC配置生成
    3. MPC配置 -> 应用到控制器
    4. 执行控制 -> 反馈调整
    """

    def __init__(self, num_pools: int = 60, horizon: int = 10, dt: float = 900.0):
        """
        初始化自适应MPC系统

        Args:
            num_pools: 渠池数量
            horizon: MPC预测时域
            dt: 时间步长
        """
        self.num_pools = num_pools
        self.horizon = horizon
        self.dt = dt

        # 组件
        self.identifier = ScenarioIdentifier()
        self.configurator = AdaptiveMPCConfigurator()
        self.mpc_manager = HotReconfigurableMPC(num_pools, horizon, dt)

        # 状态
        self.current_scenario: Optional[ScenarioDetectionResult] = None
        self.scenario_history: List[ScenarioDetectionResult] = []

        # 性能统计
        self.identification_times: List[float] = []
        self.configuration_times: List[float] = []
        self.solve_times: List[float] = []

        logger.info(f"自适应MPC系统初始化: {num_pools}池, 时域={horizon}")

    def update(self,
               levels: np.ndarray,
               flows: np.ndarray,
               season: SeasonType = SeasonType.SUMMER,
               weather: WeatherType = WeatherType.CLEAR,
               context: Dict[str, Any] = None,
               ) -> Tuple[np.ndarray, np.ndarray, ScenarioDetectionResult]:
        """
        更新系统状态并计算控制

        Args:
            levels: 当前水位数组
            flows: 当前流量数组
            season: 当前季节
            weather: 当前天气
            context: 上下文信息

        Returns:
            (入流控制, 出流控制, 场景检测结果)
        """
        context = context or {}

        # 1. 场景识别
        t0 = time.time()
        features = self.identifier.extract_features(levels, flows, season, weather)
        detection = self.identifier.identify_scenario(features, context)
        t_identify = time.time() - t0
        self.identification_times.append(t_identify)

        # 2. 检测场景变化
        scenario_changed = self._check_scenario_change(detection)

        # 3. 如果场景变化, 更新MPC配置
        if scenario_changed:
            t0 = time.time()
            self._reconfigure_all_mpc(detection)
            t_config = time.time() - t0
            self.configuration_times.append(t_config)

            logger.info(f"场景切换: {detection.detected_type.value}, "
                       f"置信度={detection.confidence:.2f}, "
                       f"严重程度={detection.severity.value}")

        # 4. 求解MPC
        t0 = time.time()
        Q_in, Q_out = self.mpc_manager.solve_all(levels, flows)
        t_solve = time.time() - t0
        self.solve_times.append(t_solve)

        # 5. 更新状态
        self.current_scenario = detection
        self.scenario_history.append(detection)

        return Q_in, Q_out, detection

    def _check_scenario_change(self, new_detection: ScenarioDetectionResult) -> bool:
        """检测场景是否变化"""
        if self.current_scenario is None:
            return True

        # 场景类型变化
        if new_detection.detected_type != self.current_scenario.detected_type:
            # 需要足够高的置信度才能切换
            if new_detection.confidence > 0.6:
                return True

        # 严重程度显著变化
        old_sev = self.current_scenario.severity
        new_sev = new_detection.severity
        severity_order = [
            ScenarioSeverity.LOW,
            ScenarioSeverity.MEDIUM,
            ScenarioSeverity.HIGH,
            ScenarioSeverity.CRITICAL,
        ]
        if abs(severity_order.index(old_sev) - severity_order.index(new_sev)) >= 2:
            return True

        # 中心位置显著变化
        if abs(new_detection.center_location - self.current_scenario.center_location) > 10:
            return True

        return False

    def _reconfigure_all_mpc(self, detection: ScenarioDetectionResult):
        """重新配置所有MPC控制器"""
        for pool_id in range(self.num_pools):
            # 生成配置
            config = self.configurator.generate_config(detection, pool_id)

            # 获取控制器
            controller = self.mpc_manager.controllers[pool_id]

            # 应用角色
            controller.apply_role(config.role, smooth_transition=False)

            # 应用权重
            controller.active_weights = config.weights

            # 应用约束
            controller.active_constraints = config.constraints

            # 应用参考偏移
            controller.Z_ref = 4.0 + config.reference_bias

            # 设置过渡参数
            controller._transition_steps = config.transition_steps

            # 更新CVXPY参数
            controller._set_parameter_values()

    def apply_event(self, event: ExtendedScenarioEvent):
        """
        直接应用事件 (用于测试)

        Args:
            event: 扩展场景事件
        """
        # 创建检测结果
        detection = ScenarioDetectionResult(
            detected_type=event.scenario_type,
            confidence=1.0,
            severity=event.severity,
            center_location=event.location,
            affected_range=event.affected_range,
        )

        # 重新配置
        self._reconfigure_all_mpc(detection)
        self.current_scenario = detection
        self.scenario_history.append(detection)

        logger.info(f"应用事件: {event.event_id}, "
                   f"类型={event.scenario_type.value}, "
                   f"位置={event.location}")

    def apply_composite_scenario(self, scenario: CompositeScenario):
        """
        应用复合场景

        Args:
            scenario: 复合场景
        """
        # 获取主要事件
        primary = scenario.get_primary_event()
        if primary is None:
            return

        # 创建检测结果 (融合所有事件)
        all_locations = [e.location for e in scenario.events]
        center = int(np.mean(all_locations))

        # 使用key函数比较严重程度
        severity_order = {
            ScenarioSeverity.LOW: 0,
            ScenarioSeverity.MEDIUM: 1,
            ScenarioSeverity.HIGH: 2,
            ScenarioSeverity.CRITICAL: 3,
        }
        max_severity = max(scenario.events, key=lambda e: severity_order.get(e.severity, 1)).severity
        total_range = max(e.affected_range for e in scenario.events)

        detection = ScenarioDetectionResult(
            detected_type=primary.scenario_type,
            confidence=1.0,
            severity=max_severity,
            center_location=center,
            affected_range=total_range,
        )

        # 重新配置
        self._reconfigure_all_mpc(detection)

        # 对于多事件场景, 逐个应用特殊配置
        for event in scenario.events:
            pool_id = event.location
            role = self.configurator._determine_role(
                pool_id, event.location, event.affected_range, event.scenario_type
            )
            config = self.configurator.generate_config(
                ScenarioDetectionResult(
                    detected_type=event.scenario_type,
                    confidence=1.0,
                    severity=event.severity,
                    center_location=event.location,
                    affected_range=event.affected_range,
                ),
                pool_id,
                role
            )

            controller = self.mpc_manager.controllers[pool_id]
            controller.apply_role(role)
            controller.active_weights = config.weights
            controller.active_constraints = config.constraints

        self.current_scenario = detection
        self.scenario_history.append(detection)

        logger.info(f"应用复合场景: {scenario.scenario_id}, "
                   f"事件数={len(scenario.events)}, "
                   f"复杂度={scenario.complexity}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'total_updates': len(self.scenario_history),
            'scenario_changes': sum(
                1 for i in range(1, len(self.scenario_history))
                if self.scenario_history[i].detected_type !=
                   self.scenario_history[i-1].detected_type
            ),
            'avg_identification_time': np.mean(self.identification_times) if self.identification_times else 0,
            'avg_configuration_time': np.mean(self.configuration_times) if self.configuration_times else 0,
            'avg_solve_time': np.mean(self.solve_times) if self.solve_times else 0,
            'current_scenario': self.current_scenario.detected_type.value if self.current_scenario else None,
        }


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "自适应MPC系统测试")
    print("=" * 70)

    # 创建系统
    system = AdaptiveMPCSystem(num_pools=60, horizon=10)

    # 测试1: 常规运行
    print(f"\n{'=' * 70}")
    print("测试1: 常规运行识别")
    print('=' * 70)

    levels = np.ones(60) * 4.0 + np.random.randn(60) * 0.1
    flows = np.ones(60) * 300.0 + np.random.randn(60) * 10

    Q_in, Q_out, detection = system.update(levels, flows)
    print(f"  检测场景: {detection.detected_type.value}")
    print(f"  置信度: {detection.confidence:.2f}")
    print(f"  严重程度: {detection.severity.value}")

    # 测试2: 污染场景
    print(f"\n{'=' * 70}")
    print("测试2: 污染场景识别")
    print('=' * 70)

    from .scenario_generator import ScenarioGenerator, ExtendedScenarioEvent

    generator = ScenarioGenerator(seed=42)
    pollution_event = generator.generate_single_event(
        scenario_type=ScenarioType.S3_POLLUTION,
        location=30,
        severity=ScenarioSeverity.CRITICAL,
    )

    system.apply_event(pollution_event)

    # 获取池30的配置
    controller_30 = system.mpc_manager.controllers[30]
    print(f"  池30角色: {controller_30.current_role.value}")
    print(f"  池30 Q_max: {controller_30.active_constraints.Q_max}")
    print(f"  池30 W_flow: {controller_30.active_weights.W_flow}")

    # 测试3: 复合场景
    print(f"\n{'=' * 70}")
    print("测试3: 复合场景处理")
    print('=' * 70)

    composite = generator.generate_dual_event_scenario()
    system.apply_composite_scenario(composite)

    print(f"  场景ID: {composite.scenario_id}")
    print(f"  事件数: {len(composite.events)}")
    print(f"  当前场景: {system.current_scenario.detected_type.value}")

    # 统计
    stats = system.get_statistics()
    print(f"\n{'=' * 70}")
    print("性能统计")
    print('=' * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

"""
运行设计域 (Operational Design Domain, ODD)

定义南水北调中线工程自主控制系统的运行边界、约束条件和能力范围

ODD层次结构:
1. 物理域 - 水力学、几何、环境约束
2. 功能域 - 系统能力、自主等级、响应时间
3. 场景域 - 支持的运行场景、边界条件
4. 安全域 - 安全约束、降级策略、紧急响应
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """自主等级定义 (参考SAE J3016)"""
    L0_MANUAL = 0          # 完全人工
    L1_ADVISORY = 1        # 辅助决策
    L2_PARTIAL = 2         # 部分自动 (PID/规则)
    L3_CONDITIONAL = 3     # 条件自动 (场景适应)
    L4_HIGH = 4            # 高度自动 (E2E)
    L5_FULL = 5            # 完全自动 (含未知场景)


class SeasonType(Enum):
    """季节类型"""
    SPRING = "spring"       # 春季 (3-5月)
    SUMMER = "summer"       # 夏季 (6-8月) - 汛期
    AUTUMN = "autumn"       # 秋季 (9-11月)
    WINTER = "winter"       # 冬季 (12-2月) - 冰期


class WeatherCondition(Enum):
    """天气条件"""
    NORMAL = "normal"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    STORM = "storm"
    DROUGHT = "drought"
    ICE = "ice"
    FOG = "fog"


@dataclass
class PhysicalConstraints:
    """物理约束"""
    # 水位约束
    level_min: float = 1.5           # m (最小水位)
    level_max: float = 5.5           # m (最大水位)
    level_change_rate_max: float = 0.3  # m/h (最大水位变化率)

    # 流量约束
    flow_min: float = 50.0           # m³/s (最小流量)
    flow_max: float = 500.0          # m³/s (最大流量)
    flow_design: float = 350.0       # m³/s (设计流量)

    # 闸门约束
    gate_opening_rate_max: float = 0.001  # 1/s (最大开度变化率)
    gate_response_time: float = 300.0     # s (闸门响应时间)

    # 几何约束
    channel_length: float = 1432000.0    # m (总长度)
    num_pools: int = 63                  # 渠池数量
    num_gates: int = 64                  # 闸门数量

    # 水力学约束
    wave_speed: float = 5.0              # m/s (典型波速)
    flow_velocity_max: float = 2.5       # m/s (最大流速)
    manning_n_range: Tuple[float, float] = (0.012, 0.018)


@dataclass
class FunctionalConstraints:
    """功能约束"""
    # 自主能力
    max_autonomy_level: AutonomyLevel = AutonomyLevel.L4_HIGH
    min_confidence_threshold: float = 0.3     # 最小置信度阈值
    decision_frequency: float = 900.0         # s (决策周期)

    # 响应时间
    perception_latency_max: float = 5.0       # s (感知延迟)
    decision_latency_max: float = 10.0        # s (决策延迟)
    execution_latency_max: float = 300.0      # s (执行延迟)
    emergency_response_time: float = 60.0     # s (应急响应)

    # 通信约束
    communication_delay_max: float = 0.5      # s (最大通信延迟)
    message_loss_rate_max: float = 0.01       # 最大丢包率
    consensus_rounds_max: int = 5             # 最大共识轮数

    # 计算约束
    prediction_horizon: int = 48              # 预测步数 (12小时)
    history_length: int = 96                  # 历史长度 (24小时)


@dataclass
class SafetyConstraints:
    """安全约束"""
    # 安全裕度
    level_safety_margin: float = 0.2         # m (水位安全裕度)
    flow_safety_margin: float = 20.0         # m³/s (流量安全裕度)

    # 降级触发条件
    degradation_triggers: Dict[str, float] = field(default_factory=lambda: {
        "confidence_too_low": 0.3,            # 置信度低于此值
        "prediction_error_high": 0.1,         # 预测误差高于此值
        "consecutive_failures": 3,            # 连续失败次数
        "communication_loss_time": 60.0,      # s (通信丢失时间)
    })

    # 紧急停止条件
    emergency_stop_triggers: Dict[str, float] = field(default_factory=lambda: {
        "level_overflow": 5.8,                # m (溢流水位)
        "level_underflow": 1.2,               # m (干渠水位)
        "flow_surge": 600.0,                  # m³/s (流量激增)
    })

    # 回退策略
    fallback_autonomy_level: AutonomyLevel = AutonomyLevel.L2_PARTIAL


@dataclass
class ScenarioConstraints:
    """场景约束"""
    # 支持的场景类型
    supported_scenarios: Set[str] = field(default_factory=lambda: {
        "S1_NORMAL",           # 常规计划输水
        "S2_DEMAND_SURGE",     # 突发增供
        "S3_POLLUTION",        # 突发污染
        "S4_FLOOD",            # 暴雨防洪
        "S5_ICE_PERIOD",       # 冰期输水
        "S6_POWER_FAILURE",    # 泵站掉电
        "S7_PLANNED_MAINTENANCE",  # 计划检修
        "S8_EMERGENCY_REPAIR", # 临时抢修
    })

    # 场景自主等级映射
    scenario_autonomy_map: Dict[str, AutonomyLevel] = field(default_factory=lambda: {
        "S1_NORMAL": AutonomyLevel.L4_HIGH,
        "S2_DEMAND_SURGE": AutonomyLevel.L4_HIGH,
        "S3_POLLUTION": AutonomyLevel.L3_CONDITIONAL,
        "S4_FLOOD": AutonomyLevel.L3_CONDITIONAL,
        "S5_ICE_PERIOD": AutonomyLevel.L3_CONDITIONAL,
        "S6_POWER_FAILURE": AutonomyLevel.L2_PARTIAL,
        "S7_PLANNED_MAINTENANCE": AutonomyLevel.L3_CONDITIONAL,
        "S8_EMERGENCY_REPAIR": AutonomyLevel.L1_ADVISORY,
    })

    # 场景切换约束
    transition_time_min: float = 60.0        # s (最小过渡时间)
    transition_time_max: float = 600.0       # s (最大过渡时间)


@dataclass
class SeasonalConstraints:
    """季节性约束"""
    season: SeasonType = SeasonType.SPRING

    # 季节特定约束
    season_specific: Dict[SeasonType, Dict[str, Any]] = field(default_factory=lambda: {
        SeasonType.SPRING: {
            "level_min": 1.5,
            "level_max": 5.5,
            "flow_max": 500.0,
            "ice_risk": False,
            "flood_risk": False,
        },
        SeasonType.SUMMER: {
            "level_min": 1.5,
            "level_max": 5.0,  # 汛期留出库容
            "flow_max": 450.0,
            "ice_risk": False,
            "flood_risk": True,
        },
        SeasonType.AUTUMN: {
            "level_min": 1.5,
            "level_max": 5.5,
            "flow_max": 500.0,
            "ice_risk": False,
            "flood_risk": False,
        },
        SeasonType.WINTER: {
            "level_min": 2.0,  # 冰期需要更高水位
            "level_max": 5.0,
            "flow_max": 300.0,  # 冰期降低流量
            "ice_risk": True,
            "flood_risk": False,
        },
    })


class OperationalDesignDomain:
    """
    运行设计域 (ODD)

    定义系统在何种条件下可以安全、可靠地运行
    """

    def __init__(
        self,
        physical: Optional[PhysicalConstraints] = None,
        functional: Optional[FunctionalConstraints] = None,
        safety: Optional[SafetyConstraints] = None,
        scenario: Optional[ScenarioConstraints] = None,
        seasonal: Optional[SeasonalConstraints] = None,
    ):
        self.physical = physical or PhysicalConstraints()
        self.functional = functional or FunctionalConstraints()
        self.safety = safety or SafetyConstraints()
        self.scenario = scenario or ScenarioConstraints()
        self.seasonal = seasonal or SeasonalConstraints()

        # 动态约束 (运行时可更新)
        self.dynamic_constraints: Dict[str, Any] = {}

        # ODD状态
        self.current_autonomy_level = AutonomyLevel.L2_PARTIAL
        self.current_scenario = "S1_NORMAL"
        self.is_within_odd = True
        self.odd_violations: List[str] = []

        # 历史记录
        self.violation_history: List[Dict[str, Any]] = []

        logger.info("OperationalDesignDomain initialized")

    def update_season(self, date: datetime):
        """根据日期更新季节"""
        month = date.month
        if month in [3, 4, 5]:
            self.seasonal.season = SeasonType.SPRING
        elif month in [6, 7, 8]:
            self.seasonal.season = SeasonType.SUMMER
        elif month in [9, 10, 11]:
            self.seasonal.season = SeasonType.AUTUMN
        else:
            self.seasonal.season = SeasonType.WINTER

        logger.info(f"Season updated to {self.seasonal.season.value}")

    def get_active_constraints(self) -> Dict[str, Any]:
        """获取当前激活的约束"""
        season_constraints = self.seasonal.season_specific.get(
            self.seasonal.season, {}
        )

        return {
            # 物理约束 (季节调整)
            "level_min": season_constraints.get("level_min", self.physical.level_min),
            "level_max": season_constraints.get("level_max", self.physical.level_max),
            "flow_max": season_constraints.get("flow_max", self.physical.flow_max),
            "flow_min": self.physical.flow_min,
            "level_change_rate_max": self.physical.level_change_rate_max,
            "gate_opening_rate_max": self.physical.gate_opening_rate_max,

            # 功能约束
            "max_autonomy_level": self.functional.max_autonomy_level.value,
            "min_confidence_threshold": self.functional.min_confidence_threshold,
            "decision_frequency": self.functional.decision_frequency,

            # 安全约束
            "level_safety_margin": self.safety.level_safety_margin,
            "flow_safety_margin": self.safety.flow_safety_margin,

            # 场景约束
            "current_scenario": self.current_scenario,
            "scenario_autonomy": self.scenario.scenario_autonomy_map.get(
                self.current_scenario, AutonomyLevel.L2_PARTIAL
            ).value,

            # 季节特定
            "ice_risk": season_constraints.get("ice_risk", False),
            "flood_risk": season_constraints.get("flood_risk", False),

            # 动态约束
            **self.dynamic_constraints,
        }

    def check_within_odd(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        检查当前状态和动作是否在ODD范围内

        Args:
            state: 系统状态
            action: 控制动作 (可选)

        Returns:
            (is_within_odd, violations): 是否在ODD内, 违规列表
        """
        violations = []
        constraints = self.get_active_constraints()

        # 检查水位约束
        levels = state.get("water_levels", [])
        for i, level in enumerate(levels):
            if level < constraints["level_min"]:
                violations.append(f"Pool {i}: level {level:.2f}m < min {constraints['level_min']}m")
            if level > constraints["level_max"]:
                violations.append(f"Pool {i}: level {level:.2f}m > max {constraints['level_max']}m")

        # 检查流量约束
        flows = state.get("flow_rates", [])
        for i, flow in enumerate(flows):
            if flow < constraints["flow_min"]:
                violations.append(f"Pool {i}: flow {flow:.1f} < min {constraints['flow_min']}")
            if flow > constraints["flow_max"]:
                violations.append(f"Pool {i}: flow {flow:.1f} > max {constraints['flow_max']}")

        # 检查置信度
        confidence = state.get("confidence", 1.0)
        if confidence < constraints["min_confidence_threshold"]:
            violations.append(f"Confidence {confidence:.2f} < threshold {constraints['min_confidence_threshold']}")

        # 检查动作约束
        if action is not None:
            gate_rates = action.get("gate_opening_rates", [])
            for i, rate in enumerate(gate_rates):
                if abs(rate) > constraints["gate_opening_rate_max"]:
                    violations.append(f"Gate {i}: rate {rate:.4f} > max {constraints['gate_opening_rate_max']}")

        self.odd_violations = violations
        self.is_within_odd = len(violations) == 0

        if violations:
            self.violation_history.append({
                "timestamp": datetime.now(),
                "violations": violations,
                "state_summary": {
                    "avg_level": np.mean(levels) if levels else 0,
                    "avg_flow": np.mean(flows) if flows else 0,
                    "confidence": confidence,
                },
            })

        return self.is_within_odd, violations

    def check_emergency_conditions(
        self,
        state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        检查紧急停止条件

        Returns:
            (is_emergency, reason): 是否紧急, 原因
        """
        triggers = self.safety.emergency_stop_triggers

        levels = state.get("water_levels", [])
        flows = state.get("flow_rates", [])

        # 检查溢流
        if levels and max(levels) > triggers["level_overflow"]:
            return True, f"OVERFLOW: max_level={max(levels):.2f}m"

        # 检查干渠
        if levels and min(levels) < triggers["level_underflow"]:
            return True, f"UNDERFLOW: min_level={min(levels):.2f}m"

        # 检查流量激增
        if flows and max(flows) > triggers["flow_surge"]:
            return True, f"FLOW_SURGE: max_flow={max(flows):.1f}m³/s"

        return False, ""

    def should_degrade_autonomy(
        self,
        state: Dict[str, Any],
        prediction_error: float = 0.0,
        consecutive_failures: int = 0,
    ) -> Tuple[bool, AutonomyLevel, str]:
        """
        判断是否需要降级自主等级

        Returns:
            (should_degrade, target_level, reason)
        """
        triggers = self.safety.degradation_triggers
        reasons = []

        # 置信度检查
        confidence = state.get("confidence", 1.0)
        if confidence < triggers["confidence_too_low"]:
            reasons.append(f"low_confidence({confidence:.2f})")

        # 预测误差检查
        if prediction_error > triggers["prediction_error_high"]:
            reasons.append(f"high_pred_error({prediction_error:.3f})")

        # 连续失败检查
        if consecutive_failures >= triggers["consecutive_failures"]:
            reasons.append(f"consecutive_failures({consecutive_failures})")

        if reasons:
            return True, self.safety.fallback_autonomy_level, "; ".join(reasons)

        return False, self.current_autonomy_level, ""

    def get_allowed_autonomy_level(
        self,
        scenario: str,
        system_health: float = 1.0,
    ) -> AutonomyLevel:
        """
        获取当前允许的最高自主等级

        Args:
            scenario: 当前场景
            system_health: 系统健康度 [0, 1]

        Returns:
            allowed_level: 允许的自主等级
        """
        # 场景限制的等级
        scenario_level = self.scenario.scenario_autonomy_map.get(
            scenario, AutonomyLevel.L2_PARTIAL
        )

        # 功能限制的等级
        functional_level = self.functional.max_autonomy_level

        # 取最小值
        allowed = min(scenario_level.value, functional_level.value)

        # 系统健康度影响
        if system_health < 0.5:
            allowed = min(allowed, AutonomyLevel.L2_PARTIAL.value)
        elif system_health < 0.8:
            allowed = min(allowed, AutonomyLevel.L3_CONDITIONAL.value)

        return AutonomyLevel(allowed)

    def update_scenario(self, scenario: str):
        """更新当前场景"""
        if scenario in self.scenario.supported_scenarios:
            self.current_scenario = scenario
            logger.info(f"Scenario updated to {scenario}")
        else:
            logger.warning(f"Unsupported scenario: {scenario}")

    def set_dynamic_constraint(self, key: str, value: Any):
        """设置动态约束"""
        self.dynamic_constraints[key] = value
        logger.info(f"Dynamic constraint set: {key}={value}")

    def clear_dynamic_constraints(self):
        """清除动态约束"""
        self.dynamic_constraints.clear()
        logger.info("Dynamic constraints cleared")

    def get_scenario_transition_time(
        self,
        from_scenario: str,
        to_scenario: str,
    ) -> float:
        """
        计算场景切换所需的过渡时间

        Args:
            from_scenario: 源场景
            to_scenario: 目标场景

        Returns:
            transition_time: 过渡时间 (s)
        """
        # 基于场景严重程度计算
        severity_map = {
            "S1_NORMAL": 1,
            "S2_DEMAND_SURGE": 2,
            "S3_POLLUTION": 4,
            "S4_FLOOD": 4,
            "S5_ICE_PERIOD": 3,
            "S6_POWER_FAILURE": 5,
            "S7_PLANNED_MAINTENANCE": 2,
            "S8_EMERGENCY_REPAIR": 5,
        }

        from_severity = severity_map.get(from_scenario, 1)
        to_severity = severity_map.get(to_scenario, 1)

        # 严重程度差异越大，过渡时间越长
        severity_diff = abs(to_severity - from_severity)
        base_time = self.scenario.transition_time_min

        transition_time = base_time + severity_diff * 60.0  # 每级60秒

        return min(transition_time, self.scenario.transition_time_max)

    def export_odd_specification(self) -> Dict[str, Any]:
        """导出ODD规格说明"""
        return {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "physical_constraints": {
                "level_min": self.physical.level_min,
                "level_max": self.physical.level_max,
                "flow_min": self.physical.flow_min,
                "flow_max": self.physical.flow_max,
                "flow_design": self.physical.flow_design,
                "channel_length": self.physical.channel_length,
                "num_pools": self.physical.num_pools,
                "num_gates": self.physical.num_gates,
            },
            "functional_constraints": {
                "max_autonomy_level": self.functional.max_autonomy_level.name,
                "min_confidence_threshold": self.functional.min_confidence_threshold,
                "decision_frequency": self.functional.decision_frequency,
                "prediction_horizon": self.functional.prediction_horizon,
            },
            "safety_constraints": {
                "degradation_triggers": self.safety.degradation_triggers,
                "emergency_stop_triggers": self.safety.emergency_stop_triggers,
                "fallback_level": self.safety.fallback_autonomy_level.name,
            },
            "supported_scenarios": list(self.scenario.supported_scenarios),
            "scenario_autonomy_map": {
                k: v.name for k, v in self.scenario.scenario_autonomy_map.items()
            },
            "seasonal_constraints": {
                season.value: constraints
                for season, constraints in self.seasonal.season_specific.items()
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """获取ODD状态"""
        return {
            "is_within_odd": self.is_within_odd,
            "current_autonomy_level": self.current_autonomy_level.name,
            "current_scenario": self.current_scenario,
            "current_season": self.seasonal.season.value,
            "active_violations": self.odd_violations,
            "dynamic_constraints": self.dynamic_constraints,
            "violation_count_24h": sum(
                1 for v in self.violation_history
                if (datetime.now() - v["timestamp"]).total_seconds() < 86400
            ),
        }

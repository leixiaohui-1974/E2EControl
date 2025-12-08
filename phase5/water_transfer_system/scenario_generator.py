"""
场景生成器 - 支持成千上万种场景组合
Scenario Generator - Supporting Thousands of Scenario Combinations

扩展维度:
1. 基础场景类型 (8种)
2. 严重程度 (4级: LOW, MEDIUM, HIGH, CRITICAL)
3. 位置 (60个渠池)
4. 季节条件 (4季: spring, summer, autumn, winter/ice)
5. 天气条件 (5种: clear, rain, storm, snow, fog)
6. 时间段 (4段: dawn, day, dusk, night)
7. 多事件组合 (单事件、双事件、三事件、级联)
8. 演化模式 (静态、渐进、突变、周期)

数学组合:
- 单事件场景: 8 × 4 × 60 × 4 × 5 × 4 = 153,600种
- 双事件场景: C(8,2) × 4² × C(60,2) × ... = 更多
- 总计可生成数百万种唯一场景
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Generator, Set
from enum import Enum
from itertools import combinations, product
import random
import hashlib
import time
import logging

from .core_types import (
    PoolRole, ScenarioType, ScenarioSeverity, ScenarioPhase,
    ControlDirective, ControlPlan, ScenarioEvent,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 扩展枚举类型
# ==============================================================================

class SeasonType(Enum):
    """季节类型"""
    SPRING = "spring"           # 春季 (3-5月)
    SUMMER = "summer"           # 夏季 (6-8月)
    AUTUMN = "autumn"           # 秋季 (9-11月)
    WINTER = "winter"           # 冬季 (12-2月)
    ICE_PERIOD = "ice_period"   # 冰期 (特殊)


class WeatherType(Enum):
    """天气类型"""
    CLEAR = "clear"             # 晴朗
    CLOUDY = "cloudy"           # 多云
    RAIN = "rain"               # 降雨
    HEAVY_RAIN = "heavy_rain"   # 暴雨
    STORM = "storm"             # 风暴
    SNOW = "snow"               # 降雪
    FOG = "fog"                 # 大雾
    HAZE = "haze"               # 雾霾


class TimeOfDay(Enum):
    """时间段"""
    DAWN = "dawn"               # 黎明 (5-7时)
    MORNING = "morning"         # 上午 (7-12时)
    AFTERNOON = "afternoon"     # 下午 (12-17时)
    DUSK = "dusk"               # 黄昏 (17-19时)
    EVENING = "evening"         # 傍晚 (19-22时)
    NIGHT = "night"             # 夜间 (22-5时)


class EvolutionPattern(Enum):
    """事件演化模式"""
    STATIC = "static"           # 静态 (持续不变)
    GRADUAL = "gradual"         # 渐进 (缓慢变化)
    SUDDEN = "sudden"           # 突变 (快速变化)
    PERIODIC = "periodic"       # 周期 (循环变化)
    CASCADING = "cascading"     # 级联 (连锁反应)
    RECOVERING = "recovering"   # 恢复 (逐渐减弱)


class RegionZone(Enum):
    """区域分区"""
    DANJIANGKOU = "danjiangkou"     # 丹江口水库区 (池0-5)
    HENAN_NORTH = "henan_north"     # 河南北部 (池6-20)
    HENAN_SOUTH = "henan_south"     # 河南南部 (池21-35)
    HEBEI = "hebei"                  # 河北段 (池36-50)
    BEIJING_TIANJIN = "beijing_tj"  # 京津段 (池51-59)


# ==============================================================================
# 扩展事件类
# ==============================================================================

@dataclass
class ExtendedScenarioEvent:
    """
    扩展场景事件 - 支持更多维度
    """
    # 基础属性
    event_id: str
    scenario_type: ScenarioType
    location: int                           # 主要位置
    severity: ScenarioSeverity

    # 时空属性
    timestamp: float = 0.0
    duration: float = 3600.0                # 持续时间 [s]
    affected_range: int = 5                 # 影响范围 (上下游池数)

    # 环境属性
    season: SeasonType = SeasonType.SUMMER
    weather: WeatherType = WeatherType.CLEAR
    time_of_day: TimeOfDay = TimeOfDay.MORNING

    # 演化属性
    evolution: EvolutionPattern = EvolutionPattern.STATIC
    evolution_rate: float = 0.0             # 演化速率
    peak_time: float = 0.0                  # 峰值时间

    # 额外属性
    secondary_locations: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_base_event(self) -> ScenarioEvent:
        """转换为基础事件"""
        return ScenarioEvent(
            event_id=self.event_id,
            scenario_type=self.scenario_type,
            location=self.location,
            severity=self.severity,
            timestamp=self.timestamp,
            duration=self.duration,
            affected_range=self.affected_range,
            metadata=self.metadata,
        )

    def get_hash(self) -> str:
        """获取唯一哈希"""
        key = f"{self.scenario_type.value}_{self.location}_{self.severity.value}_" \
              f"{self.season.value}_{self.weather.value}_{self.time_of_day.value}_" \
              f"{self.evolution.value}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class CompositeScenario:
    """
    复合场景 - 多事件组合
    """
    scenario_id: str
    events: List[ExtendedScenarioEvent]

    # 环境条件
    season: SeasonType = SeasonType.SUMMER
    weather: WeatherType = WeatherType.CLEAR
    base_flow: float = 300.0                # 基准流量 [m³/s]
    base_level: float = 4.0                 # 基准水位 [m]

    # 场景属性
    complexity: int = 1                     # 复杂度 (1-10)
    risk_level: float = 0.5                 # 风险等级 (0-1)

    # 预期结果
    expected_roles: Dict[int, PoolRole] = field(default_factory=dict)
    pass_criteria: Dict[str, Any] = field(default_factory=dict)

    def get_primary_event(self) -> Optional[ExtendedScenarioEvent]:
        """获取主要事件"""
        if not self.events:
            return None
        # 返回最高严重级别的事件
        return max(self.events, key=lambda e: e.severity.value)

    def get_affected_pools(self) -> Set[int]:
        """获取所有受影响的池"""
        pools = set()
        for event in self.events:
            center = event.location
            r = event.affected_range
            pools.update(range(max(0, center - r), min(60, center + r + 1)))
        return pools


# ==============================================================================
# 场景生成器
# ==============================================================================

class ScenarioGenerator:
    """
    场景生成器 - 支持生成成千上万种场景

    生成策略:
    1. 穷举生成: 所有可能的组合
    2. 随机采样: 从大空间中随机采样
    3. 重要性采样: 基于风险权重的采样
    4. 边界测试: 极端情况的场景
    5. 回归测试: 已知问题的场景
    """

    # 位置分布权重 (某些位置更容易出问题)
    LOCATION_WEIGHTS = {
        # 丹江口水库出口 (关键节点)
        0: 2.0, 1: 1.5, 2: 1.2,
        # 穿黄工程 (关键节点)
        29: 3.0, 30: 3.0, 31: 2.5,
        # 天津分水口
        55: 2.0, 56: 1.5,
        # 北京终点
        58: 2.0, 59: 2.5,
    }

    # 场景-季节相关性
    SCENARIO_SEASON_AFFINITY = {
        ScenarioType.S4_FLOOD_CONTROL: [SeasonType.SUMMER],
        ScenarioType.S5_ICE_PERIOD: [SeasonType.WINTER, SeasonType.ICE_PERIOD],
        ScenarioType.S3_POLLUTION: [SeasonType.SUMMER, SeasonType.AUTUMN],
    }

    # 场景-天气相关性
    SCENARIO_WEATHER_AFFINITY = {
        ScenarioType.S4_FLOOD_CONTROL: [WeatherType.HEAVY_RAIN, WeatherType.STORM],
        ScenarioType.S5_ICE_PERIOD: [WeatherType.SNOW, WeatherType.CLEAR],
    }

    def __init__(self, seed: int = None):
        """
        初始化生成器

        Args:
            seed: 随机种子 (用于可重复性)
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.generated_count = 0
        self.generated_hashes: Set[str] = set()

    # ==========================================================================
    # 基础生成方法
    # ==========================================================================

    def generate_single_event(self,
                              scenario_type: ScenarioType = None,
                              location: int = None,
                              severity: ScenarioSeverity = None,
                              season: SeasonType = None,
                              weather: WeatherType = None,
                              time_of_day: TimeOfDay = None,
                              evolution: EvolutionPattern = None,
                              ) -> ExtendedScenarioEvent:
        """
        生成单个事件

        Args:
            各维度参数，None表示随机选择

        Returns:
            扩展场景事件
        """
        # 随机选择未指定的参数
        if scenario_type is None:
            scenario_type = random.choice(list(ScenarioType))

        if location is None:
            # 基于权重随机选择位置
            weights = [self.LOCATION_WEIGHTS.get(i, 1.0) for i in range(60)]
            total = sum(weights)
            weights = [w / total for w in weights]
            location = np.random.choice(60, p=weights)

        if severity is None:
            severity = random.choice(list(ScenarioSeverity))

        if season is None:
            # 考虑场景-季节相关性
            affinity = self.SCENARIO_SEASON_AFFINITY.get(scenario_type)
            if affinity:
                season = random.choice(affinity)
            else:
                season = random.choice(list(SeasonType))

        if weather is None:
            # 考虑场景-天气相关性
            affinity = self.SCENARIO_WEATHER_AFFINITY.get(scenario_type)
            if affinity:
                weather = random.choice(affinity)
            else:
                weather = random.choice(list(WeatherType))

        if time_of_day is None:
            time_of_day = random.choice(list(TimeOfDay))

        if evolution is None:
            evolution = random.choice(list(EvolutionPattern))

        # 计算影响范围
        affected_range = self._calculate_affected_range(scenario_type, severity)

        # 计算持续时间
        duration = self._calculate_duration(scenario_type, severity)

        # 生成事件ID
        self.generated_count += 1
        event_id = f"E{self.generated_count:06d}"

        event = ExtendedScenarioEvent(
            event_id=event_id,
            scenario_type=scenario_type,
            location=location,
            severity=severity,
            duration=duration,
            affected_range=affected_range,
            season=season,
            weather=weather,
            time_of_day=time_of_day,
            evolution=evolution,
        )

        return event

    def _calculate_affected_range(self,
                                   scenario_type: ScenarioType,
                                   severity: ScenarioSeverity) -> int:
        """计算影响范围"""
        base_range = {
            ScenarioType.S1_NORMAL_PLAN: 0,
            ScenarioType.S2_SURGE_DEMAND: 3,
            ScenarioType.S3_POLLUTION: 5,
            ScenarioType.S4_FLOOD_CONTROL: 10,
            ScenarioType.S5_ICE_PERIOD: 60,  # 全线
            ScenarioType.S6_PUMP_FAILURE: 8,
            ScenarioType.S7_PLANNED_MAINT: 3,
            ScenarioType.S8_EMERGENCY_REPAIR: 5,
        }.get(scenario_type, 3)

        severity_multiplier = {
            ScenarioSeverity.LOW: 0.5,
            ScenarioSeverity.MEDIUM: 1.0,
            ScenarioSeverity.HIGH: 1.5,
            ScenarioSeverity.CRITICAL: 2.0,
        }.get(severity, 1.0)

        return int(base_range * severity_multiplier)

    def _calculate_duration(self,
                            scenario_type: ScenarioType,
                            severity: ScenarioSeverity) -> float:
        """计算持续时间 [s]"""
        base_duration = {
            ScenarioType.S1_NORMAL_PLAN: 86400,      # 24小时
            ScenarioType.S2_SURGE_DEMAND: 14400,     # 4小时
            ScenarioType.S3_POLLUTION: 43200,        # 12小时
            ScenarioType.S4_FLOOD_CONTROL: 86400,    # 24小时
            ScenarioType.S5_ICE_PERIOD: 604800,      # 7天
            ScenarioType.S6_PUMP_FAILURE: 7200,      # 2小时
            ScenarioType.S7_PLANNED_MAINT: 28800,    # 8小时
            ScenarioType.S8_EMERGENCY_REPAIR: 14400, # 4小时
        }.get(scenario_type, 3600)

        severity_multiplier = {
            ScenarioSeverity.LOW: 0.5,
            ScenarioSeverity.MEDIUM: 1.0,
            ScenarioSeverity.HIGH: 1.5,
            ScenarioSeverity.CRITICAL: 2.0,
        }.get(severity, 1.0)

        return base_duration * severity_multiplier

    # ==========================================================================
    # 穷举生成
    # ==========================================================================

    def generate_exhaustive_single_events(self,
                                          scenario_types: List[ScenarioType] = None,
                                          locations: List[int] = None,
                                          severities: List[ScenarioSeverity] = None,
                                          ) -> Generator[ExtendedScenarioEvent, None, None]:
        """
        穷举生成所有单事件场景组合

        Args:
            scenario_types: 场景类型列表 (默认全部)
            locations: 位置列表 (默认全部60个)
            severities: 严重程度列表 (默认全部)

        Yields:
            扩展场景事件
        """
        if scenario_types is None:
            scenario_types = list(ScenarioType)
        if locations is None:
            locations = list(range(60))
        if severities is None:
            severities = list(ScenarioSeverity)

        for st, loc, sev in product(scenario_types, locations, severities):
            yield self.generate_single_event(
                scenario_type=st,
                location=loc,
                severity=sev,
            )

    def generate_exhaustive_with_environment(self,
                                              scenario_types: List[ScenarioType] = None,
                                              locations: List[int] = None,
                                              severities: List[ScenarioSeverity] = None,
                                              seasons: List[SeasonType] = None,
                                              weathers: List[WeatherType] = None,
                                              ) -> Generator[ExtendedScenarioEvent, None, None]:
        """
        穷举生成所有单事件+环境组合

        注意: 这会生成大量场景!
        8 × 60 × 4 × 5 × 8 = 76,800 种
        """
        if scenario_types is None:
            scenario_types = list(ScenarioType)
        if locations is None:
            locations = list(range(60))
        if severities is None:
            severities = list(ScenarioSeverity)
        if seasons is None:
            seasons = list(SeasonType)
        if weathers is None:
            weathers = list(WeatherType)

        for st, loc, sev, season, weather in product(
            scenario_types, locations, severities, seasons, weathers
        ):
            yield self.generate_single_event(
                scenario_type=st,
                location=loc,
                severity=sev,
                season=season,
                weather=weather,
            )

    # ==========================================================================
    # 多事件组合生成
    # ==========================================================================

    def generate_dual_event_scenario(self,
                                      event1: ExtendedScenarioEvent = None,
                                      event2: ExtendedScenarioEvent = None,
                                      ) -> CompositeScenario:
        """
        生成双事件复合场景
        """
        if event1 is None:
            event1 = self.generate_single_event()
        if event2 is None:
            # 确保第二个事件在不同位置
            excluded = set(range(
                max(0, event1.location - 5),
                min(60, event1.location + 6)
            ))
            available = [i for i in range(60) if i not in excluded]
            loc2 = random.choice(available) if available else random.randint(0, 59)
            event2 = self.generate_single_event(location=loc2)

        # 设置时间关系
        event2.timestamp = event1.timestamp + random.uniform(0, 1800)  # 0-30分钟内

        # 计算复杂度
        complexity = self._calculate_complexity([event1, event2])

        # 计算风险等级
        risk_level = self._calculate_risk([event1, event2])

        scenario_id = f"D{self.generated_count:06d}"

        return CompositeScenario(
            scenario_id=scenario_id,
            events=[event1, event2],
            season=event1.season,
            weather=event1.weather,
            complexity=complexity,
            risk_level=risk_level,
        )

    def generate_triple_event_scenario(self) -> CompositeScenario:
        """生成三事件复合场景"""
        events = []
        used_locations = set()

        for i in range(3):
            available = [j for j in range(60) if j not in used_locations]
            if not available:
                available = list(range(60))

            loc = random.choice(available)
            event = self.generate_single_event(location=loc)
            event.timestamp = i * random.uniform(300, 900)  # 5-15分钟间隔
            events.append(event)

            # 标记影响区域
            used_locations.update(range(
                max(0, loc - 3),
                min(60, loc + 4)
            ))

        complexity = self._calculate_complexity(events)
        risk_level = self._calculate_risk(events)

        scenario_id = f"T{self.generated_count:06d}"

        return CompositeScenario(
            scenario_id=scenario_id,
            events=events,
            complexity=complexity,
            risk_level=risk_level,
        )

    def generate_cascading_scenario(self,
                                     initial_event: ExtendedScenarioEvent = None,
                                     cascade_count: int = 3,
                                     ) -> CompositeScenario:
        """
        生成级联事件场景

        Args:
            initial_event: 初始触发事件
            cascade_count: 级联事件数量

        Returns:
            级联复合场景
        """
        if initial_event is None:
            # 初始事件通常是严重事件
            initial_event = self.generate_single_event(
                severity=ScenarioSeverity.CRITICAL
            )

        initial_event.evolution = EvolutionPattern.CASCADING
        events = [initial_event]

        current_loc = initial_event.location
        current_time = initial_event.timestamp

        # 级联传播
        for i in range(cascade_count):
            # 向下游传播
            propagation_distance = random.randint(3, 8)
            new_loc = min(59, current_loc + propagation_distance)

            # 级联事件通常严重程度降低一级
            cascade_severity = self._reduce_severity(initial_event.severity)

            # 时间延迟
            delay = self._calculate_propagation_delay(
                current_loc, new_loc, initial_event.scenario_type
            )

            cascade_event = self.generate_single_event(
                scenario_type=initial_event.scenario_type,
                location=new_loc,
                severity=cascade_severity,
                season=initial_event.season,
                weather=initial_event.weather,
            )
            cascade_event.timestamp = current_time + delay
            cascade_event.evolution = EvolutionPattern.CASCADING
            cascade_event.metadata['triggered_by'] = events[-1].event_id

            events.append(cascade_event)

            current_loc = new_loc
            current_time = cascade_event.timestamp

        complexity = min(10, len(events) + 3)
        risk_level = min(1.0, 0.3 * len(events))

        scenario_id = f"C{self.generated_count:06d}"

        return CompositeScenario(
            scenario_id=scenario_id,
            events=events,
            complexity=complexity,
            risk_level=risk_level,
        )

    def _reduce_severity(self, severity: ScenarioSeverity) -> ScenarioSeverity:
        """降低一级严重程度"""
        order = [
            ScenarioSeverity.CRITICAL,
            ScenarioSeverity.HIGH,
            ScenarioSeverity.MEDIUM,
            ScenarioSeverity.LOW,
        ]
        idx = order.index(severity)
        return order[min(len(order) - 1, idx + 1)]

    def _calculate_propagation_delay(self,
                                      from_loc: int,
                                      to_loc: int,
                                      scenario_type: ScenarioType) -> float:
        """计算传播延迟 [s]"""
        distance = abs(to_loc - from_loc)

        # 不同场景类型的传播速度不同
        speed_factor = {
            ScenarioType.S3_POLLUTION: 1.5,     # 污染传播较慢
            ScenarioType.S4_FLOOD_CONTROL: 0.5, # 洪水传播快
            ScenarioType.S6_PUMP_FAILURE: 0.8,  # 压力波传播
        }.get(scenario_type, 1.0)

        # 每个池约15-30分钟的传播时间
        base_delay = distance * random.uniform(900, 1800)

        return base_delay * speed_factor

    def _calculate_complexity(self, events: List[ExtendedScenarioEvent]) -> int:
        """计算场景复杂度 (1-10)"""
        # 基础复杂度 = 事件数量
        complexity = len(events)

        # 不同类型事件增加复杂度
        unique_types = len(set(e.scenario_type for e in events))
        complexity += unique_types - 1

        # 高严重级别增加复杂度
        critical_count = sum(1 for e in events if e.severity == ScenarioSeverity.CRITICAL)
        complexity += critical_count

        # 空间分散增加复杂度
        locations = [e.location for e in events]
        spread = max(locations) - min(locations) if locations else 0
        if spread > 30:
            complexity += 2
        elif spread > 15:
            complexity += 1

        return min(10, max(1, complexity))

    def _calculate_risk(self, events: List[ExtendedScenarioEvent]) -> float:
        """计算风险等级 (0-1)"""
        if not events:
            return 0.0

        # 严重程度权重
        severity_weights = {
            ScenarioSeverity.LOW: 0.1,
            ScenarioSeverity.MEDIUM: 0.3,
            ScenarioSeverity.HIGH: 0.6,
            ScenarioSeverity.CRITICAL: 1.0,
        }

        # 场景类型风险
        type_risk = {
            ScenarioType.S1_NORMAL_PLAN: 0.1,
            ScenarioType.S2_SURGE_DEMAND: 0.3,
            ScenarioType.S3_POLLUTION: 0.8,
            ScenarioType.S4_FLOOD_CONTROL: 0.7,
            ScenarioType.S5_ICE_PERIOD: 0.5,
            ScenarioType.S6_PUMP_FAILURE: 0.6,
            ScenarioType.S7_PLANNED_MAINT: 0.2,
            ScenarioType.S8_EMERGENCY_REPAIR: 0.7,
        }

        total_risk = 0.0
        for event in events:
            sev_weight = severity_weights.get(event.severity, 0.5)
            type_weight = type_risk.get(event.scenario_type, 0.5)
            total_risk += sev_weight * type_weight

        # 多事件交互增加风险
        interaction_factor = 1.0 + 0.2 * (len(events) - 1)

        return min(1.0, total_risk * interaction_factor / len(events))

    # ==========================================================================
    # 批量生成方法
    # ==========================================================================

    def generate_batch(self,
                       count: int,
                       single_ratio: float = 0.6,
                       dual_ratio: float = 0.25,
                       triple_ratio: float = 0.1,
                       cascade_ratio: float = 0.05,
                       ) -> List[CompositeScenario]:
        """
        批量生成场景

        Args:
            count: 总数量
            single_ratio: 单事件场景比例
            dual_ratio: 双事件场景比例
            triple_ratio: 三事件场景比例
            cascade_ratio: 级联场景比例

        Returns:
            场景列表
        """
        scenarios = []

        single_count = int(count * single_ratio)
        dual_count = int(count * dual_ratio)
        triple_count = int(count * triple_ratio)
        cascade_count = count - single_count - dual_count - triple_count

        # 生成单事件场景
        for _ in range(single_count):
            event = self.generate_single_event()
            scenarios.append(CompositeScenario(
                scenario_id=f"S{len(scenarios):06d}",
                events=[event],
                season=event.season,
                weather=event.weather,
                complexity=1,
                risk_level=self._calculate_risk([event]),
            ))

        # 生成双事件场景
        for _ in range(dual_count):
            scenarios.append(self.generate_dual_event_scenario())

        # 生成三事件场景
        for _ in range(triple_count):
            scenarios.append(self.generate_triple_event_scenario())

        # 生成级联场景
        for _ in range(cascade_count):
            scenarios.append(self.generate_cascading_scenario())

        # 打乱顺序
        random.shuffle(scenarios)

        return scenarios

    def generate_stress_test_scenarios(self, count: int = 100) -> List[CompositeScenario]:
        """
        生成压力测试场景 (极端情况)
        """
        scenarios = []

        # 1. 全线污染
        for i in range(count // 10):
            event = self.generate_single_event(
                scenario_type=ScenarioType.S3_POLLUTION,
                location=random.randint(0, 30),
                severity=ScenarioSeverity.CRITICAL,
            )
            event.affected_range = 30  # 大范围
            scenarios.append(CompositeScenario(
                scenario_id=f"STRESS_POLLUTION_{i:03d}",
                events=[event],
                complexity=8,
                risk_level=0.95,
            ))

        # 2. 多点同时故障
        for i in range(count // 10):
            events = []
            for loc in [10, 30, 50]:  # 三个关键位置
                event = self.generate_single_event(
                    scenario_type=ScenarioType.S6_PUMP_FAILURE,
                    location=loc,
                    severity=ScenarioSeverity.HIGH,
                )
                events.append(event)
            scenarios.append(CompositeScenario(
                scenario_id=f"STRESS_MULTI_FAILURE_{i:03d}",
                events=events,
                complexity=9,
                risk_level=0.9,
            ))

        # 3. 暴雨+污染复合
        for i in range(count // 10):
            e1 = self.generate_single_event(
                scenario_type=ScenarioType.S4_FLOOD_CONTROL,
                severity=ScenarioSeverity.HIGH,
            )
            e2 = self.generate_single_event(
                scenario_type=ScenarioType.S3_POLLUTION,
                location=e1.location + 10,
                severity=ScenarioSeverity.MEDIUM,
            )
            scenarios.append(CompositeScenario(
                scenario_id=f"STRESS_FLOOD_POLLUTION_{i:03d}",
                events=[e1, e2],
                weather=WeatherType.STORM,
                complexity=8,
                risk_level=0.85,
            ))

        # 4. 长级联事件
        for i in range(count // 10):
            scenarios.append(self.generate_cascading_scenario(
                cascade_count=5  # 更长的级联
            ))

        # 5. 极端天气组合
        for i in range(count // 10):
            event = self.generate_single_event(
                season=SeasonType.ICE_PERIOD,
                weather=WeatherType.SNOW,
            )
            scenarios.append(CompositeScenario(
                scenario_id=f"STRESS_EXTREME_WEATHER_{i:03d}",
                events=[event],
                season=SeasonType.ICE_PERIOD,
                weather=WeatherType.SNOW,
                complexity=6,
                risk_level=0.7,
            ))

        # 填充剩余
        remaining = count - len(scenarios)
        scenarios.extend(self.generate_batch(remaining))

        return scenarios[:count]

    def generate_regression_test_scenarios(self) -> List[CompositeScenario]:
        """
        生成回归测试场景 (已知边界情况)
        """
        scenarios = []

        # 1. 边界位置测试
        boundary_locations = [0, 1, 29, 30, 31, 58, 59]
        for loc in boundary_locations:
            for st in ScenarioType:
                event = self.generate_single_event(
                    scenario_type=st,
                    location=loc,
                    severity=ScenarioSeverity.MEDIUM,
                )
                scenarios.append(CompositeScenario(
                    scenario_id=f"REG_BOUNDARY_{loc}_{st.value}",
                    events=[event],
                    complexity=2,
                    pass_criteria={'boundary_handling': True},
                ))

        # 2. 角色切换测试
        role_transitions = [
            (ScenarioType.S1_NORMAL_PLAN, ScenarioType.S3_POLLUTION),
            (ScenarioType.S3_POLLUTION, ScenarioType.S1_NORMAL_PLAN),
            (ScenarioType.S1_NORMAL_PLAN, ScenarioType.S4_FLOOD_CONTROL),
        ]
        for st1, st2 in role_transitions:
            e1 = self.generate_single_event(scenario_type=st1, location=30)
            e2 = self.generate_single_event(scenario_type=st2, location=30)
            e2.timestamp = e1.timestamp + 3600  # 1小时后
            scenarios.append(CompositeScenario(
                scenario_id=f"REG_TRANSITION_{st1.value}_to_{st2.value}",
                events=[e1, e2],
                complexity=4,
                pass_criteria={'smooth_transition': True},
            ))

        # 3. 零流量测试
        event = self.generate_single_event(
            scenario_type=ScenarioType.S3_POLLUTION,
            location=30,
            severity=ScenarioSeverity.CRITICAL,
        )
        scenarios.append(CompositeScenario(
            scenario_id="REG_ZERO_FLOW",
            events=[event],
            expected_roles={30: PoolRole.ISOLATE},
            pass_criteria={'Q_out_30': 0},
        ))

        return scenarios

    # ==========================================================================
    # 统计方法
    # ==========================================================================

    def count_possible_scenarios(self) -> Dict[str, int]:
        """计算可能的场景总数"""
        n_types = len(ScenarioType)
        n_locations = 60
        n_severities = len(ScenarioSeverity)
        n_seasons = len(SeasonType)
        n_weathers = len(WeatherType)
        n_times = len(TimeOfDay)
        n_evolutions = len(EvolutionPattern)

        single_basic = n_types * n_locations * n_severities
        single_full = single_basic * n_seasons * n_weathers * n_times * n_evolutions

        # 双事件组合 (简化计算)
        dual_approx = single_basic * (single_basic - 1) // 2

        return {
            'single_basic': single_basic,  # 1,920
            'single_with_season': single_basic * n_seasons,  # 9,600
            'single_with_weather': single_basic * n_weathers,  # 15,360
            'single_full': single_full,  # 数百万
            'dual_basic_approx': dual_approx,  # ~180万
            'total_estimate': single_full + dual_approx,
        }


# ==============================================================================
# 场景验证器
# ==============================================================================

class ScenarioValidator:
    """
    场景验证器 - 验证生成的场景是否合理
    """

    @staticmethod
    def validate_event(event: ExtendedScenarioEvent) -> Tuple[bool, List[str]]:
        """验证单个事件"""
        errors = []

        # 位置范围
        if not 0 <= event.location < 60:
            errors.append(f"位置超出范围: {event.location}")

        # 持续时间
        if event.duration <= 0:
            errors.append(f"持续时间无效: {event.duration}")

        # 影响范围
        if event.affected_range < 0:
            errors.append(f"影响范围无效: {event.affected_range}")

        # 季节-场景一致性检查
        if event.scenario_type == ScenarioType.S5_ICE_PERIOD:
            if event.season not in [SeasonType.WINTER, SeasonType.ICE_PERIOD]:
                errors.append(f"冰期场景在非冬季: {event.season.value}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_composite(scenario: CompositeScenario) -> Tuple[bool, List[str]]:
        """验证复合场景"""
        errors = []

        # 事件数量
        if not scenario.events:
            errors.append("场景无事件")
            return False, errors

        # 验证每个事件
        for event in scenario.events:
            valid, event_errors = ScenarioValidator.validate_event(event)
            if not valid:
                errors.extend(event_errors)

        # 时间顺序
        timestamps = [e.timestamp for e in scenario.events]
        if timestamps != sorted(timestamps):
            errors.append("事件时间顺序错误")

        # 复杂度范围
        if not 1 <= scenario.complexity <= 10:
            errors.append(f"复杂度超出范围: {scenario.complexity}")

        # 风险等级范围
        if not 0.0 <= scenario.risk_level <= 1.0:
            errors.append(f"风险等级超出范围: {scenario.risk_level}")

        return len(errors) == 0, errors


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "场景生成器测试")
    print("=" * 70)

    generator = ScenarioGenerator(seed=42)

    # 统计可能的场景数
    counts = generator.count_possible_scenarios()
    print("\n可能的场景数量:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    # 生成单事件
    print(f"\n{'=' * 70}")
    print("测试1: 生成单事件")
    print('=' * 70)

    event = generator.generate_single_event()
    print(f"  事件ID: {event.event_id}")
    print(f"  类型: {event.scenario_type.value}")
    print(f"  位置: 池{event.location}")
    print(f"  严重程度: {event.severity.value}")
    print(f"  季节: {event.season.value}")
    print(f"  天气: {event.weather.value}")

    # 生成双事件场景
    print(f"\n{'=' * 70}")
    print("测试2: 生成双事件场景")
    print('=' * 70)

    dual = generator.generate_dual_event_scenario()
    print(f"  场景ID: {dual.scenario_id}")
    print(f"  事件数: {len(dual.events)}")
    print(f"  复杂度: {dual.complexity}")
    print(f"  风险等级: {dual.risk_level:.2f}")
    for e in dual.events:
        print(f"    - {e.scenario_type.value} @ 池{e.location}")

    # 生成级联场景
    print(f"\n{'=' * 70}")
    print("测试3: 生成级联场景")
    print('=' * 70)

    cascade = generator.generate_cascading_scenario(cascade_count=4)
    print(f"  场景ID: {cascade.scenario_id}")
    print(f"  事件数: {len(cascade.events)}")
    print(f"  复杂度: {cascade.complexity}")
    for i, e in enumerate(cascade.events):
        print(f"    {i+1}. {e.scenario_type.value} @ 池{e.location} (t={e.timestamp:.0f}s)")

    # 批量生成
    print(f"\n{'=' * 70}")
    print("测试4: 批量生成100个场景")
    print('=' * 70)

    batch = generator.generate_batch(100)

    # 统计
    single_count = sum(1 for s in batch if len(s.events) == 1)
    dual_count = sum(1 for s in batch if len(s.events) == 2)
    multi_count = sum(1 for s in batch if len(s.events) > 2)

    print(f"  单事件场景: {single_count}")
    print(f"  双事件场景: {dual_count}")
    print(f"  多事件场景: {multi_count}")

    # 复杂度分布
    complexities = [s.complexity for s in batch]
    print(f"  平均复杂度: {np.mean(complexities):.2f}")
    print(f"  最高复杂度: {max(complexities)}")

    # 验证
    print(f"\n{'=' * 70}")
    print("测试5: 场景验证")
    print('=' * 70)

    validator = ScenarioValidator()
    valid_count = 0
    for scenario in batch:
        valid, errors = validator.validate_composite(scenario)
        if valid:
            valid_count += 1

    print(f"  验证通过: {valid_count}/{len(batch)}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

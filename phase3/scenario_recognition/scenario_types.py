"""
场景类型定义
定义水网系统的所有运行场景
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ScenarioCategory(Enum):
    """场景大类"""
    NORMAL = "normal"                  # 正常运行
    FLOOD_CONTROL = "flood_control"    # 防洪
    DROUGHT = "drought"                # 干旱
    ICE_PERIOD = "ice_period"          # 冰期
    EMERGENCY = "emergency"            # 应急
    MAINTENANCE = "maintenance"        # 维护


class NormalScenario(Enum):
    """正常运行场景"""
    DAILY_OPERATION = "daily_operation"      # 日常运行
    PEAK_DEMAND = "peak_demand"              # 高峰需求
    OFF_PEAK = "off_peak"                    # 低谷期
    WEEKEND_MODE = "weekend_mode"            # 周末模式
    NIGHT_MODE = "night_mode"                # 夜间模式
    TRANSITION = "transition"                # 过渡期


class FloodScenario(Enum):
    """防洪场景"""
    PRE_FLOOD = "pre_flood"              # 洪前准备
    FLOOD_PEAK = "flood_peak"            # 洪峰期
    POST_FLOOD = "post_flood"            # 洪后恢复
    FLASH_FLOOD = "flash_flood"          # 山洪


class DroughtScenario(Enum):
    """干旱场景"""
    WATER_SHORTAGE = "water_shortage"    # 水源短缺
    PRIORITY_ALLOCATION = "priority_allocation"  # 优先分配
    EMERGENCY_SUPPLY = "emergency_supply"        # 应急供水


class IceScenario(Enum):
    """冰期场景"""
    FREEZING = "freezing"                # 结冰期
    STABLE_ICE = "stable_ice"            # 稳定冰期
    THAWING = "thawing"                  # 融冰期
    ICE_JAM = "ice_jam"                  # 冰塞


class EmergencyScenario(Enum):
    """应急场景"""
    POLLUTION_DETECTED = "pollution_detected"    # 污染检测
    EQUIPMENT_FAILURE = "equipment_failure"      # 设备故障
    PIPE_BURST = "pipe_burst"                    # 管道爆裂
    POWER_OUTAGE = "power_outage"                # 停电
    CYBER_ATTACK = "cyber_attack"                # 网络攻击


class MaintenanceScenario(Enum):
    """维护场景"""
    PLANNED_MAINTENANCE = "planned_maintenance"  # 计划维护
    EMERGENCY_REPAIR = "emergency_repair"        # 紧急维修
    INSPECTION = "inspection"                    # 巡检
    CLEANING = "cleaning"                        # 清淤


@dataclass
class ScenarioDefinition:
    """场景定义"""
    category: ScenarioCategory
    sub_scenario: Enum
    name: str
    description: str
    
    # 特征范围
    level_range: tuple = (0.0, 10.0)
    flow_range: tuple = (0.0, 30.0)
    demand_range: tuple = (0.0, 20.0)
    
    # 控制参数
    control_priority: str = "normal"  # normal, high, critical
    safety_margin: float = 0.5
    response_time: float = 3600.0
    
    # 约束
    max_level_deviation: float = 0.3
    max_flow_change_rate: float = 2.0
    
    # 关键词（用于规则匹配）
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


# 场景库
SCENARIO_LIBRARY = {
    # 正常运行场景
    "daily_operation": ScenarioDefinition(
        category=ScenarioCategory.NORMAL,
        sub_scenario=NormalScenario.DAILY_OPERATION,
        name="日常运行",
        description="正常供水模式，需求稳定",
        level_range=(2.5, 3.5),
        flow_range=(4.0, 6.0),
        demand_range=(4.0, 6.0),
        control_priority="normal",
        keywords=["正常", "日常", "稳定", "常规"]
    ),
    
    "peak_demand": ScenarioDefinition(
        category=ScenarioCategory.NORMAL,
        sub_scenario=NormalScenario.PEAK_DEMAND,
        name="高峰需求",
        description="用水高峰期，需求激增",
        level_range=(2.0, 4.0),
        flow_range=(8.0, 15.0),
        demand_range=(8.0, 15.0),
        control_priority="high",
        max_flow_change_rate=3.0,
        keywords=["高峰", "用水高峰", "需求大", "激增"]
    ),
    
    "off_peak": ScenarioDefinition(
        category=ScenarioCategory.NORMAL,
        sub_scenario=NormalScenario.OFF_PEAK,
        name="低谷期",
        description="用水低谷，需求骤降",
        level_range=(2.5, 3.5),
        flow_range=(1.0, 3.0),
        demand_range=(1.0, 3.0),
        control_priority="normal",
        keywords=["低谷", "夜间", "需求小", "骤降"]
    ),
    
    "weekend_mode": ScenarioDefinition(
        category=ScenarioCategory.NORMAL,
        sub_scenario=NormalScenario.WEEKEND_MODE,
        name="周末模式",
        description="周末用水模式，需求波动",
        level_range=(2.5, 3.5),
        flow_range=(3.0, 8.0),
        demand_range=(3.0, 8.0),
        control_priority="normal",
        keywords=["周末", "休息日", "假日"]
    ),
    
    # 防洪场景
    "pre_flood": ScenarioDefinition(
        category=ScenarioCategory.FLOOD_CONTROL,
        sub_scenario=FloodScenario.PRE_FLOOD,
        name="洪前准备",
        description="预报有洪水，提前腾空",
        level_range=(1.0, 2.5),
        flow_range=(5.0, 15.0),
        demand_range=(3.0, 8.0),
        control_priority="high",
        safety_margin=1.0,
        keywords=["洪水预报", "降雨", "防汛", "腾空"]
    ),
    
    "flood_peak": ScenarioDefinition(
        category=ScenarioCategory.FLOOD_CONTROL,
        sub_scenario=FloodScenario.FLOOD_PEAK,
        name="洪峰期",
        description="洪峰来临，全力泄洪",
        level_range=(4.0, 7.0),
        flow_range=(15.0, 30.0),
        demand_range=(0.0, 5.0),
        control_priority="critical",
        safety_margin=0.3,
        max_flow_change_rate=5.0,
        keywords=["洪峰", "泄洪", "暴雨", "紧急"]
    ),
    
    "post_flood": ScenarioDefinition(
        category=ScenarioCategory.FLOOD_CONTROL,
        sub_scenario=FloodScenario.POST_FLOOD,
        name="洪后恢复",
        description="洪水退去，恢复正常",
        level_range=(2.0, 4.0),
        flow_range=(5.0, 10.0),
        demand_range=(3.0, 6.0),
        control_priority="high",
        keywords=["洪后", "恢复", "退水"]
    ),
    
    # 干旱场景
    "water_shortage": ScenarioDefinition(
        category=ScenarioCategory.DROUGHT,
        sub_scenario=DroughtScenario.WATER_SHORTAGE,
        name="水源短缺",
        description="来水不足，限制供水",
        level_range=(1.0, 2.5),
        flow_range=(1.0, 4.0),
        demand_range=(3.0, 8.0),
        control_priority="high",
        keywords=["干旱", "缺水", "来水不足", "限水"]
    ),
    
    "priority_allocation": ScenarioDefinition(
        category=ScenarioCategory.DROUGHT,
        sub_scenario=DroughtScenario.PRIORITY_ALLOCATION,
        name="优先分配",
        description="水量有限，优先保证重点",
        level_range=(1.5, 2.5),
        flow_range=(2.0, 5.0),
        demand_range=(5.0, 10.0),
        control_priority="critical",
        keywords=["优先", "分级供水", "重点保障"]
    ),
    
    # 冰期场景
    "freezing": ScenarioDefinition(
        category=ScenarioCategory.ICE_PERIOD,
        sub_scenario=IceScenario.FREEZING,
        name="结冰期",
        description="气温骤降，开始结冰",
        level_range=(2.0, 3.0),
        flow_range=(3.0, 6.0),
        demand_range=(3.0, 6.0),
        control_priority="high",
        max_flow_change_rate=1.0,
        keywords=["结冰", "气温低", "冰冻"]
    ),
    
    "stable_ice": ScenarioDefinition(
        category=ScenarioCategory.ICE_PERIOD,
        sub_scenario=IceScenario.STABLE_ICE,
        name="稳定冰期",
        description="冰层稳定，维持输水",
        level_range=(2.5, 3.5),
        flow_range=(3.0, 5.0),
        demand_range=(3.0, 5.0),
        control_priority="normal",
        max_flow_change_rate=0.5,
        keywords=["冰期", "稳定", "冬季"]
    ),
    
    "thawing": ScenarioDefinition(
        category=ScenarioCategory.ICE_PERIOD,
        sub_scenario=IceScenario.THAWING,
        name="融冰期",
        description="气温回升，冰层融化",
        level_range=(2.5, 4.0),
        flow_range=(4.0, 8.0),
        demand_range=(3.0, 6.0),
        control_priority="high",
        max_flow_change_rate=1.5,
        keywords=["融冰", "解冻", "春季"]
    ),
    
    # 应急场景
    "pollution_detected": ScenarioDefinition(
        category=ScenarioCategory.EMERGENCY,
        sub_scenario=EmergencyScenario.POLLUTION_DETECTED,
        name="污染检测",
        description="检测到水质污染，紧急处理",
        level_range=(0.5, 5.0),
        flow_range=(0.0, 10.0),
        demand_range=(0.0, 5.0),
        control_priority="critical",
        keywords=["污染", "水质", "异常", "检测"]
    ),
    
    "equipment_failure": ScenarioDefinition(
        category=ScenarioCategory.EMERGENCY,
        sub_scenario=EmergencyScenario.EQUIPMENT_FAILURE,
        name="设备故障",
        description="关键设备故障，启用备用",
        level_range=(1.0, 4.0),
        flow_range=(2.0, 10.0),
        demand_range=(3.0, 8.0),
        control_priority="critical",
        keywords=["故障", "设备", "异常", "失效"]
    ),
    
    "pipe_burst": ScenarioDefinition(
        category=ScenarioCategory.EMERGENCY,
        sub_scenario=EmergencyScenario.PIPE_BURST,
        name="管道爆裂",
        description="管道破裂，紧急隔离",
        level_range=(0.5, 3.0),
        flow_range=(0.0, 5.0),
        demand_range=(2.0, 8.0),
        control_priority="critical",
        keywords=["爆裂", "破裂", "泄漏", "管道"]
    ),
    
    # 维护场景
    "planned_maintenance": ScenarioDefinition(
        category=ScenarioCategory.MAINTENANCE,
        sub_scenario=MaintenanceScenario.PLANNED_MAINTENANCE,
        name="计划维护",
        description="计划内维护，部分停水",
        level_range=(1.5, 3.0),
        flow_range=(1.0, 5.0),
        demand_range=(2.0, 6.0),
        control_priority="high",
        keywords=["维护", "检修", "保养", "计划"]
    ),
    
    "emergency_repair": ScenarioDefinition(
        category=ScenarioCategory.MAINTENANCE,
        sub_scenario=MaintenanceScenario.EMERGENCY_REPAIR,
        name="紧急维修",
        description="紧急维修，快速恢复",
        level_range=(1.0, 3.0),
        flow_range=(0.0, 5.0),
        demand_range=(2.0, 8.0),
        control_priority="critical",
        keywords=["紧急", "维修", "抢修", "故障"]
    ),
}


def get_scenario(scenario_id: str) -> Optional[ScenarioDefinition]:
    """获取场景定义"""
    return SCENARIO_LIBRARY.get(scenario_id)


def get_scenarios_by_category(category: ScenarioCategory) -> List[ScenarioDefinition]:
    """获取某类别的所有场景"""
    return [s for s in SCENARIO_LIBRARY.values() if s.category == category]


def get_all_scenarios() -> List[ScenarioDefinition]:
    """获取所有场景"""
    return list(SCENARIO_LIBRARY.values())


def get_scenario_names() -> List[str]:
    """获取所有场景名称"""
    return list(SCENARIO_LIBRARY.keys())


# 示例使用
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("场景库统计")
    logger.info("="*60)
    
    logger.info(f"\n总场景数: {len(SCENARIO_LIBRARY)}")
    
    for category in ScenarioCategory:
        scenarios = get_scenarios_by_category(category)
        logger.info(f"\n{category.value}: {len(scenarios)}个")
        for s in scenarios:
            logger.info(f"  - {s.name}: {s.description}")
    
    logger.info("\n" + "="*60)
    logger.info("场景示例")
    logger.info("="*60)
    
    scenario = get_scenario("peak_demand")
    if scenario:
        logger.info(f"\n场景: {scenario.name}")
        logger.info(f"描述: {scenario.description}")
        logger.info(f"水位范围: {scenario.level_range}")
        logger.info(f"流量范围: {scenario.flow_range}")
        logger.info(f"优先级: {scenario.control_priority}")
        logger.info(f"关键词: {', '.join(scenario.keywords)}")

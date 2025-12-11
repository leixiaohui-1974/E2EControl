"""
全场景定义 (Complete Scenario Definitions)
南水北调中线1432公里渠道的所有运行场景

场景覆盖矩阵:
- 6大类场景
- 47个细分场景
- 覆盖所有等级(L0-L4)的测试需求
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ScenarioCategory(Enum):
    """场景大类"""
    NORMAL_OPERATION = "normal"           # 正常运行
    WATER_DEMAND = "demand"               # 需水调度
    EXTREME_WEATHER = "weather"           # 极端天气
    EQUIPMENT_FAILURE = "failure"         # 设备故障
    EMERGENCY = "emergency"               # 应急处置
    MAINTENANCE = "maintenance"           # 维护检修


class ScenarioSeverity(Enum):
    """场景严重程度"""
    LOW = 1           # 轻微
    MEDIUM = 2        # 中等
    HIGH = 3          # 严重
    CRITICAL = 4      # 危急


class RequiredLevel(Enum):
    """所需自动化等级"""
    L0_SUFFICIENT = 0    # L0即可处理
    L1_RECOMMENDED = 1   # 建议L1
    L2_REQUIRED = 2      # 需要L2
    L3_REQUIRED = 3      # 需要L3
    L4_REQUIRED = 4      # 需要L4


@dataclass
class ScenarioParameters:
    """场景参数"""
    # 水文参数
    inflow_range: Tuple[float, float] = (200.0, 350.0)      # 入流范围 m³/s
    level_range: Tuple[float, float] = (3.5, 4.5)           # 水位范围 m
    demand_range: Tuple[float, float] = (0.0, 50.0)         # 分水需求 m³/s

    # 时间参数
    duration_hours: float = 24.0                             # 持续时间
    ramp_time_hours: float = 1.0                            # 变化时间

    # 空间参数
    affected_pools: List[int] = field(default_factory=list) # 影响的渠池
    propagation_speed: float = 1.5                          # 传播速度 m/s

    # 扰动参数
    disturbance_magnitude: float = 0.0                      # 扰动幅度
    noise_level: float = 0.01                               # 噪声水平


@dataclass
class Scenario:
    """场景定义"""
    id: str
    name: str
    name_cn: str
    category: ScenarioCategory
    severity: ScenarioSeverity
    required_level: RequiredLevel
    description: str
    parameters: ScenarioParameters

    # 场景特征
    is_predictable: bool = True                # 是否可预测
    is_gradual: bool = True                    # 是否渐变
    requires_coordination: bool = False        # 是否需要多闸协调

    # 测试要求
    test_duration_min: int = 60                # 最小测试时长(分钟)
    success_criteria: Dict = field(default_factory=dict)

    # 响应要求
    max_response_time_s: float = 300.0         # 最大响应时间(秒)
    max_level_deviation_m: float = 0.3         # 最大水位偏差(米)


# ==============================================================================
# 完整场景矩阵
# ==============================================================================

COMPLETE_SCENARIO_MATRIX: Dict[str, Scenario] = {}

# -----------------------------------------------------------------------------
# 1. 正常运行场景 (8个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["NORMAL_001"] = Scenario(
    id="NORMAL_001",
    name="Steady State Operation",
    name_cn="稳态运行",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L0_SUFFICIENT,
    description="常规稳态运行，入流出流基本平衡",
    parameters=ScenarioParameters(
        inflow_range=(280.0, 320.0),
        level_range=(3.8, 4.2),
        duration_hours=24.0
    ),
    is_predictable=True,
    is_gradual=True,
    success_criteria={
        'level_stability': 0.1,      # 水位波动<0.1m
        'flow_balance': 0.95         # 流量平衡度>95%
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_002"] = Scenario(
    id="NORMAL_002",
    name="Daily Demand Variation",
    name_cn="日常需水波动",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L1_RECOMMENDED,
    description="24小时周期性需水变化",
    parameters=ScenarioParameters(
        inflow_range=(250.0, 350.0),
        level_range=(3.5, 4.5),
        demand_range=(10.0, 40.0),
        duration_hours=48.0
    ),
    is_predictable=True,
    is_gradual=True,
    success_criteria={
        'level_tracking': 0.15,
        'demand_satisfaction': 0.98
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_003"] = Scenario(
    id="NORMAL_003",
    name="Seasonal Flow Adjustment",
    name_cn="季节性流量调整",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L2_REQUIRED,
    description="季节变化导致的流量大幅调整",
    parameters=ScenarioParameters(
        inflow_range=(150.0, 400.0),
        level_range=(3.0, 5.0),
        duration_hours=168.0,  # 一周
        ramp_time_hours=24.0
    ),
    is_predictable=True,
    is_gradual=True,
    requires_coordination=True,
    success_criteria={
        'smooth_transition': 0.2,
        'no_overflow': True
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_004"] = Scenario(
    id="NORMAL_004",
    name="Night Mode Operation",
    name_cn="夜间低负荷运行",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L2_REQUIRED,
    description="夜间低需水量运行模式",
    parameters=ScenarioParameters(
        inflow_range=(150.0, 200.0),
        level_range=(3.5, 4.0),
        demand_range=(5.0, 15.0),
        duration_hours=8.0
    ),
    success_criteria={
        'energy_efficiency': 0.9,
        'level_maintenance': 0.1
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_005"] = Scenario(
    id="NORMAL_005",
    name="Peak Demand Period",
    name_cn="高峰需水期",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="夏季农业灌溉高峰期",
    parameters=ScenarioParameters(
        inflow_range=(350.0, 420.0),
        level_range=(3.0, 4.0),
        demand_range=(40.0, 80.0),
        duration_hours=72.0
    ),
    requires_coordination=True,
    success_criteria={
        'demand_met': 0.95,
        'no_dry_pool': True
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_006"] = Scenario(
    id="NORMAL_006",
    name="Multi-point Diversion",
    name_cn="多点同时分水",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="多个分水口同时开启",
    parameters=ScenarioParameters(
        inflow_range=(300.0, 380.0),
        demand_range=(20.0, 60.0),
        affected_pools=[5, 15, 25, 35, 45, 55],
        duration_hours=12.0
    ),
    requires_coordination=True,
    success_criteria={
        'all_demands_met': True,
        'level_balance': 0.2
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_007"] = Scenario(
    id="NORMAL_007",
    name="Flow Transition",
    name_cn="流量过渡调节",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="大流量到小流量的平滑过渡",
    parameters=ScenarioParameters(
        inflow_range=(200.0, 400.0),
        ramp_time_hours=2.0,
        duration_hours=6.0
    ),
    is_gradual=True,
    success_criteria={
        'no_overshoot': 0.15,
        'settling_time': 3600
    }
)

COMPLETE_SCENARIO_MATRIX["NORMAL_008"] = Scenario(
    id="NORMAL_008",
    name="Optimal Energy Operation",
    name_cn="最优能耗运行",
    category=ScenarioCategory.NORMAL_OPERATION,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L4_REQUIRED,
    description="在满足需水的同时最小化能耗",
    parameters=ScenarioParameters(
        inflow_range=(250.0, 350.0),
        duration_hours=24.0
    ),
    success_criteria={
        'energy_saving': 0.1,
        'demand_satisfaction': 0.98
    }
)

# -----------------------------------------------------------------------------
# 2. 需水调度场景 (8个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["DEMAND_001"] = Scenario(
    id="DEMAND_001",
    name="Planned Large Diversion",
    name_cn="计划大流量分水",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="按计划向下游大量供水",
    parameters=ScenarioParameters(
        demand_range=(50.0, 100.0),
        affected_pools=[30, 31, 32],
        duration_hours=24.0,
        ramp_time_hours=2.0
    ),
    is_predictable=True,
    requires_coordination=True,
    success_criteria={
        'delivery_accuracy': 0.95,
        'upstream_stability': 0.2
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_002"] = Scenario(
    id="DEMAND_002",
    name="Emergency Water Supply",
    name_cn="应急供水",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="突发用水需求的紧急响应",
    parameters=ScenarioParameters(
        demand_range=(30.0, 80.0),
        duration_hours=4.0,
        ramp_time_hours=0.25
    ),
    is_predictable=False,
    is_gradual=False,
    max_response_time_s=120.0,
    success_criteria={
        'response_time': 120,
        'supply_achieved': 0.9
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_003"] = Scenario(
    id="DEMAND_003",
    name="Demand Reduction",
    name_cn="需水骤减",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="下游突然减少用水",
    parameters=ScenarioParameters(
        demand_range=(-50.0, -20.0),
        ramp_time_hours=0.5,
        duration_hours=6.0
    ),
    is_predictable=False,
    success_criteria={
        'no_overflow': True,
        'level_control': 0.3
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_004"] = Scenario(
    id="DEMAND_004",
    name="Cascade Diversion",
    name_cn="级联分水",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="多个分水口依次开启",
    parameters=ScenarioParameters(
        demand_range=(20.0, 40.0),
        affected_pools=list(range(10, 50, 5)),
        duration_hours=12.0
    ),
    requires_coordination=True,
    success_criteria={
        'all_served': True,
        'cascade_stable': True
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_005"] = Scenario(
    id="DEMAND_005",
    name="Priority Allocation",
    name_cn="优先级分配",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L4_REQUIRED,
    description="水量不足时的优先级分配",
    parameters=ScenarioParameters(
        inflow_range=(150.0, 200.0),
        demand_range=(60.0, 100.0),
        duration_hours=24.0
    ),
    success_criteria={
        'priority_respected': True,
        'fairness_index': 0.8
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_006"] = Scenario(
    id="DEMAND_006",
    name="Dynamic Reallocation",
    name_cn="动态重分配",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L4_REQUIRED,
    description="根据实时需求动态调整分配",
    parameters=ScenarioParameters(
        demand_range=(30.0, 70.0),
        duration_hours=8.0
    ),
    is_predictable=False,
    success_criteria={
        'adaptation_time': 600,
        'efficiency': 0.9
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_007"] = Scenario(
    id="DEMAND_007",
    name="Industrial Peak Demand",
    name_cn="工业用水高峰",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="工业区集中用水时段",
    parameters=ScenarioParameters(
        demand_range=(40.0, 70.0),
        affected_pools=[20, 21, 22, 23],
        duration_hours=10.0
    ),
    success_criteria={
        'supply_pressure': 'maintained',
        'quality_standard': True
    }
)

COMPLETE_SCENARIO_MATRIX["DEMAND_008"] = Scenario(
    id="DEMAND_008",
    name="Agricultural Irrigation",
    name_cn="农业灌溉调度",
    category=ScenarioCategory.WATER_DEMAND,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="大规模农业灌溉需水",
    parameters=ScenarioParameters(
        demand_range=(50.0, 120.0),
        affected_pools=list(range(40, 60)),
        duration_hours=72.0
    ),
    is_predictable=True,
    requires_coordination=True,
    success_criteria={
        'irrigation_completed': True,
        'no_waste': 0.95
    }
)

# -----------------------------------------------------------------------------
# 3. 极端天气场景 (8个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["WEATHER_001"] = Scenario(
    id="WEATHER_001",
    name="Heavy Rainfall",
    name_cn="暴雨",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="暴雨导致入流急剧增加",
    parameters=ScenarioParameters(
        inflow_range=(400.0, 500.0),
        disturbance_magnitude=0.5,
        duration_hours=6.0,
        ramp_time_hours=0.5
    ),
    is_predictable=True,
    is_gradual=False,
    success_criteria={
        'no_overflow': True,
        'flood_control': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_002"] = Scenario(
    id="WEATHER_002",
    name="Flash Flood",
    name_cn="山洪",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="上游山洪导致突发大流量",
    parameters=ScenarioParameters(
        inflow_range=(450.0, 600.0),
        ramp_time_hours=0.25,
        duration_hours=4.0,
        disturbance_magnitude=0.8
    ),
    is_predictable=False,
    is_gradual=False,
    max_response_time_s=60.0,
    success_criteria={
        'emergency_discharge': True,
        'no_breach': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_003"] = Scenario(
    id="WEATHER_003",
    name="Drought",
    name_cn="干旱",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="持续干旱导致入流减少",
    parameters=ScenarioParameters(
        inflow_range=(80.0, 150.0),
        duration_hours=720.0,  # 30天
        ramp_time_hours=72.0
    ),
    is_predictable=True,
    success_criteria={
        'water_conservation': 0.3,
        'priority_supply': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_004"] = Scenario(
    id="WEATHER_004",
    name="Ice Formation",
    name_cn="冰冻",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="低温导致渠道结冰",
    parameters=ScenarioParameters(
        inflow_range=(100.0, 200.0),
        noise_level=0.05,
        duration_hours=168.0
    ),
    success_criteria={
        'flow_maintained': True,
        'no_ice_dam': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_005"] = Scenario(
    id="WEATHER_005",
    name="Strong Wind",
    name_cn="大风",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="大风导致水位波动",
    parameters=ScenarioParameters(
        level_range=(3.0, 5.0),
        noise_level=0.1,
        duration_hours=12.0
    ),
    success_criteria={
        'level_stability': 0.3,
        'sensor_compensation': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_006"] = Scenario(
    id="WEATHER_006",
    name="Heat Wave",
    name_cn="高温热浪",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="高温导致蒸发增加和用水需求增加",
    parameters=ScenarioParameters(
        inflow_range=(250.0, 350.0),
        demand_range=(60.0, 100.0),
        duration_hours=168.0
    ),
    success_criteria={
        'demand_satisfaction': 0.95,
        'evaporation_compensation': True
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_007"] = Scenario(
    id="WEATHER_007",
    name="Continuous Rain",
    name_cn="连续降雨",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="连续多日降雨",
    parameters=ScenarioParameters(
        inflow_range=(350.0, 420.0),
        duration_hours=120.0,
        disturbance_magnitude=0.2
    ),
    success_criteria={
        'gradual_discharge': True,
        'level_control': 0.3
    }
)

COMPLETE_SCENARIO_MATRIX["WEATHER_008"] = Scenario(
    id="WEATHER_008",
    name="Sandstorm",
    name_cn="沙尘暴",
    category=ScenarioCategory.EXTREME_WEATHER,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="沙尘暴影响传感器和水质",
    parameters=ScenarioParameters(
        noise_level=0.15,
        duration_hours=24.0
    ),
    success_criteria={
        'sensor_filtering': True,
        'operation_continuity': True
    }
)

# -----------------------------------------------------------------------------
# 4. 设备故障场景 (8个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["FAILURE_001"] = Scenario(
    id="FAILURE_001",
    name="Gate Stuck",
    name_cn="闸门卡死",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="单个闸门卡死无法动作",
    parameters=ScenarioParameters(
        affected_pools=[25],
        duration_hours=4.0
    ),
    is_predictable=False,
    requires_coordination=True,
    success_criteria={
        'bypass_achieved': True,
        'level_maintained': 0.3
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_002"] = Scenario(
    id="FAILURE_002",
    name="Multiple Gate Failure",
    name_cn="多闸门故障",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="连续多个闸门同时故障",
    parameters=ScenarioParameters(
        affected_pools=[20, 21, 22],
        duration_hours=6.0
    ),
    is_predictable=False,
    requires_coordination=True,
    success_criteria={
        'emergency_mode': True,
        'no_cascade_failure': True
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_003"] = Scenario(
    id="FAILURE_003",
    name="Sensor Failure",
    name_cn="传感器故障",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="水位传感器数据异常",
    parameters=ScenarioParameters(
        affected_pools=[15, 16],
        noise_level=0.5,
        duration_hours=2.0
    ),
    success_criteria={
        'fault_detection': 60,
        'fallback_control': True
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_004"] = Scenario(
    id="FAILURE_004",
    name="Communication Loss",
    name_cn="通信中断",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="部分站点通信中断",
    parameters=ScenarioParameters(
        affected_pools=list(range(30, 40)),
        duration_hours=1.0
    ),
    success_criteria={
        'autonomous_operation': True,
        'safe_mode': True
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_005"] = Scenario(
    id="FAILURE_005",
    name="Power Outage",
    name_cn="停电",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L3_REQUIRED,
    description="区域性停电",
    parameters=ScenarioParameters(
        affected_pools=list(range(25, 35)),
        duration_hours=2.0
    ),
    success_criteria={
        'ups_activation': True,
        'graceful_degradation': True
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_006"] = Scenario(
    id="FAILURE_006",
    name="Pump Failure",
    name_cn="泵站故障",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="加压泵站故障",
    parameters=ScenarioParameters(
        affected_pools=[45, 46],
        inflow_range=(100.0, 200.0),
        duration_hours=4.0
    ),
    success_criteria={
        'backup_activation': True,
        'flow_maintained': 0.8
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_007"] = Scenario(
    id="FAILURE_007",
    name="Control System Failure",
    name_cn="控制系统故障",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L0_SUFFICIENT,
    description="自动控制系统故障，需人工接管",
    parameters=ScenarioParameters(
        duration_hours=1.0
    ),
    success_criteria={
        'manual_takeover': 60,
        'no_incident': True
    }
)

COMPLETE_SCENARIO_MATRIX["FAILURE_008"] = Scenario(
    id="FAILURE_008",
    name="Cyber Attack",
    name_cn="网络攻击",
    category=ScenarioCategory.EQUIPMENT_FAILURE,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L3_REQUIRED,
    description="检测到异常控制指令",
    parameters=ScenarioParameters(
        duration_hours=0.5
    ),
    success_criteria={
        'attack_detected': 30,
        'isolation_complete': 60,
        'safe_state': True
    }
)

# -----------------------------------------------------------------------------
# 5. 应急处置场景 (8个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["EMERGENCY_001"] = Scenario(
    id="EMERGENCY_001",
    name="Canal Breach",
    name_cn="渠道溃口",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="渠道发生溃口",
    parameters=ScenarioParameters(
        affected_pools=[35],
        duration_hours=0.5
    ),
    is_predictable=False,
    max_response_time_s=30.0,
    success_criteria={
        'emergency_closure': 30,
        'upstream_protection': True,
        'downstream_warning': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_002"] = Scenario(
    id="EMERGENCY_002",
    name="Water Contamination",
    name_cn="水质污染",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="检测到水质污染",
    parameters=ScenarioParameters(
        affected_pools=[20],
        duration_hours=24.0
    ),
    success_criteria={
        'isolation': 300,
        'contamination_contained': True,
        'clean_water_supply': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_003"] = Scenario(
    id="EMERGENCY_003",
    name="Earthquake",
    name_cn="地震",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="地震后的紧急响应",
    parameters=ScenarioParameters(
        affected_pools=list(range(0, 63)),
        duration_hours=1.0
    ),
    is_predictable=False,
    success_criteria={
        'structural_check': True,
        'safe_shutdown': 120,
        'damage_assessment': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_004"] = Scenario(
    id="EMERGENCY_004",
    name="Upstream Dam Release",
    name_cn="上游水库泄洪",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="上游水库突发泄洪",
    parameters=ScenarioParameters(
        inflow_range=(500.0, 700.0),
        ramp_time_hours=0.5,
        duration_hours=6.0
    ),
    is_predictable=True,
    max_response_time_s=300.0,
    success_criteria={
        'pre_emptive_discharge': True,
        'no_overflow': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_005"] = Scenario(
    id="EMERGENCY_005",
    name="Fire Near Canal",
    name_cn="渠道周边火灾",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L2_REQUIRED,
    description="需要紧急供水灭火",
    parameters=ScenarioParameters(
        demand_range=(100.0, 200.0),
        affected_pools=[40],
        duration_hours=4.0
    ),
    success_criteria={
        'emergency_supply': True,
        'pressure_maintained': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_006"] = Scenario(
    id="EMERGENCY_006",
    name="Hazmat Spill",
    name_cn="危化品泄漏",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="危化品进入渠道",
    parameters=ScenarioParameters(
        affected_pools=[25, 26, 27],
        duration_hours=12.0
    ),
    success_criteria={
        'flow_reversal': True,
        'containment': True,
        'downstream_protection': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_007"] = Scenario(
    id="EMERGENCY_007",
    name="Mass Evacuation Support",
    name_cn="大规模应急供水",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="突发事件导致的大规模应急供水",
    parameters=ScenarioParameters(
        demand_range=(80.0, 150.0),
        affected_pools=list(range(30, 50)),
        duration_hours=48.0
    ),
    success_criteria={
        'supply_maintained': True,
        'priority_areas_served': True
    }
)

COMPLETE_SCENARIO_MATRIX["EMERGENCY_008"] = Scenario(
    id="EMERGENCY_008",
    name="Simultaneous Emergencies",
    name_cn="多重应急",
    category=ScenarioCategory.EMERGENCY,
    severity=ScenarioSeverity.CRITICAL,
    required_level=RequiredLevel.L4_REQUIRED,
    description="同时发生多个紧急情况",
    parameters=ScenarioParameters(
        affected_pools=[10, 30, 50],
        duration_hours=6.0
    ),
    success_criteria={
        'priority_handling': True,
        'resource_allocation': True,
        'no_cascade': True
    }
)

# -----------------------------------------------------------------------------
# 6. 维护检修场景 (7个)
# -----------------------------------------------------------------------------

COMPLETE_SCENARIO_MATRIX["MAINT_001"] = Scenario(
    id="MAINT_001",
    name="Planned Gate Maintenance",
    name_cn="计划闸门检修",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L2_REQUIRED,
    description="按计划进行闸门检修",
    parameters=ScenarioParameters(
        affected_pools=[15],
        duration_hours=8.0
    ),
    is_predictable=True,
    success_criteria={
        'flow_rerouted': True,
        'maintenance_window': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_002"] = Scenario(
    id="MAINT_002",
    name="Canal Section Dewatering",
    name_cn="渠段放空",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="放空渠段进行检修",
    parameters=ScenarioParameters(
        affected_pools=[20, 21, 22],
        duration_hours=72.0
    ),
    is_predictable=True,
    requires_coordination=True,
    success_criteria={
        'gradual_dewatering': True,
        'bypass_established': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_003"] = Scenario(
    id="MAINT_003",
    name="Sensor Calibration",
    name_cn="传感器校准",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L2_REQUIRED,
    description="传感器定期校准",
    parameters=ScenarioParameters(
        affected_pools=[10],
        duration_hours=2.0
    ),
    success_criteria={
        'calibration_complete': True,
        'backup_sensing': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_004"] = Scenario(
    id="MAINT_004",
    name="Control System Upgrade",
    name_cn="控制系统升级",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="控制系统软件升级",
    parameters=ScenarioParameters(
        duration_hours=4.0
    ),
    is_predictable=True,
    success_criteria={
        'zero_downtime': True,
        'rollback_ready': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_005"] = Scenario(
    id="MAINT_005",
    name="Annual Inspection",
    name_cn="年度检查",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.LOW,
    required_level=RequiredLevel.L2_REQUIRED,
    description="全线年度检查",
    parameters=ScenarioParameters(
        affected_pools=list(range(0, 63)),
        duration_hours=720.0  # 30天
    ),
    is_predictable=True,
    success_criteria={
        'inspection_complete': True,
        'issues_logged': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_006"] = Scenario(
    id="MAINT_006",
    name="Emergency Repair",
    name_cn="紧急维修",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.HIGH,
    required_level=RequiredLevel.L3_REQUIRED,
    description="发现问题后的紧急维修",
    parameters=ScenarioParameters(
        affected_pools=[35],
        duration_hours=12.0
    ),
    is_predictable=False,
    success_criteria={
        'repair_complete': True,
        'minimal_disruption': True
    }
)

COMPLETE_SCENARIO_MATRIX["MAINT_007"] = Scenario(
    id="MAINT_007",
    name="Capacity Expansion",
    name_cn="扩容改造",
    category=ScenarioCategory.MAINTENANCE,
    severity=ScenarioSeverity.MEDIUM,
    required_level=RequiredLevel.L3_REQUIRED,
    description="渠道扩容改造期间的运行",
    parameters=ScenarioParameters(
        affected_pools=list(range(40, 45)),
        inflow_range=(200.0, 300.0),
        duration_hours=2160.0  # 90天
    ),
    is_predictable=True,
    requires_coordination=True,
    success_criteria={
        'continuous_operation': True,
        'safety_maintained': True
    }
)


# ==============================================================================
# 场景库管理
# ==============================================================================

class ScenarioLibrary:
    """场景库管理器"""

    def __init__(self):
        self.scenarios = COMPLETE_SCENARIO_MATRIX.copy()
        self._build_indices()

    def _build_indices(self):
        """构建索引"""
        self.by_category = {}
        self.by_severity = {}
        self.by_level = {}

        for sid, scenario in self.scenarios.items():
            # 按类别
            cat = scenario.category
            if cat not in self.by_category:
                self.by_category[cat] = []
            self.by_category[cat].append(sid)

            # 按严重程度
            sev = scenario.severity
            if sev not in self.by_severity:
                self.by_severity[sev] = []
            self.by_severity[sev].append(sid)

            # 按所需等级
            level = scenario.required_level
            if level not in self.by_level:
                self.by_level[level] = []
            self.by_level[level].append(sid)

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """获取场景"""
        return self.scenarios.get(scenario_id)

    def get_by_category(self, category: ScenarioCategory) -> List[Scenario]:
        """按类别获取场景"""
        ids = self.by_category.get(category, [])
        return [self.scenarios[sid] for sid in ids]

    def get_by_level(self, level: RequiredLevel) -> List[Scenario]:
        """按等级获取场景"""
        ids = self.by_level.get(level, [])
        return [self.scenarios[sid] for sid in ids]

    def get_coverage_matrix(self) -> Dict:
        """获取覆盖矩阵"""
        matrix = {}
        for cat in ScenarioCategory:
            matrix[cat.value] = {}
            for level in RequiredLevel:
                count = sum(
                    1 for sid in self.by_category.get(cat, [])
                    if self.scenarios[sid].required_level == level
                )
                matrix[cat.value][level.name] = count
        return matrix

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_scenarios': len(self.scenarios),
            'by_category': {cat.value: len(ids) for cat, ids in self.by_category.items()},
            'by_severity': {sev.name: len(ids) for sev, ids in self.by_severity.items()},
            'by_level': {level.name: len(ids) for level, ids in self.by_level.items()},
            'coverage_matrix': self.get_coverage_matrix()
        }

    def list_all(self) -> List[Dict]:
        """列出所有场景"""
        return [
            {
                'id': s.id,
                'name': s.name,
                'name_cn': s.name_cn,
                'category': s.category.value,
                'severity': s.severity.name,
                'required_level': s.required_level.name
            }
            for s in self.scenarios.values()
        ]


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("场景库统计")
    print("=" * 70)

    library = ScenarioLibrary()
    stats = library.get_statistics()

    print(f"\n总场景数: {stats['total_scenarios']}")

    print("\n按类别分布:")
    for cat, count in stats['by_category'].items():
        print(f"  {cat}: {count}")

    print("\n按严重程度分布:")
    for sev, count in stats['by_severity'].items():
        print(f"  {sev}: {count}")

    print("\n按所需等级分布:")
    for level, count in stats['by_level'].items():
        print(f"  {level}: {count}")

    print("\n覆盖矩阵:")
    matrix = stats['coverage_matrix']
    levels = [l.name for l in RequiredLevel]
    print("         ", "  ".join(f"{l:>12}" for l in levels))
    for cat, data in matrix.items():
        row = "  ".join(f"{data[l]:>12}" for l in levels)
        print(f"{cat:>8}: {row}")

    print("\n" + "=" * 70)

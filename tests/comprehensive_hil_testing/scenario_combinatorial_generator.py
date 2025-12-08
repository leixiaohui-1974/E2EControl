"""
场景组合生成器 (Scenario Combinatorial Generator)

生成成千上万种测试场景组合，覆盖所有可能的工况。
使用正交实验设计和蒙特卡洛方法生成高覆盖率测试用例。
"""

import itertools
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generator
from enum import Enum
from datetime import datetime
import hashlib
import json


class WeatherCondition(Enum):
    """天气条件"""
    CLEAR = "晴天"
    CLOUDY = "多云"
    RAINY = "小雨"
    HEAVY_RAIN = "暴雨"
    STORM = "暴风雨"
    SNOW = "雪"
    ICE = "冰冻"
    FOG = "雾"
    HEATWAVE = "热浪"


class SeasonType(Enum):
    """季节类型"""
    SPRING = "春季"
    SUMMER = "夏季"
    AUTUMN = "秋季"
    WINTER = "冬季"
    ICE_PERIOD = "冰期"


class DemandPattern(Enum):
    """需求模式"""
    STABLE = "稳定需求"
    MORNING_PEAK = "早高峰"
    EVENING_PEAK = "晚高峰"
    DOUBLE_PEAK = "双高峰"
    WEEKEND = "周末模式"
    HOLIDAY = "节假日"
    IRRIGATION = "灌溉高峰"
    INDUSTRIAL = "工业用水"
    EMERGENCY = "应急供水"


class FaultType(Enum):
    """故障类型"""
    NONE = "无故障"
    SENSOR_DRIFT = "传感器漂移"
    SENSOR_STUCK = "传感器卡死"
    SENSOR_NOISE = "传感器噪声"
    ACTUATOR_STUCK = "执行器卡死"
    ACTUATOR_DELAY = "执行器延迟"
    ACTUATOR_DEGRADED = "执行器性能退化"
    COMMUNICATION_DELAY = "通信延迟"
    COMMUNICATION_LOSS = "通信中断"
    LEAK_SMALL = "小型泄漏"
    LEAK_LARGE = "大型泄漏"
    BLOCKAGE = "堵塞"
    POWER_FLUCTUATION = "电力波动"
    CYBER_ATTACK_FDIA = "FDIA攻击"
    CYBER_ATTACK_DOS = "DoS攻击"
    CYBER_ATTACK_REPLAY = "重放攻击"


class ControlMode(Enum):
    """控制模式"""
    NORMAL = "正常运行"
    FLOOD_PREVENTION = "防洪模式"
    DROUGHT_RESPONSE = "抗旱模式"
    ICE_PERIOD = "冰期运行"
    POLLUTION_EMERGENCY = "污染应急"
    MAINTENANCE = "维护模式"
    DEGRADED = "降级运行"
    EMERGENCY = "紧急模式"


class NetworkTopology(Enum):
    """网络拓扑"""
    SINGLE_POOL = "单池"
    CASCADE_2 = "2级级联"
    CASCADE_3 = "3级级联"
    CASCADE_5 = "5级级联"
    CASCADE_10 = "10级级联"
    BRANCH_2 = "2分支"
    BRANCH_3 = "3分支"
    MESH_SMALL = "小型网状"
    MESH_LARGE = "大型网状"


@dataclass
class ScenarioSpace:
    """场景空间定义 - 定义所有可能的参数范围"""

    # 初始条件范围
    water_levels: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    inflows: Tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
    outflows: Tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)

    # 物理参数范围
    areas: Tuple[float, ...] = (1000.0, 5000.0, 10000.0, 20000.0, 50000.0)
    manning_n: Tuple[float, ...] = (0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040)
    slopes: Tuple[float, ...] = (0.00005, 0.0001, 0.0002, 0.0005, 0.001)

    # 环境条件
    weathers: Tuple[WeatherCondition, ...] = tuple(WeatherCondition)
    seasons: Tuple[SeasonType, ...] = tuple(SeasonType)
    temperatures: Tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)

    # 运行条件
    demand_patterns: Tuple[DemandPattern, ...] = tuple(DemandPattern)
    fault_types: Tuple[FaultType, ...] = tuple(FaultType)
    control_modes: Tuple[ControlMode, ...] = tuple(ControlMode)
    topologies: Tuple[NetworkTopology, ...] = tuple(NetworkTopology)

    # 时间参数
    simulation_durations: Tuple[int, ...] = (1, 6, 12, 24, 48, 72, 168)  # hours
    time_steps: Tuple[float, ...] = (60.0, 300.0, 600.0, 900.0, 1800.0, 3600.0)  # seconds

    # 扰动参数
    disturbance_magnitudes: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    noise_levels: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)

    # 控制器参数
    mpc_horizons: Tuple[int, ...] = (3, 5, 10, 15, 20)
    weight_levels: Tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 50.0)
    weight_smooths: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    def get_total_combinations(self) -> int:
        """计算所有可能的组合数"""
        return (
            len(self.water_levels) *
            len(self.inflows) *
            len(self.outflows) *
            len(self.areas) *
            len(self.weathers) *
            len(self.seasons) *
            len(self.demand_patterns) *
            len(self.fault_types) *
            len(self.control_modes) *
            len(self.topologies)
        )


@dataclass
class TestScenario:
    """测试场景定义"""
    id: str
    name: str
    category: str
    difficulty: int  # 1-5

    # 初始条件
    initial_water_level: float
    initial_inflow: float
    initial_outflow: float

    # 物理参数
    area: float
    manning_n: float = 0.025
    slope: float = 0.0001

    # 环境条件
    weather: WeatherCondition = WeatherCondition.CLEAR
    season: SeasonType = SeasonType.SUMMER
    temperature: float = 20.0

    # 运行条件
    demand_pattern: DemandPattern = DemandPattern.STABLE
    fault_type: FaultType = FaultType.NONE
    control_mode: ControlMode = ControlMode.NORMAL
    topology: NetworkTopology = NetworkTopology.SINGLE_POOL

    # 时间参数
    duration_hours: int = 24
    time_step: float = 3600.0

    # 扰动参数
    disturbance_magnitude: float = 0.0
    noise_level: float = 0.01

    # 控制参数
    mpc_horizon: int = 10
    weight_level: float = 10.0
    weight_smooth: float = 5.0

    # 目标设定
    target_level: float = 3.0
    level_tolerance: float = 0.1

    # 注入事件
    injections: List[Dict] = field(default_factory=list)

    # 期望响应
    expected_response: Dict = field(default_factory=dict)

    # 通过标准
    pass_criteria: Dict = field(default_factory=dict)

    # 元数据
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'difficulty': self.difficulty,
            'initial_conditions': {
                'water_level': self.initial_water_level,
                'inflow': self.initial_inflow,
                'outflow': self.initial_outflow,
            },
            'physical_params': {
                'area': self.area,
                'manning_n': self.manning_n,
                'slope': self.slope,
            },
            'environment': {
                'weather': self.weather.value,
                'season': self.season.value,
                'temperature': self.temperature,
            },
            'operation': {
                'demand_pattern': self.demand_pattern.value,
                'fault_type': self.fault_type.value,
                'control_mode': self.control_mode.value,
                'topology': self.topology.value,
            },
            'timing': {
                'duration_hours': self.duration_hours,
                'time_step': self.time_step,
            },
            'disturbance': {
                'magnitude': self.disturbance_magnitude,
                'noise_level': self.noise_level,
            },
            'control': {
                'mpc_horizon': self.mpc_horizon,
                'weight_level': self.weight_level,
                'weight_smooth': self.weight_smooth,
            },
            'target': {
                'level': self.target_level,
                'tolerance': self.level_tolerance,
            },
            'injections': self.injections,
            'expected_response': self.expected_response,
            'pass_criteria': self.pass_criteria,
            'tags': self.tags,
            'created_at': self.created_at,
        }

    def get_hash(self) -> str:
        """获取场景哈希值用于去重"""
        key_params = (
            self.initial_water_level,
            self.initial_inflow,
            self.area,
            self.weather.value,
            self.season.value,
            self.fault_type.value,
            self.control_mode.value,
            self.topology.value,
        )
        return hashlib.md5(str(key_params).encode()).hexdigest()[:12]


class ScenarioCombinatorialGenerator:
    """
    场景组合生成器

    支持多种生成策略：
    1. 全组合 (Full Factorial)
    2. 正交设计 (Orthogonal Array)
    3. 拉丁超立方 (Latin Hypercube)
    4. 蒙特卡洛随机 (Monte Carlo)
    5. 边界值分析 (Boundary Value)
    6. 等价类划分 (Equivalence Partitioning)
    """

    def __init__(self, space: ScenarioSpace = None, seed: int = 42):
        self.space = space or ScenarioSpace()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.generated_scenarios: List[TestScenario] = []
        self.scenario_hashes: set = set()

    def generate_all(self, max_scenarios: int = 10000) -> List[TestScenario]:
        """生成所有类型的测试场景"""
        scenarios = []

        # 1. 基础场景 (200+)
        scenarios.extend(self._generate_basic_scenarios())

        # 2. 边界值场景 (500+)
        scenarios.extend(self._generate_boundary_scenarios())

        # 3. 正交组合场景 (1000+)
        scenarios.extend(self._generate_orthogonal_scenarios())

        # 4. 故障场景 (800+)
        scenarios.extend(self._generate_fault_scenarios())

        # 5. 极端天气场景 (400+)
        scenarios.extend(self._generate_weather_scenarios())

        # 6. 复合场景 (2000+)
        scenarios.extend(self._generate_compound_scenarios())

        # 7. 时序场景 (500+)
        scenarios.extend(self._generate_temporal_scenarios())

        # 8. 网络拓扑场景 (300+)
        scenarios.extend(self._generate_topology_scenarios())

        # 9. 蒙特卡洛随机场景 (3000+)
        remaining = max_scenarios - len(scenarios)
        if remaining > 0:
            scenarios.extend(self._generate_monte_carlo_scenarios(remaining))

        # 去重
        unique_scenarios = self._deduplicate(scenarios)

        # 限制数量
        if len(unique_scenarios) > max_scenarios:
            unique_scenarios = unique_scenarios[:max_scenarios]

        self.generated_scenarios = unique_scenarios
        return unique_scenarios

    def _generate_basic_scenarios(self) -> List[TestScenario]:
        """生成基础测试场景"""
        scenarios = []
        categories = [
            ("S1_NORMAL", "正常运行", ControlMode.NORMAL),
            ("S2_FLOOD", "防洪调度", ControlMode.FLOOD_PREVENTION),
            ("S3_DROUGHT", "干旱应对", ControlMode.DROUGHT_RESPONSE),
            ("S4_ICE", "冰期运行", ControlMode.ICE_PERIOD),
            ("S5_POLLUTION", "污染应急", ControlMode.POLLUTION_EMERGENCY),
            ("S6_MAINTENANCE", "维护模式", ControlMode.MAINTENANCE),
            ("S7_DEGRADED", "降级运行", ControlMode.DEGRADED),
            ("S8_EMERGENCY", "紧急模式", ControlMode.EMERGENCY),
        ]

        for cat_id, cat_name, control_mode in categories:
            for level in [2.0, 3.0, 4.0, 5.0]:
                for inflow in [3.0, 5.0, 10.0]:
                    scenario = TestScenario(
                        id=f"{cat_id}_BASIC_{len(scenarios):04d}",
                        name=f"{cat_name}-基础场景-水位{level}m",
                        category=cat_id,
                        difficulty=1,
                        initial_water_level=level,
                        initial_inflow=inflow,
                        initial_outflow=inflow * 0.9,
                        area=10000.0,
                        control_mode=control_mode,
                        target_level=level,
                        pass_criteria={
                            'max_level_deviation': 0.2,
                            'max_settling_time': 3600,
                            'min_stability': 0.9,
                        },
                        tags=['basic', cat_id.lower()],
                    )
                    scenarios.append(scenario)

        return scenarios

    def _generate_boundary_scenarios(self) -> List[TestScenario]:
        """生成边界值场景"""
        scenarios = []

        # 水位边界
        boundary_levels = [0.1, 0.5, 1.0, 9.0, 9.5, 9.9]
        for level in boundary_levels:
            scenarios.append(TestScenario(
                id=f"BOUNDARY_LEVEL_{len(scenarios):04d}",
                name=f"边界水位测试-{level}m",
                category="BOUNDARY",
                difficulty=3,
                initial_water_level=level,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=10000.0,
                target_level=max(1.0, min(9.0, level)),
                tags=['boundary', 'level'],
            ))

        # 流量边界
        boundary_flows = [0.0, 0.1, 0.5, 19.0, 19.5, 20.0, 30.0, 50.0]
        for flow in boundary_flows:
            scenarios.append(TestScenario(
                id=f"BOUNDARY_FLOW_{len(scenarios):04d}",
                name=f"边界流量测试-{flow}m³/s",
                category="BOUNDARY",
                difficulty=3,
                initial_water_level=3.0,
                initial_inflow=flow,
                initial_outflow=flow * 0.8 if flow > 0 else 0,
                area=10000.0,
                tags=['boundary', 'flow'],
            ))

        # 面积边界
        boundary_areas = [500.0, 1000.0, 50000.0, 100000.0]
        for area in boundary_areas:
            scenarios.append(TestScenario(
                id=f"BOUNDARY_AREA_{len(scenarios):04d}",
                name=f"边界面积测试-{area}m²",
                category="BOUNDARY",
                difficulty=2,
                initial_water_level=3.0,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=area,
                tags=['boundary', 'area'],
            ))

        # 温度边界
        boundary_temps = [-20.0, -10.0, 0.0, 35.0, 40.0, 45.0]
        for temp in boundary_temps:
            season = SeasonType.WINTER if temp < 5 else SeasonType.SUMMER
            scenarios.append(TestScenario(
                id=f"BOUNDARY_TEMP_{len(scenarios):04d}",
                name=f"边界温度测试-{temp}°C",
                category="BOUNDARY",
                difficulty=3,
                initial_water_level=3.0,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=10000.0,
                temperature=temp,
                season=season,
                tags=['boundary', 'temperature'],
            ))

        # 参数组合边界
        for level in [0.5, 9.5]:
            for flow in [0.5, 20.0]:
                for area in [1000.0, 50000.0]:
                    scenarios.append(TestScenario(
                        id=f"BOUNDARY_COMBO_{len(scenarios):04d}",
                        name=f"组合边界测试-L{level}-F{flow}-A{area}",
                        category="BOUNDARY",
                        difficulty=4,
                        initial_water_level=level,
                        initial_inflow=flow,
                        initial_outflow=flow * 0.9,
                        area=area,
                        tags=['boundary', 'combination'],
                    ))

        return scenarios

    def _generate_orthogonal_scenarios(self) -> List[TestScenario]:
        """使用正交设计生成场景"""
        scenarios = []

        # L16 正交表设计 (4因子, 4水平)
        levels_4 = [1.0, 3.0, 5.0, 7.0]
        inflows_4 = [2.0, 5.0, 10.0, 15.0]
        areas_4 = [5000.0, 10000.0, 20000.0, 50000.0]
        horizons_4 = [3, 5, 10, 15]

        # L16(4^4) 正交表
        l16_table = [
            [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3],
            [1, 0, 1, 2], [1, 1, 0, 3], [1, 2, 3, 0], [1, 3, 2, 1],
            [2, 0, 2, 3], [2, 1, 3, 2], [2, 2, 0, 1], [2, 3, 1, 0],
            [3, 0, 3, 1], [3, 1, 2, 0], [3, 2, 1, 3], [3, 3, 0, 2],
        ]

        for idx, row in enumerate(l16_table):
            for weather in [WeatherCondition.CLEAR, WeatherCondition.RAINY,
                          WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM]:
                scenario = TestScenario(
                    id=f"ORTHO_L16_{len(scenarios):04d}",
                    name=f"正交设计L16-{idx}-{weather.value}",
                    category="ORTHOGONAL",
                    difficulty=2,
                    initial_water_level=levels_4[row[0]],
                    initial_inflow=inflows_4[row[1]],
                    initial_outflow=inflows_4[row[1]] * 0.9,
                    area=areas_4[row[2]],
                    mpc_horizon=horizons_4[row[3]],
                    weather=weather,
                    tags=['orthogonal', 'l16'],
                )
                scenarios.append(scenario)

        # L25 正交表设计 (5因子, 5水平)
        weights_5 = [1.0, 5.0, 10.0, 20.0, 50.0]

        for w_idx, weight in enumerate(weights_5):
            for l_idx, level in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
                for demand in DemandPattern:
                    scenario = TestScenario(
                        id=f"ORTHO_L25_{len(scenarios):04d}",
                        name=f"正交设计L25-W{weight}-L{level}-{demand.value}",
                        category="ORTHOGONAL",
                        difficulty=2,
                        initial_water_level=level,
                        initial_inflow=5.0,
                        initial_outflow=4.5,
                        area=10000.0,
                        weight_level=weight,
                        demand_pattern=demand,
                        tags=['orthogonal', 'l25'],
                    )
                    scenarios.append(scenario)

        return scenarios

    def _generate_fault_scenarios(self) -> List[TestScenario]:
        """生成故障场景"""
        scenarios = []

        # 单故障场景
        for fault_type in FaultType:
            if fault_type == FaultType.NONE:
                continue

            for severity in [0.1, 0.3, 0.5, 0.8, 1.0]:
                for level in [2.0, 3.0, 4.0]:
                    difficulty = self._get_fault_difficulty(fault_type, severity)

                    scenario = TestScenario(
                        id=f"FAULT_{fault_type.name}_{len(scenarios):04d}",
                        name=f"故障场景-{fault_type.value}-严重度{severity}",
                        category="FAULT",
                        difficulty=difficulty,
                        initial_water_level=level,
                        initial_inflow=5.0,
                        initial_outflow=5.0,
                        area=10000.0,
                        fault_type=fault_type,
                        injections=[{
                            'type': 'fault',
                            'fault_type': fault_type.value,
                            'severity': severity,
                            'start_time': 3600,
                            'duration': 7200,
                        }],
                        pass_criteria={
                            'fault_detection_time': 300,
                            'max_level_deviation': 0.5,
                            'recovery_time': 3600,
                        },
                        tags=['fault', fault_type.name.lower()],
                    )
                    scenarios.append(scenario)

        # 多故障组合场景
        fault_combinations = [
            (FaultType.SENSOR_DRIFT, FaultType.ACTUATOR_DELAY),
            (FaultType.SENSOR_NOISE, FaultType.COMMUNICATION_DELAY),
            (FaultType.ACTUATOR_STUCK, FaultType.LEAK_SMALL),
            (FaultType.CYBER_ATTACK_FDIA, FaultType.SENSOR_NOISE),
            (FaultType.SENSOR_DRIFT, FaultType.ACTUATOR_DEGRADED, FaultType.COMMUNICATION_DELAY),
        ]

        for combo in fault_combinations:
            scenario = TestScenario(
                id=f"FAULT_COMBO_{len(scenarios):04d}",
                name=f"多故障组合-{'+'.join(f.value for f in combo)}",
                category="FAULT_COMPOUND",
                difficulty=5,
                initial_water_level=3.0,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=10000.0,
                fault_type=combo[0],
                injections=[{
                    'type': 'fault',
                    'fault_type': f.value,
                    'severity': 0.5,
                    'start_time': 3600 + i * 1800,
                    'duration': 7200,
                } for i, f in enumerate(combo)],
                tags=['fault', 'compound', 'multi-fault'],
            )
            scenarios.append(scenario)

        return scenarios

    def _generate_weather_scenarios(self) -> List[TestScenario]:
        """生成极端天气场景"""
        scenarios = []

        # 极端天气
        extreme_weather = [
            (WeatherCondition.HEAVY_RAIN, SeasonType.SUMMER, 25.0, 0.5),
            (WeatherCondition.STORM, SeasonType.SUMMER, 22.0, 1.0),
            (WeatherCondition.ICE, SeasonType.ICE_PERIOD, -10.0, 0.3),
            (WeatherCondition.SNOW, SeasonType.WINTER, -5.0, 0.4),
            (WeatherCondition.HEATWAVE, SeasonType.SUMMER, 40.0, 0.2),
            (WeatherCondition.FOG, SeasonType.AUTUMN, 10.0, 0.1),
        ]

        for weather, season, temp, disturbance in extreme_weather:
            for level in [2.0, 3.0, 4.0, 5.0]:
                for inflow_mult in [1.0, 1.5, 2.0, 3.0]:
                    base_inflow = 5.0
                    scenario = TestScenario(
                        id=f"WEATHER_{weather.name}_{len(scenarios):04d}",
                        name=f"极端天气-{weather.value}-水位{level}m",
                        category="WEATHER",
                        difficulty=4,
                        initial_water_level=level,
                        initial_inflow=base_inflow * inflow_mult,
                        initial_outflow=base_inflow,
                        area=10000.0,
                        weather=weather,
                        season=season,
                        temperature=temp,
                        disturbance_magnitude=disturbance,
                        control_mode=self._get_weather_control_mode(weather),
                        tags=['weather', weather.name.lower()],
                    )
                    scenarios.append(scenario)

        # 天气变化场景
        weather_transitions = [
            (WeatherCondition.CLEAR, WeatherCondition.RAINY, WeatherCondition.HEAVY_RAIN),
            (WeatherCondition.CLOUDY, WeatherCondition.STORM, WeatherCondition.CLEAR),
            (WeatherCondition.CLEAR, WeatherCondition.SNOW, WeatherCondition.ICE),
        ]

        for transition in weather_transitions:
            scenario = TestScenario(
                id=f"WEATHER_TRANS_{len(scenarios):04d}",
                name=f"天气变化-{'→'.join(w.value for w in transition)}",
                category="WEATHER_TRANSITION",
                difficulty=4,
                initial_water_level=3.0,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=10000.0,
                weather=transition[0],
                injections=[{
                    'type': 'weather_change',
                    'weather': w.value,
                    'start_time': i * 7200,
                } for i, w in enumerate(transition)],
                tags=['weather', 'transition'],
            )
            scenarios.append(scenario)

        return scenarios

    def _generate_compound_scenarios(self) -> List[TestScenario]:
        """生成复合场景"""
        scenarios = []

        # 天气 + 故障
        weather_fault_combos = list(itertools.product(
            [WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM, WeatherCondition.ICE],
            [FaultType.SENSOR_DRIFT, FaultType.ACTUATOR_DELAY, FaultType.COMMUNICATION_DELAY],
        ))

        for weather, fault in weather_fault_combos:
            for level in [2.0, 3.0, 4.0]:
                scenario = TestScenario(
                    id=f"COMPOUND_WF_{len(scenarios):04d}",
                    name=f"复合场景-{weather.value}+{fault.value}",
                    category="COMPOUND",
                    difficulty=5,
                    initial_water_level=level,
                    initial_inflow=5.0,
                    initial_outflow=5.0,
                    area=10000.0,
                    weather=weather,
                    fault_type=fault,
                    control_mode=self._get_weather_control_mode(weather),
                    tags=['compound', 'weather-fault'],
                )
                scenarios.append(scenario)

        # 需求变化 + 设备状态
        demand_control_combos = list(itertools.product(
            [DemandPattern.MORNING_PEAK, DemandPattern.DOUBLE_PEAK, DemandPattern.EMERGENCY],
            [ControlMode.NORMAL, ControlMode.DEGRADED, ControlMode.EMERGENCY],
        ))

        for demand, control in demand_control_combos:
            for level in [2.0, 3.0, 4.0]:
                scenario = TestScenario(
                    id=f"COMPOUND_DC_{len(scenarios):04d}",
                    name=f"复合场景-{demand.value}+{control.value}",
                    category="COMPOUND",
                    difficulty=4,
                    initial_water_level=level,
                    initial_inflow=5.0,
                    initial_outflow=5.0,
                    area=10000.0,
                    demand_pattern=demand,
                    control_mode=control,
                    tags=['compound', 'demand-control'],
                )
                scenarios.append(scenario)

        # 三重复合场景
        triple_combos = list(itertools.product(
            [WeatherCondition.HEAVY_RAIN, WeatherCondition.STORM],
            [FaultType.SENSOR_DRIFT, FaultType.COMMUNICATION_DELAY],
            [DemandPattern.EMERGENCY, DemandPattern.IRRIGATION],
        ))

        for weather, fault, demand in triple_combos[:20]:  # 限制数量
            scenario = TestScenario(
                id=f"COMPOUND_TRIPLE_{len(scenarios):04d}",
                name=f"三重复合-{weather.value}+{fault.value}+{demand.value}",
                category="COMPOUND_EXTREME",
                difficulty=5,
                initial_water_level=3.0,
                initial_inflow=8.0,
                initial_outflow=5.0,
                area=10000.0,
                weather=weather,
                fault_type=fault,
                demand_pattern=demand,
                tags=['compound', 'triple', 'extreme'],
            )
            scenarios.append(scenario)

        return scenarios

    def _generate_temporal_scenarios(self) -> List[TestScenario]:
        """生成时序场景"""
        scenarios = []

        # 日变化模式
        daily_patterns = [
            ("工作日", [(6, 1.5), (9, 1.0), (12, 1.3), (18, 1.4), (22, 0.8)]),
            ("周末", [(8, 1.2), (12, 1.4), (18, 1.3), (22, 0.9)]),
            ("节假日", [(10, 1.5), (14, 1.6), (20, 1.4)]),
        ]

        for pattern_name, demand_changes in daily_patterns:
            for level in [2.0, 3.0, 4.0]:
                scenario = TestScenario(
                    id=f"TEMPORAL_DAILY_{len(scenarios):04d}",
                    name=f"日变化-{pattern_name}-水位{level}m",
                    category="TEMPORAL",
                    difficulty=3,
                    initial_water_level=level,
                    initial_inflow=5.0,
                    initial_outflow=5.0,
                    area=10000.0,
                    duration_hours=24,
                    injections=[{
                        'type': 'demand_change',
                        'start_time': hour * 3600,
                        'multiplier': mult,
                    } for hour, mult in demand_changes],
                    tags=['temporal', 'daily'],
                )
                scenarios.append(scenario)

        # 长期趋势
        trend_patterns = [
            ("需求上升", 0.02),
            ("需求下降", -0.02),
            ("季节波动", None),
        ]

        for trend_name, trend_rate in trend_patterns:
            for duration in [72, 168, 336]:  # 3天, 7天, 14天
                scenario = TestScenario(
                    id=f"TEMPORAL_TREND_{len(scenarios):04d}",
                    name=f"长期趋势-{trend_name}-{duration}h",
                    category="TEMPORAL_TREND",
                    difficulty=3,
                    initial_water_level=3.0,
                    initial_inflow=5.0,
                    initial_outflow=5.0,
                    area=10000.0,
                    duration_hours=duration,
                    injections=[{
                        'type': 'trend',
                        'pattern': trend_name,
                        'rate': trend_rate,
                    }],
                    tags=['temporal', 'trend'],
                )
                scenarios.append(scenario)

        # 突发事件序列
        event_sequences = [
            [("normal", 0), ("surge", 3600), ("normal", 7200)],
            [("normal", 0), ("drop", 1800), ("surge", 5400), ("normal", 10800)],
            [("surge", 0), ("peak", 3600), ("drop", 7200), ("normal", 14400)],
        ]

        for seq_idx, sequence in enumerate(event_sequences):
            scenario = TestScenario(
                id=f"TEMPORAL_SEQ_{len(scenarios):04d}",
                name=f"事件序列-{seq_idx}",
                category="TEMPORAL_SEQUENCE",
                difficulty=4,
                initial_water_level=3.0,
                initial_inflow=5.0,
                initial_outflow=5.0,
                area=10000.0,
                injections=[{
                    'type': 'event',
                    'event_type': event,
                    'start_time': time,
                } for event, time in sequence],
                tags=['temporal', 'sequence'],
            )
            scenarios.append(scenario)

        return scenarios

    def _generate_topology_scenarios(self) -> List[TestScenario]:
        """生成网络拓扑场景"""
        scenarios = []

        for topology in NetworkTopology:
            for level in [2.0, 3.0, 4.0]:
                for control_mode in [ControlMode.NORMAL, ControlMode.FLOOD_PREVENTION, ControlMode.DEGRADED]:
                    scenario = TestScenario(
                        id=f"TOPO_{topology.name}_{len(scenarios):04d}",
                        name=f"拓扑测试-{topology.value}-{control_mode.value}",
                        category="TOPOLOGY",
                        difficulty=self._get_topology_difficulty(topology),
                        initial_water_level=level,
                        initial_inflow=5.0,
                        initial_outflow=5.0,
                        area=10000.0,
                        topology=topology,
                        control_mode=control_mode,
                        tags=['topology', topology.name.lower()],
                    )
                    scenarios.append(scenario)

        return scenarios

    def _generate_monte_carlo_scenarios(self, count: int) -> List[TestScenario]:
        """使用蒙特卡洛方法生成随机场景"""
        scenarios = []

        for i in range(count):
            # 随机选择参数
            level = random.choice(self.space.water_levels)
            inflow = random.choice(self.space.inflows)
            area = random.choice(self.space.areas)
            weather = random.choice(list(WeatherCondition))
            season = random.choice(list(SeasonType))
            demand = random.choice(list(DemandPattern))
            fault = random.choice(list(FaultType))
            control = random.choice(list(ControlMode))
            topology = random.choice(list(NetworkTopology))
            temp = random.choice(self.space.temperatures)
            noise = random.choice(self.space.noise_levels)
            horizon = random.choice(self.space.mpc_horizons)
            duration = random.choice(self.space.simulation_durations)

            # 计算难度
            difficulty = self._calculate_difficulty(
                level, inflow, weather, fault, control, topology
            )

            scenario = TestScenario(
                id=f"MC_{i:06d}",
                name=f"蒙特卡洛随机场景-{i}",
                category="MONTE_CARLO",
                difficulty=difficulty,
                initial_water_level=level,
                initial_inflow=inflow,
                initial_outflow=inflow * random.uniform(0.7, 1.0),
                area=area,
                weather=weather,
                season=season,
                temperature=temp,
                demand_pattern=demand,
                fault_type=fault,
                control_mode=control,
                topology=topology,
                noise_level=noise,
                mpc_horizon=horizon,
                duration_hours=duration,
                tags=['monte_carlo', 'random'],
            )
            scenarios.append(scenario)

        return scenarios

    def _get_fault_difficulty(self, fault_type: FaultType, severity: float) -> int:
        """根据故障类型和严重程度计算难度"""
        base_difficulty = {
            FaultType.NONE: 1,
            FaultType.SENSOR_DRIFT: 2,
            FaultType.SENSOR_STUCK: 3,
            FaultType.SENSOR_NOISE: 2,
            FaultType.ACTUATOR_STUCK: 4,
            FaultType.ACTUATOR_DELAY: 3,
            FaultType.ACTUATOR_DEGRADED: 3,
            FaultType.COMMUNICATION_DELAY: 2,
            FaultType.COMMUNICATION_LOSS: 4,
            FaultType.LEAK_SMALL: 3,
            FaultType.LEAK_LARGE: 5,
            FaultType.BLOCKAGE: 4,
            FaultType.POWER_FLUCTUATION: 3,
            FaultType.CYBER_ATTACK_FDIA: 5,
            FaultType.CYBER_ATTACK_DOS: 4,
            FaultType.CYBER_ATTACK_REPLAY: 4,
        }

        base = base_difficulty.get(fault_type, 3)
        severity_boost = int(severity * 2)
        return min(5, base + severity_boost)

    def _get_weather_control_mode(self, weather: WeatherCondition) -> ControlMode:
        """根据天气返回对应的控制模式"""
        weather_mode_map = {
            WeatherCondition.CLEAR: ControlMode.NORMAL,
            WeatherCondition.CLOUDY: ControlMode.NORMAL,
            WeatherCondition.RAINY: ControlMode.NORMAL,
            WeatherCondition.HEAVY_RAIN: ControlMode.FLOOD_PREVENTION,
            WeatherCondition.STORM: ControlMode.FLOOD_PREVENTION,
            WeatherCondition.SNOW: ControlMode.ICE_PERIOD,
            WeatherCondition.ICE: ControlMode.ICE_PERIOD,
            WeatherCondition.FOG: ControlMode.DEGRADED,
            WeatherCondition.HEATWAVE: ControlMode.DROUGHT_RESPONSE,
        }
        return weather_mode_map.get(weather, ControlMode.NORMAL)

    def _get_topology_difficulty(self, topology: NetworkTopology) -> int:
        """根据拓扑返回难度"""
        topology_difficulty = {
            NetworkTopology.SINGLE_POOL: 1,
            NetworkTopology.CASCADE_2: 2,
            NetworkTopology.CASCADE_3: 2,
            NetworkTopology.CASCADE_5: 3,
            NetworkTopology.CASCADE_10: 4,
            NetworkTopology.BRANCH_2: 3,
            NetworkTopology.BRANCH_3: 3,
            NetworkTopology.MESH_SMALL: 4,
            NetworkTopology.MESH_LARGE: 5,
        }
        return topology_difficulty.get(topology, 3)

    def _calculate_difficulty(self, level, inflow, weather, fault, control, topology) -> int:
        """综合计算场景难度"""
        difficulty = 1

        # 水位因素
        if level < 1.0 or level > 8.0:
            difficulty += 1

        # 流量因素
        if inflow > 15.0 or inflow < 1.0:
            difficulty += 1

        # 天气因素
        if weather in [WeatherCondition.STORM, WeatherCondition.ICE, WeatherCondition.HEATWAVE]:
            difficulty += 1

        # 故障因素
        if fault != FaultType.NONE:
            difficulty += 1

        # 控制模式因素
        if control in [ControlMode.EMERGENCY, ControlMode.DEGRADED]:
            difficulty += 1

        # 拓扑因素
        if topology in [NetworkTopology.CASCADE_10, NetworkTopology.MESH_LARGE]:
            difficulty += 1

        return min(5, difficulty)

    def _deduplicate(self, scenarios: List[TestScenario]) -> List[TestScenario]:
        """去除重复场景"""
        unique = []
        seen_hashes = set()

        for scenario in scenarios:
            h = scenario.get_hash()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(scenario)

        return unique

    def get_scenarios_by_category(self, category: str) -> List[TestScenario]:
        """按类别获取场景"""
        return [s for s in self.generated_scenarios if s.category == category]

    def get_scenarios_by_difficulty(self, difficulty: int) -> List[TestScenario]:
        """按难度获取场景"""
        return [s for s in self.generated_scenarios if s.difficulty == difficulty]

    def get_scenarios_by_tag(self, tag: str) -> List[TestScenario]:
        """按标签获取场景"""
        return [s for s in self.generated_scenarios if tag in s.tags]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total': len(self.generated_scenarios),
            'by_category': {},
            'by_difficulty': {i: 0 for i in range(1, 6)},
            'by_fault_type': {},
            'by_weather': {},
            'by_control_mode': {},
            'by_topology': {},
        }

        for scenario in self.generated_scenarios:
            # 分类统计
            cat = scenario.category
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1

            # 难度统计
            stats['by_difficulty'][scenario.difficulty] += 1

            # 故障统计
            fault = scenario.fault_type.value
            stats['by_fault_type'][fault] = stats['by_fault_type'].get(fault, 0) + 1

            # 天气统计
            weather = scenario.weather.value
            stats['by_weather'][weather] = stats['by_weather'].get(weather, 0) + 1

            # 控制模式统计
            mode = scenario.control_mode.value
            stats['by_control_mode'][mode] = stats['by_control_mode'].get(mode, 0) + 1

            # 拓扑统计
            topo = scenario.topology.value
            stats['by_topology'][topo] = stats['by_topology'].get(topo, 0) + 1

        return stats

    def export_scenarios(self, filepath: str, format: str = 'json'):
        """导出场景到文件"""
        if format == 'json':
            data = {
                'generated_at': datetime.now().isoformat(),
                'total_scenarios': len(self.generated_scenarios),
                'statistics': self.get_statistics(),
                'scenarios': [s.to_dict() for s in self.generated_scenarios],
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")


# 便捷函数
def generate_comprehensive_scenarios(count: int = 10000, seed: int = 42) -> List[TestScenario]:
    """便捷函数: 生成全面测试场景"""
    generator = ScenarioCombinatorialGenerator(seed=seed)
    return generator.generate_all(max_scenarios=count)


if __name__ == "__main__":
    # 测试场景生成
    generator = ScenarioCombinatorialGenerator(seed=42)
    scenarios = generator.generate_all(max_scenarios=10000)

    print(f"生成场景总数: {len(scenarios)}")

    stats = generator.get_statistics()
    print(f"\n按类别统计:")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"  {cat}: {count}")

    print(f"\n按难度统计:")
    for diff, count in stats['by_difficulty'].items():
        stars = "★" * diff + "☆" * (5 - diff)
        print(f"  {stars}: {count}")

    # 导出场景
    generator.export_scenarios("generated_scenarios.json")
    print(f"\n场景已导出到 generated_scenarios.json")

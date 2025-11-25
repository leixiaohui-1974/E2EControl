"""
场景生成器 (Scenario Generator)
从YAML配置文件加载和生成测试场景
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class ScenarioCategory(Enum):
    """场景大类"""
    S1_NORMAL = "S1_正常运行"
    S2_FLOOD = "S2_防洪调度"
    S3_DROUGHT = "S3_干旱应对"
    S4_ICE = "S4_冰期运行"
    S5_POLLUTION = "S5_污染应急"
    S6_EQUIPMENT = "S6_设备故障"
    S7_SECURITY = "S7_安全攻击"


class DifficultyLevel(Enum):
    """难度等级"""
    LEVEL_1 = 1  # ★☆☆☆☆
    LEVEL_2 = 2  # ★★☆☆☆
    LEVEL_3 = 3  # ★★★☆☆
    LEVEL_4 = 4  # ★★★★☆
    LEVEL_5 = 5  # ★★★★★


class AutonomousLevel(Enum):
    """自主运行等级要求"""
    L0 = 0  # 完全人工
    L1 = 1  # 辅助决策
    L2 = 2  # 部分自动
    L3 = 3  # 条件自动
    L4 = 4  # 高度自动
    L5 = 5  # 完全自主


@dataclass
class InitialState:
    """初始状态"""
    water_level: float = 3.0      # 水位 [m]
    inflow: float = 50.0          # 入流 [m³/s]
    outflow: float = 50.0         # 出流 [m³/s]
    gate_position: float = 0.5    # 闸门开度 [0-1]
    temperature: float = 15.0     # 水温 [°C]
    pollution_level: float = 0.0  # 污染物浓度 [mg/L]
    water_levels: Dict = field(default_factory=dict)    # 多池水位
    gate_positions: Dict = field(default_factory=dict)  # 多闸门位置
    inflows: Dict = field(default_factory=dict)         # 多入流

    @classmethod
    def from_dict(cls, data: Dict) -> 'InitialState':
        # 提取基本字段
        basic_fields = {k: v for k, v in data.items() if k in ['water_level', 'inflow', 'outflow',
                        'gate_position', 'temperature', 'pollution_level']}
        # 提取扩展字段
        water_levels = data.get('water_levels', {})
        gate_positions = data.get('gate_positions', {})
        inflows = data.get('inflows', {})

        return cls(
            **basic_fields,
            water_levels=water_levels,
            gate_positions=gate_positions,
            inflows=inflows
        )


@dataclass
class Injection:
    """工况注入配置"""
    type: str                     # 注入类型: step, ramp, pulse, noise, sinusoid
    target: str                   # 目标变量: inflow, outflow, sensor, actuator
    start_time: float             # 开始时间 [s]
    magnitude: float              # 幅度
    duration: float = 0.0         # 持续时间 [s]
    rate: float = 0.0             # 变化率 (for ramp)
    frequency: float = 0.0        # 频率 (for sinusoid)
    parameters: Dict = field(default_factory=dict)  # 额外参数

    @classmethod
    def from_dict(cls, data: Dict) -> 'Injection':
        params = {k: v for k, v in data.items()
                  if k not in ['type', 'target', 'start_time', 'magnitude', 'duration', 'rate', 'frequency']}
        return cls(
            type=data.get('type', 'step'),
            target=data.get('target', 'inflow'),
            start_time=data.get('start_time', 0),
            magnitude=data.get('magnitude', 0),
            duration=data.get('duration', 0),
            rate=data.get('rate', 0),
            frequency=data.get('frequency', 0),
            parameters=params
        )


@dataclass
class ExpectedResponse:
    """期望响应"""
    max_reaction_time: float = 60.0       # 最大反应时间 [s]
    max_level_overshoot: float = 0.5      # 最大水位超调 [m]
    max_settling_time: float = 1800.0     # 最大调节时间 [s]
    max_level_deviation: float = 0.1      # 最大水位偏差 [m]
    min_safety_margin: float = 0.5        # 最小安全裕度 [m]

    @classmethod
    def from_dict(cls, data: Dict) -> 'ExpectedResponse':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PassCriteria:
    """通过标准"""
    conditions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: List[str]) -> 'PassCriteria':
        return cls(conditions=data if data else [])


@dataclass
class Condition:
    """工况定义"""
    id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    autonomous_level_required: AutonomousLevel
    initial_state: InitialState
    injections: List[Injection]
    expected_response: ExpectedResponse
    pass_criteria: PassCriteria
    timeout: float = 7200.0  # 测试超时 [s]

    @classmethod
    def from_dict(cls, data: Dict) -> 'Condition':
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            difficulty=DifficultyLevel(data.get('difficulty', 1)),
            autonomous_level_required=AutonomousLevel(data.get('autonomous_level_required', 2)),
            initial_state=InitialState.from_dict(data.get('initial_state', {})),
            injections=[Injection.from_dict(inj) for inj in data.get('injections', [])],
            expected_response=ExpectedResponse.from_dict(data.get('expected_response', {})),
            pass_criteria=PassCriteria.from_dict(data.get('pass_criteria', [])),
            timeout=data.get('timeout', 7200.0)
        )


@dataclass
class Scenario:
    """场景定义"""
    id: str
    category: ScenarioCategory
    name: str
    description: str
    conditions: List[Condition] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.LEVEL_1
    autonomous_level: AutonomousLevel = AutonomousLevel.L1
    duration: float = 300.0  # 测试持续时间 [s]
    initial_state: Optional[InitialState] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'Scenario':
        category_map = {
            'S1': ScenarioCategory.S1_NORMAL,
            'S1_NORMAL': ScenarioCategory.S1_NORMAL,
            'S2': ScenarioCategory.S2_FLOOD,
            'S2_FLOOD': ScenarioCategory.S2_FLOOD,
            'S3': ScenarioCategory.S3_DROUGHT,
            'S3_DROUGHT': ScenarioCategory.S3_DROUGHT,
            'S4': ScenarioCategory.S4_ICE,
            'S4_ICE': ScenarioCategory.S4_ICE,
            'S5': ScenarioCategory.S5_POLLUTION,
            'S5_POLLUTION': ScenarioCategory.S5_POLLUTION,
            'S6': ScenarioCategory.S6_EQUIPMENT,
            'S6_EQUIPMENT': ScenarioCategory.S6_EQUIPMENT,
            'S7': ScenarioCategory.S7_SECURITY,
            'S7_SECURITY': ScenarioCategory.S7_SECURITY,
        }

        # 从ID或category字段解析类别
        cat_str = data.get('category', data.get('id', 'S1')[:2])
        if isinstance(cat_str, str):
            category = category_map.get(cat_str, category_map.get(cat_str[:2], ScenarioCategory.S1_NORMAL))
        else:
            category = cat_str

        # 解析难度
        difficulty_val = data.get('difficulty', 1)
        if isinstance(difficulty_val, int):
            difficulty = DifficultyLevel(difficulty_val)
        else:
            difficulty = difficulty_val

        # 解析自主等级
        level_val = data.get('autonomous_level', 1)
        if isinstance(level_val, str):
            level_val = int(level_val[1]) if level_val.startswith('L') else int(level_val)
        autonomous_level = AutonomousLevel(level_val)

        # 解析初始状态
        initial_state_data = data.get('initial_state')
        initial_state = None
        if initial_state_data:
            initial_state = InitialState.from_dict(initial_state_data)

        return cls(
            id=data.get('id', ''),
            category=category,
            name=data.get('name', ''),
            description=data.get('description', ''),
            conditions=[Condition.from_dict(c) for c in data.get('conditions', [])],
            difficulty=difficulty,
            autonomous_level=autonomous_level,
            duration=data.get('duration', 300.0),
            initial_state=initial_state
        )


class ScenarioGenerator:
    """
    场景生成器

    功能：
    1. 从YAML文件加载场景配置
    2. 生成测试场景实例
    3. 管理场景库
    """

    def __init__(self, scenarios_dir: str = None):
        """
        初始化场景生成器

        Args:
            scenarios_dir: 场景配置文件目录
        """
        if scenarios_dir is None:
            scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.scenarios_dir = scenarios_dir
        self.scenarios: Dict[str, Scenario] = {}
        self.conditions: Dict[str, Condition] = {}

    def create_scenario(self,
                        id: str,
                        name: str,
                        category: str,
                        difficulty: int,
                        autonomous_level: str,
                        duration: float,
                        description: str = "",
                        initial_state: Dict = None,
                        conditions: List[Dict] = None) -> Scenario:
        """
        创建新的测试场景

        Args:
            id: 场景ID
            name: 场景名称
            category: 场景类别 (S1_NORMAL, S2_FLOOD等)
            difficulty: 难度等级 (1-5)
            autonomous_level: 自主等级要求 (L0-L5)
            duration: 测试持续时间 (秒)
            description: 场景描述
            initial_state: 初始状态字典
            conditions: 工况列表

        Returns:
            Scenario: 创建的场景对象
        """
        data = {
            'id': id,
            'name': name,
            'category': category,
            'difficulty': difficulty,
            'autonomous_level': autonomous_level,
            'duration': duration,
            'description': description,
            'initial_state': initial_state or {},
            'conditions': conditions or []
        }

        scenario = Scenario.from_dict(data)
        self.scenarios[id] = scenario
        return scenario

    def load_scenario(self, filepath: str) -> List[Scenario]:
        """加载单个场景配置文件

        Returns:
            Scenario或List[Scenario]: 如果YAML包含scenarios列表则返回列表，否则返回单个场景
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        scenarios = []

        # 检查是否有scenarios列表
        if 'scenarios' in data:
            for scenario_data in data['scenarios']:
                scenario = Scenario.from_dict(scenario_data)
                self.scenarios[scenario.id] = scenario
                scenarios.append(scenario)

                # 索引所有工况
                for condition in scenario.conditions:
                    self.conditions[condition.id] = condition
        else:
            # 单个场景
            scenario = Scenario.from_dict(data.get('scenario', data))
            self.scenarios[scenario.id] = scenario
            scenarios.append(scenario)

            # 索引所有工况
            for condition in scenario.conditions:
                self.conditions[condition.id] = condition

        return scenarios

    def load_all_scenarios(self) -> Dict[str, Scenario]:
        """加载所有场景配置文件"""
        if not os.path.exists(self.scenarios_dir):
            print(f"警告: 场景目录不存在 {self.scenarios_dir}")
            return self.scenarios

        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                filepath = os.path.join(self.scenarios_dir, filename)
                try:
                    self.load_scenario(filepath)
                except Exception as e:
                    print(f"加载场景失败 {filename}: {e}")

        return self.scenarios

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """获取指定场景"""
        return self.scenarios.get(scenario_id)

    def get_condition(self, condition_id: str) -> Optional[Condition]:
        """获取指定工况"""
        return self.conditions.get(condition_id)

    def get_conditions_by_difficulty(self, max_difficulty: DifficultyLevel) -> List[Condition]:
        """获取指定难度及以下的所有工况"""
        return [c for c in self.conditions.values()
                if c.difficulty.value <= max_difficulty.value]

    def get_conditions_by_level(self, level: AutonomousLevel) -> List[Condition]:
        """获取指定自主等级要求的所有工况"""
        return [c for c in self.conditions.values()
                if c.autonomous_level_required.value <= level.value]

    def get_conditions_by_category(self, category: ScenarioCategory) -> List[Condition]:
        """获取指定类别的所有工况"""
        result = []
        for scenario in self.scenarios.values():
            if scenario.category == category:
                result.extend(scenario.conditions)
        return result

    def generate_test_suite(self,
                           target_level: AutonomousLevel,
                           categories: List[ScenarioCategory] = None) -> List[Condition]:
        """
        生成针对目标等级的测试套件

        Args:
            target_level: 目标自主等级
            categories: 限定的场景类别(可选)

        Returns:
            符合条件的工况列表
        """
        conditions = self.get_conditions_by_level(target_level)

        if categories:
            category_conditions = []
            for cat in categories:
                category_conditions.extend(self.get_conditions_by_category(cat))
            conditions = [c for c in conditions if c in category_conditions]

        # 按难度排序
        conditions.sort(key=lambda c: (c.difficulty.value, c.id))

        return conditions

    def get_statistics(self) -> Dict:
        """获取场景库统计信息"""
        stats = {
            'total_scenarios': len(self.scenarios),
            'total_conditions': len(self.conditions),
            'by_category': {},
            'by_difficulty': {d.name: 0 for d in DifficultyLevel},
            'by_level': {l.name: 0 for l in AutonomousLevel}
        }

        for scenario in self.scenarios.values():
            cat_name = scenario.category.value
            stats['by_category'][cat_name] = stats['by_category'].get(cat_name, 0) + len(scenario.conditions)

        for condition in self.conditions.values():
            stats['by_difficulty'][condition.difficulty.name] += 1
            stats['by_level'][condition.autonomous_level_required.name] += 1

        return stats

    def print_summary(self):
        """打印场景库摘要"""
        stats = self.get_statistics()

        print("=" * 60)
        print("            场景库统计信息")
        print("=" * 60)
        print(f"场景总数: {stats['total_scenarios']}")
        print(f"工况总数: {stats['total_conditions']}")

        print("\n按类别统计:")
        for cat, count in stats['by_category'].items():
            print(f"  {cat}: {count}个工况")

        print("\n按难度统计:")
        for diff, count in stats['by_difficulty'].items():
            stars = "★" * int(diff.split('_')[1]) + "☆" * (5 - int(diff.split('_')[1]))
            print(f"  {stars}: {count}个工况")

        print("\n按等级要求统计:")
        for level, count in stats['by_level'].items():
            print(f"  {level}: {count}个工况")
        print("=" * 60)


# 示例使用
if __name__ == "__main__":
    generator = ScenarioGenerator()
    generator.load_all_scenarios()
    generator.print_summary()

"""
核心数据结构和类型定义
Core Data Structures and Type Definitions

本模块定义南水北调中线全线全场景自主运行系统的基础类型：
- PoolRole: 渠池角色 (6种)
- ScenarioType: 场景类型 (8种)
- ControlDirective: 控制指令
- PoolTopology: 渠池拓扑
- CanalPoolConfig: 渠池配置
- SpecialStructure: 特殊建筑物
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ==============================================================================
# 角色定义 (Roles) - 6种
# ==============================================================================

class PoolRole(Enum):
    """
    渠池角色定义

    每个渠池在不同场景下可被分配不同角色:
    - SOURCE: 发起调节，主动供水
    - BUFFER: 利用库容吸纳水量（涨水）
    - DRAIN: 利用库容释放水量（降水）或退水
    - ISOLATE: 物理切断
    - TRANSMIT: 保持体积不变，纯粹过流
    - THROTTLE: 增加阻力消能（故障态）
    """
    SOURCE = "source"        # 源：发起调节，主动供水
    BUFFER = "buffer"        # 蓄：利用库容吸纳水量
    DRAIN = "drain"          # 排：利用库容释放水量或退水
    ISOLATE = "isolate"      # 隔：物理切断
    TRANSMIT = "transmit"    # 输：保持体积不变，纯粹过流
    THROTTLE = "throttle"    # 阻：增加阻力消能

    # 辅助角色 (用于过渡状态)
    WAIT = "wait"            # 等：等待上游来水
    FEED = "feed"            # 补：补充供水
    HOLD = "hold"            # 滞：滞留水量
    PASS = "pass"            # 过：通过
    STABLE = "stable"        # 稳：稳定运行
    PASSIVE = "passive"      # 被：被动响应
    BRAKE = "brake"          # 刹：刹车减流
    COMPENSATE = "compensate"  # 补偿：补偿流量
    STORE = "store"          # 存：存储
    CONSUME = "consume"      # 消：消耗
    BYPASS = "bypass"        # 旁：旁通
    ISLAND = "island"        # 岛：孤岛运行
    MAINT = "maint"          # 维：维护模式
    FIX = "fix"              # 修：抢修模式


# ==============================================================================
# 场景类型 (Scenarios) - 8大场景
# ==============================================================================

class ScenarioType(Enum):
    """
    场景类型定义 (8大场景)

    S1: 常规计划 - 正常供水
    S2: 突发增供 - 下游突然增加需求
    S3: 突发污染 - 水质污染事故
    S4: 暴雨防洪 - 降雨导致水位过高
    S5: 冰期输水 - 冬季冰盖条件下运行
    S6: 泵站掉电 - 沿线泵站故障
    S7: 计划检修 - 闸门或设施计划维护
    S8: 临时抢修 - 紧急故障处理
    """
    S1_NORMAL_PLAN = "S1_normal_plan"           # 常规计划
    S2_SURGE_DEMAND = "S2_surge_demand"         # 突发增供
    S3_POLLUTION = "S3_pollution"               # 突发污染
    S4_FLOOD_CONTROL = "S4_flood_control"       # 暴雨防洪
    S5_ICE_PERIOD = "S5_ice_period"             # 冰期输水
    S6_PUMP_FAILURE = "S6_pump_failure"         # 泵站掉电
    S7_PLANNED_MAINT = "S7_planned_maintenance" # 计划检修
    S8_EMERGENCY_REPAIR = "S8_emergency_repair" # 临时抢修


class ScenarioSeverity(Enum):
    """场景严重程度"""
    LOW = 1       # 低：轻微影响
    MEDIUM = 2    # 中：中等影响
    HIGH = 3      # 高：严重影响
    CRITICAL = 4  # 危急：紧急情况


class ScenarioPhase(Enum):
    """场景阶段"""
    DETECTION = "detection"       # 检测阶段
    ASSESSMENT = "assessment"     # 评估阶段
    RESPONSE = "response"         # 响应阶段
    EXECUTION = "execution"       # 执行阶段
    RECOVERY = "recovery"         # 恢复阶段
    NORMAL = "normal"             # 正常阶段


# ==============================================================================
# 控制指令 (Control Directive)
# ==============================================================================

@dataclass
class ControlDirective:
    """
    控制指令 - 从全局编排器下发给区域协调器和现地控制器

    包含:
    - pool_id: 目标渠池ID
    - role: 分配的角色
    - target_bias: 水位目标偏移量
    - weight_multipliers: MPC权重修正系数
    - hard_constraints: 临时硬约束
    - priority: 指令优先级
    - effective_time: 生效时间
    - expiry_time: 失效时间
    """
    pool_id: int
    role: PoolRole
    target_bias: float = 0.0                    # 水位目标偏移量 [m]
    weight_multipliers: Dict[str, float] = field(default_factory=dict)
    hard_constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1                           # 优先级 (1-10, 10最高)
    effective_time: Optional[float] = None      # 生效时间 [s]
    expiry_time: Optional[float] = None         # 失效时间 [s]
    source_scenario: Optional[ScenarioType] = None
    remarks: str = ""

    def is_active(self, current_time: float) -> bool:
        """检查指令是否有效"""
        if self.effective_time is not None and current_time < self.effective_time:
            return False
        if self.expiry_time is not None and current_time > self.expiry_time:
            return False
        return True

    def get_q_weight(self) -> float:
        """获取流量权重修正系数"""
        return self.weight_multipliers.get('W_Q', 1.0)

    def get_z_weight(self) -> float:
        """获取水位权重修正系数"""
        return self.weight_multipliers.get('W_Z', 1.0)

    def get_constraint(self, key: str, default: Any = None) -> Any:
        """获取硬约束值"""
        return self.hard_constraints.get(key, default)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'pool_id': self.pool_id,
            'role': self.role.value,
            'target_bias': self.target_bias,
            'weight_multipliers': self.weight_multipliers.copy(),
            'hard_constraints': self.hard_constraints.copy(),
            'priority': self.priority,
            'effective_time': self.effective_time,
            'expiry_time': self.expiry_time,
            'source_scenario': self.source_scenario.value if self.source_scenario else None,
            'remarks': self.remarks,
        }


# ==============================================================================
# 特殊建筑物类型
# ==============================================================================

class StructureType(Enum):
    """特殊建筑物类型"""
    GATE = "gate"                      # 节制闸
    INVERTED_SIPHON = "inverted_siphon"  # 倒虹吸
    AQUEDUCT = "aqueduct"              # 渡槽
    CHECK_GATE = "check_gate"          # 退水闸
    PUMP_STATION = "pump_station"      # 泵站
    DIVERSION = "diversion"            # 分水口
    CROSS_RIVER = "cross_river"        # 穿河建筑物
    TUNNEL = "tunnel"                  # 隧洞


@dataclass
class SpecialStructure:
    """
    特殊建筑物配置

    中线工程包含多种特殊建筑物:
    - 穿黄倒虹吸: 高阻尼、大滞后连通管
    - 湍河渡槽: 流量上限受限的瓶颈单元
    - 节制闸: 调节流量的主要手段
    """
    structure_id: str
    structure_type: StructureType
    chainage: float                    # 桩号 [km]
    name: str = ""

    # 水力特性
    max_flow: float = 100.0            # 最大过流能力 [m³/s]
    min_flow: float = 0.0              # 最小流量 [m³/s]
    head_loss_coefficient: float = 0.1  # 水头损失系数
    delay_time: float = 0.0            # 附加延迟时间 [s]

    # 几何参数
    length: float = 0.0                # 长度 [m]
    width: float = 0.0                 # 宽度 [m]
    invert_elevation: float = 0.0      # 底高程 [m]

    # 运行状态
    is_operational: bool = True        # 是否运行中
    current_opening: float = 1.0       # 当前开度 [0-1]

    def compute_head_loss(self, flow: float) -> float:
        """计算水头损失"""
        return self.head_loss_coefficient * (flow ** 2)

    def get_effective_flow_capacity(self) -> float:
        """获取有效过流能力"""
        return self.max_flow * self.current_opening if self.is_operational else 0.0


# ==============================================================================
# 渠池配置
# ==============================================================================

@dataclass
class CanalPoolConfig:
    """
    渠池配置 - 描述单个渠池的几何和水力特性

    基于南水北调中线实际参数:
    - 全线1432km, 约60+渠池
    - 底宽: 丹江口25m -> 北京5m (线性递减)
    - 边坡: 2.0-3.0
    - 糙率: 0.014 (基准)
    """
    pool_id: int
    name: str = ""

    # 位置信息
    chainage_start: float = 0.0        # 起始桩号 [km]
    chainage_end: float = 0.0          # 终止桩号 [km]
    region_id: int = 0                 # 所属区域ID

    # 几何参数
    length: float = 20.0               # 长度 [km]
    bottom_width: float = 15.0         # 底宽 [m]
    side_slope: float = 2.5            # 边坡 (1:m)
    bed_slope: float = 0.00004         # 底坡

    # 水力参数
    manning_n: float = 0.014           # 曼宁糙率系数 (基准)
    max_depth: float = 8.0             # 最大水深 [m]
    min_depth: float = 0.5             # 最小水深 [m]
    design_flow: float = 350.0         # 设计流量 [m³/s]
    max_flow: float = 420.0            # 最大流量 [m³/s]

    # 计算属性
    @property
    def length_m(self) -> float:
        """长度 (米)"""
        return self.length * 1000

    @property
    def surface_area(self) -> float:
        """估算水面面积 [m²] (假设平均水深4m)"""
        avg_depth = 4.0
        top_width = self.bottom_width + 2 * self.side_slope * avg_depth
        return self.length_m * (self.bottom_width + top_width) / 2

    @property
    def storage_volume(self) -> float:
        """估算蓄水量 [m³] (假设平均水深4m)"""
        avg_depth = 4.0
        return self.cross_section_area(avg_depth) * self.length_m

    def cross_section_area(self, depth: float) -> float:
        """计算过水断面面积 [m²]"""
        return (self.bottom_width + self.side_slope * depth) * depth

    def wetted_perimeter(self, depth: float) -> float:
        """计算湿周 [m]"""
        return self.bottom_width + 2 * depth * np.sqrt(1 + self.side_slope ** 2)

    def hydraulic_radius(self, depth: float) -> float:
        """计算水力半径 [m]"""
        area = self.cross_section_area(depth)
        perimeter = self.wetted_perimeter(depth)
        return area / perimeter if perimeter > 0 else 0

    def manning_flow(self, depth: float, slope: float = None) -> float:
        """使用曼宁公式计算流量 [m³/s]"""
        if slope is None:
            slope = self.bed_slope
        area = self.cross_section_area(depth)
        r = self.hydraulic_radius(depth)
        return (1.0 / self.manning_n) * area * (r ** (2/3)) * np.sqrt(slope)

    def normal_depth(self, flow: float, tolerance: float = 0.001, max_iter: int = 50) -> float:
        """使用曼宁公式计算正常水深 [m]"""
        # 二分法求解
        low, high = 0.1, self.max_depth
        for _ in range(max_iter):
            mid = (low + high) / 2
            q_calc = self.manning_flow(mid)
            if abs(q_calc - flow) < tolerance:
                return mid
            elif q_calc < flow:
                low = mid
            else:
                high = mid
        return (low + high) / 2


# ==============================================================================
# 区域配置
# ==============================================================================

@dataclass
class RegionConfig:
    """
    区域配置 - 划分为4-5个大区

    区域划分:
    - Region 0: 丹江口-淅川 (渠首段)
    - Region 1: 河南段南部
    - Region 2: 河南段北部 (含穿黄)
    - Region 3: 河北段
    - Region 4: 北京段 (终点)
    """
    region_id: int
    name: str

    # 范围
    pool_ids: List[int] = field(default_factory=list)
    chainage_start: float = 0.0        # 起始桩号 [km]
    chainage_end: float = 0.0          # 终止桩号 [km]

    # 特性
    priority: int = 1                  # 优先级
    is_critical: bool = False          # 是否关键区域

    # 协调器配置
    coordinator_horizon: int = 10      # 协调器预测时域
    coordinator_dt: float = 900.0      # 协调器时间步长 [s]

    @property
    def num_pools(self) -> int:
        """渠池数量"""
        return len(self.pool_ids)

    @property
    def length(self) -> float:
        """区域长度 [km]"""
        return self.chainage_end - self.chainage_start


# ==============================================================================
# 渠池拓扑
# ==============================================================================

@dataclass
class PoolTopology:
    """
    渠池拓扑结构 - 描述全线渠池的连接关系

    支持:
    - 串联连接 (主渠道)
    - 分水口 (支线)
    - 特殊建筑物 (倒虹吸、渡槽)
    """
    num_pools: int
    total_length: float = 1432.0       # 全线长度 [km]

    # 渠池配置
    pools: List[CanalPoolConfig] = field(default_factory=list)

    # 区域配置
    regions: List[RegionConfig] = field(default_factory=list)

    # 特殊建筑物
    special_structures: List[SpecialStructure] = field(default_factory=list)

    # 连接关系
    upstream_map: Dict[int, List[int]] = field(default_factory=dict)
    downstream_map: Dict[int, List[int]] = field(default_factory=dict)

    def get_upstream(self, pool_id: int) -> List[int]:
        """获取上游渠池列表"""
        return self.upstream_map.get(pool_id, [])

    def get_downstream(self, pool_id: int) -> List[int]:
        """获取下游渠池列表"""
        return self.downstream_map.get(pool_id, [])

    def get_pool(self, pool_id: int) -> Optional[CanalPoolConfig]:
        """获取渠池配置"""
        for pool in self.pools:
            if pool.pool_id == pool_id:
                return pool
        return None

    def get_region(self, region_id: int) -> Optional[RegionConfig]:
        """获取区域配置"""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None

    def get_pool_region(self, pool_id: int) -> Optional[RegionConfig]:
        """获取渠池所属区域"""
        for region in self.regions:
            if pool_id in region.pool_ids:
                return region
        return None

    def get_structures_in_pool(self, pool_id: int) -> List[SpecialStructure]:
        """获取渠池内的特殊建筑物"""
        pool = self.get_pool(pool_id)
        if pool is None:
            return []

        return [
            s for s in self.special_structures
            if pool.chainage_start <= s.chainage <= pool.chainage_end
        ]

    def get_distance(self, from_pool: int, to_pool: int) -> float:
        """计算两渠池之间的距离 [km]"""
        pool1 = self.get_pool(from_pool)
        pool2 = self.get_pool(to_pool)
        if pool1 is None or pool2 is None:
            return 0.0

        return abs(pool2.chainage_start - pool1.chainage_end)

    def get_travel_time(self, from_pool: int, to_pool: int,
                        velocity: float = 1.2) -> float:
        """计算水流传播时间 [s]"""
        distance_km = self.get_distance(from_pool, to_pool)
        return (distance_km * 1000) / velocity

    @staticmethod
    def create_snwd_middle_route() -> 'PoolTopology':
        """创建南水北调中线拓扑"""
        # 60个渠池 + 4个区域
        num_pools = 60

        pools = []
        for i in range(num_pools):
            # 线性插值参数
            progress = i / (num_pools - 1)
            chainage = progress * 1432  # km

            # 底宽: 25m -> 5m
            bottom_width = 25 - 20 * progress

            # 设计流量: 350 -> 70 (考虑分水)
            design_flow = 350 - 280 * progress

            pool = CanalPoolConfig(
                pool_id=i,
                name=f"Pool_{i:02d}",
                chainage_start=chainage,
                chainage_end=chainage + 1432 / num_pools,
                region_id=min(4, int(progress * 5)),
                length=1432 / num_pools,
                bottom_width=bottom_width,
                side_slope=2.5,
                bed_slope=0.00004,
                manning_n=0.014,
                max_depth=8.0,
                design_flow=design_flow,
                max_flow=design_flow * 1.2,
            )
            pools.append(pool)

        # 创建区域
        regions = [
            RegionConfig(
                region_id=0,
                name="渠首段",
                pool_ids=list(range(0, 12)),
                chainage_start=0,
                chainage_end=286,
                priority=1,
                is_critical=True,
            ),
            RegionConfig(
                region_id=1,
                name="河南段南",
                pool_ids=list(range(12, 24)),
                chainage_start=286,
                chainage_end=572,
            ),
            RegionConfig(
                region_id=2,
                name="河南段北(穿黄)",
                pool_ids=list(range(24, 36)),
                chainage_start=572,
                chainage_end=858,
                is_critical=True,
            ),
            RegionConfig(
                region_id=3,
                name="河北段",
                pool_ids=list(range(36, 48)),
                chainage_start=858,
                chainage_end=1144,
            ),
            RegionConfig(
                region_id=4,
                name="北京段",
                pool_ids=list(range(48, 60)),
                chainage_start=1144,
                chainage_end=1432,
                is_critical=True,
            ),
        ]

        # 特殊建筑物
        special_structures = [
            SpecialStructure(
                structure_id="YH_SIPHON",
                structure_type=StructureType.INVERTED_SIPHON,
                chainage=700,
                name="穿黄倒虹吸",
                max_flow=350,
                head_loss_coefficient=0.5,
                delay_time=3600,  # 1小时附加延迟
                length=4250,
            ),
            SpecialStructure(
                structure_id="TH_AQUEDUCT",
                structure_type=StructureType.AQUEDUCT,
                chainage=320,
                name="湍河渡槽",
                max_flow=280,  # 瓶颈
                length=1030,
            ),
            SpecialStructure(
                structure_id="SH_AQUEDUCT",
                structure_type=StructureType.AQUEDUCT,
                chainage=510,
                name="沙河渡槽",
                max_flow=300,
                length=2300,
            ),
        ]

        # 构建连接关系 (简单串联)
        upstream_map = {}
        downstream_map = {}
        for i in range(num_pools):
            upstream_map[i] = [i - 1] if i > 0 else []
            downstream_map[i] = [i + 1] if i < num_pools - 1 else []

        return PoolTopology(
            num_pools=num_pools,
            total_length=1432,
            pools=pools,
            regions=regions,
            special_structures=special_structures,
            upstream_map=upstream_map,
            downstream_map=downstream_map,
        )


# ==============================================================================
# 场景事件
# ==============================================================================

@dataclass
class ScenarioEvent:
    """
    场景事件 - 描述触发场景的事件
    """
    event_id: str
    scenario_type: ScenarioType
    location: int                      # 事件位置 (渠池ID)
    severity: ScenarioSeverity = ScenarioSeverity.MEDIUM
    timestamp: float = 0.0             # 事件时间 [s]

    # 事件详情
    details: Dict[str, Any] = field(default_factory=dict)

    # 影响范围
    affected_pools: List[int] = field(default_factory=list)
    affected_regions: List[int] = field(default_factory=list)

    # 状态
    phase: ScenarioPhase = ScenarioPhase.DETECTION
    is_resolved: bool = False

    def get_detail(self, key: str, default: Any = None) -> Any:
        """获取事件详情"""
        return self.details.get(key, default)


# ==============================================================================
# 控制计划
# ==============================================================================

@dataclass
class ControlPlan:
    """
    控制计划 - 由全局编排器生成，包含所有渠池的控制指令
    """
    plan_id: str
    scenario: ScenarioType
    timestamp: float

    # 指令集
    directives: Dict[int, ControlDirective] = field(default_factory=dict)

    # 元数据
    duration: float = 3600.0           # 计划持续时间 [s]
    priority: int = 1

    # 状态
    is_active: bool = True
    execution_progress: float = 0.0    # 执行进度 [0-1]

    def get_directive(self, pool_id: int) -> Optional[ControlDirective]:
        """获取指定渠池的指令"""
        return self.directives.get(pool_id)

    def add_directive(self, directive: ControlDirective):
        """添加指令"""
        self.directives[directive.pool_id] = directive

    @property
    def num_directives(self) -> int:
        """指令数量"""
        return len(self.directives)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'plan_id': self.plan_id,
            'scenario': self.scenario.value,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'priority': self.priority,
            'is_active': self.is_active,
            'num_directives': self.num_directives,
            'directives': {
                k: v.to_dict() for k, v in self.directives.items()
            },
        }


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" " * 15 + "核心数据结构测试")
    print("="*70)

    # 创建南水北调中线拓扑
    topology = PoolTopology.create_snwd_middle_route()

    print(f"\n✓ 创建南水北调中线拓扑")
    print(f"  渠池数量: {topology.num_pools}")
    print(f"  总长度: {topology.total_length} km")
    print(f"  区域数量: {len(topology.regions)}")
    print(f"  特殊建筑物: {len(topology.special_structures)}")

    # 显示区域信息
    print(f"\n区域划分:")
    for region in topology.regions:
        print(f"  {region.name}: {region.num_pools} 个渠池, "
              f"{region.chainage_start:.0f}-{region.chainage_end:.0f} km")

    # 显示特殊建筑物
    print(f"\n特殊建筑物:")
    for structure in topology.special_structures:
        print(f"  {structure.name} ({structure.structure_type.value}): "
              f"K{structure.chainage:.0f}, 最大流量 {structure.max_flow} m³/s")

    # 测试渠池配置
    pool = topology.pools[30]  # 中间位置
    print(f"\n渠池 {pool.pool_id} 配置:")
    print(f"  名称: {pool.name}")
    print(f"  桩号: K{pool.chainage_start:.0f}-K{pool.chainage_end:.0f}")
    print(f"  底宽: {pool.bottom_width:.1f} m")
    print(f"  设计流量: {pool.design_flow:.0f} m³/s")
    print(f"  水面面积: {pool.surface_area/1e6:.2f} km²")

    # 测试控制指令
    print(f"\n控制指令测试:")
    directive = ControlDirective(
        pool_id=30,
        role=PoolRole.ISOLATE,
        target_bias=-0.5,
        weight_multipliers={'W_Q': 1e9, 'W_Z': 0},
        hard_constraints={'Q_out_max': 0, 'Q_drain_min': 50},
        source_scenario=ScenarioType.S3_POLLUTION,
        remarks="污染隔离指令",
    )
    print(f"  角色: {directive.role.value}")
    print(f"  水位偏移: {directive.target_bias} m")
    print(f"  流量权重: {directive.get_q_weight()}")
    print(f"  最大出流约束: {directive.get_constraint('Q_out_max')}")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)

"""
数据类型定义 - 分布式SIL系统的核心数据结构

定义了系统内各模块间传递的数据类型，包括：
- SegmentState: 分段状态
- BoundaryCondition: 边界条件
- ControlCommand: 控制指令
- AssimilatedState: 同化后状态
- ModelError: 模型误差
- KPIMetrics: 性能指标
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


class SegmentType(Enum):
    """渠段类型"""
    CANAL = "canal"           # 普通渠道
    AQUEDUCT = "aqueduct"     # 渡槽
    TUNNEL = "tunnel"         # 隧洞
    INVERTED_SIPHON = "inverted_siphon"  # 倒虹吸


class GateType(Enum):
    """闸门类型"""
    CONTROL_GATE = "control"      # 节制闸
    DIVERSION_GATE = "diversion"  # 分水闸
    DISCHARGE_GATE = "discharge"  # 退水闸
    CHECK_GATE = "check"          # 进水闸


class ModelFidelity(Enum):
    """模型保真度"""
    REDUCED_ORDER = "reduced_order"      # 降阶模型 (IDZ)
    INTERMEDIATE = "intermediate"        # 中等保真度
    HIGH_FIDELITY = "high_fidelity"     # 高保真 (Saint-Venant)


class ErrorType(Enum):
    """误差类型"""
    MEASUREMENT_NOISE = "measurement_noise"      # 测量噪声
    MODEL_STRUCTURAL = "model_structural"        # 模型结构误差
    PARAMETER_DRIFT = "parameter_drift"          # 参数漂移
    BOUNDARY_MISMATCH = "boundary_mismatch"      # 边界不匹配
    NUMERICAL_DISPERSION = "numerical_dispersion"  # 数值色散
    LATERAL_DISTURBANCE = "lateral_disturbance"  # 侧向扰动


@dataclass
class SegmentGeometry:
    """渠段几何参数"""
    segment_id: str
    segment_type: SegmentType
    length: float              # m
    width_bottom: float        # m (底宽)
    side_slope: float          # 边坡系数
    bed_slope: float           # 底坡
    manning_n: float           # 曼宁糙率系数
    design_flow: float         # m³/s (设计流量)

    # 可选参数
    num_slices: int = 20       # 空间离散切片数
    cross_section_area: Optional[float] = None  # m² (断面面积)

    def get_hydraulic_radius(self, water_depth: float) -> float:
        """计算水力半径"""
        A = (self.width_bottom + self.side_slope * water_depth) * water_depth
        P = self.width_bottom + 2 * water_depth * np.sqrt(1 + self.side_slope**2)
        return A / P if P > 0 else 0.0


@dataclass
class GateState:
    """闸门状态"""
    gate_id: str
    gate_type: GateType
    opening: float             # 开度 [0, 1]
    flow_rate: float           # m³/s (过闸流量)
    upstream_level: float      # m (上游水位)
    downstream_level: float    # m (下游水位)
    timestamp: datetime = field(default_factory=datetime.now)

    # 闸门特性参数
    discharge_coefficient: float = 0.6
    max_opening: float = 1.0
    response_time: float = 300.0  # s (响应时间)


@dataclass
class SegmentState:
    """
    渠段状态 - 描述单个渠段的完整物理状态

    包含水位、流量、水质等多维状态信息，
    支持空间分布式表示（N个切片）
    """
    segment_id: str
    timestamp: datetime

    # 水力状态 (空间分布, 形状为 [N_slices])
    water_levels: np.ndarray       # m (各切片水位)
    flow_rates: np.ndarray         # m³/s (各切片流量)
    velocities: np.ndarray         # m/s (各切片流速)

    # 边界值 (标量)
    upstream_level: float          # m (上游边界水位)
    downstream_level: float        # m (下游边界水位)
    upstream_flow: float           # m³/s (上游入流)
    downstream_flow: float         # m³/s (下游出流)

    # 水质状态 (可选)
    concentrations: Optional[np.ndarray] = None  # mg/L (污染物浓度)
    temperatures: Optional[np.ndarray] = None    # °C (水温)
    ice_thickness: Optional[np.ndarray] = None   # m (冰层厚度)

    # 不确定性
    level_uncertainty: Optional[np.ndarray] = None  # m (水位不确定性)
    flow_uncertainty: Optional[np.ndarray] = None   # m³/s (流量不确定性)

    # 状态质量
    confidence: float = 1.0        # 状态置信度 [0, 1]
    data_quality_flag: int = 0     # 数据质量标志

    @property
    def mean_level(self) -> float:
        """平均水位"""
        return float(np.mean(self.water_levels))

    @property
    def mean_flow(self) -> float:
        """平均流量"""
        return float(np.mean(self.flow_rates))

    @property
    def volume(self) -> float:
        """估计总水量 (简化计算)"""
        return float(np.sum(self.water_levels * self.flow_rates))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "segment_id": self.segment_id,
            "timestamp": self.timestamp.isoformat(),
            "water_levels": self.water_levels.tolist(),
            "flow_rates": self.flow_rates.tolist(),
            "velocities": self.velocities.tolist(),
            "upstream_level": self.upstream_level,
            "downstream_level": self.downstream_level,
            "upstream_flow": self.upstream_flow,
            "downstream_flow": self.downstream_flow,
            "confidence": self.confidence,
        }


@dataclass
class BoundaryCondition:
    """
    边界条件 - 描述渠段间的接口状态

    支持软边界概念：边界不是硬输入，而是带置信度的估计量
    """
    boundary_id: str
    upstream_segment_id: str
    downstream_segment_id: str
    timestamp: datetime

    # 水力边界
    water_level: float             # m (边界水位)
    flow_rate: float               # m³/s (边界流量)

    # 不确定性描述
    level_std: float = 0.05        # m (水位标准差)
    flow_std: float = 0.5          # m³/s (流量标准差)
    confidence: float = 1.0        # 置信度 [0, 1]

    # 边界类型
    is_measured: bool = False      # 是否为实测值
    is_filtered: bool = False      # 是否经过滤波
    filter_type: Optional[str] = None  # 滤波器类型

    # 偏置估计
    level_bias: float = 0.0        # m (水位偏置估计)
    flow_bias: float = 0.0         # m³/s (流量偏置估计)

    # 侧向扰动
    lateral_inflow: float = 0.0    # m³/s (侧向入流)
    lateral_outflow: float = 0.0   # m³/s (侧向出流/取水)

    @property
    def net_lateral(self) -> float:
        """净侧向流量"""
        return self.lateral_inflow - self.lateral_outflow

    @property
    def corrected_flow(self) -> float:
        """校正后流量"""
        return self.flow_rate - self.flow_bias

    @property
    def corrected_level(self) -> float:
        """校正后水位"""
        return self.water_level - self.level_bias

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "boundary_id": self.boundary_id,
            "upstream_segment_id": self.upstream_segment_id,
            "downstream_segment_id": self.downstream_segment_id,
            "timestamp": self.timestamp.isoformat(),
            "water_level": self.water_level,
            "flow_rate": self.flow_rate,
            "level_std": self.level_std,
            "flow_std": self.flow_std,
            "confidence": self.confidence,
            "level_bias": self.level_bias,
            "flow_bias": self.flow_bias,
            "lateral_inflow": self.lateral_inflow,
            "lateral_outflow": self.lateral_outflow,
        }


@dataclass
class ControlCommand:
    """
    控制指令 - 发送给闸门/执行器的控制命令
    """
    command_id: str
    gate_id: str
    timestamp: datetime

    # 控制目标
    target_opening: Optional[float] = None     # 目标开度 [0, 1]
    target_flow: Optional[float] = None        # 目标流量 m³/s
    target_level: Optional[float] = None       # 目标水位 m

    # 控制模式
    control_mode: str = "opening"  # opening, flow, level

    # 约束
    max_rate: float = 0.01         # 最大变化率 (开度/秒)
    priority: int = 1              # 优先级 (1最高)

    # 来源
    source_controller: str = "unknown"
    source_level: str = "L2"       # L1, L2, L3, L4

    # 有效期
    valid_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "command_id": self.command_id,
            "gate_id": self.gate_id,
            "timestamp": self.timestamp.isoformat(),
            "target_opening": self.target_opening,
            "target_flow": self.target_flow,
            "target_level": self.target_level,
            "control_mode": self.control_mode,
            "source_controller": self.source_controller,
            "source_level": self.source_level,
        }


@dataclass
class AssimilatedState:
    """
    同化后状态 - 接口同化器输出的校正状态

    结合模型预测与观测数据，输出最优状态估计
    """
    boundary_id: str
    timestamp: datetime

    # 同化后的状态估计
    estimated_level: float         # m (估计水位)
    estimated_flow: float          # m³/s (估计流量)

    # 协方差/不确定性
    level_variance: float          # m² (水位方差)
    flow_variance: float           # m³/s² (流量方差)
    level_flow_covariance: float   # 交叉协方差

    # 创新量 (观测-预测)
    innovation_level: float = 0.0  # m
    innovation_flow: float = 0.0   # m³/s

    # 增益
    kalman_gain_level: float = 0.5
    kalman_gain_flow: float = 0.5

    # 同化质量
    assimilation_quality: float = 1.0  # [0, 1]
    rejected_outliers: int = 0

    @property
    def state_vector(self) -> np.ndarray:
        """状态向量 [h, q]"""
        return np.array([self.estimated_level, self.estimated_flow])

    @property
    def covariance_matrix(self) -> np.ndarray:
        """协方差矩阵"""
        return np.array([
            [self.level_variance, self.level_flow_covariance],
            [self.level_flow_covariance, self.flow_variance]
        ])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "boundary_id": self.boundary_id,
            "timestamp": self.timestamp.isoformat(),
            "estimated_level": self.estimated_level,
            "estimated_flow": self.estimated_flow,
            "level_variance": self.level_variance,
            "flow_variance": self.flow_variance,
            "innovation_level": self.innovation_level,
            "innovation_flow": self.innovation_flow,
            "assimilation_quality": self.assimilation_quality,
        }


@dataclass
class ModelError:
    """
    模型误差 - 描述模型不确定性和偏差

    用于误差预算管理和控制鲁棒性分析
    """
    error_id: str
    segment_id: str
    error_type: ErrorType
    timestamp: datetime

    # 误差统计
    mean_error: float              # 平均误差
    std_error: float               # 误差标准差
    max_error: float               # 最大误差
    rmse: float                    # 均方根误差

    # 误差特性
    correlation_length: float = 1000.0  # m (空间相关长度)
    correlation_time: float = 3600.0    # s (时间相关长度)
    spectral_type: str = "white"        # white, pink, red

    # 误差预算
    budget_limit: float = 0.1      # 允许的误差上限
    is_within_budget: bool = True

    # 偏置建模
    bias_state: float = 0.0        # 当前偏置状态
    bias_drift_rate: float = 0.0   # 偏置漂移率

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_id": self.error_id,
            "segment_id": self.segment_id,
            "error_type": self.error_type.value,
            "timestamp": self.timestamp.isoformat(),
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "max_error": self.max_error,
            "rmse": self.rmse,
            "is_within_budget": self.is_within_budget,
        }


@dataclass
class KPIMetrics:
    """
    KPI指标 - 全线一致性性能指标

    用于SIL测试的自动化评估和回归测试
    """
    metric_id: str
    timestamp: datetime
    time_window: float             # s (统计时间窗口)

    # 水量平衡KPI
    mass_balance_error: float      # m³ (水量平衡误差)
    mass_balance_error_rate: float # m³/s (误差率)
    mass_conservation_ratio: float # 质量守恒比

    # 接口残差KPI
    interface_level_residuals: Dict[str, float] = field(default_factory=dict)
    interface_flow_residuals: Dict[str, float] = field(default_factory=dict)
    max_interface_residual: float = 0.0
    mean_interface_residual: float = 0.0

    # 漂移率KPI
    level_drift_rate: float = 0.0    # m/h (水位漂移率)
    flow_drift_rate: float = 0.0     # m³/s/h (流量漂移率)
    cumulative_drift: float = 0.0    # 累计漂移

    # 控制性能KPI
    target_deviation: float = 0.0    # 目标偏差
    overshoot: float = 0.0           # 超调量
    settling_time: float = 0.0       # 调节时间
    constraint_violation_count: int = 0  # 约束违背次数
    constraint_violation_rate: float = 0.0  # 约束违背率

    # 数值稳定性KPI
    max_cfl_number: float = 0.0      # 最大CFL数
    numerical_dissipation: float = 0.0  # 数值耗散
    energy_balance_error: float = 0.0   # 能量平衡误差

    # 综合评分
    overall_score: float = 1.0       # 综合得分 [0, 1]
    is_valid: bool = True            # 是否有效

    # 门槛判定
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "mass_balance_error": 100.0,      # m³
        "max_interface_residual": 0.1,    # m
        "level_drift_rate": 0.01,         # m/h
        "constraint_violation_rate": 0.05,
    })

    def check_thresholds(self) -> Tuple[bool, List[str]]:
        """检查是否超出门槛"""
        violations = []

        if abs(self.mass_balance_error) > self.thresholds["mass_balance_error"]:
            violations.append(f"mass_balance_error: {self.mass_balance_error:.2f} > {self.thresholds['mass_balance_error']}")

        if self.max_interface_residual > self.thresholds["max_interface_residual"]:
            violations.append(f"max_interface_residual: {self.max_interface_residual:.4f} > {self.thresholds['max_interface_residual']}")

        if abs(self.level_drift_rate) > self.thresholds["level_drift_rate"]:
            violations.append(f"level_drift_rate: {self.level_drift_rate:.6f} > {self.thresholds['level_drift_rate']}")

        if self.constraint_violation_rate > self.thresholds["constraint_violation_rate"]:
            violations.append(f"constraint_violation_rate: {self.constraint_violation_rate:.2%} > {self.thresholds['constraint_violation_rate']:.2%}")

        return len(violations) == 0, violations

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        is_pass, violations = self.check_thresholds()
        return {
            "metric_id": self.metric_id,
            "timestamp": self.timestamp.isoformat(),
            "time_window": self.time_window,
            "mass_balance_error": self.mass_balance_error,
            "mass_conservation_ratio": self.mass_conservation_ratio,
            "max_interface_residual": self.max_interface_residual,
            "mean_interface_residual": self.mean_interface_residual,
            "level_drift_rate": self.level_drift_rate,
            "flow_drift_rate": self.flow_drift_rate,
            "target_deviation": self.target_deviation,
            "overshoot": self.overshoot,
            "constraint_violation_count": self.constraint_violation_count,
            "overall_score": self.overall_score,
            "is_valid": self.is_valid,
            "threshold_passed": is_pass,
            "violations": violations,
        }


@dataclass
class ScenarioConfig:
    """场景配置"""
    scenario_id: str
    name: str
    duration: float                # s (场景持续时间)

    # 初始条件
    initial_states: Dict[str, SegmentState] = field(default_factory=dict)
    initial_boundaries: Dict[str, BoundaryCondition] = field(default_factory=dict)

    # 边界驱动
    upstream_boundary_profile: Optional[np.ndarray] = None  # 时间序列
    lateral_inflow_profiles: Dict[str, np.ndarray] = field(default_factory=dict)
    demand_profiles: Dict[str, np.ndarray] = field(default_factory=dict)

    # 扰动配置
    disturbance_config: Dict[str, Any] = field(default_factory=dict)

    # 参数不确定性
    parameter_ensemble_size: int = 10
    manning_n_range: Tuple[float, float] = (0.012, 0.018)
    discharge_coeff_range: Tuple[float, float] = (0.55, 0.65)

    # 评估标准
    pass_criteria: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """仿真结果"""
    scenario_id: str
    start_time: datetime
    end_time: datetime

    # 状态历史
    state_history: List[Dict[str, SegmentState]] = field(default_factory=list)
    boundary_history: List[Dict[str, BoundaryCondition]] = field(default_factory=list)
    control_history: List[ControlCommand] = field(default_factory=list)

    # KPI历史
    kpi_history: List[KPIMetrics] = field(default_factory=list)

    # 最终评估
    final_kpi: Optional[KPIMetrics] = None
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            "scenario_id": self.scenario_id,
            "duration": (self.end_time - self.start_time).total_seconds(),
            "num_steps": len(self.state_history),
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "final_score": self.final_kpi.overall_score if self.final_kpi else 0.0,
        }

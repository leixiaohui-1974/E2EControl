"""
场景生成器 (Scenario Generator)

功能:
1. 真值系统/场景生成 - 驱动SIL测试的标准输入
2. 边界条件生成 - 带置信度的随机过程
3. 扰动注入 - 参数随机化、偏置、侧向扰动
4. 历史数据回放 - 支持真实运行数据回放
5. 极端场景生成 - 洪水、干旱、冰期等

设计原则:
- 边界条件不是硬输入,而是带噪声和偏置的估计量
- 支持参数集合(ensemble)用于鲁棒性测试
- 形成误差预算表,每类不确定性有上限、谱型、相关长度
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

from ..interfaces.data_types import (
    SegmentState, BoundaryCondition, SegmentGeometry,
    ScenarioConfig, ErrorType, GateState, GateType
)

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """噪声类型"""
    WHITE = "white"          # 白噪声
    PINK = "pink"            # 1/f噪声 (红噪声)
    BROWNIAN = "brownian"    # 布朗运动
    ORNSTEIN_UHLENBECK = "ou"  # OU过程 (均值回归)


class ScenarioType(Enum):
    """场景类型"""
    STEADY_STATE = "steady_state"      # 稳态运行
    STEP_CHANGE = "step_change"        # 阶跃变化
    RAMP_CHANGE = "ramp_change"        # 斜坡变化
    PERIODIC = "periodic"              # 周期变化
    FLOOD = "flood"                    # 洪水
    DROUGHT = "drought"                # 干旱
    ICE_PERIOD = "ice_period"          # 冰期
    DEMAND_SURGE = "demand_surge"      # 需求激增
    GATE_FAILURE = "gate_failure"      # 闸门故障
    SENSOR_FAULT = "sensor_fault"      # 传感器故障
    COMPOUND = "compound"              # 复合场景


@dataclass
class NoiseModel:
    """噪声模型配置"""
    noise_type: NoiseType = NoiseType.WHITE
    amplitude: float = 0.01            # 噪声幅度
    correlation_time: float = 300.0    # 时间相关长度 (s)
    correlation_length: float = 1000.0 # 空间相关长度 (m)
    mean: float = 0.0                  # 均值
    seed: Optional[int] = None


@dataclass
class BiasModel:
    """偏置模型配置"""
    initial_bias: float = 0.0          # 初始偏置
    drift_rate: float = 0.0            # 漂移率 (单位/s)
    max_bias: float = 0.1              # 最大偏置
    random_walk_std: float = 0.001     # 随机游走标准差


@dataclass
class DisturbanceConfig:
    """扰动配置"""
    # 侧向扰动
    lateral_inflow_mean: float = 0.0     # m³/s
    lateral_inflow_std: float = 0.5      # m³/s
    lateral_outflow_mean: float = 0.0    # m³/s
    lateral_outflow_std: float = 0.5     # m³/s

    # 参数扰动
    manning_n_perturbation: float = 0.002  # 曼宁系数扰动
    discharge_coeff_perturbation: float = 0.02  # 流量系数扰动

    # 未建模动态
    unmodeled_dynamics_amplitude: float = 0.01


@dataclass
class ErrorBudget:
    """误差预算表"""
    error_type: ErrorType
    max_amplitude: float           # 最大幅度
    spectral_type: str            # 频谱类型
    correlation_time: float       # 时间相关长度
    correlation_length: float     # 空间相关长度
    probability: float = 1.0      # 出现概率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "max_amplitude": self.max_amplitude,
            "spectral_type": self.spectral_type,
            "correlation_time": self.correlation_time,
            "correlation_length": self.correlation_length,
            "probability": self.probability,
        }


class ScenarioGenerator:
    """
    场景生成器

    负责生成SIL测试所需的各种场景，包括:
    - 真值边界条件
    - 带噪声的测量值
    - 参数不确定性样本
    - 扰动注入
    """

    def __init__(
        self,
        num_segments: int = 63,
        segment_length: float = 22730.0,  # m (约22.73km)
        dt: float = 900.0,  # s (15分钟)
        seed: Optional[int] = 42
    ):
        """
        初始化场景生成器

        Args:
            num_segments: 渠段数量 (中线约63个)
            segment_length: 平均渠段长度 (m)
            dt: 时间步长 (s)
            seed: 随机种子
        """
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.dt = dt
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # 默认误差预算表
        self.error_budgets = self._init_default_error_budgets()

        # 渠段几何参数 (默认)
        self.segment_geometries = self._init_default_geometries()

        # 噪声生成器状态
        self._noise_states: Dict[str, np.ndarray] = {}

        logger.info(f"ScenarioGenerator initialized: {num_segments} segments, dt={dt}s")

    def _init_default_error_budgets(self) -> Dict[ErrorType, ErrorBudget]:
        """初始化默认误差预算表"""
        return {
            ErrorType.MEASUREMENT_NOISE: ErrorBudget(
                error_type=ErrorType.MEASUREMENT_NOISE,
                max_amplitude=0.02,  # 2cm水位噪声
                spectral_type="white",
                correlation_time=60.0,
                correlation_length=100.0,
            ),
            ErrorType.MODEL_STRUCTURAL: ErrorBudget(
                error_type=ErrorType.MODEL_STRUCTURAL,
                max_amplitude=0.05,  # 5%结构误差
                spectral_type="pink",
                correlation_time=3600.0,
                correlation_length=5000.0,
            ),
            ErrorType.PARAMETER_DRIFT: ErrorBudget(
                error_type=ErrorType.PARAMETER_DRIFT,
                max_amplitude=0.1,   # 10%参数漂移
                spectral_type="brownian",
                correlation_time=86400.0,  # 1天
                correlation_length=10000.0,
            ),
            ErrorType.BOUNDARY_MISMATCH: ErrorBudget(
                error_type=ErrorType.BOUNDARY_MISMATCH,
                max_amplitude=0.1,   # 0.1m边界不匹配
                spectral_type="pink",
                correlation_time=1800.0,
                correlation_length=2000.0,
            ),
            ErrorType.NUMERICAL_DISPERSION: ErrorBudget(
                error_type=ErrorType.NUMERICAL_DISPERSION,
                max_amplitude=0.02,
                spectral_type="white",
                correlation_time=900.0,
                correlation_length=1000.0,
            ),
            ErrorType.LATERAL_DISTURBANCE: ErrorBudget(
                error_type=ErrorType.LATERAL_DISTURBANCE,
                max_amplitude=2.0,   # 2 m³/s侧向扰动
                spectral_type="ou",
                correlation_time=7200.0,
                correlation_length=20000.0,
            ),
        }

    def _init_default_geometries(self) -> List[SegmentGeometry]:
        """初始化默认渠段几何参数"""
        geometries = []
        for i in range(self.num_segments):
            # 根据位置变化几何参数 (北方渠道逐渐变窄)
            width_factor = 1.0 - 0.3 * (i / self.num_segments)
            geometries.append(SegmentGeometry(
                segment_id=f"SEG_{i:03d}",
                segment_type=SegmentGeometry.SegmentType.CANAL if hasattr(SegmentGeometry, 'SegmentType') else "canal",
                length=self.segment_length,
                width_bottom=20.0 * width_factor,  # 底宽 20m -> 14m
                side_slope=2.0,  # 1:2边坡
                bed_slope=1.0 / 25000,  # 底坡 1:25000
                manning_n=0.014 + 0.002 * np.random.randn(),  # 曼宁系数
                design_flow=350.0 * width_factor,  # 设计流量
                num_slices=20,
            ))
        return geometries

    def generate_noise(
        self,
        noise_model: NoiseModel,
        num_steps: int,
        shape: Tuple[int, ...] = (1,)
    ) -> np.ndarray:
        """
        生成噪声序列

        Args:
            noise_model: 噪声模型配置
            num_steps: 时间步数
            shape: 每步的形状

        Returns:
            noise: 噪声序列 [num_steps, *shape]
        """
        if noise_model.seed is not None:
            rng = np.random.RandomState(noise_model.seed)
        else:
            rng = np.random.RandomState()

        full_shape = (num_steps,) + shape
        noise = np.zeros(full_shape)

        if noise_model.noise_type == NoiseType.WHITE:
            noise = rng.randn(*full_shape) * noise_model.amplitude + noise_model.mean

        elif noise_model.noise_type == NoiseType.PINK:
            # 1/f噪声通过滤波白噪声实现
            white = rng.randn(*full_shape)
            # 简化的1/f滤波
            alpha = 1.0 - self.dt / noise_model.correlation_time
            for t in range(1, num_steps):
                noise[t] = alpha * noise[t-1] + np.sqrt(1 - alpha**2) * white[t]
            noise = noise * noise_model.amplitude + noise_model.mean

        elif noise_model.noise_type == NoiseType.BROWNIAN:
            # 布朗运动
            increments = rng.randn(*full_shape) * noise_model.amplitude * np.sqrt(self.dt)
            noise = np.cumsum(increments, axis=0) + noise_model.mean

        elif noise_model.noise_type == NoiseType.ORNSTEIN_UHLENBECK:
            # OU过程 (均值回归)
            theta = 1.0 / noise_model.correlation_time  # 回归速率
            sigma = noise_model.amplitude * np.sqrt(2 * theta)
            x = np.zeros(full_shape)
            x[0] = noise_model.mean
            for t in range(1, num_steps):
                dx = theta * (noise_model.mean - x[t-1]) * self.dt + sigma * np.sqrt(self.dt) * rng.randn(*shape)
                x[t] = x[t-1] + dx
            noise = x

        return noise

    def generate_bias_evolution(
        self,
        bias_model: BiasModel,
        num_steps: int
    ) -> np.ndarray:
        """
        生成偏置演化序列

        Args:
            bias_model: 偏置模型配置
            num_steps: 时间步数

        Returns:
            bias: 偏置序列 [num_steps]
        """
        bias = np.zeros(num_steps)
        bias[0] = bias_model.initial_bias

        for t in range(1, num_steps):
            # 线性漂移 + 随机游走
            drift = bias_model.drift_rate * self.dt
            random_walk = bias_model.random_walk_std * np.sqrt(self.dt) * np.random.randn()
            bias[t] = bias[t-1] + drift + random_walk

            # 限制最大偏置
            bias[t] = np.clip(bias[t], -bias_model.max_bias, bias_model.max_bias)

        return bias

    def generate_boundary_profile(
        self,
        scenario_type: ScenarioType,
        base_value: float,
        duration: float,
        **kwargs
    ) -> np.ndarray:
        """
        生成边界条件时间序列

        Args:
            scenario_type: 场景类型
            base_value: 基准值
            duration: 持续时间 (s)
            **kwargs: 场景特定参数

        Returns:
            profile: 时间序列 [num_steps]
        """
        num_steps = int(duration / self.dt)
        t = np.arange(num_steps) * self.dt

        if scenario_type == ScenarioType.STEADY_STATE:
            profile = np.ones(num_steps) * base_value

        elif scenario_type == ScenarioType.STEP_CHANGE:
            step_time = kwargs.get("step_time", duration / 2)
            step_magnitude = kwargs.get("step_magnitude", 0.1 * base_value)
            profile = np.where(t >= step_time, base_value + step_magnitude, base_value)

        elif scenario_type == ScenarioType.RAMP_CHANGE:
            ramp_start = kwargs.get("ramp_start", duration / 4)
            ramp_end = kwargs.get("ramp_end", 3 * duration / 4)
            ramp_magnitude = kwargs.get("ramp_magnitude", 0.2 * base_value)
            profile = np.zeros(num_steps)
            for i, ti in enumerate(t):
                if ti < ramp_start:
                    profile[i] = base_value
                elif ti < ramp_end:
                    progress = (ti - ramp_start) / (ramp_end - ramp_start)
                    profile[i] = base_value + progress * ramp_magnitude
                else:
                    profile[i] = base_value + ramp_magnitude

        elif scenario_type == ScenarioType.PERIODIC:
            period = kwargs.get("period", 86400.0)  # 默认24小时周期
            amplitude = kwargs.get("amplitude", 0.1 * base_value)
            profile = base_value + amplitude * np.sin(2 * np.pi * t / period)

        elif scenario_type == ScenarioType.FLOOD:
            peak_time = kwargs.get("peak_time", duration / 2)
            peak_factor = kwargs.get("peak_factor", 2.0)
            rise_time = kwargs.get("rise_time", duration / 6)
            # 洪水过程线 (三角形近似)
            profile = np.zeros(num_steps)
            for i, ti in enumerate(t):
                if ti < peak_time - rise_time:
                    profile[i] = base_value
                elif ti < peak_time:
                    progress = (ti - (peak_time - rise_time)) / rise_time
                    profile[i] = base_value + (peak_factor - 1) * base_value * progress
                elif ti < peak_time + 2 * rise_time:
                    progress = (ti - peak_time) / (2 * rise_time)
                    profile[i] = peak_factor * base_value - (peak_factor - 1) * base_value * progress
                else:
                    profile[i] = base_value

        elif scenario_type == ScenarioType.DROUGHT:
            min_factor = kwargs.get("min_factor", 0.5)
            transition_time = kwargs.get("transition_time", duration / 4)
            profile = np.zeros(num_steps)
            for i, ti in enumerate(t):
                if ti < transition_time:
                    progress = ti / transition_time
                    profile[i] = base_value - (1 - min_factor) * base_value * progress
                else:
                    profile[i] = min_factor * base_value

        elif scenario_type == ScenarioType.DEMAND_SURGE:
            surge_start = kwargs.get("surge_start", duration / 3)
            surge_duration = kwargs.get("surge_duration", duration / 6)
            surge_factor = kwargs.get("surge_factor", 1.5)
            profile = np.where(
                (t >= surge_start) & (t < surge_start + surge_duration),
                base_value * surge_factor,
                base_value
            )

        else:
            # 默认稳态
            profile = np.ones(num_steps) * base_value

        return profile

    def generate_parameter_ensemble(
        self,
        base_params: Dict[str, float],
        ensemble_size: int = 10,
        perturbation_config: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[Dict[str, float]]:
        """
        生成参数集合用于鲁棒性测试

        Args:
            base_params: 基准参数
            ensemble_size: 集合大小
            perturbation_config: 扰动配置 {param_name: (min_factor, max_factor)}

        Returns:
            ensemble: 参数集合列表
        """
        if perturbation_config is None:
            perturbation_config = {
                "manning_n": (0.85, 1.15),
                "discharge_coefficient": (0.9, 1.1),
                "bed_slope": (0.95, 1.05),
            }

        ensemble = []
        for i in range(ensemble_size):
            params = base_params.copy()
            for param_name, (min_f, max_f) in perturbation_config.items():
                if param_name in params:
                    factor = np.random.uniform(min_f, max_f)
                    params[param_name] = params[param_name] * factor
            ensemble.append(params)

        return ensemble

    def generate_lateral_disturbances(
        self,
        duration: float,
        disturbance_config: DisturbanceConfig,
        segment_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        生成侧向扰动 (取退水、渗漏等)

        Args:
            duration: 持续时间 (s)
            disturbance_config: 扰动配置
            segment_indices: 需要扰动的渠段索引

        Returns:
            disturbances: {segment_id: [inflow, outflow]} 时间序列
        """
        if segment_indices is None:
            # 默认在部分渠段有侧向扰动
            segment_indices = list(range(0, self.num_segments, 5))

        num_steps = int(duration / self.dt)
        disturbances = {}

        for seg_idx in segment_indices:
            segment_id = f"SEG_{seg_idx:03d}"

            # 使用OU过程生成侧向扰动
            noise_model = NoiseModel(
                noise_type=NoiseType.ORNSTEIN_UHLENBECK,
                amplitude=disturbance_config.lateral_inflow_std,
                mean=disturbance_config.lateral_inflow_mean,
                correlation_time=7200.0,  # 2小时相关时间
            )

            inflow = self.generate_noise(noise_model, num_steps)
            inflow = np.maximum(inflow.flatten(), 0)  # 入流非负

            noise_model.mean = disturbance_config.lateral_outflow_mean
            noise_model.amplitude = disturbance_config.lateral_outflow_std
            outflow = self.generate_noise(noise_model, num_steps)
            outflow = np.maximum(outflow.flatten(), 0)  # 出流非负

            disturbances[segment_id] = {
                "inflow": inflow,
                "outflow": outflow,
            }

        return disturbances

    def create_scenario(
        self,
        scenario_type: ScenarioType,
        duration: float = 86400.0,  # 默认24小时
        base_flow: float = 300.0,   # m³/s
        base_level: float = 4.0,    # m
        ensemble_size: int = 1,
        add_noise: bool = True,
        add_bias: bool = True,
        add_lateral: bool = True,
        **kwargs
    ) -> ScenarioConfig:
        """
        创建完整的测试场景

        Args:
            scenario_type: 场景类型
            duration: 持续时间 (s)
            base_flow: 基准流量 (m³/s)
            base_level: 基准水位 (m)
            ensemble_size: 参数集合大小
            add_noise: 是否添加噪声
            add_bias: 是否添加偏置
            add_lateral: 是否添加侧向扰动
            **kwargs: 场景特定参数

        Returns:
            scenario: 场景配置
        """
        scenario_id = f"{scenario_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        num_steps = int(duration / self.dt)

        # 生成上游边界流量
        upstream_flow = self.generate_boundary_profile(
            scenario_type, base_flow, duration, **kwargs
        )

        # 添加噪声
        if add_noise:
            noise_model = NoiseModel(
                noise_type=NoiseType.WHITE,
                amplitude=self.error_budgets[ErrorType.MEASUREMENT_NOISE].max_amplitude * base_flow,
            )
            upstream_flow += self.generate_noise(noise_model, num_steps).flatten()

        # 生成初始状态
        initial_states = {}
        initial_boundaries = {}

        for i in range(self.num_segments):
            segment_id = f"SEG_{i:03d}"
            # 沿程水位递减
            level_gradient = 0.00004  # 水面比降
            local_level = base_level - i * self.segment_length * level_gradient

            initial_states[segment_id] = SegmentState(
                segment_id=segment_id,
                timestamp=datetime.now(),
                water_levels=np.ones(20) * local_level,
                flow_rates=np.ones(20) * base_flow,
                velocities=np.ones(20) * 1.0,
                upstream_level=local_level,
                downstream_level=local_level - self.segment_length * level_gradient,
                upstream_flow=base_flow,
                downstream_flow=base_flow,
            )

            if i < self.num_segments - 1:
                boundary_id = f"BND_{i:03d}_{i+1:03d}"
                initial_boundaries[boundary_id] = BoundaryCondition(
                    boundary_id=boundary_id,
                    upstream_segment_id=segment_id,
                    downstream_segment_id=f"SEG_{i+1:03d}",
                    timestamp=datetime.now(),
                    water_level=local_level - self.segment_length * level_gradient,
                    flow_rate=base_flow,
                )

        # 侧向扰动
        lateral_inflow_profiles = {}
        if add_lateral:
            disturbance_config = DisturbanceConfig()
            lateral_disturbances = self.generate_lateral_disturbances(
                duration, disturbance_config
            )
            for seg_id, dist in lateral_disturbances.items():
                lateral_inflow_profiles[seg_id] = dist["inflow"] - dist["outflow"]

        # 扰动配置
        disturbance_dict = {
            "add_noise": add_noise,
            "add_bias": add_bias,
            "add_lateral": add_lateral,
            "noise_amplitude": self.error_budgets[ErrorType.MEASUREMENT_NOISE].max_amplitude,
        }

        # 偏置配置
        if add_bias:
            bias_model = BiasModel(
                initial_bias=0.0,
                drift_rate=0.001 / 3600,  # 0.001单位/小时
                max_bias=0.05,
            )
            disturbance_dict["bias_model"] = {
                "drift_rate": bias_model.drift_rate,
                "max_bias": bias_model.max_bias,
            }

        # 评估标准
        pass_criteria = {
            "max_level_deviation": 0.1,  # m
            "max_flow_deviation": 5.0,   # m³/s
            "settling_time": 7200.0,     # s
            "overshoot": 0.05,           # 5%
            "mass_balance_error": 100.0, # m³
        }

        return ScenarioConfig(
            scenario_id=scenario_id,
            name=f"{scenario_type.value} scenario",
            duration=duration,
            initial_states=initial_states,
            initial_boundaries=initial_boundaries,
            upstream_boundary_profile=upstream_flow,
            lateral_inflow_profiles=lateral_inflow_profiles,
            disturbance_config=disturbance_dict,
            parameter_ensemble_size=ensemble_size,
            pass_criteria=pass_criteria,
        )

    def create_ensemble_scenarios(
        self,
        base_scenario: ScenarioConfig,
        ensemble_size: int = 10
    ) -> List[ScenarioConfig]:
        """
        从基础场景创建参数集合

        用于测试控制算法的鲁棒性
        """
        base_params = {
            "manning_n": 0.014,
            "discharge_coefficient": 0.6,
            "bed_slope": 1.0 / 25000,
        }

        param_ensemble = self.generate_parameter_ensemble(base_params, ensemble_size)

        scenarios = []
        for i, params in enumerate(param_ensemble):
            scenario = ScenarioConfig(
                scenario_id=f"{base_scenario.scenario_id}_ens{i:03d}",
                name=f"{base_scenario.name} (ensemble {i})",
                duration=base_scenario.duration,
                initial_states=base_scenario.initial_states.copy(),
                initial_boundaries=base_scenario.initial_boundaries.copy(),
                upstream_boundary_profile=base_scenario.upstream_boundary_profile.copy(),
                lateral_inflow_profiles=base_scenario.lateral_inflow_profiles.copy(),
                disturbance_config={
                    **base_scenario.disturbance_config,
                    "ensemble_params": params,
                },
                parameter_ensemble_size=1,
                pass_criteria=base_scenario.pass_criteria.copy(),
            )
            scenarios.append(scenario)

        return scenarios

    def inject_fault(
        self,
        scenario: ScenarioConfig,
        fault_type: str,
        segment_id: str,
        start_time: float,
        duration: float,
        magnitude: float = 0.1
    ) -> ScenarioConfig:
        """
        向场景注入故障

        Args:
            scenario: 基础场景
            fault_type: 故障类型 ("sensor_drift", "gate_stuck", "level_bias")
            segment_id: 故障渠段
            start_time: 故障开始时间 (s)
            duration: 故障持续时间 (s)
            magnitude: 故障幅度

        Returns:
            modified_scenario: 修改后的场景
        """
        # 复制场景
        modified_scenario = ScenarioConfig(
            scenario_id=f"{scenario.scenario_id}_fault_{fault_type}",
            name=f"{scenario.name} with {fault_type}",
            duration=scenario.duration,
            initial_states=scenario.initial_states.copy(),
            initial_boundaries=scenario.initial_boundaries.copy(),
            upstream_boundary_profile=scenario.upstream_boundary_profile.copy() if scenario.upstream_boundary_profile is not None else None,
            lateral_inflow_profiles=scenario.lateral_inflow_profiles.copy(),
            disturbance_config=scenario.disturbance_config.copy(),
            parameter_ensemble_size=scenario.parameter_ensemble_size,
            pass_criteria=scenario.pass_criteria.copy(),
        )

        # 添加故障配置
        fault_config = {
            "fault_type": fault_type,
            "segment_id": segment_id,
            "start_time": start_time,
            "duration": duration,
            "magnitude": magnitude,
        }

        if "faults" not in modified_scenario.disturbance_config:
            modified_scenario.disturbance_config["faults"] = []
        modified_scenario.disturbance_config["faults"].append(fault_config)

        return modified_scenario

    def export_scenario(self, scenario: ScenarioConfig, filepath: str):
        """导出场景配置到JSON文件"""
        data = {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "duration": scenario.duration,
            "upstream_boundary_profile": scenario.upstream_boundary_profile.tolist() if scenario.upstream_boundary_profile is not None else None,
            "lateral_inflow_profiles": {
                k: v.tolist() for k, v in scenario.lateral_inflow_profiles.items()
            },
            "disturbance_config": scenario.disturbance_config,
            "parameter_ensemble_size": scenario.parameter_ensemble_size,
            "pass_criteria": scenario.pass_criteria,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Scenario exported to {filepath}")

    def get_error_budget_report(self) -> Dict[str, Any]:
        """获取误差预算报告"""
        report = {
            "total_error_types": len(self.error_budgets),
            "budgets": {}
        }

        for error_type, budget in self.error_budgets.items():
            report["budgets"][error_type.value] = budget.to_dict()

        return report

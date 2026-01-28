"""
接口同化器 (Boundary Assimilator)

核心功能:
1. 软边界融合 - IDZ与高保真模型状态融合
2. 守恒纠偏 - 体积/质量守恒约束
3. 偏置估计 - 在线偏置状态跟踪
4. 异常过滤 - 剔除离群值

设计原则:
- 边界不是硬输入,而是带置信度的估计量
- 在接口处加"误差断路器",避免误差沿线传播
- 融合系数自适应,根据工况动态调整
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from scipy.linalg import inv, sqrtm
from collections import deque

from .data_types import (
    BoundaryCondition, AssimilatedState, SegmentState,
    ModelError, ErrorType
)

logger = logging.getLogger(__name__)


class FusionMode(Enum):
    """融合模式"""
    FIXED_WEIGHT = "fixed_weight"          # 固定权重
    ADAPTIVE = "adaptive"                   # 自适应权重
    KALMAN = "kalman"                       # 卡尔曼滤波
    ENSEMBLE = "ensemble"                   # 集合同化


@dataclass
class FusionConfig:
    """融合配置"""
    mode: FusionMode = FusionMode.ADAPTIVE

    # 固定权重 (用于FIXED_WEIGHT模式)
    alpha_flow: float = 0.5        # 流量融合系数 (高保真权重)
    beta_level: float = 0.5        # 水位融合系数 (高保真权重)

    # 自适应参数
    alpha_min: float = 0.2
    alpha_max: float = 0.8
    gate_action_threshold: float = 0.02  # 闸门动作阈值

    # 卡尔曼滤波参数
    process_noise_level: float = 0.01
    process_noise_flow: float = 1.0
    measurement_noise_level: float = 0.02
    measurement_noise_flow: float = 2.0

    # 守恒纠偏参数
    conservation_correction_gain: float = 0.5
    max_volume_correction: float = 1000.0  # m³

    # 异常检测参数
    outlier_threshold: float = 3.0  # 标准差倍数


@dataclass
class InterfaceState:
    """接口状态"""
    boundary_id: str
    timestamp: datetime

    # IDZ模型状态
    level_idz: float = 0.0
    flow_idz: float = 0.0

    # 高保真模型状态
    level_fine: float = 0.0
    flow_fine: float = 0.0

    # 融合后状态
    level_fused: float = 0.0
    flow_fused: float = 0.0

    # 融合系数 (实际使用的)
    alpha: float = 0.5
    beta: float = 0.5

    # 偏置状态
    level_bias: float = 0.0
    flow_bias: float = 0.0

    # 体积误差累计
    volume_error_cumulative: float = 0.0

    # 置信度
    confidence: float = 1.0


class BoundaryAssimilator:
    """
    边界同化器

    负责IDZ与高保真模型之间的状态融合与守恒约束
    核心作用：在接口处切断误差传播
    """

    def __init__(
        self,
        num_interfaces: int = 62,
        config: Optional[FusionConfig] = None,
        dt: float = 900.0,
    ):
        """
        初始化边界同化器

        Args:
            num_interfaces: 接口数量 (渠段数-1)
            config: 融合配置
            dt: 时间步长 (s)
        """
        self.num_interfaces = num_interfaces
        self.config = config or FusionConfig()
        self.dt = dt

        # 接口状态
        self.interface_states: Dict[str, InterfaceState] = {}

        # 历史数据 (用于异常检测和自适应)
        self.history_window = 20
        self.level_history: Dict[str, deque] = {}
        self.flow_history: Dict[str, deque] = {}

        # 卡尔曼滤波器状态
        self.kf_states: Dict[str, np.ndarray] = {}  # [h, q]
        self.kf_covariances: Dict[str, np.ndarray] = {}  # P矩阵

        # 统计
        self.total_corrections = 0
        self.outliers_rejected = 0

        # 初始化接口
        self._init_interfaces()

        logger.info(f"BoundaryAssimilator initialized: {num_interfaces} interfaces")

    def _init_interfaces(self):
        """初始化所有接口"""
        for i in range(self.num_interfaces):
            boundary_id = f"BND_{i:03d}_{i+1:03d}"

            self.interface_states[boundary_id] = InterfaceState(
                boundary_id=boundary_id,
                timestamp=datetime.now(),
            )

            self.level_history[boundary_id] = deque(maxlen=self.history_window)
            self.flow_history[boundary_id] = deque(maxlen=self.history_window)

            # 初始化卡尔曼滤波器
            self.kf_states[boundary_id] = np.array([4.0, 300.0])  # [h, q]
            self.kf_covariances[boundary_id] = np.eye(2) * 0.1

    def assimilate(
        self,
        boundary_id: str,
        idz_state: Dict[str, float],
        fine_state: Optional[Dict[str, float]] = None,
        gate_action: float = 0.0,
        is_control_active: bool = False,
    ) -> AssimilatedState:
        """
        执行状态同化

        Args:
            boundary_id: 接口ID
            idz_state: IDZ模型状态 {"level": h, "flow": q}
            fine_state: 高保真模型状态 (可选)
            gate_action: 闸门动作幅度 (用于自适应权重)
            is_control_active: 是否处于控制激活状态

        Returns:
            assimilated: 同化后的状态
        """
        if boundary_id not in self.interface_states:
            self._init_interface(boundary_id)

        interface = self.interface_states[boundary_id]
        interface.timestamp = datetime.now()

        # 更新IDZ状态
        interface.level_idz = idz_state.get("level", 0.0)
        interface.flow_idz = idz_state.get("flow", 0.0)

        # 如果有高保真状态
        if fine_state is not None:
            interface.level_fine = fine_state.get("level", interface.level_idz)
            interface.flow_fine = fine_state.get("flow", interface.flow_idz)
        else:
            # 没有高保真数据时使用IDZ
            interface.level_fine = interface.level_idz
            interface.flow_fine = interface.flow_idz

        # 异常检测
        is_level_outlier = self._detect_outlier(
            boundary_id, "level", interface.level_fine
        )
        is_flow_outlier = self._detect_outlier(
            boundary_id, "flow", interface.flow_fine
        )

        if is_level_outlier or is_flow_outlier:
            self.outliers_rejected += 1
            logger.warning(f"Outlier detected at {boundary_id}")
            # 使用IDZ值
            interface.level_fine = interface.level_idz
            interface.flow_fine = interface.flow_idz

        # 计算融合系数
        alpha, beta = self._compute_fusion_weights(
            boundary_id, gate_action, is_control_active
        )
        interface.alpha = alpha
        interface.beta = beta

        # 执行融合
        if self.config.mode == FusionMode.KALMAN:
            fused_level, fused_flow, P = self._kalman_fusion(
                boundary_id, interface
            )
        else:
            # 加权融合
            fused_level = (1 - beta) * interface.level_idz + beta * interface.level_fine
            fused_flow = (1 - alpha) * interface.flow_idz + alpha * interface.flow_fine

        # 偏置校正
        fused_level -= interface.level_bias
        fused_flow -= interface.flow_bias

        interface.level_fused = fused_level
        interface.flow_fused = fused_flow

        # 更新历史
        self.level_history[boundary_id].append(fused_level)
        self.flow_history[boundary_id].append(fused_flow)

        # 计算创新量
        innovation_level = interface.level_fine - interface.level_idz
        innovation_flow = interface.flow_fine - interface.flow_idz

        # 构建输出
        return AssimilatedState(
            boundary_id=boundary_id,
            timestamp=datetime.now(),
            estimated_level=fused_level,
            estimated_flow=fused_flow,
            level_variance=self.config.process_noise_level**2,
            flow_variance=self.config.process_noise_flow**2,
            level_flow_covariance=0.0,
            innovation_level=innovation_level,
            innovation_flow=innovation_flow,
            kalman_gain_level=beta,
            kalman_gain_flow=alpha,
            assimilation_quality=interface.confidence,
            rejected_outliers=1 if (is_level_outlier or is_flow_outlier) else 0,
        )

    def _init_interface(self, boundary_id: str):
        """初始化单个接口"""
        self.interface_states[boundary_id] = InterfaceState(
            boundary_id=boundary_id,
            timestamp=datetime.now(),
        )
        self.level_history[boundary_id] = deque(maxlen=self.history_window)
        self.flow_history[boundary_id] = deque(maxlen=self.history_window)
        self.kf_states[boundary_id] = np.array([4.0, 300.0])
        self.kf_covariances[boundary_id] = np.eye(2) * 0.1

    def _compute_fusion_weights(
        self,
        boundary_id: str,
        gate_action: float,
        is_control_active: bool
    ) -> Tuple[float, float]:
        """
        计算自适应融合系数

        规则:
        - 闸门动作剧烈/控制激活: 更信高保真 (α, β增大)
        - 工况平稳: 更信IDZ (α, β减小)
        """
        if self.config.mode == FusionMode.FIXED_WEIGHT:
            return self.config.alpha_flow, self.config.beta_level

        # 基础权重
        alpha = (self.config.alpha_min + self.config.alpha_max) / 2
        beta = (self.config.alpha_min + self.config.alpha_max) / 2

        # 根据闸门动作调整
        if abs(gate_action) > self.config.gate_action_threshold:
            # 闸门动作大,更信高保真
            action_factor = min(abs(gate_action) / self.config.gate_action_threshold, 2.0)
            alpha = min(self.config.alpha_max, alpha + 0.1 * action_factor)
            beta = min(self.config.alpha_max, beta + 0.1 * action_factor)

        # 控制激活时偏向高保真
        if is_control_active:
            alpha = min(self.config.alpha_max, alpha + 0.1)
            beta = min(self.config.alpha_max, beta + 0.1)

        # 根据历史波动调整
        if boundary_id in self.level_history and len(self.level_history[boundary_id]) > 5:
            level_std = np.std(list(self.level_history[boundary_id]))
            if level_std > 0.1:  # 波动大
                beta = max(self.config.alpha_min, beta - 0.1)

        return alpha, beta

    def _kalman_fusion(
        self,
        boundary_id: str,
        interface: InterfaceState
    ) -> Tuple[float, float, np.ndarray]:
        """
        卡尔曼滤波融合

        状态: x = [h, q]
        预测: IDZ模型
        观测: 高保真模型
        """
        # 获取当前状态
        x = self.kf_states[boundary_id]
        P = self.kf_covariances[boundary_id]

        # 状态转移 (简化为恒等)
        F = np.eye(2)
        Q = np.diag([
            self.config.process_noise_level**2,
            self.config.process_noise_flow**2
        ])

        # 预测步 (使用IDZ)
        x_pred = np.array([interface.level_idz, interface.flow_idz])
        P_pred = F @ P @ F.T + Q

        # 观测矩阵
        H = np.eye(2)
        R = np.diag([
            self.config.measurement_noise_level**2,
            self.config.measurement_noise_flow**2
        ])

        # 观测 (高保真)
        z = np.array([interface.level_fine, interface.flow_fine])

        # 卡尔曼增益
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)

        # 更新步
        y = z - H @ x_pred  # 创新
        x_new = x_pred + K @ y
        P_new = (np.eye(2) - K @ H) @ P_pred

        # 保存状态
        self.kf_states[boundary_id] = x_new
        self.kf_covariances[boundary_id] = P_new

        return x_new[0], x_new[1], P_new

    def _detect_outlier(
        self,
        boundary_id: str,
        variable: str,
        value: float
    ) -> bool:
        """检测异常值"""
        history = (
            self.level_history[boundary_id] if variable == "level"
            else self.flow_history[boundary_id]
        )

        if len(history) < 5:
            return False

        hist_array = np.array(list(history))
        mean = np.mean(hist_array)
        std = np.std(hist_array)

        if std < 1e-6:
            return False

        z_score = abs(value - mean) / std
        return z_score > self.config.outlier_threshold

    def apply_conservation_correction(
        self,
        boundary_id: str,
        volume_error: float,
    ) -> float:
        """
        应用守恒纠偏

        将体积误差吃进接口偏置,而不是向下游传播

        Args:
            boundary_id: 接口ID
            volume_error: 体积误差 (m³)

        Returns:
            correction: 实际应用的校正量
        """
        if boundary_id not in self.interface_states:
            return 0.0

        interface = self.interface_states[boundary_id]

        # 限制单次校正量
        correction = np.clip(
            volume_error * self.config.conservation_correction_gain,
            -self.config.max_volume_correction,
            self.config.max_volume_correction
        )

        # 累计体积误差
        interface.volume_error_cumulative += volume_error

        # 转换为流量偏置 (在下一时间步应用)
        flow_bias_increment = correction / self.dt
        interface.flow_bias += flow_bias_increment

        self.total_corrections += 1

        logger.debug(
            f"Conservation correction at {boundary_id}: "
            f"volume_error={volume_error:.2f}m³, correction={correction:.2f}m³"
        )

        return correction

    def update_bias_estimates(
        self,
        boundary_id: str,
        level_innovation: float,
        flow_innovation: float,
        learning_rate: float = 0.01
    ):
        """
        更新偏置估计 (慢变偏置跟踪)

        Args:
            boundary_id: 接口ID
            level_innovation: 水位创新量
            flow_innovation: 流量创新量
            learning_rate: 学习率
        """
        if boundary_id not in self.interface_states:
            return

        interface = self.interface_states[boundary_id]

        # 指数滑动平均更新偏置
        interface.level_bias = (
            (1 - learning_rate) * interface.level_bias +
            learning_rate * level_innovation
        )
        interface.flow_bias = (
            (1 - learning_rate) * interface.flow_bias +
            learning_rate * flow_innovation
        )

    def get_interface_residuals(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有接口的残差统计

        用于SIL KPI监控
        """
        residuals = {}

        for boundary_id, interface in self.interface_states.items():
            level_residual = interface.level_fine - interface.level_idz
            flow_residual = interface.flow_fine - interface.flow_idz

            residuals[boundary_id] = {
                "level_residual": level_residual,
                "flow_residual": flow_residual,
                "level_bias": interface.level_bias,
                "flow_bias": interface.flow_bias,
                "volume_error_cumulative": interface.volume_error_cumulative,
                "fusion_alpha": interface.alpha,
                "fusion_beta": interface.beta,
            }

        return residuals

    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取汇总统计"""
        all_level_residuals = []
        all_flow_residuals = []

        for interface in self.interface_states.values():
            all_level_residuals.append(
                interface.level_fine - interface.level_idz
            )
            all_flow_residuals.append(
                interface.flow_fine - interface.flow_idz
            )

        return {
            "num_interfaces": len(self.interface_states),
            "total_corrections": self.total_corrections,
            "outliers_rejected": self.outliers_rejected,
            "level_residual_mean": np.mean(all_level_residuals) if all_level_residuals else 0.0,
            "level_residual_std": np.std(all_level_residuals) if all_level_residuals else 0.0,
            "level_residual_max": np.max(np.abs(all_level_residuals)) if all_level_residuals else 0.0,
            "flow_residual_mean": np.mean(all_flow_residuals) if all_flow_residuals else 0.0,
            "flow_residual_std": np.std(all_flow_residuals) if all_flow_residuals else 0.0,
            "flow_residual_max": np.max(np.abs(all_flow_residuals)) if all_flow_residuals else 0.0,
        }

    def reset(self):
        """重置所有接口状态"""
        for boundary_id in self.interface_states:
            self.interface_states[boundary_id] = InterfaceState(
                boundary_id=boundary_id,
                timestamp=datetime.now(),
            )
            self.level_history[boundary_id].clear()
            self.flow_history[boundary_id].clear()
            self.kf_states[boundary_id] = np.array([4.0, 300.0])
            self.kf_covariances[boundary_id] = np.eye(2) * 0.1

        self.total_corrections = 0
        self.outliers_rejected = 0

        logger.info("BoundaryAssimilator reset")

    def get_fused_boundary(self, boundary_id: str) -> Optional[BoundaryCondition]:
        """获取融合后的边界条件"""
        if boundary_id not in self.interface_states:
            return None

        interface = self.interface_states[boundary_id]

        # 提取上下游段ID
        parts = boundary_id.split("_")
        upstream_id = f"SEG_{parts[1]}"
        downstream_id = f"SEG_{parts[2]}"

        return BoundaryCondition(
            boundary_id=boundary_id,
            upstream_segment_id=upstream_id,
            downstream_segment_id=downstream_id,
            timestamp=interface.timestamp,
            water_level=interface.level_fused,
            flow_rate=interface.flow_fused,
            level_std=self.config.process_noise_level,
            flow_std=self.config.process_noise_flow,
            confidence=interface.confidence,
            is_measured=False,
            is_filtered=True,
            filter_type=self.config.mode.value,
            level_bias=interface.level_bias,
            flow_bias=interface.flow_bias,
        )

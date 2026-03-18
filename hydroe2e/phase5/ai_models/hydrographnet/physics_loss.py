"""
物理守恒损失函数 (Physics Conservation Loss)
基于 HydroGraphNet 的物理约束损失，适配水网世界模型

包含的物理约束:
1. 全局质量守恒: ΔV = Q_in - Q_out + Rainfall
2. 连续性方程: ∂ρ/∂t + ∇·(ρv) = 0
3. 水量平衡: dZ/dt = (Q_in - Q_out) / A

参考:
- HydroGraphNet: physics_loss/global_mass_conservation.py
- Saint-Venant 方程
- 曼宁公式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# 全局质量守恒损失 (来自 HydroGraphNet)
# ==============================================================================

class GlobalMassConservationLoss(nn.Module):
    """
    全局质量守恒损失

    确保预测的水量变化满足质量守恒:
    ΔV = Q_in * Δt - Q_out * Δt + Rainfall

    模式:
    - 训练模式: 返回绝对值损失 (凸函数，便于优化)
    - 测试模式: 返回带符号的原始值 (便于分析)
    """

    def __init__(self,
                 mode: Literal['train', 'test'] = 'train',
                 delta_t: float = 900.0):
        """
        Args:
            mode: 'train' 或 'test'
            delta_t: 时间步长 [秒] (默认15分钟)
        """
        super().__init__()
        self.mode = mode
        self.delta_t = delta_t

    def forward(self,
                pred_volume: torch.Tensor,      # 预测的水量 (t+1)
                current_volume: torch.Tensor,   # 当前水量 (t)
                total_inflow: torch.Tensor,     # 总入流 [m³/s]
                total_outflow: torch.Tensor,    # 总出流 [m³/s]
                total_rainfall: torch.Tensor = None,  # 总降雨体积 [m³]
                ) -> torch.Tensor:
        """
        计算全局质量守恒损失

        Args:
            pred_volume: 预测的下一时刻总水量 [m³]
            current_volume: 当前总水量 [m³]
            total_inflow: 入流流量 [m³/s]
            total_outflow: 出流流量 [m³/s]
            total_rainfall: 降雨贡献体积 [m³] (可选)

        Returns:
            质量守恒误差
        """
        # 计算水量变化
        delta_v = pred_volume - current_volume

        # 计算理论水量变化
        inflow_volume = total_inflow * self.delta_t
        outflow_volume = total_outflow * self.delta_t

        if total_rainfall is not None:
            rf_volume = total_rainfall
        else:
            rf_volume = torch.zeros_like(delta_v)

        # 质量守恒误差
        # ΔV_pred = ΔV_theory => ΔV_pred - (Q_in - Q_out)*Δt - RF = 0
        global_volume_error = delta_v - inflow_volume + outflow_volume - rf_volume

        if self.mode == 'train':
            global_volume_error = torch.abs(global_volume_error)

        return global_volume_error.mean()


# ==============================================================================
# 连续性方程损失
# ==============================================================================

class ContinuityEquationLoss(nn.Module):
    """
    连续性方程损失

    连续性方程: ∂ρ/∂t + ∇·(ρv) = 0
    对于不可压缩流体: ∇·v = 0

    简化形式 (一维): ∂A/∂t + ∂Q/∂x = q
    其中:
    - A: 过水断面面积
    - Q: 流量
    - q: 侧向入流 (降雨、分水等)
    """

    def __init__(self,
                 dx: float = 1000.0,  # 空间步长 [m]
                 dt: float = 900.0):  # 时间步长 [s]
        super().__init__()
        self.dx = dx
        self.dt = dt

    def forward(self,
                pred_area: torch.Tensor,      # 预测断面面积 [batch, num_nodes]
                current_area: torch.Tensor,   # 当前断面面积
                flow_upstream: torch.Tensor,  # 上游流量 [batch, num_nodes]
                flow_downstream: torch.Tensor,  # 下游流量
                lateral_inflow: torch.Tensor = None,  # 侧向入流
                ) -> torch.Tensor:
        """
        计算连续性方程残差

        ∂A/∂t + ∂Q/∂x = q
        => (A_new - A_old)/Δt + (Q_down - Q_up)/Δx = q
        """
        # 时间导数
        dA_dt = (pred_area - current_area) / self.dt

        # 空间导数
        dQ_dx = (flow_downstream - flow_upstream) / self.dx

        # 侧向入流
        if lateral_inflow is None:
            lateral_inflow = torch.zeros_like(dA_dt)

        # 连续性方程残差
        residual = dA_dt + dQ_dx - lateral_inflow

        return (residual ** 2).mean()


# ==============================================================================
# 水量平衡损失 (适配 LSTM 模型)
# ==============================================================================

class WaterBalanceLoss(nn.Module):
    """
    水量平衡损失

    适配现有 LSTM 模型的物理约束损失
    基于渠池的积分形式: dZ/dt = (Q_in - Q_out) / A_s

    可作为 neural_physics_engine.py 中 PINN Loss 的补充
    """

    def __init__(self,
                 dt: float = 900.0,
                 surface_area: float = 100000.0,
                 mode: str = 'soft'):
        """
        Args:
            dt: 时间步长 [s]
            surface_area: 水面面积 [m²]
            mode: 'soft' (L2损失) 或 'hard' (ReLU惩罚)
        """
        super().__init__()
        self.dt = dt
        self.A_s = surface_area
        self.mode = mode

    def forward(self,
                pred_level: torch.Tensor,     # 预测水位 [batch, 1]
                current_level: torch.Tensor,  # 当前水位 [batch, 1]
                q_in: torch.Tensor,           # 入流 [batch, 1]
                q_out: torch.Tensor,          # 出流 [batch, 1]
                ) -> torch.Tensor:
        """
        计算水量平衡损失

        理论水位变化: ΔZ = (Q_in - Q_out) * Δt / A_s
        """
        # 理论水位变化
        expected_dZ = (q_in - q_out) * self.dt / self.A_s

        # 实际水位变化
        actual_dZ = pred_level - current_level

        # 残差
        residual = actual_dZ - expected_dZ

        if self.mode == 'soft':
            return (residual ** 2).mean()
        else:  # hard
            return F.relu(torch.abs(residual) - 0.01).mean()


# ==============================================================================
# 综合物理损失 (HydroPhysicsLoss)
# ==============================================================================

class HydroPhysicsLoss(nn.Module):
    """
    综合水力学物理损失

    结合多种物理约束:
    1. 质量守恒
    2. 水量平衡
    3. 流量-水位关系 (曼宁公式)
    4. 动量守恒 (可选)

    用于增强神经网络模型的物理一致性
    """

    def __init__(self,
                 dt: float = 900.0,
                 mass_weight: float = 1.0,
                 balance_weight: float = 1.0,
                 manning_weight: float = 0.1,
                 momentum_weight: float = 0.0):
        """
        Args:
            dt: 时间步长 [s]
            mass_weight: 质量守恒权重
            balance_weight: 水量平衡权重
            manning_weight: 曼宁公式权重
            momentum_weight: 动量守恒权重
        """
        super().__init__()
        self.dt = dt
        self.mass_weight = mass_weight
        self.balance_weight = balance_weight
        self.manning_weight = manning_weight
        self.momentum_weight = momentum_weight

        # 子损失函数
        self.mass_loss = GlobalMassConservationLoss(mode='train', delta_t=dt)
        self.balance_loss = WaterBalanceLoss(dt=dt)

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                physics_params: Dict[str, torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        计算综合物理损失

        Args:
            predictions: 模型预测
                - 'level': 预测水位 [batch, 1]
                - 'flow': 预测流量 [batch, 1] (可选)
            targets: 目标值
                - 'level': 目标水位
                - 'flow': 目标流量 (可选)
            physics_params: 物理参数
                - 'q_in': 入流
                - 'q_out': 出流
                - 'current_level': 当前水位
                - 'surface_area': 水面面积
                - 'manning_n': 曼宁系数
                - 'slope': 底坡

        Returns:
            损失字典
        """
        losses = {}

        pred_level = predictions.get('level')
        current_level = physics_params.get('current_level') if physics_params else None
        q_in = physics_params.get('q_in') if physics_params else None
        q_out = physics_params.get('q_out') if physics_params else None

        # 1. 数据损失 (MSE)
        if pred_level is not None and 'level' in targets:
            data_loss = F.mse_loss(pred_level, targets['level'])
            losses['data_loss'] = data_loss

        # 2. 水量平衡损失
        if (pred_level is not None and current_level is not None and
            q_in is not None and q_out is not None):
            balance_loss = self.balance_loss(pred_level, current_level, q_in, q_out)
            losses['balance_loss'] = balance_loss * self.balance_weight

        # 3. 质量守恒损失 (全局)
        if physics_params and 'total_volume' in physics_params:
            pred_volume = physics_params.get('pred_volume', pred_level.sum())
            current_volume = physics_params.get('current_volume', current_level.sum())
            total_inflow = physics_params.get('total_inflow', q_in.sum())
            total_outflow = physics_params.get('total_outflow', q_out.sum())

            mass_loss = self.mass_loss(
                pred_volume, current_volume, total_inflow, total_outflow
            )
            losses['mass_loss'] = mass_loss * self.mass_weight

        # 4. 曼宁公式约束 (流量-水位关系)
        if (self.manning_weight > 0 and 'flow' in predictions and
            physics_params and 'manning_n' in physics_params):
            pred_flow = predictions['flow']
            manning_n = physics_params['manning_n']
            slope = physics_params.get('slope', 0.001)
            width = physics_params.get('width', 50.0)

            # 简化曼宁公式: Q = (1/n) * A * R^(2/3) * S^(1/2)
            # 近似: Q ≈ k * h^(5/3) 其中 h 是水深
            h = pred_level.clamp(min=0.1)
            expected_flow = (1.0 / manning_n) * width * h * (h ** (2/3)) * (slope ** 0.5)

            manning_loss = F.mse_loss(pred_flow, expected_flow)
            losses['manning_loss'] = manning_loss * self.manning_weight

        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses


# ==============================================================================
# 图结构物理损失 (用于 GNN)
# ==============================================================================

class GraphPhysicsLoss(nn.Module):
    """
    图结构物理损失

    专为图神经网络设计的物理约束损失
    处理节点级和边级的物理约束
    """

    def __init__(self, delta_t: float = 900.0):
        super().__init__()
        self.delta_t = delta_t

    def forward(self,
                pred: torch.Tensor,           # 节点预测 [num_nodes, features]
                physics_data: Dict[str, torch.Tensor],
                batch_vector: torch.Tensor,   # 批次归属向量
                ) -> torch.Tensor:
        """
        计算图结构物理损失

        参考 HydroGraphNet 的 compute_physics_loss 实现
        """
        unique_ids = torch.unique(batch_vector)
        physics_losses = []

        # 假设 pred[:, 1] 是体积变化预测
        predicted_diff = pred[:, 1] if pred.dim() > 1 else pred

        for uid in unique_ids:
            mask = (batch_vector == uid)
            pred_diff_sum = predicted_diff[mask].sum()

            idx = (unique_ids == uid).nonzero(as_tuple=False).item()

            # 获取物理参数
            past_volume = physics_data.get("past_volume", torch.zeros(len(unique_ids)))[idx]
            future_volume = physics_data.get("future_volume", torch.zeros(len(unique_ids)))[idx]
            inflow = physics_data.get("avg_inflow", torch.zeros(len(unique_ids)))[idx]
            precip = physics_data.get("avg_precipitation", torch.zeros(len(unique_ids)))[idx]
            volume_std = physics_data.get("volume_std", torch.ones(len(unique_ids)))[idx]
            area_sum = physics_data.get("area_sum", torch.ones(len(unique_ids)))[idx]

            # 预测总体积
            pred_total_volume = past_volume + volume_std * pred_diff_sum

            # 连续性约束
            term1 = F.relu((pred_total_volume - (
                past_volume + self.delta_t * inflow)) / area_sum) ** 2
            term2 = F.relu((future_volume - pred_total_volume - self.delta_t * (
                inflow + precip)) / area_sum) ** 2

            physics_losses.append(term1 + term2)

        if physics_losses:
            return torch.stack(physics_losses).mean()
        else:
            return torch.tensor(0.0, device=pred.device)


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info(" " * 15 + "物理守恒损失测试")
    logger.info("=" * 70)

    batch_size = 32

    # 1. 测试 GlobalMassConservationLoss
    logger.info("\n1. GlobalMassConservationLoss 测试")
    logger.info("-" * 50)

    mass_loss = GlobalMassConservationLoss(mode='train', delta_t=900.0)

    pred_volume = torch.randn(batch_size) * 1000 + 10000
    current_volume = torch.randn(batch_size) * 1000 + 10000
    q_in = torch.ones(batch_size) * 100
    q_out = torch.ones(batch_size) * 100

    loss = mass_loss(pred_volume, current_volume, q_in, q_out)
    logger.info(f"  质量守恒损失: {loss.item():.4f}")

    # 2. 测试 WaterBalanceLoss
    logger.info("\n2. WaterBalanceLoss 测试")
    logger.info("-" * 50)

    balance_loss = WaterBalanceLoss(dt=900.0, surface_area=100000.0)

    pred_level = torch.randn(batch_size, 1) * 0.5 + 4.0
    current_level = torch.randn(batch_size, 1) * 0.5 + 4.0
    q_in = torch.ones(batch_size, 1) * 100
    q_out = torch.ones(batch_size, 1) * 100

    loss = balance_loss(pred_level, current_level, q_in, q_out)
    logger.info(f"  水量平衡损失: {loss.item():.4f}")

    # 3. 测试 HydroPhysicsLoss
    logger.info("\n3. HydroPhysicsLoss 测试")
    logger.info("-" * 50)

    hydro_loss = HydroPhysicsLoss(
        dt=900.0,
        mass_weight=1.0,
        balance_weight=1.0,
        manning_weight=0.0
    )

    predictions = {'level': pred_level}
    targets = {'level': current_level + 0.1}
    physics_params = {
        'current_level': current_level,
        'q_in': q_in,
        'q_out': q_out,
    }

    losses = hydro_loss(predictions, targets, physics_params)
    logger.info(f"  数据损失: {losses.get('data_loss', 0):.4f}")
    logger.info(f"  平衡损失: {losses.get('balance_loss', 0):.4f}")
    logger.info(f"  总损失: {losses['total_loss'].item():.4f}")

    # 4. 验证质量守恒
    logger.info("\n4. 质量守恒验证")
    logger.info("-" * 50)

    # 理想情况: 入流=出流，水量不变
    pred_v = torch.tensor([10000.0])
    curr_v = torch.tensor([10000.0])
    q_in_t = torch.tensor([100.0])
    q_out_t = torch.tensor([100.0])

    loss_balanced = mass_loss(pred_v, curr_v, q_in_t, q_out_t)
    logger.info(f"  平衡状态损失: {loss_balanced.item():.6f}")

    # 非平衡情况: 入流>出流，水量应增加
    pred_v_wrong = torch.tensor([10000.0])  # 预测不变 (错误)
    q_in_large = torch.tensor([200.0])  # 入流增大

    loss_unbalanced = mass_loss(pred_v_wrong, curr_v, q_in_large, q_out_t)
    logger.info(f"  非平衡状态损失: {loss_unbalanced.item():.6f}")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)

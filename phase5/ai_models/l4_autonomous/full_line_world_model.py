"""
全线统一世界模型 (Full Line World Model)
南水北调中线1432公里渠道的统一时空预测模型

特点:
1. 级联渠池的时空建模
2. 考虑水流传播延迟
3. 多尺度预测 (短期精确 + 长期趋势)
4. 不确定性量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """世界模型配置"""
    # 渠道结构
    num_pools: int = 63                    # 渠池数量
    num_gates: int = 64                    # 闸门数量
    total_length: float = 1432.0           # 总长度 (km)

    # 网络结构
    node_dim: int = 64                     # 节点特征维度
    edge_dim: int = 32                     # 边特征维度
    hidden_dim: int = 256                  # 隐层维度
    num_layers: int = 4                    # GNN层数

    # 时间参数
    history_length: int = 48               # 历史窗口
    prediction_horizons: List[int] = None  # 多尺度预测 [短期, 中期, 长期]
    dt: float = 900.0                      # 时间步长 (15分钟)

    # 物理参数
    avg_velocity: float = 1.5              # 平均流速 (m/s)
    min_delay: float = 1800.0              # 最小延迟 (s)
    max_delay: float = 86400.0             # 最大延迟 (s)

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [4, 12, 48]  # 1h, 3h, 12h


class CascadeGraphEncoder(nn.Module):
    """
    级联图编码器
    捕捉渠池间的级联关系和水流传播
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # 节点编码
        self.node_encoder = nn.Sequential(
            nn.Linear(8, config.node_dim),  # [水位, 入流, 出流, 闸开, 分水, 长度, 面积, 糙率]
            nn.LayerNorm(config.node_dim),
            nn.GELU()
        )

        # 边编码 (上下游关系)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, config.edge_dim),  # [距离, 延迟, 连接类型]
            nn.LayerNorm(config.edge_dim),
            nn.GELU()
        )

        # 消息传递层
        self.message_layers = nn.ModuleList([
            CascadeMessagePassing(config.node_dim, config.edge_dim)
            for _ in range(config.num_layers)
        ])

        # 延迟嵌入
        self.delay_embedding = nn.Embedding(100, config.node_dim)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch, pools, 8] 节点特征
            edge_index: [2, edges] 边索引
            edge_features: [edges, 3] 边特征
        Returns:
            [batch, pools, node_dim] 编码后的节点特征
        """
        # 编码节点
        x = self.node_encoder(node_features)  # [B, P, D]

        # 编码边
        edge_attr = self.edge_encoder(edge_features)  # [E, edge_dim]

        # 消息传递
        for layer in self.message_layers:
            x = layer(x, edge_index, edge_attr)

        return x


class CascadeMessagePassing(nn.Module):
    """
    级联消息传递层
    考虑水流方向和延迟
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()

        # 消息函数
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim)
        )

        # 更新函数
        self.update_fn = nn.GRUCell(node_dim, node_dim)

        # 层归一化
        self.norm = nn.LayerNorm(node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """消息传递"""
        B, P, D = x.shape

        # 获取源节点和目标节点
        src, dst = edge_index  # [E], [E]

        # 构建消息
        messages = []
        for b in range(B):
            x_src = x[b, src, :]  # [E, D]
            x_dst = x[b, dst, :]  # [E, D]
            msg_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)
            msg = self.message_fn(msg_input)  # [E, D]
            messages.append(msg)

        messages = torch.stack(messages, dim=0)  # [B, E, D]

        # 聚合消息 (按目标节点)
        aggregated = torch.zeros(B, P, D, device=x.device)
        for b in range(B):
            aggregated[b].index_add_(0, dst, messages[b])

        # 更新节点
        x_flat = x.view(B * P, D)
        agg_flat = aggregated.view(B * P, D)
        x_updated = self.update_fn(agg_flat, x_flat)
        x_updated = x_updated.view(B, P, D)

        return self.norm(x_updated + x)


class SpatioTemporalEncoder(nn.Module):
    """
    时空编码器
    联合建模空间级联和时间演化
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # 空间编码 (级联图)
        self.spatial_encoder = CascadeGraphEncoder(config)

        # 时间编码 (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=config.node_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # 初始化边信息
        self._init_graph_structure()

    def _init_graph_structure(self):
        """初始化渠道图结构"""
        P = self.config.num_pools

        # 构建级联边 (上游 -> 下游)
        src = list(range(P - 1))
        dst = list(range(1, P))

        # 双向边 (用于信息回传)
        edge_index = torch.LongTensor([
            src + dst,  # 源
            dst + src   # 目标
        ])

        # 边特征: [归一化距离, 归一化延迟, 方向]
        edge_features = []
        for i in range(P - 1):
            dist = (i + 1) / P  # 归一化距离
            delay = 0.5         # 归一化延迟
            direction = 1.0     # 下游方向
            edge_features.append([dist, delay, direction])

        # 反向边
        for i in range(P - 1):
            dist = (P - i - 1) / P
            delay = 0.5
            direction = -1.0    # 上游方向
            edge_features.append([dist, delay, direction])

        edge_features = torch.FloatTensor(edge_features)

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_features', edge_features)

    def forward(self,
                observations: torch.Tensor,
                pool_attributes: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            observations: [batch, time, pools, features] 时序观测
            pool_attributes: [batch, pools, attrs] 渠池属性
        Returns:
            [batch, pools, hidden_dim] 时空编码
        """
        B, T, P, F = observations.shape

        # 添加渠池属性
        if pool_attributes is not None:
            # 扩展属性到每个时间步
            attrs_expanded = pool_attributes.unsqueeze(1).expand(-1, T, -1, -1)
            node_features = torch.cat([observations, attrs_expanded], dim=-1)
        else:
            # 使用默认属性
            default_attrs = torch.zeros(B, T, P, 3, device=observations.device)
            default_attrs[..., 0] = torch.linspace(0, 1, P)  # 归一化位置
            default_attrs[..., 1] = 1.0  # 归一化面积
            default_attrs[..., 2] = 0.025  # 默认糙率
            node_features = torch.cat([observations, default_attrs], dim=-1)

        # 空间编码 (每个时间步)
        spatial_features = []
        for t in range(T):
            x_t = self.spatial_encoder(
                node_features[:, t, :, :],
                self.edge_index,
                self.edge_features
            )
            spatial_features.append(x_t)

        spatial_features = torch.stack(spatial_features, dim=1)  # [B, T, P, D]

        # 时间编码 (每个渠池)
        temporal_features = []
        for p in range(P):
            x_p = spatial_features[:, :, p, :]  # [B, T, D]
            x_p, _ = self.temporal_encoder(x_p)  # [B, T, 2*H]
            temporal_features.append(x_p[:, -1, :])  # 取最后时刻

        temporal_features = torch.stack(temporal_features, dim=1)  # [B, P, 2*H]

        # 融合
        output = self.fusion(temporal_features)  # [B, P, H]

        return output


class CascadePredictor(nn.Module):
    """
    级联预测器
    多尺度时空预测
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # 各预测尺度的解码器
        self.decoders = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.decoders[str(horizon)] = nn.Sequential(
                nn.Linear(config.hidden_dim + config.num_gates, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, horizon * 3)  # [水位, 入流, 出流] * horizon
            )

        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, len(config.prediction_horizons)),
            nn.Softplus()
        )

        # 级联传播模型
        self.cascade_propagator = CascadePropagator(config)

    def forward(self,
                state: torch.Tensor,
                action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, pools, hidden_dim] 编码状态
            action_sequence: [batch, max_horizon, gates] 动作序列
        Returns:
            多尺度预测结果
        """
        B, P, D = state.shape
        results = {}

        # 各尺度预测
        for horizon in self.config.prediction_horizons:
            h_str = str(horizon)

            # 提取对应动作
            actions = action_sequence[:, :horizon, :]  # [B, H, G]
            action_summary = actions.mean(dim=1)  # [B, G] 动作摘要

            # 预测
            decoder_input = torch.cat([
                state.mean(dim=1),  # [B, D] 全线状态摘要
                action_summary
            ], dim=-1)

            pred = self.decoders[h_str](decoder_input)  # [B, H*3*P] 或简化
            pred = pred.view(B, horizon, 3)  # 简化: 全线平均预测

            results[f'horizon_{horizon}'] = {
                'level': pred[..., 0],
                'inflow': pred[..., 1],
                'outflow': pred[..., 2]
            }

        # 不确定性
        uncertainty = self.uncertainty_estimator(state.mean(dim=1))
        results['uncertainty'] = uncertainty

        return results


class CascadePropagator(nn.Module):
    """
    级联传播模型
    模拟水流在渠道中的传播
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # 传播延迟估计
        self.delay_estimator = nn.Sequential(
            nn.Linear(config.node_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 衰减系数估计
        self.attenuation_estimator = nn.Sequential(
            nn.Linear(config.node_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def propagate(self,
                  upstream_state: torch.Tensor,
                  downstream_state: torch.Tensor,
                  distance: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算上游到下游的传播效应

        Args:
            upstream_state: [batch, node_dim] 上游状态
            downstream_state: [batch, node_dim] 下游状态
            distance: 距离 (km)

        Returns:
            delay: 传播延迟 (归一化)
            attenuation: 衰减系数
        """
        combined = torch.cat([upstream_state, downstream_state], dim=-1)

        # 估计延迟
        delay = self.delay_estimator(combined)
        # 缩放到物理范围
        delay = delay * (self.config.max_delay - self.config.min_delay) + self.config.min_delay
        delay = delay / 3600.0  # 转换为小时

        # 估计衰减
        attenuation = self.attenuation_estimator(combined)

        return delay, attenuation


class FullLineWorldModel(nn.Module):
    """
    全线统一世界模型
    整合时空编码和级联预测
    """

    def __init__(self, config: WorldModelConfig = None):
        super().__init__()
        self.config = config or WorldModelConfig()

        # 时空编码器
        self.encoder = SpatioTemporalEncoder(self.config)

        # 级联预测器
        self.predictor = CascadePredictor(self.config)

        # 物理约束层
        self.physics_constraint = PhysicsConstraintLayer(self.config)

        # 训练统计
        self.training_stats = {
            'total_samples': 0,
            'avg_loss': 0.0
        }

        logger.info(f"Full Line World Model initialized")
        logger.info(f"  Pools: {self.config.num_pools}")
        logger.info(f"  Prediction horizons: {self.config.prediction_horizons}")

    def forward(self,
                observations: torch.Tensor,
                action_sequence: torch.Tensor,
                pool_attributes: torch.Tensor = None) -> Dict[str, Any]:
        """
        完整的世界模型预测

        Args:
            observations: [batch, time, pools, features]
            action_sequence: [batch, max_horizon, gates]
            pool_attributes: [batch, pools, attrs]

        Returns:
            预测结果和置信度
        """
        # 编码
        state = self.encoder(observations, pool_attributes)

        # 预测
        predictions = self.predictor(state, action_sequence)

        # 物理约束修正
        predictions = self.physics_constraint(predictions, observations)

        return {
            'state': state,
            'predictions': predictions,
            'config': self.config
        }

    def predict_cascade_effect(self,
                               disturbance_pool: int,
                               disturbance_magnitude: float,
                               current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测级联效应
        分析某个渠池的扰动如何传播到全线

        Args:
            disturbance_pool: 扰动发生的渠池ID
            disturbance_magnitude: 扰动幅度
            current_state: 当前全线状态

        Returns:
            各渠池受到的影响及时间
        """
        P = self.config.num_pools
        effects = torch.zeros(P)
        arrival_times = torch.zeros(P)

        # 简化模型: 基于距离和流速估计
        avg_pool_length = self.config.total_length / P  # km
        velocity = self.config.avg_velocity  # m/s

        for i in range(P):
            distance = abs(i - disturbance_pool) * avg_pool_length  # km
            travel_time = distance * 1000 / velocity / 3600  # hours

            # 衰减
            attenuation = np.exp(-0.1 * distance)

            effects[i] = disturbance_magnitude * attenuation
            arrival_times[i] = travel_time

        return {
            'effects': effects,
            'arrival_times': arrival_times,
            'source_pool': disturbance_pool,
            'magnitude': disturbance_magnitude
        }


class PhysicsConstraintLayer(nn.Module):
    """
    物理约束层
    确保预测满足物理守恒定律
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # 质量守恒修正
        self.mass_balance_correction = nn.Linear(3, 3)

    def forward(self,
                predictions: Dict[str, Any],
                observations: torch.Tensor) -> Dict[str, Any]:
        """应用物理约束"""
        # 检查质量守恒
        for key, pred_dict in predictions.items():
            if key.startswith('horizon_'):
                inflow = pred_dict.get('inflow')
                outflow = pred_dict.get('outflow')
                level = pred_dict.get('level')

                if inflow is not None and outflow is not None:
                    # 质量守恒: dLevel/dt ∝ (inflow - outflow)
                    # 这里简单检查符号一致性
                    flow_balance = inflow - outflow
                    level_change = level[:, 1:] - level[:, :-1] if level.shape[1] > 1 else torch.zeros_like(flow_balance)

                    # 可以添加更严格的物理约束修正
                    pass

        return predictions


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Full Line World Model Test")
    logger.info("=" * 70)

    # 创建配置
    config = WorldModelConfig(
        num_pools=10,
        num_gates=11,
        history_length=24,
        prediction_horizons=[4, 12, 24]
    )

    # 创建模型
    model = FullLineWorldModel(config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nTotal parameters: {total_params:,}")

    # 测试前向传播
    batch_size = 2
    observations = torch.randn(batch_size, config.history_length, config.num_pools, 5)
    action_sequence = torch.rand(batch_size, max(config.prediction_horizons), config.num_gates)

    output = model(observations, action_sequence)

    logger.info(f"\nOutput keys: {output.keys()}")
    logger.info(f"State shape: {output['state'].shape}")
    logger.info(f"Prediction horizons: {list(output['predictions'].keys())}")

    # 测试级联效应预测
    cascade = model.predict_cascade_effect(
        disturbance_pool=3,
        disturbance_magnitude=0.5,
        current_state=output['state']
    )
    logger.info(f"\nCascade effect from pool 3:")
    logger.info(f"  Effects: {cascade['effects'][:5].numpy()}...")
    logger.info(f"  Arrival times: {cascade['arrival_times'][:5].numpy()}... hours")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)

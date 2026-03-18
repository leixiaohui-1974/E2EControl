"""
端到端自主控制器 (E2E Autonomous Controller)
南水北调中线L4级自主运行核心组件

设计理念:
1. 感知-预测-决策-执行 全链路自主
2. 多尺度时空推理
3. 不确定性感知决策
4. 可解释性保障
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """自主等级"""
    L0_MANUAL = 0           # 完全人工
    L1_ASSISTED = 1         # 辅助决策
    L2_PARTIAL = 2          # 部分自动 (PID/MPC)
    L3_CONDITIONAL = 3      # 条件自动 (神经代理)
    L4_HIGH = 4             # 高度自动 (端到端自主)
    L5_FULL = 5             # 完全自动


@dataclass
class AutonomousConfig:
    """自主控制配置"""
    # 网络结构
    num_pools: int = 63                    # 中线渠池数量
    num_gates: int = 64                    # 闸门数量
    state_dim: int = 128                   # 状态编码维度
    action_dim: int = 64                   # 动作编码维度
    hidden_dim: int = 256                  # 隐层维度
    num_heads: int = 8                     # 注意力头数
    num_layers: int = 6                    # Transformer层数

    # 时空参数
    history_length: int = 96               # 历史窗口 (96*15min=24h)
    prediction_horizon: int = 48           # 预测视野 (48*15min=12h)
    dt: float = 900.0                      # 时间步长 (15分钟)

    # 决策参数
    confidence_threshold: float = 0.3      # 自主决策置信度阈值 (未训练模型使用较低阈值)
    safety_margin: float = 0.2             # 安全裕度

    # 目标权重
    level_weight: float = 1.0              # 水位控制权重
    flow_weight: float = 0.5               # 流量平稳权重
    energy_weight: float = 0.1             # 能耗优化权重
    safety_weight: float = 10.0            # 安全约束权重

    # 学习参数
    learning_rate: float = 1e-4
    gamma: float = 0.99                    # 折扣因子
    tau: float = 0.005                     # 软更新系数


class SpatioTemporalAttention(nn.Module):
    """
    时空注意力机制
    捕捉渠池间的空间关联和时间演化
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # 空间注意力 (渠池间)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=config.state_dim,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 时间注意力 (历史序列)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.state_dim,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.state_dim),
            nn.LayerNorm(config.state_dim),
            nn.GELU()
        )

        # 位置编码
        self.spatial_pos = nn.Parameter(
            torch.randn(1, config.num_pools, config.state_dim) * 0.02
        )
        self.temporal_pos = nn.Parameter(
            torch.randn(1, config.history_length, config.state_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time, pools, state_dim]
        Returns:
            [batch, pools, state_dim]
        """
        B, T, P, D = x.shape

        # 空间注意力: 每个时刻的渠池间关系
        spatial_out = []
        for t in range(T):
            x_t = x[:, t, :, :] + self.spatial_pos[:, :P, :]
            attn_out, _ = self.spatial_attention(x_t, x_t, x_t)
            spatial_out.append(attn_out)
        spatial_features = torch.stack(spatial_out, dim=1)  # [B, T, P, D]

        # 时间注意力: 每个渠池的历史演化
        temporal_out = []
        for p in range(P):
            x_p = spatial_features[:, :, p, :] + self.temporal_pos[:, :T, :]
            attn_out, _ = self.temporal_attention(x_p, x_p, x_p)
            temporal_out.append(attn_out[:, -1, :])  # 取最后时刻
        temporal_features = torch.stack(temporal_out, dim=1)  # [B, P, D]

        # 融合最后时刻的空间特征
        last_spatial = spatial_features[:, -1, :, :]  # [B, P, D]

        # 时空融合
        fused = self.fusion(torch.cat([last_spatial, temporal_features], dim=-1))

        return fused


class PerceptionModule(nn.Module):
    """
    感知模块
    整合多源观测数据
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # 观测编码器
        # 输入: [水位, 入流, 出流, 闸门开度, 分水流量, 传感器状态]
        self.obs_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, config.state_dim),
            nn.LayerNorm(config.state_dim)
        )

        # 外部条件编码器 (天气、需水计划等)
        self.external_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, config.state_dim)
        )

        # 时空注意力
        self.st_attention = SpatioTemporalAttention(config)

    def forward(self,
                observations: torch.Tensor,
                external_conditions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            observations: [batch, time, pools, 6] 渠池观测
            external_conditions: [batch, 16] 外部条件
        Returns:
            state: [batch, pools, state_dim] 全线状态表示
        """
        B, T, P, _ = observations.shape

        # 编码观测
        obs_encoded = self.obs_encoder(observations)  # [B, T, P, state_dim]

        # 时空注意力
        state = self.st_attention(obs_encoded)  # [B, P, state_dim]

        # 融合外部条件
        if external_conditions is not None:
            ext_encoded = self.external_encoder(external_conditions)  # [B, state_dim]
            state = state + ext_encoded.unsqueeze(1)  # 广播到所有渠池

        return state


class PredictionModule(nn.Module):
    """
    预测模块
    多步时空预测
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.state_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers // 2
        )

        # 动作条件编码
        self.action_encoder = nn.Linear(config.num_gates, config.state_dim)

        # 预测头
        self.level_head = nn.Linear(config.state_dim, 1)
        self.flow_head = nn.Linear(config.state_dim, 2)  # 入流, 出流

        # 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # 位置编码
        self.time_pos = nn.Parameter(
            torch.randn(1, config.prediction_horizon, config.state_dim) * 0.02
        )

    def forward(self,
                state: torch.Tensor,
                action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, pools, state_dim] 当前状态
            action_sequence: [batch, horizon, gates] 计划动作序列
        Returns:
            predictions: 包含水位、流量、不确定性的预测
        """
        B, P, D = state.shape
        H = action_sequence.shape[1]

        # 编码动作序列
        action_encoded = self.action_encoder(action_sequence)  # [B, H, D]
        action_encoded = action_encoded + self.time_pos[:, :H, :]

        # 扩展状态为memory
        memory = state  # [B, P, D]

        # Transformer解码预测
        predictions = self.decoder(action_encoded, memory)  # [B, H, D]

        # 扩展到每个渠池
        predictions_expanded = predictions.unsqueeze(2).expand(-1, -1, P, -1)
        predictions_expanded = predictions_expanded + state.unsqueeze(1)

        # 预测各变量
        level_pred = self.level_head(predictions_expanded).squeeze(-1)  # [B, H, P]
        flow_pred = self.flow_head(predictions_expanded)  # [B, H, P, 2]
        uncertainty = self.uncertainty_head(predictions_expanded).squeeze(-1)  # [B, H, P]

        return {
            'level': level_pred,
            'inflow': flow_pred[..., 0],
            'outflow': flow_pred[..., 1],
            'uncertainty': uncertainty
        }


class DecisionModule(nn.Module):
    """
    决策模块
    基于预测的最优控制决策
    """

    def __init__(self, config: AutonomousConfig):
        super().__init__()
        self.config = config

        # 策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(config.state_dim * config.num_pools, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_gates * 2)  # mean, log_std
        )

        # 价值网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(config.state_dim * config.num_pools + config.num_gates,
                     config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # 置信度评估
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.state_dim * config.num_pools, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # 初始化置信度估计器的偏置，使未训练模型有合理的初始置信度
        with torch.no_grad():
            self.confidence_estimator[2].bias.fill_(0.5)  # sigmoid(0.5) ≈ 0.62

        # 目标编码器
        self.target_encoder = nn.Linear(config.num_pools, config.state_dim)

    def forward(self,
                state: torch.Tensor,
                target_levels: torch.Tensor = None,
                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, pools, state_dim]
            target_levels: [batch, pools] 目标水位
            deterministic: 是否使用确定性策略
        Returns:
            action: 闸门控制动作
            confidence: 决策置信度
        """
        B, P, D = state.shape

        # 展平状态
        state_flat = state.view(B, -1)  # [B, P*D]

        # 如果有目标水位，融合目标信息
        if target_levels is not None:
            target_encoded = self.target_encoder(target_levels)  # [B, D]
            state_with_target = state_flat + target_encoded.repeat(1, P)
        else:
            state_with_target = state_flat

        # Actor输出
        actor_output = self.actor(state_with_target)  # [B, gates*2]
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        # 采样动作
        if deterministic:
            action = torch.sigmoid(mean)  # 闸门开度 [0, 1]
        else:
            dist = torch.distributions.Normal(mean, std)
            action_raw = dist.rsample()
            action = torch.sigmoid(action_raw)

        # 计算log概率 (用于训练)
        if not deterministic:
            log_prob = dist.log_prob(action_raw).sum(dim=-1, keepdim=True)
            # 修正sigmoid变换
            log_prob -= torch.log(action * (1 - action) + 1e-8).sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        # 评估置信度
        confidence = self.confidence_estimator(state_flat)

        # Critic评估
        state_action = torch.cat([state_flat, action], dim=-1)
        value = self.critic(state_action)

        return {
            'action': action,
            'mean': mean,
            'std': std,
            'log_prob': log_prob,
            'value': value,
            'confidence': confidence
        }


class E2EAutonomousController(nn.Module):
    """
    端到端自主控制器
    整合感知-预测-决策的完整自主系统
    """

    def __init__(self, config: AutonomousConfig = None):
        super().__init__()
        self.config = config or AutonomousConfig()

        # 核心模块
        self.perception = PerceptionModule(self.config)
        self.prediction = PredictionModule(self.config)
        self.decision = DecisionModule(self.config)

        # 当前自主等级
        self.autonomy_level = AutonomyLevel.L4_HIGH

        # 运行统计
        self.stats = {
            'total_steps': 0,
            'autonomous_steps': 0,
            'override_steps': 0,
            'avg_confidence': 0.0
        }

        logger.info(f"E2E Autonomous Controller initialized")
        logger.info(f"  Pools: {self.config.num_pools}")
        logger.info(f"  Gates: {self.config.num_gates}")
        logger.info(f"  Autonomy Level: {self.autonomy_level.name}")

    def forward(self,
                observations: torch.Tensor,
                target_levels: torch.Tensor,
                external_conditions: torch.Tensor = None,
                planned_actions: torch.Tensor = None) -> Dict[str, Any]:
        """
        完整的自主控制流程

        Args:
            observations: [batch, time, pools, 6] 历史观测
            target_levels: [batch, pools] 目标水位
            external_conditions: [batch, 16] 外部条件
            planned_actions: [batch, horizon, gates] 可选的计划动作

        Returns:
            control_output: 控制输出和诊断信息
        """
        # 1. 感知: 构建全线状态表示
        state = self.perception(observations, external_conditions)

        # 2. 决策: 生成控制动作
        decision_output = self.decision(state, target_levels, deterministic=True)
        action = decision_output['action']
        confidence = decision_output['confidence']

        # 3. 预测: 评估动作效果
        if planned_actions is None:
            # 使用当前动作扩展为预测序列
            planned_actions = action.unsqueeze(1).expand(
                -1, self.config.prediction_horizon, -1
            )

        predictions = self.prediction(state, planned_actions)

        # 4. 安全检查
        safety_check = self._check_safety(predictions, target_levels)

        # 5. 决定是否自主执行
        is_autonomous = (
            confidence.mean() >= self.config.confidence_threshold and
            safety_check['is_safe']
        )

        # 更新统计
        self.stats['total_steps'] += 1
        if is_autonomous:
            self.stats['autonomous_steps'] += 1
        else:
            self.stats['override_steps'] += 1
        self.stats['avg_confidence'] = (
            0.99 * self.stats['avg_confidence'] +
            0.01 * confidence.mean().item()
        )

        return {
            'action': action,
            'confidence': confidence,
            'predictions': predictions,
            'state': state,
            'is_autonomous': is_autonomous,
            'safety_check': safety_check,
            'decision_info': decision_output,
            'autonomy_rate': self.stats['autonomous_steps'] / max(1, self.stats['total_steps'])
        }

    def _check_safety(self,
                      predictions: Dict[str, torch.Tensor],
                      target_levels: torch.Tensor) -> Dict[str, Any]:
        """安全检查"""
        level_pred = predictions['level']
        uncertainty = predictions['uncertainty']

        # 水位边界检查
        level_min, level_max = 1.0, 6.5  # 安全水位范围
        margin = self.config.safety_margin

        # 考虑不确定性的边界
        level_upper = level_pred + 2 * uncertainty
        level_lower = level_pred - 2 * uncertainty

        # 检查是否在安全范围内
        is_safe_upper = (level_upper <= level_max - margin).all()
        is_safe_lower = (level_lower >= level_min + margin).all()
        is_safe = is_safe_upper and is_safe_lower

        # 计算与目标的偏差
        target_expanded = target_levels.unsqueeze(1).expand_as(level_pred)
        level_error = (level_pred - target_expanded).abs().mean()

        return {
            'is_safe': is_safe.item() if isinstance(is_safe, torch.Tensor) else is_safe,
            'level_error': level_error.item(),
            'max_level': level_pred.max().item(),
            'min_level': level_pred.min().item(),
            'max_uncertainty': uncertainty.max().item()
        }

    def get_autonomy_status(self) -> Dict[str, Any]:
        """获取自主运行状态"""
        return {
            'level': self.autonomy_level.name,
            'level_value': self.autonomy_level.value,
            'stats': self.stats.copy(),
            'confidence_threshold': self.config.confidence_threshold
        }

    def set_autonomy_level(self, level: AutonomyLevel):
        """设置自主等级"""
        self.autonomy_level = level
        logger.info(f"Autonomy level set to: {level.name}")

    def step(self,
             current_obs: Dict[str, np.ndarray],
             target_levels: np.ndarray,
             history_buffer: List[Dict] = None) -> Dict[str, Any]:
        """
        单步控制接口 (与现有系统兼容)

        Args:
            current_obs: 当前观测 {'levels': [...], 'inflows': [...], ...}
            target_levels: 目标水位数组
            history_buffer: 历史观测缓存

        Returns:
            gate_commands: 闸门控制命令
        """
        # 构建观测张量
        if history_buffer is None or len(history_buffer) < self.config.history_length:
            # 填充历史
            obs_list = [current_obs] * self.config.history_length
        else:
            obs_list = history_buffer[-self.config.history_length:]

        # 转换为张量
        observations = self._obs_to_tensor(obs_list)
        targets = torch.FloatTensor(target_levels).unsqueeze(0)

        # 前向推理
        with torch.no_grad():
            output = self.forward(observations, targets)

        # 转换为numpy输出
        action = output['action'].squeeze(0).numpy()

        return {
            'gate_openings': action,
            'confidence': output['confidence'].item(),
            'is_autonomous': output['is_autonomous'],
            'predictions': {
                k: v.squeeze(0).numpy()
                for k, v in output['predictions'].items()
            }
        }

    def _obs_to_tensor(self, obs_list: List[Dict]) -> torch.Tensor:
        """将观测列表转换为张量"""
        T = len(obs_list)
        P = self.config.num_pools

        observations = np.zeros((1, T, P, 6))

        for t, obs in enumerate(obs_list):
            if 'levels' in obs:
                observations[0, t, :len(obs['levels']), 0] = obs['levels'][:P]
            if 'inflows' in obs:
                observations[0, t, :len(obs['inflows']), 1] = obs['inflows'][:P]
            if 'outflows' in obs:
                observations[0, t, :len(obs['outflows']), 2] = obs['outflows'][:P]
            if 'gate_openings' in obs:
                observations[0, t, :len(obs['gate_openings']), 3] = obs['gate_openings'][:P]
            if 'diversions' in obs:
                observations[0, t, :len(obs['diversions']), 4] = obs['diversions'][:P]
            # 传感器状态默认为1 (正常)
            observations[0, t, :, 5] = 1.0

        return torch.FloatTensor(observations)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'stats': self.stats,
            'autonomy_level': self.autonomy_level.value
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.stats = checkpoint.get('stats', self.stats)
        self.autonomy_level = AutonomyLevel(checkpoint.get('autonomy_level', 4))
        logger.info(f"Model loaded from {path}")


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("E2E Autonomous Controller Test")
    logger.info("=" * 70)

    # 创建配置
    config = AutonomousConfig(
        num_pools=10,
        num_gates=11,
        history_length=24,
        prediction_horizon=12
    )

    # 创建控制器
    controller = E2EAutonomousController(config)

    # 打印模型信息
    total_params = sum(p.numel() for p in controller.parameters())
    logger.info(f"\nTotal parameters: {total_params:,}")

    # 测试前向传播
    batch_size = 2
    observations = torch.randn(batch_size, config.history_length, config.num_pools, 6)
    target_levels = torch.ones(batch_size, config.num_pools) * 4.0

    output = controller(observations, target_levels)

    logger.info(f"\nOutput shapes:")
    logger.info(f"  Action: {output['action'].shape}")
    logger.info(f"  Confidence: {output['confidence'].shape}")
    logger.info(f"  Predictions level: {output['predictions']['level'].shape}")
    logger.info(f"  Is autonomous: {output['is_autonomous']}")

    # 测试step接口
    current_obs = {
        'levels': np.random.uniform(3.5, 4.5, config.num_pools),
        'inflows': np.random.uniform(200, 300, config.num_pools),
        'outflows': np.random.uniform(180, 280, config.num_pools),
        'gate_openings': np.random.uniform(0.7, 1.0, config.num_pools),
        'diversions': np.random.uniform(0, 10, config.num_pools)
    }

    result = controller.step(current_obs, np.ones(config.num_pools) * 4.0)
    logger.info(f"\nStep output:")
    logger.info(f"  Gate openings: {result['gate_openings'][:5]}...")
    logger.info(f"  Confidence: {result['confidence']:.3f}")
    logger.info(f"  Is autonomous: {result['is_autonomous']}")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)

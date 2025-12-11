"""
生成式场景引擎 (Scenario VAE)
Generative Scenario Engine using Variational Autoencoders

基于 VAE/CVAE 的时间序列生成器:
- 将历史水文/故障序列压缩为隐向量
- 从隐空间采样生成新场景
- 条件生成 (季节、故障类型等)
- 多维输出: 上游来水、下游需水、糙率变化、传感器噪声

技术特点:
1. 时序VAE: 使用LSTM编码器/解码器处理时间序列
2. 条件生成: CVAE支持按标签生成特定类型场景
3. 多通道输出: 同时生成多个相关时间序列
4. 与现有ScenarioGenerator集成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置数据类
# ==============================================================================

@dataclass
class ScenarioVAEConfig:
    """场景VAE配置"""
    # 输入/输出维度
    sequence_length: int = 96          # 序列长度 (例如: 96个15分钟 = 24小时)
    num_channels: int = 4              # 通道数 [来水, 需水, 糙率, 噪声]

    # 编码器
    encoder_hidden_dim: int = 128
    encoder_num_layers: int = 2

    # 隐空间
    latent_dim: int = 32               # 隐向量维度

    # 解码器
    decoder_hidden_dim: int = 128
    decoder_num_layers: int = 2

    # 条件
    num_conditions: int = 8            # 条件类别数 (场景类型)
    condition_embed_dim: int = 16      # 条件嵌入维度

    # 训练
    learning_rate: float = 1e-3
    kl_weight: float = 0.001           # KL散度权重 (beta-VAE)
    dropout: float = 0.1

    # 生成控制
    temperature: float = 1.0           # 采样温度


# ==============================================================================
# 场景条件定义
# ==============================================================================

class ScenarioConditions:
    """场景条件标签定义"""

    # 季节条件
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3
    ICE_PERIOD = 4

    # 事件类型
    NORMAL = 5
    FLOOD = 6
    DROUGHT = 7
    POLLUTION = 8
    EQUIPMENT_FAULT = 9
    SENSOR_FAULT = 10
    GATE_STUCK = 11

    # 严重程度
    LOW = 12
    MEDIUM = 13
    HIGH = 14
    CRITICAL = 15

    @classmethod
    def encode(cls, conditions: List[str]) -> torch.Tensor:
        """将条件字符串编码为one-hot向量"""
        condition_map = {
            'spring': cls.SPRING,
            'summer': cls.SUMMER,
            'autumn': cls.AUTUMN,
            'winter': cls.WINTER,
            'ice_period': cls.ICE_PERIOD,
            'normal': cls.NORMAL,
            'flood': cls.FLOOD,
            'drought': cls.DROUGHT,
            'pollution': cls.POLLUTION,
            'equipment_fault': cls.EQUIPMENT_FAULT,
            'sensor_fault': cls.SENSOR_FAULT,
            'gate_stuck': cls.GATE_STUCK,
            'low': cls.LOW,
            'medium': cls.MEDIUM,
            'high': cls.HIGH,
            'critical': cls.CRITICAL,
        }

        vector = torch.zeros(16)
        for cond in conditions:
            cond_lower = cond.lower()
            if cond_lower in condition_map:
                vector[condition_map[cond_lower]] = 1.0

        return vector


# ==============================================================================
# VAE 编码器
# ==============================================================================

class SequenceEncoder(nn.Module):
    """
    时序编码器

    将时间序列压缩为隐空间向量
    """

    def __init__(self, config: ScenarioVAEConfig):
        super().__init__()
        self.config = config

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(config.num_channels, config.encoder_hidden_dim),
            nn.LayerNorm(config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=config.encoder_hidden_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            batch_first=True,
            dropout=config.dropout if config.encoder_num_layers > 1 else 0,
            bidirectional=True
        )

        # 隐向量参数 (均值和方差)
        hidden_out_dim = config.encoder_hidden_dim * 2  # 双向
        self.fc_mu = nn.Linear(hidden_out_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(hidden_out_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码序列

        Args:
            x: 输入序列 [batch, seq_len, channels]

        Returns:
            (mu, logvar): 隐空间的均值和对数方差
        """
        # 输入投影
        h = self.input_proj(x)

        # LSTM编码
        _, (h_n, _) = self.lstm(h)

        # 合并双向隐状态
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        # 计算均值和方差
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)

        return mu, logvar


# ==============================================================================
# VAE 解码器
# ==============================================================================

class SequenceDecoder(nn.Module):
    """
    时序解码器

    从隐向量重建时间序列
    """

    def __init__(self, config: ScenarioVAEConfig):
        super().__init__()
        self.config = config

        # 隐向量投影
        self.latent_proj = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder_hidden_dim),
            nn.ReLU()
        )

        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=config.decoder_hidden_dim,
            hidden_size=config.decoder_hidden_dim,
            num_layers=config.decoder_num_layers,
            batch_first=True,
            dropout=config.dropout if config.decoder_num_layers > 1 else 0
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.num_channels * 2),  # 均值和方差
        )

    def forward(self,
                z: torch.Tensor,
                seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码隐向量

        Args:
            z: 隐向量 [batch, latent_dim]
            seq_len: 输出序列长度

        Returns:
            (recon_mu, recon_logvar): 重建序列的均值和对数方差
        """
        seq_len = seq_len or self.config.sequence_length
        batch_size = z.size(0)

        # 投影隐向量
        h = self.latent_proj(z)

        # 扩展为序列
        h = h.unsqueeze(1).expand(-1, seq_len, -1)

        # LSTM解码
        lstm_out, _ = self.lstm(h)

        # 输出投影
        output = self.output_proj(lstm_out)

        # 分离均值和方差
        recon_mu = output[:, :, :self.config.num_channels]
        recon_logvar = output[:, :, self.config.num_channels:]

        return recon_mu, recon_logvar


# ==============================================================================
# 条件编码器 (CVAE)
# ==============================================================================

class ConditionEncoder(nn.Module):
    """条件编码器"""

    def __init__(self, config: ScenarioVAEConfig):
        super().__init__()

        # 条件嵌入
        self.condition_embed = nn.Sequential(
            nn.Linear(16, config.condition_embed_dim),  # 16个条件标签
            nn.ReLU(),
            nn.Linear(config.condition_embed_dim, config.condition_embed_dim)
        )

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        编码条件

        Args:
            conditions: 条件向量 [batch, 16]

        Returns:
            条件嵌入 [batch, embed_dim]
        """
        return self.condition_embed(conditions)


# ==============================================================================
# 场景VAE主模型
# ==============================================================================

class ScenarioVAE(nn.Module):
    """
    场景变分自编码器

    用于生成水文时间序列场景
    """

    def __init__(self, config: ScenarioVAEConfig = None):
        super().__init__()
        self.config = config or ScenarioVAEConfig()

        # 编码器
        self.encoder = SequenceEncoder(self.config)

        # 解码器
        self.decoder = SequenceDecoder(self.config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 [batch, seq_len, channels]

        Returns:
            包含重建和隐空间参数的字典
        """
        # 编码
        mu, logvar = self.encoder(x)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_mu, recon_logvar = self.decoder(z, x.size(1))

        return {
            'recon_mu': recon_mu,
            'recon_logvar': recon_logvar,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码为隐向量"""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """从隐向量解码"""
        recon_mu, _ = self.decoder(z, seq_len)
        return recon_mu

    def sample(self,
               num_samples: int = 1,
               seq_len: int = None,
               temperature: float = 1.0) -> torch.Tensor:
        """
        从先验分布采样生成新场景

        Args:
            num_samples: 采样数量
            seq_len: 序列长度
            temperature: 采样温度

        Returns:
            生成的场景 [num_samples, seq_len, channels]
        """
        seq_len = seq_len or self.config.sequence_length

        # 从标准正态分布采样
        z = torch.randn(num_samples, self.config.latent_dim) * temperature
        z = z.to(next(self.parameters()).device)

        # 解码
        with torch.no_grad():
            generated = self.decode(z, seq_len)

        return generated


# ==============================================================================
# 条件场景VAE (CVAE)
# ==============================================================================

class ConditionalScenarioVAE(nn.Module):
    """
    条件场景变分自编码器 (CVAE)

    支持按条件生成特定类型的场景
    """

    def __init__(self, config: ScenarioVAEConfig = None):
        super().__init__()
        self.config = config or ScenarioVAEConfig()

        # 条件编码器
        self.condition_encoder = ConditionEncoder(self.config)

        # 修改编码器输入维度 (加入条件)
        encoder_config = ScenarioVAEConfig(
            sequence_length=self.config.sequence_length,
            num_channels=self.config.num_channels + self.config.condition_embed_dim,
            encoder_hidden_dim=self.config.encoder_hidden_dim,
            encoder_num_layers=self.config.encoder_num_layers,
            latent_dim=self.config.latent_dim,
            dropout=self.config.dropout
        )
        self.encoder = SequenceEncoder(encoder_config)

        # 修改解码器隐向量维度 (加入条件)
        decoder_config = ScenarioVAEConfig(
            sequence_length=self.config.sequence_length,
            num_channels=self.config.num_channels,
            decoder_hidden_dim=self.config.decoder_hidden_dim,
            decoder_num_layers=self.config.decoder_num_layers,
            latent_dim=self.config.latent_dim + self.config.condition_embed_dim,
            dropout=self.config.dropout
        )
        self.decoder = SequenceDecoder(decoder_config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                x: torch.Tensor,
                conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 [batch, seq_len, channels]
            conditions: 条件向量 [batch, 16]

        Returns:
            包含重建和隐空间参数的字典
        """
        batch_size, seq_len, _ = x.shape

        # 编码条件
        cond_embed = self.condition_encoder(conditions)

        # 将条件扩展并拼接到输入
        cond_expanded = cond_embed.unsqueeze(1).expand(-1, seq_len, -1)
        x_cond = torch.cat([x, cond_expanded], dim=-1)

        # 编码
        mu, logvar = self.encoder(x_cond)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 将条件拼接到隐向量
        z_cond = torch.cat([z, cond_embed], dim=-1)

        # 解码
        recon_mu, recon_logvar = self.decoder(z_cond, seq_len)

        return {
            'recon_mu': recon_mu,
            'recon_logvar': recon_logvar,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def generate(self,
                 conditions: List[str],
                 num_samples: int = 1,
                 seq_len: int = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        按条件生成场景

        Args:
            conditions: 条件列表 ['winter', 'flood']
            num_samples: 采样数量
            seq_len: 序列长度
            temperature: 采样温度

        Returns:
            生成的场景 [num_samples, seq_len, channels]
        """
        seq_len = seq_len or self.config.sequence_length
        device = next(self.parameters()).device

        # 编码条件
        cond_vector = ScenarioConditions.encode(conditions)
        cond_vector = cond_vector.unsqueeze(0).expand(num_samples, -1).to(device)
        cond_embed = self.condition_encoder(cond_vector)

        # 从先验采样
        z = torch.randn(num_samples, self.config.latent_dim, device=device) * temperature

        # 拼接条件
        z_cond = torch.cat([z, cond_embed], dim=-1)

        # 解码
        with torch.no_grad():
            recon_mu, _ = self.decoder(z_cond, seq_len)

        return recon_mu


# ==============================================================================
# 场景隐空间
# ==============================================================================

class ScenarioLatentSpace:
    """
    场景隐空间管理

    用于场景插值、变换和检索
    """

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.stored_vectors: Dict[str, torch.Tensor] = {}
        self.stored_labels: Dict[str, List[str]] = {}

    def store(self,
              name: str,
              z: torch.Tensor,
              labels: List[str] = None):
        """存储隐向量"""
        self.stored_vectors[name] = z.detach().cpu()
        if labels:
            self.stored_labels[name] = labels

    def interpolate(self,
                    z1: torch.Tensor,
                    z2: torch.Tensor,
                    steps: int = 10) -> torch.Tensor:
        """
        在两个隐向量之间插值

        Args:
            z1, z2: 起点和终点隐向量
            steps: 插值步数

        Returns:
            插值序列 [steps, latent_dim]
        """
        alphas = torch.linspace(0, 1, steps)
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated.append(z_interp)

        return torch.stack(interpolated)

    def arithmetic(self,
                   z_base: torch.Tensor,
                   z_add: torch.Tensor = None,
                   z_sub: torch.Tensor = None) -> torch.Tensor:
        """
        隐空间算术运算

        例如: 正常场景 + (洪水场景 - 正常场景) = 洪水场景
        """
        result = z_base.clone()
        if z_add is not None:
            result = result + z_add
        if z_sub is not None:
            result = result - z_sub
        return result


# ==============================================================================
# VAE损失函数
# ==============================================================================

class VAELoss(nn.Module):
    """VAE损失函数"""

    def __init__(self, kl_weight: float = 0.001):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self,
                recon_mu: torch.Tensor,
                recon_logvar: torch.Tensor,
                target: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算VAE损失

        Args:
            recon_mu: 重建均值
            recon_logvar: 重建对数方差
            target: 目标序列
            mu: 隐空间均值
            logvar: 隐空间对数方差

        Returns:
            损失字典
        """
        # 重建损失 (负对数似然)
        recon_std = torch.exp(0.5 * recon_logvar)
        recon_dist = Normal(recon_mu, recon_std + 1e-6)
        recon_loss = -recon_dist.log_prob(target).sum(dim=[1, 2]).mean()

        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # 总损失
        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


# ==============================================================================
# 场景生成器包装类 (与现有接口集成)
# ==============================================================================

class AIScenarioGenerator:
    """
    AI场景生成器

    包装ScenarioVAE，提供与现有ScenarioGenerator兼容的接口
    """

    def __init__(self,
                 model: ConditionalScenarioVAE = None,
                 config: ScenarioVAEConfig = None,
                 device: str = None):
        """
        初始化AI场景生成器

        Args:
            model: 预训练的CVAE模型
            config: 配置
            device: 计算设备
        """
        self.config = config or ScenarioVAEConfig()

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # 创建或加载模型
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = ConditionalScenarioVAE(self.config).to(self.device)

        # 输出通道映射
        self.channel_names = ['upstream_flow', 'downstream_demand', 'roughness', 'sensor_noise']

        logger.info(f"AIScenarioGenerator 初始化完成, 设备: {self.device}")

    def generate_from_ai(self,
                         conditions: List[str],
                         num_scenarios: int = 1,
                         duration_hours: float = 24.0,
                         dt_minutes: float = 15.0) -> Dict[str, np.ndarray]:
        """
        使用AI生成场景 (与现有ScenarioGenerator集成的方法)

        Args:
            conditions: 条件列表 ['winter', 'sensor_fault']
            num_scenarios: 生成场景数量
            duration_hours: 场景时长 (小时)
            dt_minutes: 时间步长 (分钟)

        Returns:
            场景数据字典:
            {
                'upstream_flow': [num_scenarios, seq_len],  # 上游来水曲线
                'downstream_demand': [num_scenarios, seq_len],  # 下游需水曲线
                'roughness': [num_scenarios, seq_len],  # 糙率变化曲线
                'sensor_noise': [num_scenarios, seq_len],  # 传感器噪声序列
                'time': [seq_len],  # 时间轴
                'conditions': conditions,
            }
        """
        # 计算序列长度
        seq_len = int(duration_hours * 60 / dt_minutes)

        # 生成
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                conditions=conditions,
                num_samples=num_scenarios,
                seq_len=seq_len,
                temperature=self.config.temperature
            )

        # 转换为numpy并后处理
        generated_np = generated.cpu().numpy()

        # 分离各通道并进行物理约束
        result = {
            'upstream_flow': self._postprocess_flow(generated_np[:, :, 0]),
            'downstream_demand': self._postprocess_flow(generated_np[:, :, 1]),
            'roughness': self._postprocess_roughness(generated_np[:, :, 2]),
            'sensor_noise': generated_np[:, :, 3],
            'time': np.arange(seq_len) * dt_minutes / 60,  # 小时
            'conditions': conditions,
        }

        return result

    def _postprocess_flow(self, flow: np.ndarray) -> np.ndarray:
        """后处理流量数据"""
        # 缩放到合理范围 [50, 400] m³/s
        flow = flow * 100 + 200  # 假设VAE输出归一化数据
        flow = np.clip(flow, 50, 400)
        return flow

    def _postprocess_roughness(self, roughness: np.ndarray) -> np.ndarray:
        """后处理糙率数据"""
        # 缩放到合理范围 [0.012, 0.025]
        roughness = roughness * 0.003 + 0.015
        roughness = np.clip(roughness, 0.012, 0.025)
        return roughness

    def load_model(self, path: str):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型加载成功: {path}")

    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"模型保存成功: {path}")


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "场景VAE测试")
    print("=" * 70)

    # 创建配置
    config = ScenarioVAEConfig(
        sequence_length=96,
        num_channels=4,
        latent_dim=32,
    )

    # 测试基础VAE
    print("\n1. 基础VAE测试")
    print("-" * 70)

    vae = ScenarioVAE(config)

    # 创建测试数据
    batch_size = 8
    x = torch.randn(batch_size, config.sequence_length, config.num_channels)

    # 前向传播
    output = vae(x)
    print(f"  输入形状: {x.shape}")
    print(f"  重建形状: {output['recon_mu'].shape}")
    print(f"  隐向量形状: {output['z'].shape}")

    # 采样
    samples = vae.sample(num_samples=4)
    print(f"  采样形状: {samples.shape}")

    # 测试CVAE
    print("\n2. 条件VAE测试")
    print("-" * 70)

    cvae = ConditionalScenarioVAE(config)

    # 条件编码
    conditions = ScenarioConditions.encode(['winter', 'flood'])
    conditions = conditions.unsqueeze(0).expand(batch_size, -1)

    # 前向传播
    output = cvae(x, conditions)
    print(f"  条件输入: ['winter', 'flood']")
    print(f"  重建形状: {output['recon_mu'].shape}")

    # 条件生成
    generated = cvae.generate(['summer', 'normal'], num_samples=4)
    print(f"  条件生成形状: {generated.shape}")

    # 测试AI场景生成器
    print("\n3. AI场景生成器测试")
    print("-" * 70)

    generator = AIScenarioGenerator(config=config)

    scenarios = generator.generate_from_ai(
        conditions=['winter', 'sensor_fault'],
        num_scenarios=3,
        duration_hours=24.0
    )

    print(f"  生成场景数: {len(scenarios['upstream_flow'])}")
    print(f"  序列长度: {len(scenarios['time'])}")
    print(f"  上游流量范围: [{scenarios['upstream_flow'].min():.1f}, {scenarios['upstream_flow'].max():.1f}]")
    print(f"  糙率范围: [{scenarios['roughness'].min():.4f}, {scenarios['roughness'].max():.4f}]")

    # 测试损失函数
    print("\n4. VAE损失函数测试")
    print("-" * 70)

    criterion = VAELoss(kl_weight=0.001)
    losses = criterion(
        output['recon_mu'],
        output['recon_logvar'],
        x,
        output['mu'],
        output['logvar']
    )

    print(f"  重建损失: {losses['recon_loss'].item():.4f}")
    print(f"  KL损失: {losses['kl_loss'].item():.4f}")
    print(f"  总损失: {losses['total_loss'].item():.4f}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

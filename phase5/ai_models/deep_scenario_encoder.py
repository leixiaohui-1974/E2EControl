"""
深度场景编码器 (Deep Scenario Encoder)
Contrastive Learning based Scenario Recognition

基于对比学习的场景识别层:
- 将时序数据编码为高维向量
- 使用对比损失学习场景表示
- 向量数据库检索相似场景
- 未知场景/异常检测

技术特点:
1. 1D-CNN / Transformer 时序特征提取
2. SimCLR / Triplet Loss 对比学习
3. FAISS / 余弦相似度检索
4. 与现有 rule_engine.py 集成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置数据类
# ==============================================================================

@dataclass
class DeepEncoderConfig:
    """深度编码器配置"""
    # 输入参数
    sequence_length: int = 96        # 序列长度
    input_channels: int = 5          # 输入通道数 [水位, 流量, 闸门, 需求, 天气]

    # 编码器架构
    encoder_type: str = 'cnn'        # 'cnn', 'transformer', 'hybrid'
    hidden_dim: int = 128            # 隐层维度
    embedding_dim: int = 64          # 嵌入向量维度
    num_layers: int = 3              # 层数
    num_heads: int = 4               # Transformer注意力头数
    dropout: float = 0.1

    # 对比学习
    contrastive_loss: str = 'simclr'  # 'simclr', 'triplet', 'ntxent'
    temperature: float = 0.1          # SimCLR温度参数
    margin: float = 1.0               # Triplet Loss margin

    # 检索参数
    similarity_threshold: float = 0.7  # 相似度阈值
    anomaly_threshold: float = 0.3     # 异常检测阈值
    top_k: int = 5                     # 返回Top-K结果


# ==============================================================================
# 1D-CNN 编码器
# ==============================================================================

class CNN1DEncoder(nn.Module):
    """
    1D卷积时序编码器

    适合捕捉局部时序模式
    """

    def __init__(self, config: DeepEncoderConfig):
        super().__init__()
        self.config = config

        # 卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = config.input_channels
        out_channels = config.hidden_dim // 2

        for i in range(config.num_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(config.dropout)
            ))
            in_channels = out_channels
            out_channels = min(out_channels * 2, config.hidden_dim)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(in_channels, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch, seq_len, channels]

        Returns:
            嵌入向量 [batch, embedding_dim]
        """
        # 转换维度 [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # 卷积编码
        for conv in self.conv_layers:
            x = conv(x)

        # 全局池化
        x = self.global_pool(x).squeeze(-1)

        # 投影
        embedding = self.projection(x)

        # L2归一化
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


# ==============================================================================
# Transformer 编码器
# ==============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer时序编码器

    适合捕捉长程依赖关系
    """

    def __init__(self, config: DeepEncoderConfig):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.input_embed = nn.Linear(config.input_channels, config.hidden_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            config.sequence_length,
            config.dropout
        )

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch, seq_len, channels]

        Returns:
            嵌入向量 [batch, embedding_dim]
        """
        batch_size = x.size(0)

        # 输入嵌入
        x = self.input_embed(x)
        x = self.pos_encoding(x)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer编码
        x = self.transformer(x)

        # 取CLS token输出
        cls_output = x[:, 0]

        # 投影
        embedding = self.projection(cls_output)

        # L2归一化
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ==============================================================================
# 混合编码器 (CNN + Transformer)
# ==============================================================================

class HybridEncoder(nn.Module):
    """
    混合编码器

    CNN提取局部特征 + Transformer捕捉全局关系
    """

    def __init__(self, config: DeepEncoderConfig):
        super().__init__()
        self.config = config

        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(config.input_channels, config.hidden_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(config.hidden_dim // 2, config.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Transformer
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.sequence_length // 4, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 投影头
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.cnn(x)
        x = x.transpose(1, 2)  # [batch, seq_len//4, hidden_dim]

        # Transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # 投影
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len//4]
        embedding = self.projection(x)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding


# ==============================================================================
# 对比学习损失函数
# ==============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数

    支持多种对比损失:
    - SimCLR (NT-Xent)
    - Triplet Loss
    """

    def __init__(self, config: DeepEncoderConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.margin = config.margin

    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor = None,
                positives: torch.Tensor = None,
                negatives: torch.Tensor = None) -> torch.Tensor:
        """
        计算对比损失

        Args:
            embeddings: 嵌入向量 [batch, embedding_dim]
            labels: 类别标签 [batch] (用于监督对比学习)
            positives: 正样本嵌入 [batch, embedding_dim]
            negatives: 负样本嵌入 [batch, embedding_dim]

        Returns:
            损失值
        """
        if self.config.contrastive_loss == 'simclr':
            return self._simclr_loss(embeddings, positives)
        elif self.config.contrastive_loss == 'triplet':
            return self._triplet_loss(embeddings, positives, negatives)
        elif self.config.contrastive_loss == 'supervised':
            return self._supervised_contrastive_loss(embeddings, labels)
        else:
            return self._simclr_loss(embeddings, positives)

    def _simclr_loss(self,
                    z_i: torch.Tensor,
                    z_j: torch.Tensor) -> torch.Tensor:
        """
        SimCLR损失 (NT-Xent)

        正样本: 同一场景的不同增强版本
        负样本: 批次内其他样本
        """
        batch_size = z_i.size(0)

        # 合并正样本对
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch, embedding_dim]

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]

        # 创建正样本掩码
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 正样本位置
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)

        # NT-Xent损失
        positives = sim_matrix[pos_mask].view(2 * batch_size, 1)
        negatives = sim_matrix[~pos_mask & ~mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def _triplet_loss(self,
                     anchor: torch.Tensor,
                     positive: torch.Tensor,
                     negative: torch.Tensor) -> torch.Tensor:
        """
        Triplet Loss

        anchor: 锚点样本
        positive: 正样本 (同类)
        negative: 负样本 (异类)
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()

    def _supervised_contrastive_loss(self,
                                     embeddings: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        """
        监督对比损失

        利用标签信息：同类为正样本，异类为负样本
        """
        batch_size = embeddings.size(0)

        # 相似度矩阵
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # 标签掩码
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()

        # 移除自身
        mask_self = torch.eye(batch_size, device=embeddings.device)
        mask_pos = mask_pos - mask_self

        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (1 - mask_self)  # 移除自身

        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)

        # 只计算有正样本的
        num_pos = mask_pos.sum(dim=1)
        loss = (loss * (num_pos > 0).float()).sum() / (num_pos > 0).sum().clamp(min=1)

        return loss


# ==============================================================================
# 场景向量数据库
# ==============================================================================

class ScenarioVectorDB:
    """
    场景向量数据库

    存储和检索场景的嵌入向量
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

        # 存储
        self.vectors: List[np.ndarray] = []
        self.scenario_ids: List[str] = []
        self.metadata: List[Dict] = []

        # 索引矩阵 (用于快速检索)
        self._index_matrix: Optional[np.ndarray] = None
        self._index_dirty = True

    def add(self,
            embedding: np.ndarray,
            scenario_id: str,
            metadata: Dict = None):
        """
        添加场景向量

        Args:
            embedding: 嵌入向量 [embedding_dim]
            scenario_id: 场景ID
            metadata: 元数据
        """
        embedding = np.array(embedding).flatten()
        assert embedding.shape[0] == self.embedding_dim

        self.vectors.append(embedding)
        self.scenario_ids.append(scenario_id)
        self.metadata.append(metadata or {})
        self._index_dirty = True

    def add_batch(self,
                  embeddings: np.ndarray,
                  scenario_ids: List[str],
                  metadata_list: List[Dict] = None):
        """批量添加"""
        metadata_list = metadata_list or [{}] * len(scenario_ids)

        for emb, sid, meta in zip(embeddings, scenario_ids, metadata_list):
            self.add(emb, sid, meta)

    def _rebuild_index(self):
        """重建索引"""
        if len(self.vectors) > 0:
            self._index_matrix = np.stack(self.vectors, axis=0)
        else:
            self._index_matrix = np.zeros((0, self.embedding_dim))
        self._index_dirty = False

    def search(self,
               query: np.ndarray,
               top_k: int = 5,
               threshold: float = None) -> List[Tuple[str, float, Dict]]:
        """
        检索相似场景

        Args:
            query: 查询向量 [embedding_dim]
            top_k: 返回数量
            threshold: 相似度阈值

        Returns:
            [(scenario_id, similarity, metadata), ...]
        """
        if self._index_dirty:
            self._rebuild_index()

        if len(self.vectors) == 0:
            return []

        query = np.array(query).flatten()
        query = query / (np.linalg.norm(query) + 1e-8)

        # 余弦相似度
        similarities = np.dot(self._index_matrix, query)

        # 排序
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if threshold is not None and sim < threshold:
                break
            results.append((
                self.scenario_ids[idx],
                sim,
                self.metadata[idx]
            ))

        return results

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self._index_dirty:
            self._rebuild_index()

        return {
            'num_vectors': len(self.vectors),
            'embedding_dim': self.embedding_dim,
            'scenario_types': len(set(self.scenario_ids)),
        }

    def save(self, path: str):
        """保存数据库"""
        np.savez(path,
                 vectors=np.array(self.vectors),
                 scenario_ids=np.array(self.scenario_ids),
                 metadata=np.array(self.metadata, dtype=object))
        logger.info(f"向量数据库已保存: {path}")

    def load(self, path: str):
        """加载数据库"""
        data = np.load(path, allow_pickle=True)
        self.vectors = list(data['vectors'])
        self.scenario_ids = list(data['scenario_ids'])
        self.metadata = list(data['metadata'])
        self._index_dirty = True
        logger.info(f"向量数据库已加载: {path}, {len(self.vectors)}个向量")


# ==============================================================================
# 深度场景编码器 (主类)
# ==============================================================================

class DeepScenarioEncoder:
    """
    深度场景编码器

    主要功能:
    1. 将时序数据编码为向量
    2. 检索相似场景
    3. 检测异常/未知场景
    4. 与现有规则引擎集成
    """

    def __init__(self,
                 config: DeepEncoderConfig = None,
                 device: str = None):
        """
        初始化深度场景编码器

        Args:
            config: 配置
            device: 计算设备
        """
        self.config = config or DeepEncoderConfig()

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # 创建编码器
        self.encoder = self._create_encoder()
        self.encoder.to(self.device)

        # 对比损失
        self.contrastive_loss = ContrastiveLoss(self.config)

        # 向量数据库
        self.vector_db = ScenarioVectorDB(self.config.embedding_dim)

        logger.info(f"DeepScenarioEncoder 初始化完成")
        logger.info(f"  编码器类型: {self.config.encoder_type}")
        logger.info(f"  嵌入维度: {self.config.embedding_dim}")
        logger.info(f"  设备: {self.device}")

    def _create_encoder(self) -> nn.Module:
        """创建编码器"""
        if self.config.encoder_type == 'cnn':
            return CNN1DEncoder(self.config)
        elif self.config.encoder_type == 'transformer':
            return TransformerEncoder(self.config)
        elif self.config.encoder_type == 'hybrid':
            return HybridEncoder(self.config)
        else:
            return CNN1DEncoder(self.config)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        编码时序数据

        Args:
            data: 输入数据 [batch, seq_len, channels] 或 [seq_len, channels]

        Returns:
            嵌入向量 [batch, embedding_dim] 或 [embedding_dim]
        """
        # 处理维度
        squeeze = False
        if data.ndim == 2:
            data = data[np.newaxis, :]
            squeeze = True

        # 转换为张量
        x = torch.FloatTensor(data).to(self.device)

        # 编码
        self.encoder.eval()
        with torch.no_grad():
            embedding = self.encoder(x)

        embedding = embedding.cpu().numpy()

        if squeeze:
            embedding = embedding[0]

        return embedding

    def recognize(self,
                  data: np.ndarray,
                  top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        识别场景

        Args:
            data: 输入数据 [seq_len, channels]
            top_k: 返回数量

        Returns:
            [(scenario_id, similarity, metadata), ...]
        """
        top_k = top_k or self.config.top_k

        # 编码
        embedding = self.encode(data)

        # 检索
        results = self.vector_db.search(
            embedding,
            top_k=top_k,
            threshold=self.config.similarity_threshold
        )

        return results

    def detect_anomaly(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        检测异常/未知场景

        Args:
            data: 输入数据 [seq_len, channels]

        Returns:
            (is_anomaly, anomaly_score)
        """
        # 编码
        embedding = self.encode(data)

        # 检索最相似场景
        results = self.vector_db.search(embedding, top_k=1)

        if len(results) == 0:
            return True, 1.0

        _, similarity, _ = results[0]

        # 低相似度 = 异常
        is_anomaly = similarity < self.config.anomaly_threshold
        anomaly_score = 1.0 - similarity

        return is_anomaly, anomaly_score

    def add_scenario(self,
                     data: np.ndarray,
                     scenario_id: str,
                     metadata: Dict = None):
        """
        添加新场景到数据库

        Args:
            data: 场景数据 [seq_len, channels]
            scenario_id: 场景ID
            metadata: 元数据
        """
        embedding = self.encode(data)
        self.vector_db.add(embedding, scenario_id, metadata)

    def load_model(self, path: str):
        """加载编码器模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"编码器模型已加载: {path}")

    def save_model(self, path: str):
        """保存编码器模型"""
        checkpoint = {
            'model_state_dict': self.encoder.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"编码器模型已保存: {path}")


# ==============================================================================
# 数据增强 (用于对比学习)
# ==============================================================================

class TimeSeriesAugmentation:
    """
    时序数据增强

    用于生成对比学习的正样本
    """

    @staticmethod
    def add_noise(x: np.ndarray, std: float = 0.1) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, std, x.shape)
        return x + noise

    @staticmethod
    def scale(x: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(*scale_range)
        return x * scale

    @staticmethod
    def time_shift(x: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """时间平移"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=0)

    @staticmethod
    def time_warp(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """时间扭曲"""
        seq_len = x.shape[0]
        warp = np.cumsum(np.random.normal(1, sigma, seq_len))
        warp = warp / warp[-1] * (seq_len - 1)
        warp = np.clip(warp, 0, seq_len - 1).astype(int)
        return x[warp]

    @staticmethod
    def channel_dropout(x: np.ndarray, p: float = 0.1) -> np.ndarray:
        """通道dropout"""
        mask = np.random.binomial(1, 1 - p, x.shape[-1])
        return x * mask

    @staticmethod
    def augment(x: np.ndarray) -> np.ndarray:
        """应用随机增强组合"""
        augmentations = [
            TimeSeriesAugmentation.add_noise,
            TimeSeriesAugmentation.scale,
            TimeSeriesAugmentation.time_shift,
        ]

        # 随机选择1-2个增强
        num_aug = np.random.randint(1, 3)
        selected = np.random.choice(len(augmentations), num_aug, replace=False)

        result = x.copy()
        for idx in selected:
            result = augmentations[idx](result)

        return result


# ==============================================================================
# 与规则引擎集成的混合识别器
# ==============================================================================

class HybridScenarioRecognizer:
    """
    混合场景识别器

    结合规则引擎和深度编码器:
    - 规则引擎: 处理已知模式 (高置信度)
    - 深度编码器: 处理复杂/未知模式 (泛化能力)
    """

    def __init__(self,
                 deep_encoder: DeepScenarioEncoder,
                 rule_engine=None,
                 rule_weight: float = 0.6,
                 deep_weight: float = 0.4):
        """
        Args:
            deep_encoder: 深度场景编码器
            rule_engine: 规则引擎 (可选)
            rule_weight: 规则权重
            deep_weight: 深度学习权重
        """
        self.deep_encoder = deep_encoder
        self.rule_engine = rule_engine
        self.rule_weight = rule_weight
        self.deep_weight = deep_weight

    def recognize(self, data: np.ndarray, state=None) -> Dict[str, Any]:
        """
        混合识别

        Args:
            data: 时序数据 [seq_len, channels]
            state: 系统状态 (用于规则引擎)

        Returns:
            识别结果
        """
        results = {
            'scenario_id': None,
            'confidence': 0.0,
            'source': 'unknown',
            'is_anomaly': False,
            'details': {}
        }

        # 深度编码器识别
        deep_results = self.deep_encoder.recognize(data)
        is_anomaly, anomaly_score = self.deep_encoder.detect_anomaly(data)

        if len(deep_results) > 0:
            deep_id, deep_sim, deep_meta = deep_results[0]
            results['details']['deep'] = {
                'scenario_id': deep_id,
                'similarity': deep_sim,
                'metadata': deep_meta
            }

        # 规则引擎识别 (如果可用)
        if self.rule_engine is not None and state is not None:
            try:
                rule_results = self.rule_engine.recognize(state)
                if len(rule_results) > 0:
                    rule_id, rule_conf = rule_results[0]
                    results['details']['rule'] = {
                        'scenario_id': rule_id,
                        'confidence': rule_conf
                    }
            except Exception as e:
                logger.warning(f"规则引擎识别失败: {e}")

        # 融合结果
        results = self._fuse_results(results, deep_results, is_anomaly, anomaly_score)

        return results

    def _fuse_results(self,
                      results: Dict,
                      deep_results: List,
                      is_anomaly: bool,
                      anomaly_score: float) -> Dict:
        """融合结果"""
        results['is_anomaly'] = is_anomaly
        results['anomaly_score'] = anomaly_score

        # 如果是异常，标记为未知
        if is_anomaly:
            results['scenario_id'] = 'UNKNOWN'
            results['confidence'] = anomaly_score
            results['source'] = 'anomaly_detection'
            return results

        # 否则使用深度编码器结果
        if len(deep_results) > 0:
            results['scenario_id'] = deep_results[0][0]
            results['confidence'] = deep_results[0][1]
            results['source'] = 'deep_encoder'

        # 如果规则引擎有高置信度结果，优先使用
        if 'rule' in results['details']:
            rule_conf = results['details']['rule']['confidence']
            if rule_conf > 0.9:
                results['scenario_id'] = results['details']['rule']['scenario_id']
                results['confidence'] = rule_conf
                results['source'] = 'rule_engine'

        return results


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "深度场景编码器测试")
    print("=" * 70)

    # 创建配置
    config = DeepEncoderConfig(
        sequence_length=96,
        input_channels=5,
        encoder_type='cnn',
        embedding_dim=64,
    )

    # 创建编码器
    encoder = DeepScenarioEncoder(config)

    # 测试编码
    print("\n1. 编码测试")
    print("-" * 50)

    test_data = np.random.randn(96, 5).astype(np.float32)
    embedding = encoder.encode(test_data)
    print(f"  输入形状: {test_data.shape}")
    print(f"  嵌入形状: {embedding.shape}")
    print(f"  嵌入范数: {np.linalg.norm(embedding):.4f}")

    # 批量编码
    batch_data = np.random.randn(8, 96, 5).astype(np.float32)
    batch_embedding = encoder.encode(batch_data)
    print(f"  批量输入: {batch_data.shape}")
    print(f"  批量嵌入: {batch_embedding.shape}")

    # 添加场景到数据库
    print("\n2. 向量数据库测试")
    print("-" * 50)

    scenarios = ['flood', 'drought', 'normal', 'ice_period', 'pollution']
    for i, scenario in enumerate(scenarios):
        for j in range(10):
            data = np.random.randn(96, 5).astype(np.float32)
            # 添加场景特定模式
            if scenario == 'flood':
                data[:, 0] += 2.0  # 高水位
            elif scenario == 'drought':
                data[:, 0] -= 2.0  # 低水位
            encoder.add_scenario(data, scenario, {'index': j})

    stats = encoder.vector_db.get_statistics()
    print(f"  向量数量: {stats['num_vectors']}")
    print(f"  场景类型: {stats['scenario_types']}")

    # 测试检索
    print("\n3. 场景识别测试")
    print("-" * 50)

    # 生成类似flood的数据
    flood_like = np.random.randn(96, 5).astype(np.float32)
    flood_like[:, 0] += 2.0

    results = encoder.recognize(flood_like)
    print(f"  输入: 类似洪水场景")
    print(f"  Top-3 结果:")
    for scenario_id, similarity, _ in results[:3]:
        print(f"    {scenario_id}: {similarity:.4f}")

    # 测试异常检测
    print("\n4. 异常检测测试")
    print("-" * 50)

    # 正常数据
    normal_data = np.random.randn(96, 5).astype(np.float32)
    is_anomaly, score = encoder.detect_anomaly(normal_data)
    print(f"  正常数据: is_anomaly={is_anomaly}, score={score:.4f}")

    # 异常数据 (极端值)
    anomaly_data = np.random.randn(96, 5).astype(np.float32) * 10
    is_anomaly, score = encoder.detect_anomaly(anomaly_data)
    print(f"  异常数据: is_anomaly={is_anomaly}, score={score:.4f}")

    # 测试数据增强
    print("\n5. 数据增强测试")
    print("-" * 50)

    original = np.random.randn(96, 5).astype(np.float32)
    augmented = TimeSeriesAugmentation.augment(original)
    print(f"  原始数据范围: [{original.min():.2f}, {original.max():.2f}]")
    print(f"  增强数据范围: [{augmented.min():.2f}, {augmented.max():.2f}]")

    # 测试对比损失
    print("\n6. 对比损失测试")
    print("-" * 50)

    criterion = ContrastiveLoss(config)

    z_i = torch.randn(32, 64)
    z_j = torch.randn(32, 64)

    loss = criterion(z_i, positives=z_j)
    print(f"  SimCLR损失: {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

"""
神经代理模型 (Neural Surrogate Model)
Neural Physics Engine for Water Network World Model

用深度学习模型替换/增强原有的基于微分方程的物理模型:
- LSTM/GRU 网络捕捉水流时滞和记忆效应
- PINN (Physics-Informed Neural Networks) 物理约束
- 与原 PhysicsModel 相同的 step() 接口

技术特点:
1. 时序建模: 使用 LSTM 捕捉变时滞非线性动态
2. 物理约束: Loss 函数中加入水量平衡残差作为正则项
3. Teacher-Student: 利用现有 physics_model.py 生成训练数据
4. 接口兼容: NeuralPhysicsEngine 保持与原 PhysicsModel 相同接口
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)


# ==============================================================================
# 配置数据类
# ==============================================================================

@dataclass
class NeuralPhysicsConfig:
    """神经物理引擎配置"""
    # 网络架构
    input_dim: int = 5          # [上游流量历史, 当前水位, 闸门动作, 分水扰动, 时间特征]
    hidden_dim: int = 128       # LSTM隐层维度
    num_layers: int = 2         # LSTM层数
    output_dim: int = 1         # 下一时刻水位
    dropout: float = 0.1        # Dropout率

    # 序列参数
    sequence_length: int = 16   # 历史序列长度 (用于捕捉时滞)
    prediction_horizon: int = 1 # 预测步数

    # 物理约束
    use_physics_loss: bool = True     # 是否使用物理约束损失
    physics_loss_weight: float = 0.1  # 物理损失权重
    mass_balance_tol: float = 0.01    # 质量平衡容差

    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 100

    # 模型路径
    model_path: str = "neural_physics_model.pt"


# ==============================================================================
# LSTM 代理模型
# ==============================================================================

class LSTMSurrogateModel(nn.Module):
    """
    LSTM 神经代理模型

    用于学习渠池水位-流量动态关系:
    - 输入: [上游流量历史序列, 当前水位, 闸门开度, 分水扰动]
    - 输出: 下一时刻水位预测

    架构特点:
    - Bi-directional LSTM 捕捉双向时序依赖
    - Attention 机制聚焦关键时刻
    - Residual 连接稳定训练
    """

    def __init__(self, config: NeuralPhysicsConfig):
        super().__init__()
        self.config = config

        # 输入嵌入层
        self.input_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )

        # 时间注意力机制
        self.attention = TemporalAttention(config.hidden_dim * 2)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

        # 残差连接的水位映射
        self.level_residual = nn.Linear(1, config.output_dim)

    def forward(self,
                x: torch.Tensor,
                current_level: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            current_level: 当前水位 [batch, 1] (用于残差连接)

        Returns:
            预测水位 [batch, output_dim]
        """
        batch_size = x.size(0)

        # 输入嵌入
        embedded = self.input_embed(x)

        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # 注意力聚合
        context = self.attention(lstm_out)

        # 预测输出
        output = self.output_layer(context)

        # 残差连接 (物理约束: 水位变化应该是渐变的)
        if current_level is not None:
            residual = self.level_residual(current_level)
            output = output + residual

        return output

    def predict_sequence(self,
                        x: torch.Tensor,
                        steps: int = 1) -> torch.Tensor:
        """
        多步预测

        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            steps: 预测步数

        Returns:
            预测序列 [batch, steps, output_dim]
        """
        predictions = []
        current_x = x

        for _ in range(steps):
            # 单步预测
            pred = self.forward(current_x)
            predictions.append(pred)

            # 更新输入序列 (滚动窗口)
            # 这里简化处理，实际应用中需要更新流量等特征
            current_x = torch.roll(current_x, shifts=-1, dims=1)
            current_x[:, -1, 0] = pred.squeeze()

        return torch.stack(predictions, dim=1)


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]

        Returns:
            context: [batch, hidden_dim]
        """
        # 计算注意力权重
        weights = self.attention(lstm_output)  # [batch, seq_len, 1]
        weights = F.softmax(weights, dim=1)

        # 加权求和
        context = torch.sum(weights * lstm_output, dim=1)

        return context


# ==============================================================================
# GRU 代理模型 (轻量级替代方案)
# ==============================================================================

class GRUSurrogateModel(nn.Module):
    """
    GRU 神经代理模型 (轻量级)

    比 LSTM 参数更少，训练更快，适合实时推理
    """

    def __init__(self, config: NeuralPhysicsConfig):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.input_embed = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        # GRU 编码器
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        embedded = self.input_embed(x)
        _, h_n = self.gru(embedded)
        output = self.output_layer(h_n[-1])
        return output


# ==============================================================================
# 物理信息神经网络损失 (PINN Loss)
# ==============================================================================

class PINNLoss(nn.Module):
    """
    Physics-Informed Neural Network Loss
    物理信息神经网络损失函数

    在标准 MSE 损失基础上，加入物理约束:
    1. 质量守恒: dV/dt = Q_in - Q_out
    2. 水位非负约束
    3. 变化率约束 (防止突变)
    """

    def __init__(self,
                 mass_balance_weight: float = 0.1,
                 level_constraint_weight: float = 0.05,
                 smoothness_weight: float = 0.01,
                 dt: float = 900.0,
                 A_s: float = 100000.0):
        super().__init__()
        self.mass_balance_weight = mass_balance_weight
        self.level_constraint_weight = level_constraint_weight
        self.smoothness_weight = smoothness_weight
        self.dt = dt
        self.A_s = A_s

    def forward(self,
                pred_level: torch.Tensor,
                target_level: torch.Tensor,
                q_in: torch.Tensor,
                q_out: torch.Tensor,
                prev_level: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算带物理约束的损失

        Args:
            pred_level: 预测水位 [batch, 1]
            target_level: 目标水位 [batch, 1]
            q_in: 入流 [batch, 1]
            q_out: 出流 [batch, 1]
            prev_level: 上一时刻水位 [batch, 1]

        Returns:
            损失字典
        """
        losses = {}

        # 1. 数据拟合损失 (MSE)
        mse_loss = F.mse_loss(pred_level, target_level)
        losses['mse_loss'] = mse_loss

        # 2. 质量守恒约束
        # dZ = (Q_in - Q_out) * dt / A_s
        if prev_level is not None:
            expected_dZ = (q_in - q_out) * self.dt / self.A_s
            actual_dZ = pred_level - prev_level
            mass_balance_residual = F.mse_loss(actual_dZ, expected_dZ)
            losses['mass_balance_loss'] = mass_balance_residual * self.mass_balance_weight
        else:
            losses['mass_balance_loss'] = torch.tensor(0.0)

        # 3. 水位非负约束 (软约束)
        level_violation = F.relu(-pred_level)  # 负水位惩罚
        losses['level_constraint_loss'] = level_violation.mean() * self.level_constraint_weight

        # 4. 平滑性约束 (防止突变)
        if prev_level is not None:
            level_change = torch.abs(pred_level - prev_level)
            max_change = 0.5  # 最大允许变化 [m]
            smoothness_violation = F.relu(level_change - max_change)
            losses['smoothness_loss'] = smoothness_violation.mean() * self.smoothness_weight
        else:
            losses['smoothness_loss'] = torch.tensor(0.0)

        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses


# ==============================================================================
# 神经物理引擎 (主接口类)
# ==============================================================================

class NeuralPhysicsEngine:
    """
    神经物理引擎

    提供与原 PhysicsModel 兼容的接口，内部使用神经网络进行推理
    支持:
    - 单步模拟 step()
    - 多步预测 predict()
    - 模型加载/保存
    - 与传统模型的混合推理
    """

    def __init__(self,
                 config: NeuralPhysicsConfig = None,
                 device: str = None):
        """
        初始化神经物理引擎

        Args:
            config: 配置
            device: 计算设备 ('cpu', 'cuda', 'mps')
        """
        self.config = config or NeuralPhysicsConfig()

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # 创建模型
        self.model = LSTMSurrogateModel(self.config).to(self.device)

        # 历史数据缓冲 (用于维护输入序列)
        self._history_buffer: List[np.ndarray] = []
        self._max_history = self.config.sequence_length

        # 当前状态
        self.current_level: float = 4.0
        self.current_inflow: float = 100.0
        self.current_outflow: float = 100.0

        # 归一化参数 (需要从训练数据学习)
        self._input_mean = np.zeros(self.config.input_dim)
        self._input_std = np.ones(self.config.input_dim)
        self._output_mean = 0.0
        self._output_std = 1.0

        logger.info(f"NeuralPhysicsEngine 初始化完成, 设备: {self.device}")

    def reset(self,
              initial_level: float = 4.0,
              initial_inflow: float = 100.0):
        """
        重置引擎状态

        Args:
            initial_level: 初始水位 [m]
            initial_inflow: 初始入流 [m³/s]
        """
        self.current_level = initial_level
        self.current_inflow = initial_inflow
        self.current_outflow = initial_inflow

        # 清空历史缓冲，用初始状态填充
        initial_state = np.array([
            initial_inflow,     # 上游流量
            initial_level,      # 当前水位
            1.0,                # 闸门开度
            0.0,                # 分水扰动
            0.0                 # 时间特征
        ])
        self._history_buffer = [initial_state.copy() for _ in range(self._max_history)]

        logger.debug(f"引擎重置: 水位={initial_level}m, 流量={initial_inflow}m³/s")

    def step(self,
             q_in: float,
             q_out: float = None,
             gate_opening: float = 1.0,
             diversion: float = 0.0,
             time_feature: float = 0.0) -> float:
        """
        执行一步模拟 (与原 PhysicsModel.step() 接口兼容)

        Args:
            q_in: 入流 [m³/s]
            q_out: 出流 [m³/s] (如果为None则由模型预测)
            gate_opening: 闸门开度 [0-1]
            diversion: 分水扰动 [m³/s]
            time_feature: 时间特征 (0-1归一化的时间)

        Returns:
            新的水位 [m]
        """
        # 构建当前状态向量
        current_state = np.array([
            q_in,
            self.current_level,
            gate_opening,
            diversion,
            time_feature
        ])

        # 更新历史缓冲
        self._history_buffer.append(current_state)
        if len(self._history_buffer) > self._max_history:
            self._history_buffer.pop(0)

        # 构建输入序列
        input_seq = np.stack(self._history_buffer, axis=0)

        # 归一化
        input_seq_normalized = (input_seq - self._input_mean) / (self._input_std + 1e-8)

        # 转换为张量
        input_tensor = torch.FloatTensor(input_seq_normalized).unsqueeze(0).to(self.device)
        current_level_tensor = torch.FloatTensor([[self.current_level]]).to(self.device)

        # 模型推理
        self.model.eval()
        with torch.no_grad():
            pred_normalized = self.model(input_tensor, current_level_tensor)

        # 反归一化
        pred_level = pred_normalized.cpu().numpy()[0, 0] * self._output_std + self._output_mean

        # 物理约束: 水位不能为负
        pred_level = max(0.0, pred_level)

        # 更新状态
        self.current_level = pred_level
        self.current_inflow = q_in
        if q_out is not None:
            self.current_outflow = q_out

        return self.current_level

    def predict(self,
                steps: int = 10,
                future_inflows: np.ndarray = None) -> np.ndarray:
        """
        多步预测

        Args:
            steps: 预测步数
            future_inflows: 未来入流序列 [steps]

        Returns:
            预测水位序列 [steps]
        """
        predictions = []

        # 保存当前状态
        saved_history = self._history_buffer.copy()
        saved_level = self.current_level

        for i in range(steps):
            q_in = future_inflows[i] if future_inflows is not None else self.current_inflow
            pred = self.step(q_in)
            predictions.append(pred)

        # 恢复状态
        self._history_buffer = saved_history
        self.current_level = saved_level

        return np.array(predictions)

    def load_model(self, path: str = None):
        """加载预训练模型"""
        path = path or self.config.model_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._input_mean = checkpoint.get('input_mean', self._input_mean)
            self._input_std = checkpoint.get('input_std', self._input_std)
            self._output_mean = checkpoint.get('output_mean', self._output_mean)
            self._output_std = checkpoint.get('output_std', self._output_std)
            logger.info(f"模型加载成功: {path}")
        else:
            logger.warning(f"模型文件不存在: {path}")

    def save_model(self, path: str = None):
        """保存模型"""
        path = path or self.config.model_path
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_mean': self._input_mean,
            'input_std': self._input_std,
            'output_mean': self._output_mean,
            'output_std': self._output_std,
        }
        torch.save(checkpoint, path)
        logger.info(f"模型保存成功: {path}")

    def set_normalization_params(self,
                                 input_mean: np.ndarray,
                                 input_std: np.ndarray,
                                 output_mean: float,
                                 output_std: float):
        """设置归一化参数"""
        self._input_mean = input_mean
        self._input_std = input_std
        self._output_mean = output_mean
        self._output_std = output_std

    def get_state(self) -> Dict[str, float]:
        """获取当前状态 (与原 PhysicsModel 兼容)"""
        return {
            'level': self.current_level,
            'inflow': self.current_inflow,
            'outflow': self.current_outflow,
        }


# ==============================================================================
# 混合物理引擎 (传统+神经)
# ==============================================================================

class HybridPhysicsEngine:
    """
    混合物理引擎

    结合传统机理模型和神经网络模型:
    - 正常工况: 使用传统模型 (更可解释)
    - 复杂工况: 使用神经模型 (更准确)
    - 残差学习: 神经网络学习传统模型的误差
    """

    def __init__(self,
                 traditional_model,
                 neural_config: NeuralPhysicsConfig = None,
                 blend_weight: float = 0.5):
        """
        Args:
            traditional_model: 传统物理模型实例
            neural_config: 神经模型配置
            blend_weight: 混合权重 (0=纯传统, 1=纯神经)
        """
        self.traditional = traditional_model
        self.neural = NeuralPhysicsEngine(neural_config)
        self.blend_weight = blend_weight

    def step(self, q_in: float, q_out: float = None, **kwargs) -> float:
        """混合推理"""
        # 传统模型预测
        trad_pred = self.traditional.step(q_in, q_out)

        # 神经模型预测
        neural_pred = self.neural.step(q_in, q_out, **kwargs)

        # 加权混合
        final_pred = (1 - self.blend_weight) * trad_pred + self.blend_weight * neural_pred

        return final_pred

    def set_blend_weight(self, weight: float):
        """动态调整混合权重"""
        self.blend_weight = np.clip(weight, 0, 1)


# ==============================================================================
# 训练器
# ==============================================================================

class NeuralPhysicsTrainer:
    """神经物理模型训练器"""

    def __init__(self,
                 model: LSTMSurrogateModel,
                 config: NeuralPhysicsConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 损失函数
        self.criterion = PINNLoss() if config.use_physics_loss else nn.MSELoss()

        # 训练历史
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x, y = batch['input'].to(self.device), batch['target'].to(self.device)
            current_level = batch.get('current_level')
            if current_level is not None:
                current_level = current_level.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model(x, current_level)

            # 计算损失
            if isinstance(self.criterion, PINNLoss):
                q_in = batch['q_in'].to(self.device)
                q_out = batch['q_out'].to(self.device)
                prev_level = batch.get('prev_level')
                if prev_level is not None:
                    prev_level = prev_level.to(self.device)
                losses = self.criterion(pred, y, q_in, q_out, prev_level)
                loss = losses['total_loss']
            else:
                loss = self.criterion(pred, y)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, dataloader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch['input'].to(self.device), batch['target'].to(self.device)
                pred = self.model(x)
                loss = F.mse_loss(pred, y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self,
              train_loader,
              val_loader,
              num_epochs: int = None) -> Dict:
        """完整训练流程"""
        num_epochs = num_epochs or self.config.num_epochs
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        return self.history


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "神经物理引擎测试")
    print("=" * 70)

    # 创建配置
    config = NeuralPhysicsConfig(
        input_dim=5,
        hidden_dim=64,
        num_layers=2,
        sequence_length=16
    )

    # 创建引擎
    engine = NeuralPhysicsEngine(config)
    engine.reset(initial_level=4.0, initial_inflow=100.0)

    print(f"\n神经物理引擎创建成功")
    print(f"  设备: {engine.device}")
    print(f"  输入维度: {config.input_dim}")
    print(f"  序列长度: {config.sequence_length}")

    # 模拟测试
    print(f"\n运行100步模拟测试...")
    levels = []
    for i in range(100):
        # 模拟入流变化
        q_in = 100 + 20 * np.sin(2 * np.pi * i / 50)
        level = engine.step(q_in, gate_opening=0.8)
        levels.append(level)

    print(f"  初始水位: {levels[0]:.3f}m")
    print(f"  最终水位: {levels[-1]:.3f}m")
    print(f"  平均水位: {np.mean(levels):.3f}m")
    print(f"  水位标准差: {np.std(levels):.3f}m")

    # 测试PINN损失
    print(f"\n{'=' * 70}")
    print("PINN损失函数测试")
    print('=' * 70)

    pinn_loss = PINNLoss()

    # 创建测试数据
    batch_size = 32
    pred_level = torch.randn(batch_size, 1) * 0.5 + 4.0
    target_level = torch.randn(batch_size, 1) * 0.5 + 4.0
    q_in = torch.ones(batch_size, 1) * 100
    q_out = torch.ones(batch_size, 1) * 100
    prev_level = pred_level - 0.1

    losses = pinn_loss(pred_level, target_level, q_in, q_out, prev_level)

    print(f"  MSE损失: {losses['mse_loss'].item():.6f}")
    print(f"  质量平衡损失: {losses['mass_balance_loss'].item():.6f}")
    print(f"  水位约束损失: {losses['level_constraint_loss'].item():.6f}")
    print(f"  平滑性损失: {losses['smoothness_loss'].item():.6f}")
    print(f"  总损失: {losses['total_loss'].item():.6f}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

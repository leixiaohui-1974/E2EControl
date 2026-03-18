"""
深度学习异常检测器
包括：LSTM Autoencoder, GRU-based, Transformer等
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import Optional, List
from collections import deque

from .base_detector import (
    BaseDetector, AnomalyReport, AnomalyType,
    SeverityLevel, calculate_severity
)


class LSTMAutoencoderDetector(BaseDetector):
    """
    LSTM Autoencoder 异常检测器
    
    原理：
    - 使用LSTM学习时间序列的正常模式
    - 异常序列的重构误差会显著增大
    
    优势：
    - 捕捉时间依赖关系
    - 适合时间序列数据
    - 可学习复杂模式
    
    注：这是简化实现，生产环境应使用PyTorch或TensorFlow
    """
    
    def __init__(self, name: str = "LSTM-AE", sequence_length: int = 10,
                 hidden_dim: int = 16, threshold_percentile: float = 95):
        """
        初始化LSTM Autoencoder
        
        Args:
            name: 检测器名称
            sequence_length: 序列长度
            hidden_dim: 隐藏层维度
            threshold_percentile: 阈值百分位
        """
        super().__init__(name, window_size=sequence_length * 2)
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.threshold_percentile = threshold_percentile
        
        # 模型参数（简化版本）
        self.encoder_weights = None
        self.decoder_weights = None
        self.mean = 0.0
        self.std = 1.0
        self.threshold = 0.0
        
        # 序列缓存
        self.sequence_buffer = deque(maxlen=sequence_length)
        
    def fit(self, data: np.ndarray):
        """
        训练LSTM Autoencoder
        
        在生产环境中，应该使用：
        ```python
        import torch
        import torch.nn as nn
        
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
            
            def forward(self, x):
                encoded, _ = self.encoder(x)
                decoded, _ = self.decoder(encoded)
                return decoded
        ```
        
        Args:
            data: 训练数据
        """
        if len(data) < self.sequence_length * 2:
            return
        
        # 数据标准化
        self.mean = np.mean(data)
        self.std = np.std(data)
        normalized_data = (data - self.mean) / (self.std + 1e-8)
        
        # 构建序列
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = normalized_data[i:i + self.sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        # 简化的"训练"：使用滑动平均作为重构
        # 实际应该训练LSTM网络
        reconstructions = []
        for seq in sequences:
            # 简化重构：使用移动平均
            reconstruction = np.convolve(seq, np.ones(3)/3, mode='same')
            reconstructions.append(reconstruction)
        
        # 计算重构误差
        errors = []
        for i in range(len(sequences)):
            mse = np.mean((sequences[i] - reconstructions[i])**2)
            errors.append(mse)
        
        # 设置阈值
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        # "保存"模型参数（简化）
        self.encoder_weights = np.random.randn(self.hidden_dim, self.sequence_length)
        self.decoder_weights = np.random.randn(self.sequence_length, self.hidden_dim)
        
        self.is_trained = True
    
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        if not self.is_trained:
            return None
        
        # 添加到序列缓存
        normalized_value = (value - self.mean) / (self.std + 1e-8)
        self.sequence_buffer.append(normalized_value)
        
        # 需要完整序列
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # 获取当前序列
        current_sequence = np.array(list(self.sequence_buffer))
        
        # 简化的重构
        reconstruction = np.convolve(current_sequence, np.ones(3)/3, mode='same')
        
        # 计算重构误差
        mse = np.mean((current_sequence - reconstruction)**2)
        
        # 判断是否异常
        if mse > self.threshold:
            # 反标准化
            actual_value = value
            expected_value = reconstruction[-1] * self.std + self.mean
            
            severity = calculate_severity(mse - self.threshold, self.threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=actual_value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, (mse - self.threshold) / self.threshold),
                description=f"LSTM-AE检测: 重构误差MSE={mse:.4f}, 阈值={self.threshold:.4f}",
                threshold=self.threshold,
                expected_value=expected_value,
                deviation=mse - self.threshold
            )
        
        return None


class GRUDetector(BaseDetector):
    """
    GRU-based 异常检测器
    
    原理：使用GRU预测下一个时间点的值，预测误差大则异常
    """
    
    def __init__(self, name: str = "GRU", sequence_length: int = 10,
                 threshold_percentile: float = 95):
        """初始化GRU检测器"""
        super().__init__(name, window_size=sequence_length * 2)
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile
        
        # 模型参数（简化）
        self.mean = 0.0
        self.std = 1.0
        self.threshold = 0.0
        
        # 序列缓存
        self.sequence_buffer = deque(maxlen=sequence_length)
        
    def fit(self, data: np.ndarray):
        """训练GRU预测模型（简化版）"""
        if len(data) < self.sequence_length * 2:
            return
        
        # 标准化
        self.mean = np.mean(data)
        self.std = np.std(data)
        normalized_data = (data - self.mean) / (self.std + 1e-8)
        
        # 计算预测误差
        errors = []
        for i in range(self.sequence_length, len(data)):
            # 简化预测：使用指数加权移动平均
            seq = normalized_data[i-self.sequence_length:i]
            weights = np.exp(np.linspace(-1, 0, self.sequence_length))
            weights /= weights.sum()
            prediction = np.sum(seq * weights)
            
            error = abs(normalized_data[i] - prediction)
            errors.append(error)
        
        # 设置阈值
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        self.is_trained = True
    
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        if not self.is_trained:
            return None
        
        # 添加到序列缓存
        normalized_value = (value - self.mean) / (self.std + 1e-8)
        self.sequence_buffer.append(normalized_value)
        
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        # 预测（使用指数加权）
        seq = np.array(list(self.sequence_buffer)[:-1])
        weights = np.exp(np.linspace(-1, 0, len(seq)))
        weights /= weights.sum()
        prediction = np.sum(seq * weights)
        
        # 计算预测误差
        error = abs(normalized_value - prediction)
        
        # 判断是否异常
        if error > self.threshold:
            # 反标准化
            actual_value = value
            expected_value = prediction * self.std + self.mean
            
            severity = calculate_severity(error - self.threshold, self.threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=actual_value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, error / self.threshold),
                description=f"GRU检测: 预测误差={error:.4f}, 阈值={self.threshold:.4f}",
                threshold=self.threshold,
                expected_value=expected_value,
                deviation=error - self.threshold
            )
        
        return None


class VariationalAutoencoderDetector(BaseDetector):
    """
    Variational Autoencoder (VAE) 检测器
    
    原理：学习正常数据的概率分布，异常点的概率密度低
    """
    
    def __init__(self, name: str = "VAE", window_size: int = 50,
                 latent_dim: int = 5, threshold_percentile: float = 95):
        """初始化VAE检测器"""
        super().__init__(name, window_size)
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        
        # 模型参数（简化：使用高斯混合模型近似）
        self.mean = 0.0
        self.std = 1.0
        self.threshold = 0.0
        
    def fit(self, data: np.ndarray):
        """训练VAE（简化为高斯模型）"""
        if len(data) < 20:
            return
        
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # 计算负对数似然
        normalized_data = (data - self.mean) / (self.std + 1e-8)
        neg_log_likelihood = -(-0.5 * normalized_data**2 - 0.5 * np.log(2 * np.pi))
        
        # 设置阈值
        self.threshold = np.percentile(neg_log_likelihood, self.threshold_percentile)
        
        self.is_trained = True
    
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        if not self.is_trained:
            return None
        
        # 计算负对数似然
        normalized_value = (value - self.mean) / (self.std + 1e-8)
        neg_log_likelihood = -(-0.5 * normalized_value**2 - 0.5 * np.log(2 * np.pi))
        
        # 判断是否异常
        if neg_log_likelihood > self.threshold:
            severity = calculate_severity(neg_log_likelihood - self.threshold, self.threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, (neg_log_likelihood - self.threshold) / self.threshold),
                description=f"VAE检测: 负对数似然={neg_log_likelihood:.4f}, 阈值={self.threshold:.4f}",
                threshold=self.threshold,
                expected_value=self.mean,
                deviation=neg_log_likelihood - self.threshold
            )
        
        return None


# 演示
if __name__ == "__main__":
    logger.info("="*80)
    logger.info(" "*20 + "深度学习异常检测器演示")
    logger.info("="*80)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    
    # 正常数据：带趋势和周期性
    t = np.arange(n_samples)
    normal_data = 3.0 + 0.5 * np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(n_samples)
    
    # 创建检测器
    detectors = [
        LSTMAutoencoderDetector(sequence_length=10),
        GRUDetector(sequence_length=10),
        VariationalAutoencoderDetector()
    ]
    
    # 训练
    logger.info("\n训练检测器...")
    for detector in detectors:
        detector.fit(normal_data)
        logger.info(f"  ✓ {detector.name} 已训练")
    
    # 测试数据（含异常）
    logger.info("\n" + "-"*80)
    logger.info("测试异常检测...")
    logger.info("-"*80)
    
    test_data = list(normal_data[-30:])
    test_data[10] = 6.0  # 注入异常
    test_data[20] = 1.0  # 注入异常
    
    anomaly_counts = {d.name: 0 for d in detectors}
    
    for t, value in enumerate(test_data):
        if t % 10 == 0 or value in [6.0, 1.0]:
            logger.info(f"\n[T={t}] 值={value:.2f}")
            
        for detector in detectors:
            report = detector.detect(value, t, "test_var")
            if report:
                anomaly_counts[detector.name] += 1
                if value in [6.0, 1.0]:
                    logger.info(f"  🔴 {detector.name}: {report.description[:60]}...")
    
    # 统计
    logger.info("\n" + "="*80)
    logger.info("检测统计")
    logger.info("="*80)
    
    for detector in detectors:
        count = anomaly_counts[detector.name]
        logger.info(f"\n{detector.name}:")
        logger.info(f"  异常检出次数: {count}")
        logger.info(f"  检出率: {count / 2 * 100:.0f}% (预期2个异常)")
    
    logger.info("\n✅ 演示完成！")
    logger.info("\n注：这是简化实现，生产环境应使用PyTorch/TensorFlow")
    logger.info("="*80)

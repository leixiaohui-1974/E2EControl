"""
统计方法异常检测器
包括：3-sigma、CUSUM、EWMA等经典统计方法
"""

import logging

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from base_detector import (
        BaseDetector, AnomalyReport, AnomalyType,
        SeverityLevel, calculate_severity
    )
except ImportError:
    from .base_detector import (
        BaseDetector, AnomalyReport, AnomalyType,
        SeverityLevel, calculate_severity
    )


class ThreeSigmaDetector(BaseDetector):
    """
    3-Sigma检测器
    基于正态分布假设，超过3个标准差视为异常
    """
    
    def __init__(self, name: str = "3-sigma", window_size: int = 20, 
                 n_sigma: float = 3.0):
        """
        初始化3-Sigma检测器
        
        Args:
            name: 检测器名称
            window_size: 滑动窗口大小
            n_sigma: 标准差倍数（默认3）
        """
        super().__init__(name, window_size)
        self.n_sigma = n_sigma
        self.mean = 0.0
        self.std = 0.0
        
    def fit(self, data: np.ndarray):
        """训练检测器"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.is_trained = True
        
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        # 使用滑动窗口更新统计量
        if len(self.history) >= 5:
            history_array = self.get_history_array()
            self.mean = np.mean(history_array)
            self.std = np.std(history_array)
        
        if self.std == 0:
            return None
        
        deviation = abs(value - self.mean)
        threshold = self.n_sigma * self.std
        
        if deviation > threshold:
            self.anomaly_count += 1
            
            severity = calculate_severity(deviation, threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, deviation / threshold / 2),
                description=f"{self.name}检测: 值偏离均值{deviation:.2f}, 超过{self.n_sigma}σ阈值",
                threshold=threshold,
                expected_value=self.mean,
                deviation=deviation
            )
        
        return None


class CUSUMDetector(BaseDetector):
    """
    CUSUM检测器 (Cumulative Sum)
    用于检测过程均值的持续偏移
    """
    
    def __init__(self, name: str = "CUSUM", threshold: float = 5.0, 
                 drift: float = 1.0):
        """
        初始化CUSUM检测器
        
        Args:
            name: 检测器名称
            threshold: 检测阈值
            drift: 漂移量
        """
        super().__init__(name, window_size=100)
        self.threshold = threshold
        self.drift = drift
        self.target_mean = 0.0
        self.cumsum_pos = 0.0  # 正向累积和
        self.cumsum_neg = 0.0  # 负向累积和
        
    def fit(self, data: np.ndarray):
        """训练检测器"""
        self.target_mean = np.mean(data)
        self.is_trained = True
        
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        # 更新目标均值（滑动窗口）
        if len(self.history) >= 10:
            self.target_mean = np.mean(self.get_history_array())
        
        # 计算偏差
        deviation = value - self.target_mean
        
        # 更新累积和
        self.cumsum_pos = max(0, self.cumsum_pos + deviation - self.drift)
        self.cumsum_neg = max(0, self.cumsum_neg - deviation - self.drift)
        
        # 检测
        if self.cumsum_pos > self.threshold:
            self.anomaly_count += 1
            self.cumsum_pos = 0  # 重置
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.DRIFT,
                severity=SeverityLevel.MODERATE,
                confidence=min(1.0, self.cumsum_pos / self.threshold),
                description=f"CUSUM检测: 检测到正向漂移，累积和={self.cumsum_pos:.2f}",
                threshold=self.threshold,
                expected_value=self.target_mean,
                deviation=deviation
            )
        
        elif self.cumsum_neg > self.threshold:
            self.anomaly_count += 1
            self.cumsum_neg = 0  # 重置
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.DRIFT,
                severity=SeverityLevel.MODERATE,
                confidence=min(1.0, self.cumsum_neg / self.threshold),
                description=f"CUSUM检测: 检测到负向漂移，累积和={self.cumsum_neg:.2f}",
                threshold=self.threshold,
                expected_value=self.target_mean,
                deviation=deviation
            )
        
        return None


class EWMADetector(BaseDetector):
    """
    EWMA检测器 (Exponentially Weighted Moving Average)
    指数加权移动平均，对最近数据赋予更大权重
    """
    
    def __init__(self, name: str = "EWMA", alpha: float = 0.3, 
                 threshold_factor: float = 3.0):
        """
        初始化EWMA检测器
        
        Args:
            name: 检测器名称
            alpha: 平滑系数 (0, 1]，越大对最近数据越敏感
            threshold_factor: 阈值因子
        """
        super().__init__(name, window_size=50)
        self.alpha = alpha
        self.threshold_factor = threshold_factor
        self.ewma = 0.0
        self.ewma_variance = 0.0
        self.initialized = False
        
    def fit(self, data: np.ndarray):
        """训练检测器"""
        self.ewma = np.mean(data[:10]) if len(data) >= 10 else np.mean(data)
        self.ewma_variance = np.var(data)
        self.initialized = True
        self.is_trained = True
        
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        # 初始化
        if not self.initialized:
            if len(self.history) >= 5:
                self.ewma = np.mean(self.get_history_array())
                self.ewma_variance = np.var(self.get_history_array())
                self.initialized = True
            return None
        
        # 计算偏差
        deviation = abs(value - self.ewma)
        threshold = self.threshold_factor * np.sqrt(self.ewma_variance)
        
        # 更新EWMA
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma
        self.ewma_variance = self.alpha * (value - self.ewma)**2 + (1 - self.alpha) * self.ewma_variance
        
        # 检测
        if deviation > threshold:
            self.anomaly_count += 1
            
            severity = calculate_severity(deviation, threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, deviation / threshold),
                description=f"EWMA检测: 值偏离EWMA {deviation:.2f}, 超过阈值",
                threshold=threshold,
                expected_value=self.ewma,
                deviation=deviation
            )
        
        return None


class RangeDetector(BaseDetector):
    """
    范围检测器
    检测值是否超出预定义的正常范围
    """
    
    def __init__(self, name: str = "Range", min_value: float = 0.0, 
                 max_value: float = 10.0):
        """
        初始化范围检测器
        
        Args:
            name: 检测器名称
            min_value: 最小正常值
            max_value: 最大正常值
        """
        super().__init__(name, window_size=10)
        self.min_value = min_value
        self.max_value = max_value
        
    def fit(self, data: np.ndarray):
        """训练检测器（可选）"""
        # 自动调整范围
        self.min_value = np.min(data) * 0.8
        self.max_value = np.max(data) * 1.2
        self.is_trained = True
        
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        if value < self.min_value:
            self.anomaly_count += 1
            deviation = self.min_value - value
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.LEVEL,
                severity=calculate_severity(deviation, self.min_value),
                confidence=1.0,
                description=f"范围检测: 值{value:.2f}低于下限{self.min_value:.2f}",
                threshold=self.min_value,
                expected_value=(self.min_value + self.max_value) / 2,
                deviation=deviation
            )
        
        elif value > self.max_value:
            self.anomaly_count += 1
            deviation = value - self.max_value
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.LEVEL,
                severity=calculate_severity(deviation, self.max_value),
                confidence=1.0,
                description=f"范围检测: 值{value:.2f}超过上限{self.max_value:.2f}",
                threshold=self.max_value,
                expected_value=(self.min_value + self.max_value) / 2,
                deviation=deviation
            )
        
        return None


class RateOfChangeDetector(BaseDetector):
    """
    变化率检测器
    检测值的变化速率是否异常
    """
    
    def __init__(self, name: str = "RateOfChange", max_rate: float = 2.0):
        """
        初始化变化率检测器
        
        Args:
            name: 检测器名称
            max_rate: 最大允许变化率
        """
        super().__init__(name, window_size=10)
        self.max_rate = max_rate
        self.last_value = None
        
    def fit(self, data: np.ndarray):
        """训练检测器"""
        # 计算历史最大变化率
        if len(data) > 1:
            rates = np.abs(np.diff(data))
            self.max_rate = np.percentile(rates, 95) * 1.5  # 95分位数*1.5
        self.is_trained = True
        
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        self.add_history(value)
        
        if self.last_value is None:
            self.last_value = value
            return None
        
        # 计算变化率
        rate = abs(value - self.last_value)
        self.last_value = value
        
        if rate > self.max_rate:
            self.anomaly_count += 1
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=calculate_severity(rate, self.max_rate),
                confidence=min(1.0, rate / self.max_rate),
                description=f"变化率检测: 变化率{rate:.2f}超过阈值{self.max_rate:.2f}",
                threshold=self.max_rate,
                deviation=rate
            )
        
        return None


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "统计方法检测器演示")
    logger.info("="*70)
    
    # 生成测试数据
    np.random.seed(42)
    normal_data = np.random.normal(3.0, 0.2, 100)
    
    # 创建异常数据
    test_data = list(normal_data[:50])
    test_data.extend([3.0] * 10)  # 正常
    test_data.append(5.5)  # 突变异常
    test_data.extend([3.0] * 10)
    test_data.extend(np.linspace(3.0, 4.5, 20))  # 漂移
    
    # 测试各个检测器
    detectors = [
        ThreeSigmaDetector(),
        CUSUMDetector(),
        EWMADetector(),
        RangeDetector(min_value=1.0, max_value=5.0),
        RateOfChangeDetector(max_rate=0.5)
    ]
    
    # 训练
    logger.info("\n训练检测器...")
    for detector in detectors:
        detector.fit(normal_data)
        logger.info(f"  ✓ {detector.name} 已训练")
    
    # 检测
    logger.info("\n" + "-"*70)
    logger.info("运行异常检测...")
    logger.info("-"*70)
    
    anomaly_count = {d.name: 0 for d in detectors}
    
    for t, value in enumerate(test_data):
        for detector in detectors:
            report = detector.detect(value, t, "test_variable")
            if report:
                anomaly_count[detector.name] += 1
                if anomaly_count[detector.name] <= 3:  # 只打印前3个
                    logger.info(f"\n[{t}] {detector.name} 检测到异常:")
                    logger.info(f"  值: {report.value:.2f}")
                    logger.info(f"  类型: {report.anomaly_type.value}")
                    logger.info(f"  严重程度: {report.severity.name}")
                    logger.info(f"  描述: {report.description}")
    
    # 统计
    logger.info("\n" + "="*70)
    logger.info("检测统计")
    logger.info("="*70)
    
    for detector in detectors:
        stats = detector.get_statistics()
        logger.info(f"\n{detector.name}:")
        logger.info(f"  总检测次数: {stats['total_count']}")
        logger.info(f"  异常次数: {stats['anomaly_count']}")
        logger.info(f"  异常率: {stats['anomaly_rate']:.2%}")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)

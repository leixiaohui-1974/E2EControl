"""
异常检测基础框架
定义异常检测器的基类和通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型"""
    SENSOR = "sensor"              # 传感器异常
    ACTUATOR = "actuator"          # 执行器异常
    LEVEL = "level"                # 水位异常
    FLOW = "flow"                  # 流量异常
    SUDDEN_CHANGE = "sudden_change"  # 突变
    DRIFT = "drift"                # 漂移
    SPIKE = "spike"                # 尖峰
    STUCK = "stuck"                # 卡死
    NOISE = "noise"                # 噪声异常


class SeverityLevel(Enum):
    """严重程度"""
    NORMAL = 0      # 正常
    MINOR = 1       # 轻微
    MODERATE = 2    # 中度
    SEVERE = 3      # 严重
    CRITICAL = 4    # 危急


@dataclass
class AnomalyReport:
    """异常报告"""
    timestamp: int
    variable_name: str
    value: float
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float  # 置信度 [0, 1]
    description: str
    threshold: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None


class BaseDetector(ABC):
    """异常检测器基类"""
    
    def __init__(self, name: str, window_size: int = 20):
        """
        初始化检测器
        
        Args:
            name: 检测器名称
            window_size: 时间窗口大小
        """
        self.name = name
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.is_trained = False
        self.anomaly_count = 0
        self.total_count = 0
        
    @abstractmethod
    def fit(self, data: np.ndarray):
        """
        训练检测器（使用正常数据）
        
        Args:
            data: 训练数据
        """
        pass
    
    @abstractmethod
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """
        检测单个数据点是否异常
        
        Args:
            value: 当前值
            timestamp: 时间戳
            variable_name: 变量名
            
        Returns:
            异常报告（如果检测到异常），否则返回None
        """
        pass
    
    def add_history(self, value: float):
        """添加历史数据"""
        self.history.append(value)
        self.total_count += 1
    
    def get_history_array(self) -> np.ndarray:
        """获取历史数据数组"""
        return np.array(list(self.history))
    
    def get_statistics(self) -> Dict:
        """获取检测统计信息"""
        return {
            'name': self.name,
            'total_count': self.total_count,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(1, self.total_count),
            'is_trained': self.is_trained,
            'window_size': self.window_size
        }
    
    def reset(self):
        """重置检测器"""
        self.history.clear()
        self.anomaly_count = 0
        self.total_count = 0


class MultiVariateDetector(ABC):
    """多变量异常检测器基类"""
    
    def __init__(self, name: str, variable_names: List[str], window_size: int = 20):
        """
        初始化多变量检测器
        
        Args:
            name: 检测器名称
            variable_names: 变量名列表
            window_size: 时间窗口大小
        """
        self.name = name
        self.variable_names = variable_names
        self.window_size = window_size
        self.history = {var: deque(maxlen=window_size) for var in variable_names}
        self.is_trained = False
        
    @abstractmethod
    def fit(self, data: Dict[str, np.ndarray]):
        """
        训练检测器
        
        Args:
            data: 训练数据字典 {variable_name: array}
        """
        pass
    
    @abstractmethod
    def detect(self, values: Dict[str, float], timestamp: int) -> List[AnomalyReport]:
        """
        检测多变量数据
        
        Args:
            values: 当前值字典 {variable_name: value}
            timestamp: 时间戳
            
        Returns:
            异常报告列表
        """
        pass
    
    def add_history(self, values: Dict[str, float]):
        """添加历史数据"""
        for var, val in values.items():
            if var in self.history:
                self.history[var].append(val)
    
    def get_history_arrays(self) -> Dict[str, np.ndarray]:
        """获取历史数据数组"""
        return {var: np.array(list(hist)) for var, hist in self.history.items()}


class EnsembleDetector:
    """
    集成检测器
    结合多个检测器的结果
    """
    
    def __init__(self, name: str = "ensemble"):
        """初始化集成检测器"""
        self.name = name
        self.detectors: List[BaseDetector] = []
        self.weights: List[float] = []
        
    def add_detector(self, detector: BaseDetector, weight: float = 1.0):
        """
        添加检测器
        
        Args:
            detector: 检测器实例
            weight: 权重
        """
        self.detectors.append(detector)
        self.weights.append(weight)
    
    def detect(self, value: float, timestamp: int, variable_name: str, 
               method: str = 'voting') -> Optional[AnomalyReport]:
        """
        集成检测
        
        Args:
            value: 当前值
            timestamp: 时间戳
            variable_name: 变量名
            method: 集成方法 ('voting', 'weighted', 'unanimous')
            
        Returns:
            异常报告（如果检测到异常）
        """
        reports = []
        for detector in self.detectors:
            report = detector.detect(value, timestamp, variable_name)
            if report is not None:
                reports.append(report)
        
        if len(reports) == 0:
            return None
        
        # 根据集成方法决定
        if method == 'voting':
            # 投票：超过半数认为异常则判定为异常
            if len(reports) > len(self.detectors) / 2:
                return self._merge_reports(reports)
            return None
        
        elif method == 'weighted':
            # 加权：根据权重计算异常分数
            anomaly_scores = []
            for i, detector in enumerate(self.detectors):
                report = detector.detect(value, timestamp, variable_name)
                score = self.weights[i] * (1.0 if report is not None else 0.0)
                anomaly_scores.append(score)
            
            total_score = sum(anomaly_scores)
            threshold = sum(self.weights) / 2
            
            if total_score > threshold:
                return self._merge_reports(reports)
            return None
        
        elif method == 'unanimous':
            # 一致：所有检测器都认为异常
            if len(reports) == len(self.detectors):
                return self._merge_reports(reports)
            return None
        
        return None
    
    def _merge_reports(self, reports: List[AnomalyReport]) -> AnomalyReport:
        """合并多个异常报告"""
        if len(reports) == 0:
            return None
        
        # 取最严重的异常
        most_severe = max(reports, key=lambda r: r.severity.value)
        
        # 平均置信度
        avg_confidence = np.mean([r.confidence for r in reports])
        
        # 合并描述
        descriptions = [r.description for r in reports]
        merged_description = f"集成检测: {'; '.join(set(descriptions))}"
        
        return AnomalyReport(
            timestamp=most_severe.timestamp,
            variable_name=most_severe.variable_name,
            value=most_severe.value,
            anomaly_type=most_severe.anomaly_type,
            severity=most_severe.severity,
            confidence=avg_confidence,
            description=merged_description
        )


# 工具函数

def calculate_severity(deviation: float, threshold: float) -> SeverityLevel:
    """
    根据偏差计算严重程度
    
    Args:
        deviation: 偏差值
        threshold: 阈值
        
    Returns:
        严重程度等级
    """
    ratio = abs(deviation) / threshold
    
    if ratio < 1.0:
        return SeverityLevel.NORMAL
    elif ratio < 1.5:
        return SeverityLevel.MINOR
    elif ratio < 2.0:
        return SeverityLevel.MODERATE
    elif ratio < 3.0:
        return SeverityLevel.SEVERE
    else:
        return SeverityLevel.CRITICAL


def adaptive_threshold(data: np.ndarray, base_threshold: float, 
                       adaptation_rate: float = 0.1) -> float:
    """
    自适应阈值调整
    
    Args:
        data: 历史数据
        base_threshold: 基础阈值
        adaptation_rate: 适应率
        
    Returns:
        调整后的阈值
    """
    if len(data) < 5:
        return base_threshold
    
    recent_std = np.std(data[-10:])
    historical_std = np.std(data)
    
    # 根据最近波动调整阈值
    if recent_std > historical_std:
        adjusted_threshold = base_threshold * (1 + adaptation_rate * (recent_std / historical_std - 1))
    else:
        adjusted_threshold = base_threshold
    
    return adjusted_threshold


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "异常检测基础框架演示")
    logger.info("="*70)
    
    # 展示异常类型
    logger.info("\n支持的异常类型:")
    for anomaly_type in AnomalyType:
        logger.info(f"  • {anomaly_type.value}")
    
    # 展示严重程度
    logger.info("\n严重程度等级:")
    for severity in SeverityLevel:
        logger.info(f"  • {severity.name}: {severity.value}")
    
    # 创建示例报告
    logger.info("\n" + "-"*70)
    logger.info("异常报告示例:")
    logger.info("-"*70)
    
    report = AnomalyReport(
        timestamp=100,
        variable_name="pool_1_level",
        value=6.5,
        anomaly_type=AnomalyType.LEVEL,
        severity=SeverityLevel.SEVERE,
        confidence=0.95,
        description="水位严重超标，超过安全上限",
        threshold=5.0,
        expected_value=3.0,
        deviation=3.5
    )
    
    logger.info(f"时间戳: {report.timestamp}")
    logger.info(f"变量: {report.variable_name}")
    logger.info(f"当前值: {report.value}")
    logger.info(f"异常类型: {report.anomaly_type.value}")
    logger.info(f"严重程度: {report.severity.name}")
    logger.info(f"置信度: {report.confidence:.2%}")
    logger.info(f"描述: {report.description}")
    logger.info(f"阈值: {report.threshold}")
    logger.info(f"期望值: {report.expected_value}")
    logger.info(f"偏差: {report.deviation}")
    
    logger.info("\n" + "="*70)
    logger.info("基础框架加载完成！")
    logger.info("="*70)

"""
机器学习异常检测器
包括：Isolation Forest, One-Class SVM, LOF等无监督方法
"""

import sys
sys.path.append('..')

import numpy as np
from typing import Optional, List, Tuple
from collections import deque

from anomaly_detection.base_detector import (
    BaseDetector, MultiVariateDetector, AnomalyReport, 
    AnomalyType, SeverityLevel, calculate_severity
)


class IsolationForestDetector(MultiVariateDetector):
    """
    Isolation Forest 异常检测器
    
    原理：异常点更容易被隔离，需要更少的分割次数
    优势：
    - 无监督学习
    - 对高维数据有效
    - 训练速度快
    """
    
    def __init__(self, variable_names: List[str], window_size: int = 50,
                 contamination: float = 0.1, n_estimators: int = 100):
        """
        初始化Isolation Forest检测器
        
        Args:
            variable_names: 变量名列表
            window_size: 窗口大小
            contamination: 污染率（预期异常比例）
            n_estimators: 树的数量
        """
        super().__init__("IsolationForest", variable_names, window_size)
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # 模型参数（简化实现，生产环境应使用sklearn）
        self.trees = []
        self.threshold = 0.0
        
    def fit(self, data: dict):
        """
        训练Isolation Forest模型
        
        Args:
            data: 训练数据字典 {variable_name: array}
        """
        # 构建特征矩阵
        X = self._build_feature_matrix(data)
        
        if len(X) < 10:
            return
        
        # 简化的Isolation Forest实现
        # 生产环境应使用：
        # from sklearn.ensemble import IsolationForest
        # self.model = IsolationForest(contamination=self.contamination, 
        #                              n_estimators=self.n_estimators)
        # self.model.fit(X)
        
        # 这里使用简化版本：基于统计距离
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T)
        self.cov_inv = np.linalg.pinv(self.cov)
        
        # 计算马氏距离作为异常分数
        distances = []
        for x in X:
            diff = x - self.mean
            distance = np.sqrt(diff @ self.cov_inv @ diff.T)
            distances.append(distance)
        
        # 设置阈值（基于污染率）
        self.threshold = np.percentile(distances, (1 - self.contamination) * 100)
        
        self.is_trained = True
    
    def detect(self, values: dict, timestamp: int) -> List[AnomalyReport]:
        """
        检测异常
        
        Args:
            values: 当前值字典
            timestamp: 时间戳
            
        Returns:
            异常报告列表
        """
        if not self.is_trained:
            return []
        
        self.add_history(values)
        
        reports = []
        
        # 构建特征向量
        feature_vector = np.array([values.get(var, 0.0) for var in self.variable_names])
        
        # 计算异常分数（马氏距离）
        diff = feature_vector - self.mean
        anomaly_score = np.sqrt(diff @ self.cov_inv @ diff.T)
        
        # 判断是否异常
        if anomaly_score > self.threshold:
            # 识别最异常的变量
            normalized_diff = np.abs(diff) / (np.sqrt(np.diag(self.cov)) + 1e-6)
            most_anomalous_idx = np.argmax(normalized_diff)
            most_anomalous_var = self.variable_names[most_anomalous_idx]
            
            severity = calculate_severity(anomaly_score - self.threshold, self.threshold)
            
            report = AnomalyReport(
                timestamp=timestamp,
                variable_name=most_anomalous_var,
                value=values[most_anomalous_var],
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, (anomaly_score - self.threshold) / self.threshold),
                description=f"IsolationForest检测: 异常分数={anomaly_score:.2f}, 阈值={self.threshold:.2f}",
                threshold=self.threshold,
                deviation=anomaly_score - self.threshold
            )
            reports.append(report)
        
        return reports
    
    def _build_feature_matrix(self, data: dict) -> np.ndarray:
        """构建特征矩阵"""
        arrays = [data[var] for var in self.variable_names if var in data]
        return np.column_stack(arrays)


class OneClassSVMDetector(MultiVariateDetector):
    """
    One-Class SVM 检测器
    
    原理：学习正常数据的边界，超出边界的被视为异常
    优势：
    - 非线性边界
    - 核技巧处理复杂分布
    """
    
    def __init__(self, variable_names: List[str], window_size: int = 50,
                 nu: float = 0.1):
        """
        初始化One-Class SVM检测器
        
        Args:
            variable_names: 变量名列表
            window_size: 窗口大小
            nu: 异常比例上界（类似contamination）
        """
        super().__init__("OneClassSVM", variable_names, window_size)
        self.nu = nu
        
        # 模型参数
        self.support_vectors = None
        self.center = None
        self.radius = 0.0
        
    def fit(self, data: dict):
        """训练One-Class SVM模型"""
        X = self._build_feature_matrix(data)
        
        if len(X) < 10:
            return
        
        # 简化实现：使用球形边界（SVDD - Support Vector Data Description）
        # 生产环境应使用：
        # from sklearn.svm import OneClassSVM
        # self.model = OneClassSVM(nu=self.nu, kernel='rbf')
        # self.model.fit(X)
        
        # 计算数据中心
        self.center = np.mean(X, axis=0)
        
        # 计算到中心的距离
        distances = np.sqrt(np.sum((X - self.center)**2, axis=1))
        
        # 设置半径（基于nu）
        self.radius = np.percentile(distances, (1 - self.nu) * 100)
        
        self.is_trained = True
    
    def detect(self, values: dict, timestamp: int) -> List[AnomalyReport]:
        """检测异常"""
        if not self.is_trained:
            return []
        
        self.add_history(values)
        
        reports = []
        
        # 构建特征向量
        feature_vector = np.array([values.get(var, 0.0) for var in self.variable_names])
        
        # 计算到中心的距离
        distance = np.sqrt(np.sum((feature_vector - self.center)**2))
        
        # 判断是否超出边界
        if distance > self.radius:
            # 识别偏离最大的变量
            diff = np.abs(feature_vector - self.center)
            most_anomalous_idx = np.argmax(diff)
            most_anomalous_var = self.variable_names[most_anomalous_idx]
            
            severity = calculate_severity(distance - self.radius, self.radius)
            
            report = AnomalyReport(
                timestamp=timestamp,
                variable_name=most_anomalous_var,
                value=values[most_anomalous_var],
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, (distance - self.radius) / self.radius),
                description=f"OneClassSVM检测: 距离={distance:.2f}, 半径={self.radius:.2f}",
                threshold=self.radius,
                deviation=distance - self.radius
            )
            reports.append(report)
        
        return reports
    
    def _build_feature_matrix(self, data: dict) -> np.ndarray:
        """构建特征矩阵"""
        arrays = [data[var] for var in self.variable_names if var in data]
        return np.column_stack(arrays)


class LOFDetector(MultiVariateDetector):
    """
    Local Outlier Factor (LOF) 检测器
    
    原理：基于局部密度的异常检测
    优势：
    - 检测局部异常
    - 适应不同密度区域
    """
    
    def __init__(self, variable_names: List[str], window_size: int = 50,
                 n_neighbors: int = 20, contamination: float = 0.1):
        """
        初始化LOF检测器
        
        Args:
            variable_names: 变量名列表
            window_size: 窗口大小
            n_neighbors: 邻居数量
            contamination: 污染率
        """
        super().__init__("LOF", variable_names, window_size)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        # 训练数据（用于计算LOF）
        self.training_data = None
        self.threshold = 0.0
        
    def fit(self, data: dict):
        """训练LOF模型"""
        X = self._build_feature_matrix(data)
        
        if len(X) < self.n_neighbors + 5:
            return
        
        self.training_data = X
        
        # 计算所有点的LOF分数
        lof_scores = []
        for i in range(len(X)):
            lof = self._compute_lof(X[i], X)
            lof_scores.append(lof)
        
        # 设置阈值
        self.threshold = np.percentile(lof_scores, (1 - self.contamination) * 100)
        
        self.is_trained = True
    
    def detect(self, values: dict, timestamp: int) -> List[AnomalyReport]:
        """检测异常"""
        if not self.is_trained:
            return []
        
        self.add_history(values)
        
        reports = []
        
        # 构建特征向量
        feature_vector = np.array([values.get(var, 0.0) for var in self.variable_names])
        
        # 计算LOF分数
        lof_score = self._compute_lof(feature_vector, self.training_data)
        
        # 判断是否异常
        if lof_score > self.threshold:
            # 识别异常变量（简化）
            most_anomalous_var = self.variable_names[0]
            
            severity = calculate_severity(lof_score - self.threshold, self.threshold)
            
            report = AnomalyReport(
                timestamp=timestamp,
                variable_name=most_anomalous_var,
                value=values[most_anomalous_var],
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, (lof_score - self.threshold) / self.threshold),
                description=f"LOF检测: LOF分数={lof_score:.2f}, 阈值={self.threshold:.2f}",
                threshold=self.threshold,
                deviation=lof_score - self.threshold
            )
            reports.append(report)
        
        return reports
    
    def _compute_lof(self, point: np.ndarray, data: np.ndarray) -> float:
        """计算LOF分数（简化版本）"""
        # 计算到所有点的距离
        distances = np.sqrt(np.sum((data - point)**2, axis=1))
        
        # 找到k近邻
        k = min(self.n_neighbors, len(data) - 1)
        nearest_indices = np.argpartition(distances, k)[:k]
        nearest_distances = distances[nearest_indices]
        
        # 计算局部可达密度（简化）
        lrd = 1.0 / (np.mean(nearest_distances) + 1e-6)
        
        # 计算邻居的lrd
        neighbor_lrds = []
        for idx in nearest_indices:
            neighbor_distances = np.sqrt(np.sum((data - data[idx])**2, axis=1))
            neighbor_k_distances = np.partition(neighbor_distances, k)[:k]
            neighbor_lrd = 1.0 / (np.mean(neighbor_k_distances) + 1e-6)
            neighbor_lrds.append(neighbor_lrd)
        
        # 计算LOF
        if lrd == 0:
            return 1.0
        
        lof = np.mean(neighbor_lrds) / lrd
        
        return lof
    
    def _build_feature_matrix(self, data: dict) -> np.ndarray:
        """构建特征矩阵"""
        arrays = [data[var] for var in self.variable_names if var in data]
        return np.column_stack(arrays)


class AutoencoderDetector(BaseDetector):
    """
    Autoencoder 异常检测器（简化版本）
    
    原理：学习正常数据的低维表示，异常数据重构误差大
    """
    
    def __init__(self, name: str = "Autoencoder", window_size: int = 50,
                 encoding_dim: int = 5, threshold_percentile: float = 95):
        """
        初始化Autoencoder检测器
        
        Args:
            name: 检测器名称
            window_size: 窗口大小
            encoding_dim: 编码维度
            threshold_percentile: 阈值百分位
        """
        super().__init__(name, window_size)
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        
        # 模型参数（简化：使用PCA代替）
        self.mean = None
        self.components = None
        self.threshold = 0.0
        
    def fit(self, data: np.ndarray):
        """训练Autoencoder（使用PCA简化）"""
        if len(data) < 20:
            return
        
        # 中心化
        self.mean = np.mean(data)
        centered_data = data - self.mean
        
        # 简化：使用移动平均作为"重构"
        window = min(5, len(data) // 2)
        reconstructions = []
        
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            reconstruction = np.mean(data[start:end])
            reconstructions.append(reconstruction)
        
        # 计算重构误差
        errors = np.abs(data - np.array(reconstructions))
        
        # 设置阈值
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        self.is_trained = True
    
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """检测异常"""
        if not self.is_trained:
            self.add_history(value)
            return None
        
        self.add_history(value)
        
        # 重构（使用历史平均）
        if len(self.history) < 3:
            return None
        
        reconstruction = np.mean(list(self.history)[-5:])
        
        # 计算重构误差
        error = abs(value - reconstruction)
        
        # 判断是否异常
        if error > self.threshold:
            severity = calculate_severity(error, self.threshold)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                severity=severity,
                confidence=min(1.0, error / self.threshold),
                description=f"Autoencoder检测: 重构误差={error:.3f}, 阈值={self.threshold:.3f}",
                threshold=self.threshold,
                expected_value=reconstruction,
                deviation=error
            )
        
        return None


# 演示
if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "机器学习异常检测器演示")
    print("="*80)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    
    # 正常数据
    normal_data = {
        'level': np.random.normal(3.0, 0.2, n_samples),
        'flow': np.random.normal(50.0, 5.0, n_samples),
        'demand': np.random.normal(48.0, 4.0, n_samples)
    }
    
    # 创建检测器
    variable_names = ['level', 'flow', 'demand']
    
    detectors = [
        IsolationForestDetector(variable_names, contamination=0.1),
        OneClassSVMDetector(variable_names, nu=0.1),
        LOFDetector(variable_names, n_neighbors=10, contamination=0.1)
    ]
    
    # 训练
    print("\n训练检测器...")
    for detector in detectors:
        detector.fit(normal_data)
        print(f"  ✓ {detector.name} 已训练")
    
    # 测试异常数据
    print("\n" + "-"*80)
    print("测试异常检测...")
    print("-"*80)
    
    # 注入异常
    test_cases = [
        {'level': 3.0, 'flow': 50.0, 'demand': 48.0},  # 正常
        {'level': 5.5, 'flow': 50.0, 'demand': 48.0},  # 水位异常
        {'level': 3.0, 'flow': 90.0, 'demand': 48.0},  # 流量异常
        {'level': 3.0, 'flow': 50.0, 'demand': 90.0},  # 需求异常
    ]
    
    for t, test_values in enumerate(test_cases):
        print(f"\n[T={t}] 测试: level={test_values['level']}, "
              f"flow={test_values['flow']}, demand={test_values['demand']}")
        
        for detector in detectors:
            reports = detector.detect(test_values, t)
            if reports:
                for report in reports:
                    print(f"  🔴 {detector.name}: {report.description}")
            else:
                print(f"  ✓ {detector.name}: 正常")
    
    # 统计
    print("\n" + "="*80)
    print("检测统计")
    print("="*80)
    
    for detector in detectors:
        print(f"\n{detector.name}:")
        print(f"  训练状态: {'已训练' if detector.is_trained else '未训练'}")
    
    print("\n✅ 演示完成！")
    print("="*80)

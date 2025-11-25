"""
集成学习异常检测器
结合统计方法、机器学习和深度学习的优势
"""

import sys
sys.path.append('..')

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from anomaly_detection.base_detector import (
    BaseDetector, AnomalyReport, AnomalyType, SeverityLevel
)
from anomaly_detection.statistical_detectors import (
    ThreeSigmaDetector, CUSUMDetector, EWMADetector
)
from anomaly_detection.ml_detectors import AutoencoderDetector
from anomaly_detection.deep_learning_detectors import (
    LSTMAutoencoderDetector, GRUDetector
)


class HybridEnsembleDetector(BaseDetector):
    """
    混合集成检测器
    
    架构：
    1. 统计层：快速响应突变（3-Sigma, CUSUM, EWMA）
    2. 机器学习层：检测复杂模式（Autoencoder）
    3. 深度学习层：时序模式（LSTM, GRU）
    
    融合策略：
    - 投票法（Voting）
    - 加权法（Weighted）
    - 堆叠法（Stacking）
    """
    
    def __init__(self, name: str = "HybridEnsemble", window_size: int = 50,
                 fusion_method: str = 'weighted'):
        """
        初始化混合集成检测器
        
        Args:
            name: 检测器名称
            window_size: 窗口大小
            fusion_method: 融合方法 ('voting', 'weighted', 'stacking')
        """
        super().__init__(name, window_size)
        self.fusion_method = fusion_method
        
        # 统计层检测器
        self.statistical_detectors = [
            ThreeSigmaDetector(n_sigma=3.0),
            CUSUMDetector(threshold=5.0, drift=1.0),
            EWMADetector(alpha=0.3)
        ]
        
        # 机器学习层检测器
        self.ml_detectors = [
            AutoencoderDetector()
        ]
        
        # 深度学习层检测器
        self.dl_detectors = [
            LSTMAutoencoderDetector(sequence_length=10),
            GRUDetector(sequence_length=10)
        ]
        
        # 所有检测器
        self.all_detectors = (self.statistical_detectors + 
                             self.ml_detectors + 
                             self.dl_detectors)
        
        # 权重（基于各层的可靠性）
        self.weights = {
            'statistical': 0.3,
            'ml': 0.3,
            'dl': 0.4
        }
        
        # 性能统计（用于动态调整权重）
        self.performance_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        
    def fit(self, data: np.ndarray):
        """训练所有检测器"""
        print(f"  训练 {self.name}...")
        
        # 统计层
        for detector in self.statistical_detectors:
            detector.fit(data)
            print(f"    ✓ {detector.name}")
        
        # 机器学习层
        for detector in self.ml_detectors:
            detector.fit(data)
            print(f"    ✓ {detector.name}")
        
        # 深度学习层
        for detector in self.dl_detectors:
            detector.fit(data)
            print(f"    ✓ {detector.name}")
        
        self.is_trained = True
    
    def detect(self, value: float, timestamp: int, variable_name: str) -> Optional[AnomalyReport]:
        """
        集成检测
        
        Returns:
            融合后的异常报告
        """
        if not self.is_trained:
            self.add_history(value)
            return None
        
        self.add_history(value)
        
        # 收集各检测器的结果
        statistical_reports = []
        ml_reports = []
        dl_reports = []
        
        for detector in self.statistical_detectors:
            report = detector.detect(value, timestamp, variable_name)
            if report:
                statistical_reports.append(report)
        
        for detector in self.ml_detectors:
            report = detector.detect(value, timestamp, variable_name)
            if report:
                ml_reports.append(report)
        
        for detector in self.dl_detectors:
            report = detector.detect(value, timestamp, variable_name)
            if report:
                dl_reports.append(report)
        
        # 融合决策
        if self.fusion_method == 'voting':
            return self._voting_fusion(statistical_reports, ml_reports, dl_reports,
                                      value, timestamp, variable_name)
        elif self.fusion_method == 'weighted':
            return self._weighted_fusion(statistical_reports, ml_reports, dl_reports,
                                        value, timestamp, variable_name)
        else:  # stacking
            return self._stacking_fusion(statistical_reports, ml_reports, dl_reports,
                                        value, timestamp, variable_name)
    
    def _voting_fusion(self, stat_reports, ml_reports, dl_reports,
                      value, timestamp, variable_name) -> Optional[AnomalyReport]:
        """投票融合：多数检测器认为异常则判定为异常"""
        # 计算各层的投票
        stat_vote = 1 if len(stat_reports) > len(self.statistical_detectors) / 2 else 0
        ml_vote = 1 if len(ml_reports) > 0 else 0
        dl_vote = 1 if len(dl_reports) > 0 else 0
        
        total_votes = stat_vote + ml_vote + dl_vote
        
        # 需要至少2层认为异常
        if total_votes >= 2:
            # 合并所有报告
            all_reports = stat_reports + ml_reports + dl_reports
            
            if len(all_reports) == 0:
                return None
            
            # 选择最严重的
            most_severe = max(all_reports, key=lambda r: r.severity.value)
            
            # 计算平均置信度
            avg_confidence = np.mean([r.confidence for r in all_reports])
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=most_severe.anomaly_type,
                severity=most_severe.severity,
                confidence=avg_confidence,
                description=f"集成检测(投票): {len(all_reports)}/{len(self.all_detectors)}检测器报警",
                threshold=most_severe.threshold
            )
        
        return None
    
    def _weighted_fusion(self, stat_reports, ml_reports, dl_reports,
                        value, timestamp, variable_name) -> Optional[AnomalyReport]:
        """加权融合：根据各层权重计算异常分数"""
        # 计算各层的异常分数
        stat_score = len(stat_reports) / len(self.statistical_detectors) if self.statistical_detectors else 0
        ml_score = len(ml_reports) / len(self.ml_detectors) if self.ml_detectors else 0
        dl_score = len(dl_reports) / len(self.dl_detectors) if self.dl_detectors else 0
        
        # 加权求和
        weighted_score = (stat_score * self.weights['statistical'] +
                         ml_score * self.weights['ml'] +
                         dl_score * self.weights['dl'])
        
        # 阈值：加权分数>0.5认为异常
        if weighted_score > 0.5:
            all_reports = stat_reports + ml_reports + dl_reports
            
            if len(all_reports) == 0:
                return None
            
            # 选择最严重的
            most_severe = max(all_reports, key=lambda r: r.severity.value)
            
            # 置信度基于加权分数
            confidence = min(1.0, weighted_score * 1.5)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=most_severe.anomaly_type,
                severity=most_severe.severity,
                confidence=confidence,
                description=f"集成检测(加权): 分数={weighted_score:.2f}, "
                           f"统计={stat_score:.2f}, ML={ml_score:.2f}, DL={dl_score:.2f}",
                threshold=0.5
            )
        
        return None
    
    def _stacking_fusion(self, stat_reports, ml_reports, dl_reports,
                        value, timestamp, variable_name) -> Optional[AnomalyReport]:
        """堆叠融合：使用元学习器组合基学习器的输出"""
        # 构建特征向量
        features = [
            len(stat_reports) / len(self.statistical_detectors),
            len(ml_reports) / len(self.ml_detectors) if self.ml_detectors else 0,
            len(dl_reports) / len(self.dl_detectors) if self.dl_detectors else 0,
            np.mean([r.confidence for r in stat_reports]) if stat_reports else 0,
            np.mean([r.confidence for r in ml_reports]) if ml_reports else 0,
            np.mean([r.confidence for r in dl_reports]) if dl_reports else 0
        ]
        
        # 简化的元学习器：线性组合
        meta_weights = np.array([0.2, 0.15, 0.25, 0.15, 0.1, 0.15])
        meta_score = np.dot(features, meta_weights)
        
        # 阈值
        if meta_score > 0.3:
            all_reports = stat_reports + ml_reports + dl_reports
            
            if len(all_reports) == 0:
                return None
            
            most_severe = max(all_reports, key=lambda r: r.severity.value)
            
            return AnomalyReport(
                timestamp=timestamp,
                variable_name=variable_name,
                value=value,
                anomaly_type=most_severe.anomaly_type,
                severity=most_severe.severity,
                confidence=min(1.0, meta_score * 2),
                description=f"集成检测(堆叠): 元分数={meta_score:.2f}",
                threshold=0.3
            )
        
        return None
    
    def update_weights(self, detector_name: str, is_correct: bool):
        """
        动态更新权重（基于在线反馈）
        
        Args:
            detector_name: 检测器名称
            is_correct: 是否正确检测
        """
        # 更新性能统计
        if is_correct:
            self.performance_stats[detector_name]['tp'] += 1
        else:
            self.performance_stats[detector_name]['fp'] += 1
        
        # 重新计算权重（基于F1分数）
        # 这里省略详细实现
        pass
    
    def get_statistics(self) -> Dict:
        """获取集成检测器统计信息"""
        stats = super().get_statistics()
        
        # 添加各层统计
        stats['layers'] = {
            'statistical': len(self.statistical_detectors),
            'ml': len(self.ml_detectors),
            'dl': len(self.dl_detectors)
        }
        
        stats['weights'] = self.weights.copy()
        stats['fusion_method'] = self.fusion_method
        
        return stats


# 演示
if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "集成学习异常检测器演示")
    print("="*80)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 150
    
    # 正常数据：带周期性
    t = np.arange(n_samples)
    normal_data = 3.0 + 0.3 * np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(n_samples)
    
    # 创建集成检测器（3种融合方法）
    detectors = [
        HybridEnsembleDetector(fusion_method='voting'),
        HybridEnsembleDetector(fusion_method='weighted'),
        HybridEnsembleDetector(fusion_method='stacking')
    ]
    
    # 训练
    print("\n训练集成检测器...")
    print("="*80)
    for detector in detectors:
        detector.fit(normal_data)
        print()
    
    # 测试数据（含异常）
    print("="*80)
    print("测试异常检测...")
    print("="*80)
    
    test_data = list(normal_data[-30:])
    test_data[10] = 5.5  # 异常1
    test_data[20] = 1.2  # 异常2
    
    anomaly_counts = {d.fusion_method: 0 for d in detectors}
    
    for t, value in enumerate(test_data):
        for detector in detectors:
            report = detector.detect(value, t, "test_var")
            if report:
                anomaly_counts[detector.fusion_method] += 1
                if value in [5.5, 1.2]:
                    print(f"\n[T={t}] 值={value:.2f}")
                    print(f"  {detector.fusion_method}: {report.description}")
    
    # 统计
    print("\n" + "="*80)
    print("检测统计对比")
    print("="*80)
    
    for detector in detectors:
        count = anomaly_counts[detector.fusion_method]
        stats = detector.get_statistics()
        
        print(f"\n{detector.fusion_method.upper()} 融合:")
        print(f"  异常检出次数: {count}")
        print(f"  检测器层数: {sum(stats['layers'].values())}层")
        print(f"    - 统计层: {stats['layers']['statistical']}个")
        print(f"    - ML层: {stats['layers']['ml']}个")
        print(f"    - DL层: {stats['layers']['dl']}个")
        print(f"  融合权重:")
        for layer, weight in stats['weights'].items():
            print(f"    - {layer}: {weight:.2f}")
    
    print("\n✅ 集成检测器演示完成！")
    print("="*80)

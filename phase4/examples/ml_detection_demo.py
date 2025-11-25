"""
Phase 4.2 机器学习异常检测完整演示
对比统计方法 vs ML方法 vs 集成方法
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 导入检测器
from anomaly_detection.statistical_detectors import ThreeSigmaDetector, CUSUMDetector
from anomaly_detection.ml_detectors import (
    IsolationForestDetector, OneClassSVMDetector, LOFDetector
)
from anomaly_detection.deep_learning_detectors import (
    LSTMAutoencoderDetector, GRUDetector
)
from anomaly_detection.ensemble_detector import HybridEnsembleDetector


def generate_complex_data(n_samples: int = 500) -> tuple:
    """
    生成复杂测试数据
    
    包含：正常模式 + 多种异常类型
    """
    np.random.seed(42)
    t = np.arange(n_samples)
    
    # 基础信号：趋势 + 周期 + 噪声
    trend = 0.002 * t
    seasonal = 0.5 * np.sin(2 * np.pi * t / 50)
    noise = 0.1 * np.random.randn(n_samples)
    
    data = 3.0 + trend + seasonal + noise
    
    # 注入各类异常
    anomalies = []
    
    # 1. 突变异常 (T=100, 250, 400)
    data[100] = 6.0
    anomalies.append((100, 'spike', 'Sudden spike'))
    
    data[250] = 1.5
    anomalies.append((250, 'drop', 'Sudden drop'))
    
    data[400] = 5.5
    anomalies.append((400, 'spike', 'Sudden spike'))
    
    # 2. 漂移异常 (T=150-180)
    data[150:180] += np.linspace(0, 1.5, 30)
    for t in range(150, 180, 10):
        anomalies.append((t, 'drift', 'Gradual drift'))
    
    # 3. 模式变化 (T=300-330)
    data[300:330] = 3.0 + 0.3 * np.sin(2 * np.pi * np.arange(30) / 5)  # 高频振荡
    for t in range(300, 330, 10):
        anomalies.append((t, 'pattern_change', 'Pattern change'))
    
    return data, anomalies


def run_detection_comparison():
    """运行检测器对比"""
    print("="*80)
    print(" "*15 + "Phase 4.2 机器学习异常检测完整演示")
    print("="*80)
    
    # 生成数据
    print("\n生成测试数据...")
    data, true_anomalies = generate_complex_data()
    
    # 分割训练/测试
    train_size = 200
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"  训练数据: {train_size}个样本")
    print(f"  测试数据: {len(test_data)}个样本")
    print(f"  真实异常: {len([a for a in true_anomalies if a[0] >= train_size])}个")
    
    # 创建检测器
    print("\n初始化检测器...")
    print("-"*80)
    
    detectors = {
        '3-Sigma (统计)': ThreeSigmaDetector(),
        'CUSUM (统计)': CUSUMDetector(),
        'IsolationForest (ML)': IsolationForestDetector(['value'], window_size=50),
        'OneClassSVM (ML)': OneClassSVMDetector(['value'], window_size=50),
        'LOF (ML)': LOFDetector(['value'], window_size=50, n_neighbors=15),
        'LSTM-AE (DL)': LSTMAutoencoderDetector(sequence_length=10),
        'GRU (DL)': GRUDetector(sequence_length=10),
        '集成(Weighted)': HybridEnsembleDetector(fusion_method='weighted')
    }
    
    # 训练
    print("\n训练检测器...")
    print("="*80)
    
    for name, detector in detectors.items():
        if 'Isolation' in name or 'OneClass' in name or 'LOF' in name:
            # 多变量检测器
            detector.fit({'value': train_data})
        else:
            # 单变量检测器
            detector.fit(train_data)
        
        if name != '集成(Weighted)':
            print(f"  ✓ {name}")
    
    # 测试
    print("\n运行异常检测...")
    print("="*80)
    
    results = {name: [] for name in detectors.keys()}
    
    for t, value in enumerate(test_data):
        actual_t = t + train_size
        
        for name, detector in detectors.items():
            if 'Isolation' in name or 'OneClass' in name or 'LOF' in name:
                # 多变量检测器
                reports = detector.detect({'value': value}, actual_t)
                detected = len(reports) > 0
            else:
                # 单变量检测器
                report = detector.detect(value, actual_t, 'value')
                detected = report is not None
            
            results[name].append(1 if detected else 0)
    
    # 计算性能指标
    print("\n计算性能指标...")
    print("-"*80)
    
    performance = {}
    
    true_labels = np.zeros(len(test_data))
    for t_anomaly, _, _ in true_anomalies:
        if t_anomaly >= train_size:
            true_labels[t_anomaly - train_size] = 1
    
    for name, detections in results.items():
        pred_labels = np.array(detections)
        
        tp = np.sum((true_labels == 1) & (pred_labels == 1))
        fp = np.sum((true_labels == 0) & (pred_labels == 1))
        tn = np.sum((true_labels == 0) & (pred_labels == 0))
        fn = np.sum((true_labels == 1) & (pred_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(test_data)
        
        performance[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'fp': fp
        }
        
        print(f"\n{name}:")
        print(f"  精确率: {precision:.2%}")
        print(f"  召回率: {recall:.2%}")
        print(f"  F1分数: {f1:.2%}")
        print(f"  准确率: {accuracy:.2%}")
        print(f"  误报数: {fp}")
    
    # 可视化
    print("\n生成可视化...")
    visualize_results(data, test_data, train_size, results, true_anomalies, performance)
    
    print("\n✅ 演示完成！")
    print("="*80)
    
    return results, performance


def visualize_results(data, test_data, train_size, results, true_anomalies, performance):
    """可视化检测结果"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                 left=0.06, right=0.98, top=0.96, bottom=0.05)
    
    fig.suptitle('Phase 4.2 机器学习异常检测性能对比', 
                 fontsize=16, fontweight='bold')
    
    # 1. 原始数据 + 真实异常
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(data))
    ax1.plot(t, data, 'b-', linewidth=1, alpha=0.7, label='时间序列')
    ax1.axvline(train_size, color='g', linestyle='--', linewidth=2, label='训练/测试分割')
    
    # 标注异常
    for t_anomaly, atype, desc in true_anomalies:
        if t_anomaly >= train_size:
            ax1.axvline(t_anomaly, color='r', linestyle=':', alpha=0.5)
            if atype == 'spike':
                ax1.scatter(t_anomaly, data[t_anomaly], color='r', s=100, marker='^', 
                           zorder=5, label='真实异常' if t_anomaly == true_anomalies[0][0] else '')
    
    ax1.set_xlabel('时间步', fontsize=11, fontweight='bold')
    ax1.set_ylabel('值', fontsize=11, fontweight='bold')
    ax1.set_title('① 原始数据与真实异常', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2-7. 各检测器的检测结果
    detector_names = list(results.keys())[:6]
    
    for i, name in enumerate(detector_names):
        ax = fig.add_subplot(gs[(i // 3) + 1, i % 3])
        
        # 绘制测试数据
        test_t = np.arange(train_size, train_size + len(test_data))
        ax.plot(test_t, test_data, 'b-', linewidth=1, alpha=0.5)
        
        # 标注检测结果
        detections = np.array(results[name])
        detected_t = test_t[detections == 1]
        detected_values = test_data[detections == 1]
        
        ax.scatter(detected_t, detected_values, color='r', s=30, marker='x', 
                  label='检测到异常', zorder=5)
        
        # 标注真实异常
        for t_anomaly, _, _ in true_anomalies:
            if t_anomaly >= train_size:
                ax.axvline(t_anomaly, color='orange', linestyle=':', alpha=0.3)
        
        # 显示性能指标
        perf = performance[name]
        ax.text(0.02, 0.98, 
               f"F1={perf['f1']:.2f}\nP={perf['precision']:.2f}\nR={perf['recall']:.2f}",
               transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.set_title(f'{chr(9312+i+1)} {name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 8. 性能对比雷达图
    ax8 = fig.add_subplot(gs[2, 2], projection='polar')
    
    categories = ['Precision', 'Recall', 'F1', 'Accuracy']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # 绘制几个关键检测器
    key_detectors = ['3-Sigma (统计)', 'IsolationForest (ML)', 'LSTM-AE (DL)', '集成(Weighted)']
    colors = ['blue', 'green', 'orange', 'red']
    
    for name, color in zip(key_detectors, colors):
        if name in performance:
            perf = performance[name]
            values = [perf['precision'], perf['recall'], perf['f1'], perf['accuracy']]
            values += values[:1]
            ax8.plot(angles, values, 'o-', linewidth=2, label=name.split('(')[0].strip(), color=color)
            ax8.fill(angles, values, alpha=0.1, color=color)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, fontsize=9)
    ax8.set_ylim(0, 1)
    ax8.set_title('⑧ 性能对比雷达图', fontsize=11, fontweight='bold', pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax8.grid(True)
    
    plt.savefig('/workspace/phase4_ml_detection_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存可视化: /workspace/phase4_ml_detection_comparison.png")


if __name__ == "__main__":
    results, performance = run_detection_comparison()

"""
特征提取器
从时序数据中提取场景识别特征
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import deque
from dataclasses import dataclass


@dataclass
class TimeSeriesFeatures:
    """时序特征"""
    # 统计特征
    mean: float
    std: float
    min: float
    max: float
    range: float
    
    # 趋势特征
    trend: float          # 线性趋势斜率
    trend_strength: float # 趋势强度(R²)
    
    # 波动特征
    volatility: float     # 波动率
    cv: float             # 变异系数
    
    # 周期特征
    has_cycle: bool       # 是否有周期性
    cycle_period: int     # 周期长度
    
    # 变化特征
    change_rate: float    # 变化率
    acceleration: float   # 加速度


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, window_size: int = 24):
        """
        初始化
        
        Args:
            window_size: 时间窗口大小（小时）
        """
        self.window_size = window_size
        
        # 历史数据缓存
        self.level_history = deque(maxlen=window_size)
        self.flow_history = deque(maxlen=window_size)
        self.demand_history = deque(maxlen=window_size)
    
    def add_observation(self, levels: List[float], flows: List[float], demands: List[float]):
        """添加观测值"""
        self.level_history.append(np.mean(levels))
        self.flow_history.append(np.mean(flows))
        self.demand_history.append(np.mean(demands))
    
    def extract_features(self) -> Dict[str, TimeSeriesFeatures]:
        """
        提取所有特征
        
        Returns:
            {'level': features, 'flow': features, 'demand': features}
        """
        features = {}
        
        if len(self.level_history) >= 10:
            features['level'] = self._extract_ts_features(list(self.level_history))
        
        if len(self.flow_history) >= 10:
            features['flow'] = self._extract_ts_features(list(self.flow_history))
        
        if len(self.demand_history) >= 10:
            features['demand'] = self._extract_ts_features(list(self.demand_history))
        
        return features
    
    def _extract_ts_features(self, data: List[float]) -> TimeSeriesFeatures:
        """提取单个时间序列的特征"""
        arr = np.array(data)
        n = len(arr)
        
        # 统计特征
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        range_val = max_val - min_val
        
        # 趋势特征
        x = np.arange(n)
        trend_coef = np.polyfit(x, arr, 1)[0]  # 斜率
        
        # 计算R²
        y_pred = np.poly1d(np.polyfit(x, arr, 1))(x)
        ss_res = np.sum((arr - y_pred)**2)
        ss_tot = np.sum((arr - mean_val)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 波动特征
        volatility = float(np.std(np.diff(arr))) if n > 1 else 0.0
        cv = std_val / mean_val if mean_val > 0 else 0.0
        
        # 周期检测（简化版）
        has_cycle, cycle_period = self._detect_cycle(arr)
        
        # 变化率
        if n >= 2:
            change_rate = float((arr[-1] - arr[0]) / n)
        else:
            change_rate = 0.0
        
        # 加速度
        if n >= 3:
            diff1 = np.diff(arr)
            diff2 = np.diff(diff1)
            acceleration = float(np.mean(diff2))
        else:
            acceleration = 0.0
        
        return TimeSeriesFeatures(
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            range=range_val,
            trend=float(trend_coef),
            trend_strength=float(r_squared),
            volatility=volatility,
            cv=cv,
            has_cycle=has_cycle,
            cycle_period=cycle_period,
            change_rate=change_rate,
            acceleration=acceleration
        )
    
    def _detect_cycle(self, data: np.ndarray) -> Tuple[bool, int]:
        """
        检测周期性
        
        Returns:
            (是否有周期, 周期长度)
        """
        n = len(data)
        
        if n < 24:
            return False, 0
        
        # 简化：检查24小时周期（日周期）
        if n >= 24:
            # 计算自相关
            acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]  # 归一化
            
            # 检查24小时处的自相关
            if len(acf) >= 24:
                if acf[24] > 0.5:  # 相关性阈值
                    return True, 24
        
        return False, 0
    
    def get_scenario_indicators(self) -> Dict[str, float]:
        """
        获取场景指示器
        
        Returns:
            各种场景的可能性指标
        """
        if len(self.demand_history) < 10:
            return {}
        
        features = self.extract_features()
        indicators = {}
        
        # 高峰指标
        if 'demand' in features:
            demand_feat = features['demand']
            indicators['peak_likelihood'] = min(1.0, demand_feat.mean / 10.0)
            
            # 波动指标
            indicators['volatility_level'] = min(1.0, demand_feat.volatility / 2.0)
            
            # 趋势指标
            if demand_feat.trend > 0.1:
                indicators['rising_trend'] = min(1.0, abs(demand_feat.trend) / 0.5)
            elif demand_feat.trend < -0.1:
                indicators['falling_trend'] = min(1.0, abs(demand_feat.trend) / 0.5)
        
        # 稳定性指标
        if 'level' in features:
            level_feat = features['level']
            indicators['stability'] = 1.0 - min(1.0, level_feat.cv)
            
            # 异常指标
            if level_feat.range > 2.0:
                indicators['level_anomaly'] = min(1.0, level_feat.range / 4.0)
        
        # 流量异常指标
        if 'flow' in features:
            flow_feat = features['flow']
            if flow_feat.volatility > 2.0:
                indicators['flow_anomaly'] = min(1.0, flow_feat.volatility / 4.0)
        
        return indicators


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*25 + "特征提取器演示")
    print("="*70)
    
    # 创建提取器
    extractor = FeatureExtractor(window_size=24)
    
    # 模拟数据：日周期需求
    print("\n生成模拟数据（24小时日周期）...")
    for t in range(36):
        hour = t % 24
        
        # 模拟日周期需求
        if 8 <= hour <= 10 or 18 <= hour <= 20:
            demand = 10.0 + np.random.normal(0, 0.5)  # 高峰
        elif 0 <= hour <= 5:
            demand = 3.0 + np.random.normal(0, 0.3)   # 低谷
        else:
            demand = 6.0 + np.random.normal(0, 0.4)   # 正常
        
        level = 3.0 + np.random.normal(0, 0.2)
        flow = demand + np.random.normal(0, 0.3)
        
        extractor.add_observation([level], [flow], [demand])
    
    # 提取特征
    print("\n提取特征...")
    features = extractor.extract_features()
    
    for name, feat in features.items():
        print(f"\n{name.upper()} 特征:")
        print(f"  均值: {feat.mean:.2f}")
        print(f"  标准差: {feat.std:.2f}")
        print(f"  范围: [{feat.min:.2f}, {feat.max:.2f}]")
        print(f"  趋势斜率: {feat.trend:.4f}")
        print(f"  趋势强度(R²): {feat.trend_strength:.4f}")
        print(f"  波动率: {feat.volatility:.4f}")
        print(f"  变异系数: {feat.cv:.4f}")
        print(f"  周期性: {'是' if feat.has_cycle else '否'} " +
              (f"(周期={feat.cycle_period}h)" if feat.has_cycle else ""))
    
    # 场景指标
    print("\n场景指标:")
    indicators = extractor.get_scenario_indicators()
    for key, value in indicators.items():
        print(f"  {key}: {value:.2%}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

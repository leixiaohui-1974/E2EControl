"""
场景识别器
整合规则引擎和特征提取的综合识别系统
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from scenario_types import ScenarioDefinition, SCENARIO_LIBRARY
from rule_engine import RuleEngine, SystemState
from feature_extractor import FeatureExtractor


@dataclass
class RecognitionResult:
    """识别结果"""
    scenario_id: str
    scenario_name: str
    confidence: float
    method: str  # 'rule', 'feature', 'hybrid'
    
    # 详细信息
    category: str
    priority: str
    description: str
    
    # 建议
    recommended_actions: List[str] = None
    
    def __post_init__(self):
        if self.recommended_actions is None:
            self.recommended_actions = []


class ScenarioRecognizer:
    """场景识别器（混合方法）"""
    
    def __init__(self, window_size: int = 24):
        """
        初始化识别器
        
        Args:
            window_size: 特征提取窗口大小
        """
        self.rule_engine = RuleEngine()
        self.feature_extractor = FeatureExtractor(window_size)
        
        # 识别历史
        self.recognition_history = deque(maxlen=100)
        
        # 权重配置
        self.rule_weight = 0.7
        self.feature_weight = 0.3
    
    def recognize(self, state: SystemState) -> RecognitionResult:
        """
        识别当前场景（综合方法）
        
        Args:
            state: 系统状态
            
        Returns:
            识别结果
        """
        # 1. 基于规则的识别
        rule_matches = self.rule_engine.recognize(state)
        
        # 2. 更新特征提取器
        self.feature_extractor.add_observation(
            state.levels, state.flows, state.demands
        )
        
        # 3. 基于特征的识别
        feature_indicators = self.feature_extractor.get_scenario_indicators()
        feature_matches = self._feature_based_recognition(feature_indicators, state)
        
        # 4. 融合结果
        final_scenario_id, final_confidence, method = self._fuse_results(
            rule_matches, feature_matches
        )
        
        # 5. 获取场景定义
        scenario = SCENARIO_LIBRARY.get(final_scenario_id)
        if not scenario:
            scenario = SCENARIO_LIBRARY['daily_operation']
            final_scenario_id = 'daily_operation'
        
        # 6. 生成建议
        actions = self._generate_recommendations(scenario, state)
        
        # 7. 创建结果
        result = RecognitionResult(
            scenario_id=final_scenario_id,
            scenario_name=scenario.name,
            confidence=final_confidence,
            method=method,
            category=scenario.category.value,
            priority=scenario.control_priority,
            description=scenario.description,
            recommended_actions=actions
        )
        
        # 8. 记录历史
        self.recognition_history.append(result)
        
        return result
    
    def _feature_based_recognition(self, indicators: Dict[str, float], 
                                   state: SystemState) -> List[Tuple[str, float]]:
        """基于特征指标的识别"""
        matches = []
        
        # 根据指标判断场景
        if 'peak_likelihood' in indicators and indicators['peak_likelihood'] > 0.7:
            matches.append(('peak_demand', indicators['peak_likelihood']))
        
        if 'falling_trend' in indicators and indicators['falling_trend'] > 0.7:
            matches.append(('off_peak', indicators['falling_trend']))
        
        if 'volatility_level' in indicators and indicators['volatility_level'] > 0.8:
            # 高波动可能是异常
            if 'level_anomaly' in indicators or 'flow_anomaly' in indicators:
                matches.append(('equipment_failure', 0.7))
        
        if 'level_anomaly' in indicators and indicators['level_anomaly'] > 0.7:
            # 水位异常，检查是洪水还是干旱
            if np.mean(state.levels) > 5.0:
                matches.append(('flood_peak', indicators['level_anomaly']))
            elif np.mean(state.levels) < 2.0:
                matches.append(('water_shortage', indicators['level_anomaly']))
        
        if 'stability' in indicators and indicators['stability'] > 0.8:
            matches.append(('daily_operation', indicators['stability'] * 0.6))
        
        # 按置信度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def _fuse_results(self, rule_matches: List[Tuple[str, float]],
                     feature_matches: List[Tuple[str, float]]) -> Tuple[str, float, str]:
        """
        融合规则和特征识别结果
        
        Returns:
            (scenario_id, confidence, method)
        """
        # 转换为字典
        rule_dict = {sid: conf for sid, conf in rule_matches}
        feature_dict = {sid: conf for sid, conf in feature_matches}
        
        # 收集所有候选场景
        all_scenarios = set(rule_dict.keys()) | set(feature_dict.keys())
        
        # 计算融合分数
        fused_scores = {}
        for sid in all_scenarios:
            rule_score = rule_dict.get(sid, 0.0)
            feature_score = feature_dict.get(sid, 0.0)
            
            # 加权融合
            fused_score = (self.rule_weight * rule_score + 
                          self.feature_weight * feature_score)
            
            fused_scores[sid] = fused_score
        
        # 选择最高分
        if fused_scores:
            best_scenario = max(fused_scores.items(), key=lambda x: x[1])
            scenario_id, confidence = best_scenario
            
            # 确定方法
            if scenario_id in rule_dict and scenario_id in feature_dict:
                method = 'hybrid'
            elif scenario_id in rule_dict:
                method = 'rule'
            else:
                method = 'feature'
            
            return scenario_id, confidence, method
        else:
            # 默认
            return 'daily_operation', 0.5, 'default'
    
    def _generate_recommendations(self, scenario: ScenarioDefinition,
                                  state: SystemState) -> List[str]:
        """生成控制建议"""
        actions = []
        
        # 根据场景类型给出建议
        if scenario.control_priority == 'critical':
            actions.append("⚠️ 紧急情况，启用应急预案")
            actions.append("📞 通知相关人员")
        
        if scenario.category.value == 'flood_control':
            actions.append("🌊 增大泄洪流量")
            actions.append("📊 密切监控水位变化")
            if np.mean(state.levels) > 6.0:
                actions.append("🚨 水位过高，考虑启用备用泄洪通道")
        
        elif scenario.category.value == 'drought':
            actions.append("💧 启动节水措施")
            actions.append("📋 执行分级供水计划")
            actions.append("🔍 寻找备用水源")
        
        elif scenario.category.value == 'ice_period':
            actions.append("❄️ 减缓流量调整速度")
            actions.append("🌡️ 监控水温变化")
            actions.append("⚠️ 防止冰塞形成")
        
        elif scenario.category.value == 'emergency':
            actions.append("🛑 启动应急响应")
            if "pollution" in scenario.sub_scenario.value:
                actions.append("💦 隔离污染水源")
                actions.append("🧪 加强水质监测")
            elif "failure" in scenario.sub_scenario.value:
                actions.append("🔧 启用备用设备")
                actions.append("📞 联系维修人员")
        
        elif scenario.category.value == 'normal':
            if scenario.sub_scenario.value == 'peak_demand':
                actions.append("📈 增加供水流量")
                actions.append("🔄 调整各池协同策略")
            elif scenario.sub_scenario.value == 'off_peak':
                actions.append("📉 降低流量，节约能源")
                actions.append("🔧 适合进行维护检查")
        
        # 通用建议
        if scenario.safety_margin > 0.8:
            actions.append(f"🛡️ 保持{scenario.safety_margin}m安全裕度")
        
        return actions
    
    def get_scenario_trend(self, window: int = 5) -> Dict:
        """
        分析场景变化趋势
        
        Args:
            window: 分析窗口
            
        Returns:
            趋势信息
        """
        if len(self.recognition_history) < window:
            return {'status': 'insufficient_data'}
        
        recent = list(self.recognition_history)[-window:]
        
        # 统计场景类别
        categories = [r.category for r in recent]
        unique_categories = list(set(categories))
        
        # 检查是否稳定
        if len(unique_categories) == 1:
            status = 'stable'
        elif len(unique_categories) <= 2:
            status = 'transitioning'
        else:
            status = 'volatile'
        
        # 计算平均置信度
        avg_confidence = np.mean([r.confidence for r in recent])
        
        # 最常见场景
        from collections import Counter
        scenario_counts = Counter([r.scenario_id for r in recent])
        most_common = scenario_counts.most_common(1)[0]
        
        return {
            'status': status,
            'avg_confidence': avg_confidence,
            'most_common_scenario': most_common[0],
            'scenario_frequency': most_common[1] / window,
            'unique_scenarios': len(set([r.scenario_id for r in recent]))
        }


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "场景识别器综合演示")
    print("="*70)
    
    # 创建识别器
    recognizer = ScenarioRecognizer(window_size=24)
    
    # 测试场景序列
    scenarios_to_test = [
        {
            'name': '正常运行',
            'state': SystemState(
                time=10,
                levels=[3.0, 3.0, 3.0],
                flows=[5.0, 5.0, 5.0],
                demands=[5.0, 5.0, 5.0]
            )
        },
        {
            'name': '高峰需求',
            'state': SystemState(
                time=18,  # 傍晚
                levels=[2.8, 2.9, 3.0],
                flows=[12.0, 11.0, 10.0],
                demands=[12.0, 11.5, 11.0]
            )
        },
        {
            'name': '洪峰来临',
            'state': SystemState(
                time=100,
                levels=[6.5, 6.0, 5.8],
                flows=[20.0, 18.0, 17.0],
                demands=[5.0, 5.0, 5.0],
                weather={'rainfall': 20.0}
            )
        },
        {
            'name': '设备故障',
            'state': SystemState(
                time=200,
                levels=[2.0, 3.0, 2.5],
                flows=[5.0, 5.0, 5.0],
                demands=[6.0, 6.0, 6.0],
                alerts=["闸门1响应异常", "控制失效"]
            )
        },
    ]
    
    # 逐个测试
    for i, test in enumerate(scenarios_to_test, 1):
        print(f"\n{'='*70}")
        print(f"测试场景 {i}: {test['name']}")
        print('='*70)
        
        result = recognizer.recognize(test['state'])
        
        print(f"\n识别结果:")
        print(f"  场景: {result.scenario_name}")
        print(f"  置信度: {result.confidence:.2%}")
        print(f"  识别方法: {result.method}")
        print(f"  类别: {result.category}")
        print(f"  优先级: {result.priority}")
        print(f"  描述: {result.description}")
        
        if result.recommended_actions:
            print(f"\n建议措施:")
            for action in result.recommended_actions:
                print(f"    {action}")
    
    # 分析趋势
    print(f"\n{'='*70}")
    print("场景变化趋势分析")
    print('='*70)
    
    trend = recognizer.get_scenario_trend(window=4)
    print(f"  状态: {trend['status']}")
    print(f"  平均置信度: {trend['avg_confidence']:.2%}")
    print(f"  最常见场景: {trend['most_common_scenario']}")
    print(f"  出现频率: {trend['scenario_frequency']:.1%}")
    print(f"  场景变化: {trend['unique_scenarios']}种不同场景")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

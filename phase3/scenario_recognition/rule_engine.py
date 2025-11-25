"""
规则引擎
基于规则的场景识别
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from scenario_types import ScenarioDefinition, SCENARIO_LIBRARY, ScenarioCategory


@dataclass
class SystemState:
    """系统状态"""
    time: int
    levels: List[float]          # 各池水位
    flows: List[float]           # 各闸门流量
    demands: List[float]         # 需求
    weather: Dict = None         # 天气信息
    alerts: List[str] = None     # 告警信息
    
    def __post_init__(self):
        if self.weather is None:
            self.weather = {}
        if self.alerts is None:
            self.alerts = []


@dataclass
class Rule:
    """识别规则"""
    name: str
    scenario_id: str
    conditions: List[callable]
    confidence: float = 0.8
    priority: int = 1  # 优先级，数字越大越优先


class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        """初始化规则引擎"""
        self.rules: List[Rule] = []
        self.state_history = deque(maxlen=100)
        self._build_rules()
    
    def _build_rules(self):
        """构建规则库"""
        
        # 规则1: 高峰需求
        self.rules.append(Rule(
            name="高峰需求识别",
            scenario_id="peak_demand",
            conditions=[
                lambda state: np.mean(state.demands) > 8.0,  # 需求大
                lambda state: self._is_peak_hour(state.time),  # 高峰时段
            ],
            confidence=0.9,
            priority=3
        ))
        
        # 规则2: 低谷期
        self.rules.append(Rule(
            name="低谷期识别",
            scenario_id="off_peak",
            conditions=[
                lambda state: np.mean(state.demands) < 3.0,  # 需求小
                lambda state: self._is_night(state.time),  # 夜间
            ],
            confidence=0.9,
            priority=3
        ))
        
        # 规则3: 洪峰期
        self.rules.append(Rule(
            name="洪峰期识别",
            scenario_id="flood_peak",
            conditions=[
                lambda state: np.mean(state.levels) > 5.0,  # 水位很高
                lambda state: np.mean(state.flows) > 15.0,  # 流量很大
                lambda state: self._has_rainfall(state),  # 有降雨
            ],
            confidence=0.95,
            priority=5  # 高优先级
        ))
        
        # 规则4: 洪前准备
        self.rules.append(Rule(
            name="洪前准备识别",
            scenario_id="pre_flood",
            conditions=[
                lambda state: self._has_rainfall_forecast(state),  # 有降雨预报
                lambda state: np.mean(state.levels) > 2.0,  # 水位偏高
            ],
            confidence=0.85,
            priority=4
        ))
        
        # 规则5: 水源短缺
        self.rules.append(Rule(
            name="水源短缺识别",
            scenario_id="water_shortage",
            conditions=[
                lambda state: np.mean(state.levels) < 2.0,  # 水位低
                lambda state: np.mean(state.demands) > np.mean(state.flows),  # 供不应求
                lambda state: self._is_low_water_season(state.time),  # 枯水期
            ],
            confidence=0.85,
            priority=4
        ))
        
        # 规则6: 结冰期
        self.rules.append(Rule(
            name="结冰期识别",
            scenario_id="freezing",
            conditions=[
                lambda state: self._is_winter(state.time),  # 冬季
                lambda state: self._has_low_temperature(state),  # 低温
                lambda state: self._flow_change_slow(state),  # 流量变化慢
            ],
            confidence=0.80,
            priority=3
        ))
        
        # 规则7: 融冰期
        self.rules.append(Rule(
            name="融冰期识别",
            scenario_id="thawing",
            conditions=[
                lambda state: self._is_spring(state.time),  # 春季
                lambda state: self._temperature_rising(state),  # 气温回升
            ],
            confidence=0.80,
            priority=3
        ))
        
        # 规则8: 设备故障
        self.rules.append(Rule(
            name="设备故障识别",
            scenario_id="equipment_failure",
            conditions=[
                lambda state: "故障" in state.alerts or "异常" in state.alerts,
                lambda state: self._has_abnormal_response(state),
            ],
            confidence=0.95,
            priority=5
        ))
        
        # 规则9: 污染检测
        self.rules.append(Rule(
            name="污染检测识别",
            scenario_id="pollution_detected",
            conditions=[
                lambda state: "污染" in state.alerts or "水质" in state.alerts,
            ],
            confidence=0.98,
            priority=5
        ))
        
        # 规则10: 管道爆裂
        self.rules.append(Rule(
            name="管道爆裂识别",
            scenario_id="pipe_burst",
            conditions=[
                lambda state: "泄漏" in state.alerts or "爆裂" in state.alerts,
                lambda state: np.mean(state.levels) < 1.5 and np.mean(state.flows) > 10.0,
            ],
            confidence=0.95,
            priority=5
        ))
        
        # 规则11: 周末模式
        self.rules.append(Rule(
            name="周末模式识别",
            scenario_id="weekend_mode",
            conditions=[
                lambda state: self._is_weekend(state.time),
                lambda state: 3.0 < np.mean(state.demands) < 8.0,
            ],
            confidence=0.85,
            priority=2
        ))
        
        # 规则12: 日常运行（默认）
        self.rules.append(Rule(
            name="日常运行识别",
            scenario_id="daily_operation",
            conditions=[
                lambda state: 4.0 < np.mean(state.demands) < 6.0,
                lambda state: 2.5 < np.mean(state.levels) < 3.5,
            ],
            confidence=0.7,
            priority=1  # 最低优先级
        ))
    
    def recognize(self, state: SystemState) -> List[Tuple[str, float]]:
        """
        识别当前场景
        
        Args:
            state: 系统状态
            
        Returns:
            [(scenario_id, confidence), ...] 按confidence降序排列
        """
        # 记录历史
        self.state_history.append(state)
        
        # 评估所有规则
        matched_scenarios = []
        
        for rule in self.rules:
            try:
                # 检查所有条件
                all_matched = all(cond(state) for cond in rule.conditions)
                
                if all_matched:
                    # 计算置信度（考虑优先级）
                    confidence = rule.confidence * (1.0 + 0.1 * rule.priority)
                    confidence = min(1.0, confidence)  # 限制在[0,1]
                    
                    matched_scenarios.append((rule.scenario_id, confidence))
            except Exception as e:
                # 规则评估失败，跳过
                pass
        
        # 按置信度排序
        matched_scenarios.sort(key=lambda x: x[1], reverse=True)
        
        return matched_scenarios
    
    def get_best_scenario(self, state: SystemState) -> Tuple[str, float]:
        """
        获取最佳匹配场景
        
        Returns:
            (scenario_id, confidence)
        """
        matches = self.recognize(state)
        
        if matches:
            return matches[0]
        else:
            # 默认返回日常运行
            return ("daily_operation", 0.5)
    
    # 辅助判断函数
    
    def _is_peak_hour(self, time: int) -> bool:
        """是否高峰时段"""
        hour = time % 24
        return (7 <= hour <= 9) or (17 <= hour <= 20)
    
    def _is_night(self, time: int) -> bool:
        """是否夜间"""
        hour = time % 24
        return 0 <= hour <= 5 or hour >= 22
    
    def _is_weekend(self, time: int) -> bool:
        """是否周末"""
        # 简化：假设每7天一个周期，最后2天是周末
        day = (time // 24) % 7
        return day >= 5
    
    def _is_winter(self, time: int) -> bool:
        """是否冬季"""
        # 简化：假设一年360天，11-2月是冬季
        day_of_year = (time // 24) % 360
        return day_of_year < 60 or day_of_year >= 300
    
    def _is_spring(self, time: int) -> bool:
        """是否春季"""
        day_of_year = (time // 24) % 360
        return 60 <= day_of_year < 150
    
    def _is_low_water_season(self, time: int) -> bool:
        """是否枯水期"""
        # 简化：1-3月是枯水期
        day_of_year = (time // 24) % 360
        return day_of_year < 90
    
    def _has_rainfall(self, state: SystemState) -> bool:
        """是否有降雨"""
        if state.weather and 'rainfall' in state.weather:
            return state.weather['rainfall'] > 10.0  # mm/h
        return False
    
    def _has_rainfall_forecast(self, state: SystemState) -> bool:
        """是否有降雨预报"""
        if state.weather and 'rainfall_forecast' in state.weather:
            forecast = state.weather['rainfall_forecast']
            return np.sum(forecast) > 50.0  # 未来降雨量
        return False
    
    def _has_low_temperature(self, state: SystemState) -> bool:
        """是否低温"""
        if state.weather and 'temperature' in state.weather:
            return state.weather['temperature'] < 0.0  # 摄氏度
        return False
    
    def _temperature_rising(self, state: SystemState) -> bool:
        """气温是否回升"""
        if len(self.state_history) < 5:
            return False
        
        recent_temps = []
        for s in list(self.state_history)[-5:]:
            if s.weather and 'temperature' in s.weather:
                recent_temps.append(s.weather['temperature'])
        
        if len(recent_temps) >= 3:
            # 检查是否有上升趋势
            return recent_temps[-1] > recent_temps[0]
        return False
    
    def _flow_change_slow(self, state: SystemState) -> bool:
        """流量变化是否缓慢"""
        if len(self.state_history) < 3:
            return False
        
        recent_flows = [s.flows for s in list(self.state_history)[-3:]]
        
        # 计算流量变化率
        changes = []
        for i in range(1, len(recent_flows)):
            avg_change = np.mean(np.abs(np.array(recent_flows[i]) - np.array(recent_flows[i-1])))
            changes.append(avg_change)
        
        return np.mean(changes) < 0.5  # 变化小于0.5
    
    def _has_abnormal_response(self, state: SystemState) -> bool:
        """是否有异常响应"""
        # 检查水位-流量关系是否异常
        if len(state.levels) == 0 or len(state.flows) == 0:
            return False
        
        # 简化判断：如果流量很大但水位没增加，可能有问题
        if np.mean(state.flows) > 10.0 and np.mean(state.levels) < 2.0:
            return True
        
        return False


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*25 + "规则引擎演示")
    print("="*70)
    
    # 创建规则引擎
    engine = RuleEngine()
    print(f"\n规则库: {len(engine.rules)}条规则")
    
    # 测试场景1: 高峰需求
    print("\n" + "-"*70)
    print("场景1: 高峰需求")
    print("-"*70)
    
    state1 = SystemState(
        time=8,  # 8点
        levels=[3.0, 3.0, 3.0],
        flows=[10.0, 10.0, 10.0],
        demands=[12.0, 11.0, 10.0],
        weather={}
    )
    
    matches = engine.recognize(state1)
    print(f"识别结果: {len(matches)}个匹配")
    for scenario_id, conf in matches[:3]:
        scenario = SCENARIO_LIBRARY.get(scenario_id)
        if scenario:
            print(f"  - {scenario.name}: {conf:.2%}")
    
    # 测试场景2: 洪峰期
    print("\n" + "-"*70)
    print("场景2: 洪峰期")
    print("-"*70)
    
    state2 = SystemState(
        time=100,
        levels=[6.0, 5.5, 5.8],
        flows=[18.0, 17.0, 16.0],
        demands=[5.0, 5.0, 5.0],
        weather={'rainfall': 15.0}
    )
    
    matches = engine.recognize(state2)
    print(f"识别结果: {len(matches)}个匹配")
    for scenario_id, conf in matches[:3]:
        scenario = SCENARIO_LIBRARY.get(scenario_id)
        if scenario:
            print(f"  - {scenario.name}: {conf:.2%}")
    
    # 测试场景3: 设备故障
    print("\n" + "-"*70)
    print("场景3: 设备故障")
    print("-"*70)
    
    state3 = SystemState(
        time=200,
        levels=[2.0, 3.0, 2.5],
        flows=[5.0, 5.0, 5.0],
        demands=[5.0, 5.0, 5.0],
        alerts=["闸门1故障", "响应异常"]
    )
    
    best_scenario, conf = engine.get_best_scenario(state3)
    scenario = SCENARIO_LIBRARY.get(best_scenario)
    if scenario:
        print(f"最佳匹配: {scenario.name} (置信度: {conf:.2%})")
        print(f"描述: {scenario.description}")
        print(f"优先级: {scenario.control_priority}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

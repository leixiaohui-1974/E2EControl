"""
简单知识库
存储场景-策略-经验的知识
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ControlStrategy:
    """控制策略"""
    strategy_id: str
    name: str
    scenario_id: str
    
    # MPC参数
    horizon: int = 10
    weights: Dict[str, float] = None
    
    # 约束调整
    max_level_deviation: float = 0.3
    max_flow_change_rate: float = 2.0
    safety_margin: float = 0.5
    
    # 其他配置
    update_frequency: float = 3600.0  # 秒
    use_feedforward: bool = False
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'level_tracking': 10.0,
                'flow_smoothness': 5.0,
                'energy_cost': 0.3
            }


@dataclass
class CaseRecord:
    """案例记录"""
    case_id: str
    scenario_id: str
    timestamp: int
    
    # 状态
    initial_state: Dict
    
    # 采取的措施
    strategy_used: str
    actions_taken: List[str]
    
    # 结果
    outcome: str  # 'success', 'partial', 'failed'
    performance: Dict  # 性能指标
    
    # 经验教训
    lessons_learned: List[str] = None
    
    def __post_init__(self):
        if self.lessons_learned is None:
            self.lessons_learned = []


class SimpleKnowledgeBase:
    """简单知识库"""
    
    def __init__(self):
        """初始化知识库"""
        self.strategies: Dict[str, ControlStrategy] = {}
        self.cases: Dict[str, CaseRecord] = {}
        self.scenario_strategy_map: Dict[str, List[str]] = defaultdict(list)
        
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """初始化默认策略"""
        
        # 策略1: 日常运行
        self.add_strategy(ControlStrategy(
            strategy_id="daily_normal",
            name="日常正常运行策略",
            scenario_id="daily_operation",
            horizon=10,
            weights={
                'level_tracking': 10.0,
                'flow_smoothness': 5.0,
                'energy_cost': 0.3,
                'water_delivery': 2.0
            },
            max_level_deviation=0.3,
            max_flow_change_rate=2.0,
            safety_margin=0.5
        ))
        
        # 策略2: 高峰需求
        self.add_strategy(ControlStrategy(
            strategy_id="peak_response",
            name="高峰快速响应策略",
            scenario_id="peak_demand",
            horizon=8,
            weights={
                'level_tracking': 8.0,
                'flow_smoothness': 3.0,
                'energy_cost': 0.1,
                'water_delivery': 15.0  # 重视供水
            },
            max_level_deviation=0.5,
            max_flow_change_rate=3.0,
            safety_margin=0.3,
            use_feedforward=True
        ))
        
        # 策略3: 低谷期
        self.add_strategy(ControlStrategy(
            strategy_id="off_peak_efficient",
            name="低谷节能策略",
            scenario_id="off_peak",
            horizon=12,
            weights={
                'level_tracking': 10.0,
                'flow_smoothness': 8.0,
                'energy_cost': 2.0,  # 重视节能
                'water_delivery': 1.0
            },
            max_level_deviation=0.2,
            max_flow_change_rate=1.0,
            safety_margin=0.6
        ))
        
        # 策略4: 防洪泄洪
        self.add_strategy(ControlStrategy(
            strategy_id="flood_control",
            name="防洪泄洪策略",
            scenario_id="flood_peak",
            horizon=5,  # 短时域快速响应
            weights={
                'level_tracking': 20.0,  # 最重视水位
                'flow_smoothness': 1.0,
                'energy_cost': 0.0,
                'water_delivery': 0.5
            },
            max_level_deviation=0.8,
            max_flow_change_rate=5.0,
            safety_margin=0.2,
            update_frequency=1800.0  # 更频繁更新
        ))
        
        # 策略5: 干旱限水
        self.add_strategy(ControlStrategy(
            strategy_id="drought_conservation",
            name="干旱节水策略",
            scenario_id="water_shortage",
            horizon=15,  # 长时域规划
            weights={
                'level_tracking': 12.0,
                'flow_smoothness': 6.0,
                'energy_cost': 1.0,
                'water_delivery': 10.0
            },
            max_level_deviation=0.4,
            max_flow_change_rate=1.5,
            safety_margin=0.8
        ))
        
        # 策略6: 冰期稳定
        self.add_strategy(ControlStrategy(
            strategy_id="ice_stable",
            name="冰期稳定运行策略",
            scenario_id="stable_ice",
            horizon=12,
            weights={
                'level_tracking': 10.0,
                'flow_smoothness': 10.0,  # 重视平稳
                'energy_cost': 0.5,
                'water_delivery': 2.0
            },
            max_level_deviation=0.2,
            max_flow_change_rate=0.5,  # 变化要慢
            safety_margin=0.6
        ))
        
        # 策略7: 应急响应
        self.add_strategy(ControlStrategy(
            strategy_id="emergency_response",
            name="应急快速响应策略",
            scenario_id="equipment_failure",
            horizon=5,
            weights={
                'level_tracking': 15.0,
                'flow_smoothness': 2.0,
                'energy_cost': 0.0,
                'water_delivery': 8.0
            },
            max_level_deviation=0.6,
            max_flow_change_rate=4.0,
            safety_margin=0.4,
            update_frequency=600.0  # 10分钟更新一次
        ))
    
    def add_strategy(self, strategy: ControlStrategy):
        """添加策略"""
        self.strategies[strategy.strategy_id] = strategy
        self.scenario_strategy_map[strategy.scenario_id].append(strategy.strategy_id)
    
    def get_strategy(self, strategy_id: str) -> Optional[ControlStrategy]:
        """获取策略"""
        return self.strategies.get(strategy_id)
    
    def get_strategies_for_scenario(self, scenario_id: str) -> List[ControlStrategy]:
        """获取某场景的所有策略"""
        strategy_ids = self.scenario_strategy_map.get(scenario_id, [])
        return [self.strategies[sid] for sid in strategy_ids if sid in self.strategies]
    
    def add_case(self, case: CaseRecord):
        """添加案例"""
        self.cases[case.case_id] = case
    
    def find_similar_cases(self, scenario_id: str, limit: int = 5) -> List[CaseRecord]:
        """查找相似案例"""
        similar = [c for c in self.cases.values() if c.scenario_id == scenario_id]
        
        # 按成功率排序
        similar.sort(key=lambda c: 1 if c.outcome == 'success' else 
                                   0.5 if c.outcome == 'partial' else 0, 
                    reverse=True)
        
        return similar[:limit]
    
    def get_best_practices(self, scenario_id: str) -> Dict:
        """获取最佳实践"""
        strategies = self.get_strategies_for_scenario(scenario_id)
        cases = self.find_similar_cases(scenario_id)
        
        # 统计成功案例使用的策略
        strategy_success_count = defaultdict(int)
        strategy_total_count = defaultdict(int)
        
        for case in cases:
            strategy_total_count[case.strategy_used] += 1
            if case.outcome == 'success':
                strategy_success_count[case.strategy_used] += 1
        
        # 计算成功率
        strategy_success_rate = {}
        for sid, total in strategy_total_count.items():
            success = strategy_success_count[sid]
            strategy_success_rate[sid] = success / total if total > 0 else 0
        
        # 推荐策略
        if strategy_success_rate:
            recommended = max(strategy_success_rate.items(), key=lambda x: x[1])
            recommended_strategy_id = recommended[0]
        elif strategies:
            recommended_strategy_id = strategies[0].strategy_id
        else:
            recommended_strategy_id = "daily_normal"
        
        # 收集经验教训
        all_lessons = []
        for case in cases:
            if case.outcome == 'success':
                all_lessons.extend(case.lessons_learned)
        
        return {
            'recommended_strategy': recommended_strategy_id,
            'success_rate': strategy_success_rate.get(recommended_strategy_id, 0),
            'total_cases': len(cases),
            'lessons_learned': list(set(all_lessons))[:5]  # 去重，取前5条
        }
    
    def save_to_file(self, filepath: str):
        """保存知识库到文件"""
        data = {
            'strategies': {sid: asdict(s) for sid, s in self.strategies.items()},
            'cases': {cid: asdict(c) for cid, c in self.cases.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str):
        """从文件加载知识库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载策略
        for sid, s_dict in data['strategies'].items():
            strategy = ControlStrategy(**s_dict)
            self.add_strategy(strategy)
        
        # 加载案例
        for cid, c_dict in data['cases'].items():
            case = CaseRecord(**c_dict)
            self.add_case(case)


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*25 + "知识库演示")
    print("="*70)
    
    # 创建知识库
    kb = SimpleKnowledgeBase()
    
    print(f"\n默认策略数: {len(kb.strategies)}")
    
    # 查看策略
    print("\n策略列表:")
    for sid, strategy in kb.strategies.items():
        print(f"  {strategy.name} ({sid})")
        print(f"    场景: {strategy.scenario_id}")
        print(f"    时域: {strategy.horizon}")
        print(f"    权重: {strategy.weights}")
    
    # 查询特定场景的策略
    print("\n" + "-"*70)
    print("查询'高峰需求'场景的策略")
    print("-"*70)
    
    peak_strategies = kb.get_strategies_for_scenario("peak_demand")
    for strategy in peak_strategies:
        print(f"\n策略: {strategy.name}")
        print(f"  时域: {strategy.horizon}步")
        print(f"  供水权重: {strategy.weights['water_delivery']}")
        print(f"  最大流量变化率: {strategy.max_flow_change_rate} m³/s/h")
        print(f"  前馈补偿: {'启用' if strategy.use_feedforward else '禁用'}")
    
    # 添加案例
    print("\n" + "-"*70)
    print("添加历史案例")
    print("-"*70)
    
    case1 = CaseRecord(
        case_id="case_001",
        scenario_id="peak_demand",
        timestamp=1000,
        initial_state={'levels': [3.0, 3.0], 'demands': [12.0, 11.0]},
        strategy_used="peak_response",
        actions_taken=["增加供水流量", "调整各池协同"],
        outcome="success",
        performance={'rmse': 0.08, 'delivery_rate': 0.98},
        lessons_learned=["提前预测高峰有助于平稳过渡", "前馈补偿效果明显"]
    )
    
    kb.add_case(case1)
    
    # 获取最佳实践
    best_practices = kb.get_best_practices("peak_demand")
    print(f"\n最佳实践:")
    print(f"  推荐策略: {best_practices['recommended_strategy']}")
    print(f"  成功率: {best_practices['success_rate']:.1%}")
    print(f"  案例数: {best_practices['total_cases']}")
    
    if best_practices['lessons_learned']:
        print(f"  经验教训:")
        for lesson in best_practices['lessons_learned']:
            print(f"    - {lesson}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

"""
决策引擎
基于场景识别和知识库的智能决策
"""

import logging

logger = logging.getLogger(__name__)

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from hydroe2e.phase3.scenario_recognition.scenario_types import SCENARIO_LIBRARY
from hydroe2e.phase3.scenario_recognition.scenario_recognizer import ScenarioRecognizer, RecognitionResult
from hydroe2e.phase3.scenario_recognition.rule_engine import SystemState
from hydroe2e.phase3.knowledge.simple_kb import SimpleKnowledgeBase, ControlStrategy


@dataclass
class Decision:
    """决策结果"""
    scenario: RecognitionResult
    strategy: ControlStrategy
    
    # 控制参数
    mpc_config: Dict
    
    # 风险评估
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    risk_factors: List[str]
    
    # 执行建议
    immediate_actions: List[str]
    monitoring_points: List[str]
    
    # 决策依据
    reasoning: List[str]


class DecisionEngine:
    """智能决策引擎"""
    
    def __init__(self):
        """初始化决策引擎"""
        self.recognizer = ScenarioRecognizer(window_size=24)
        self.knowledge_base = SimpleKnowledgeBase()
        
        # 决策历史
        self.decision_history = []
    
    def make_decision(self, state: SystemState) -> Decision:
        """
        做出智能决策
        
        Args:
            state: 当前系统状态
            
        Returns:
            决策结果
        """
        # 1. 场景识别
        scenario = self.recognizer.recognize(state)
        
        # 2. 检索策略
        strategy = self._select_strategy(scenario, state)
        
        # 3. 配置MPC参数
        mpc_config = self._configure_mpc(strategy, scenario, state)
        
        # 4. 评估风险
        risk_level, risk_factors = self._assess_risk(scenario, state)
        
        # 5. 生成行动建议
        immediate_actions = self._generate_actions(scenario, strategy, risk_level)
        monitoring_points = self._identify_monitoring_points(scenario, state)
        
        # 6. 决策推理
        reasoning = self._explain_reasoning(scenario, strategy, risk_level)
        
        # 7. 创建决策
        decision = Decision(
            scenario=scenario,
            strategy=strategy,
            mpc_config=mpc_config,
            risk_level=risk_level,
            risk_factors=risk_factors,
            immediate_actions=immediate_actions,
            monitoring_points=monitoring_points,
            reasoning=reasoning
        )
        
        # 8. 记录历史
        self.decision_history.append(decision)
        
        return decision
    
    def _select_strategy(self, scenario: RecognitionResult, state: SystemState) -> ControlStrategy:
        """选择控制策略"""
        # 1. 获取最佳实践
        best_practices = self.knowledge_base.get_best_practices(scenario.scenario_id)
        
        # 2. 获取推荐策略
        strategy_id = best_practices['recommended_strategy']
        strategy = self.knowledge_base.get_strategy(strategy_id)
        
        # 3. 如果没有找到，使用默认策略
        if not strategy:
            strategies = self.knowledge_base.get_strategies_for_scenario(scenario.scenario_id)
            if strategies:
                strategy = strategies[0]
            else:
                # 最后兜底：使用日常策略
                strategy = self.knowledge_base.get_strategy("daily_normal")
        
        return strategy
    
    def _configure_mpc(self, strategy: ControlStrategy, 
                       scenario: RecognitionResult, state: SystemState) -> Dict:
        """配置MPC参数"""
        config = {
            'horizon': strategy.horizon,
            'dt': strategy.update_frequency,
            'weights': strategy.weights.copy(),
            'constraints': {
                'max_level_deviation': strategy.max_level_deviation,
                'max_flow_change_rate': strategy.max_flow_change_rate,
                'safety_margin': strategy.safety_margin
            },
            'use_feedforward': strategy.use_feedforward,
            'mode': 'weighted_sum'  # 默认模式
        }
        
        # 根据场景动态调整
        if scenario.priority == 'critical':
            config['horizon'] = min(config['horizon'], 5)  # 缩短时域
            config['mode'] = 'adaptive'  # 使用自适应模式
        
        # 根据风险调整
        if scenario.confidence < 0.7:
            # 识别不确定，增加鲁棒性
            config['constraints']['safety_margin'] *= 1.5
        
        return config
    
    def _assess_risk(self, scenario: RecognitionResult, 
                     state: SystemState) -> Tuple[str, List[str]]:
        """评估风险"""
        risk_factors = []
        risk_score = 0
        
        # 1. 场景本身的风险
        if scenario.priority == 'critical':
            risk_score += 3
            risk_factors.append("场景优先级为紧急")
        elif scenario.priority == 'high':
            risk_score += 2
            risk_factors.append("场景优先级较高")
        
        # 2. 识别置信度
        if scenario.confidence < 0.6:
            risk_score += 2
            risk_factors.append(f"场景识别置信度较低 ({scenario.confidence:.1%})")
        
        # 3. 水位风险
        avg_level = np.mean(state.levels)
        if avg_level > 6.0:
            risk_score += 2
            risk_factors.append(f"水位过高 ({avg_level:.2f}m)")
        elif avg_level < 1.5:
            risk_score += 2
            risk_factors.append(f"水位过低 ({avg_level:.2f}m)")
        
        # 4. 流量风险
        avg_flow = np.mean(state.flows)
        if avg_flow > 15.0:
            risk_score += 1
            risk_factors.append(f"流量过大 ({avg_flow:.2f} m³/s)")
        
        # 5. 需求风险
        avg_demand = np.mean(state.demands)
        supply_demand_ratio = avg_flow / avg_demand if avg_demand > 0 else 1.0
        if supply_demand_ratio < 0.8:
            risk_score += 2
            risk_factors.append("供水不足，供需比 < 0.8")
        
        # 6. 告警
        if state.alerts:
            risk_score += len(state.alerts)
            for alert in state.alerts:
                risk_factors.append(f"告警: {alert}")
        
        # 确定风险等级
        if risk_score >= 6:
            risk_level = 'critical'
        elif risk_score >= 4:
            risk_level = 'high'
        elif risk_score >= 2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return risk_level, risk_factors
    
    def _generate_actions(self, scenario: RecognitionResult, 
                         strategy: ControlStrategy, risk_level: str) -> List[str]:
        """生成立即行动建议"""
        actions = []
        
        # 使用场景推荐的行动
        actions.extend(scenario.recommended_actions)
        
        # 根据风险等级添加行动
        if risk_level == 'critical':
            actions.insert(0, "🚨 启动应急响应程序")
            actions.append("📞 立即通知值班人员")
        elif risk_level == 'high':
            actions.insert(0, "⚠️ 提高监控频率")
            actions.append("📋 准备应急预案")
        
        # 根据策略添加行动
        if strategy.use_feedforward:
            actions.append("📡 启用前馈补偿控制")
        
        if strategy.update_frequency < 1800:
            actions.append(f"⏱️ 加快更新频率至{strategy.update_frequency/60:.0f}分钟")
        
        return actions
    
    def _identify_monitoring_points(self, scenario: RecognitionResult,
                                    state: SystemState) -> List[str]:
        """识别监控要点"""
        points = []
        
        # 通用监控点
        points.append("💧 各池水位变化")
        points.append("🌊 流量波动情况")
        points.append("📊 供需平衡状态")
        
        # 根据场景添加
        if 'flood' in scenario.scenario_id:
            points.append("🌧️ 上游来水情况")
            points.append("📈 水位上升速率")
        
        if 'drought' in scenario.scenario_id:
            points.append("💦 水源储备量")
            points.append("👥 用户需求变化")
        
        if 'ice' in scenario.scenario_id:
            points.append("❄️ 水温和气温")
            points.append("🧊 冰层形成情况")
        
        if 'emergency' in scenario.category or scenario.priority == 'critical':
            points.append("⚙️ 设备运行状态")
            points.append("🧪 水质参数")
        
        return points
    
    def _explain_reasoning(self, scenario: RecognitionResult,
                          strategy: ControlStrategy, risk_level: str) -> List[str]:
        """解释决策推理过程"""
        reasoning = []
        
        reasoning.append(f"1. 场景识别: 识别为'{scenario.scenario_name}'，" +
                        f"置信度{scenario.confidence:.1%}，" +
                        f"使用{scenario.method}方法")
        
        reasoning.append(f"2. 策略选择: 选用'{strategy.name}'，" +
                        f"基于历史最佳实践")
        
        reasoning.append(f"3. 风险评估: 当前风险等级为'{risk_level}'，" +
                        f"需采取相应措施")
        
        reasoning.append(f"4. 参数配置: MPC时域={strategy.horizon}步，" +
                        f"更新频率={strategy.update_frequency/60:.0f}分钟")
        
        reasoning.append(f"5. 优先级: 场景优先级为'{scenario.priority}'，" +
                        f"需快速响应" if scenario.priority in ['high', 'critical'] 
                        else f"5. 优先级: 正常优先级，按计划执行")
        
        return reasoning


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*25 + "决策引擎演示")
    logger.info("="*70)
    
    # 创建决策引擎
    engine = DecisionEngine()
    
    # 测试场景
    test_scenarios = [
        {
            'name': '日常运行',
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
                time=18,
                levels=[2.9, 3.0, 3.1],
                flows=[11.0, 10.5, 10.0],
                demands=[12.0, 11.5, 11.0]
            )
        },
        {
            'name': '洪峰危险',
            'state': SystemState(
                time=100,
                levels=[6.5, 6.2, 6.0],
                flows=[20.0, 19.0, 18.0],
                demands=[5.0, 5.0, 5.0],
                weather={'rainfall': 25.0},
                alerts=["水位接近警戒线"]
            )
        }
    ]
    
    for i, test in enumerate(test_scenarios, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"场景 {i}: {test['name']}")
        logger.info('='*70)
        
        decision = engine.make_decision(test['state'])
        
        logger.info(f"\n📋 场景识别:")
        logger.info(f"  识别结果: {decision.scenario.scenario_name}")
        logger.info(f"  置信度: {decision.scenario.confidence:.2%}")
        logger.info(f"  优先级: {decision.scenario.priority}")
        
        logger.info(f"\n🎯 策略选择:")
        logger.info(f"  策略: {decision.strategy.name}")
        logger.info(f"  MPC时域: {decision.mpc_config['horizon']}步")
        logger.info(f"  更新频率: {decision.mpc_config['dt']/60:.0f}分钟")
        
        logger.info(f"\n⚠️ 风险评估:")
        logger.info(f"  风险等级: {decision.risk_level}")
        if decision.risk_factors:
            logger.info(f"  风险因素:")
            for factor in decision.risk_factors:
                logger.info(f"    • {factor}")
        
        logger.info(f"\n🚀 立即行动:")
        for action in decision.immediate_actions[:5]:
            logger.info(f"  {action}")
        
        logger.info(f"\n🔍 监控要点:")
        for point in decision.monitoring_points[:5]:
            logger.info(f"  {point}")
        
        logger.info(f"\n💡 决策推理:")
        for reason in decision.reasoning:
            logger.info(f"  {reason}")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)

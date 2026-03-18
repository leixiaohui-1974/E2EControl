"""
Phase 3 综合演示
展示场景识别和智能决策的完整流程
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from hydroe2e.phase3.scenario_recognition.rule_engine import SystemState
from hydroe2e.phase3.decision.decision_engine import DecisionEngine


def run_integrated_demo():
    """运行综合演示"""
    
    logger.info("="*80)
    logger.info(" "*25 + "Phase 3 智能决策系统演示")
    logger.info("="*80)
    
    # 创建决策引擎
    logger.info("\n🚀 初始化智能决策引擎...")
    engine = DecisionEngine()
    
    logger.info(f"  ✓ 场景识别器已加载 (窗口={engine.recognizer.feature_extractor.window_size}h)")
    logger.info(f"  ✓ 知识库已加载 ({len(engine.knowledge_base.strategies)}个策略)")
    logger.info(f"  ✓ 规则引擎已加载 ({len(engine.recognizer.rule_engine.rules)}条规则)")
    
    # 场景序列
    scenarios = [
        {
            'name': '1️⃣  正常运行 → 日常供水',
            'description': '清晨时段，需求平稳',
            'state': SystemState(
                time=6,  # 早上6点
                levels=[3.0, 3.0, 3.0],
                flows=[5.0, 5.0, 5.0],
                demands=[4.5, 5.0, 5.2]
            )
        },
        {
            'name': '2️⃣  高峰来临 → 需求激增',
            'description': '早高峰，用水激增',
            'state': SystemState(
                time=8,  # 早上8点
                levels=[2.9, 2.8, 2.9],
                flows=[10.0, 9.5, 9.0],
                demands=[11.5, 12.0, 11.8]
            )
        },
        {
            'name': '3️⃣  预警！上游来水增大',
            'description': '检测到降雨，需防汛',
            'state': SystemState(
                time=50,
                levels=[3.5, 3.4, 3.3],
                flows=[8.0, 8.5, 9.0],
                demands=[6.0, 6.0, 6.0],
                weather={'rainfall_forecast': [15.0, 20.0, 25.0]}
            )
        },
        {
            'name': '4️⃣  危机！洪峰抵达',
            'description': '大流量来水，水位飙升',
            'state': SystemState(
                time=60,
                levels=[6.2, 5.8, 5.5],
                flows=[22.0, 20.0, 18.0],
                demands=[5.0, 5.0, 5.0],
                weather={'rainfall': 30.0},
                alerts=["水位快速上升", "接近警戒线"]
            )
        },
        {
            'name': '5️⃣  洪后恢复',
            'description': '洪水退去，恢复正常',
            'state': SystemState(
                time=80,
                levels=[4.0, 3.8, 3.6],
                flows=[8.0, 7.5, 7.0],
                demands=[5.5, 5.5, 5.5]
            )
        },
        {
            'name': '6️⃣  夜间低谷',
            'description': '深夜，需求降低',
            'state': SystemState(
                time=100,  # 夜间
                levels=[3.2, 3.1, 3.0],
                flows=[2.5, 2.5, 2.5],
                demands=[2.0, 2.2, 2.5]
            )
        }
    ]
    
    # 逐场景演示
    for i, scenario in enumerate(scenarios):
        logger.info(f"\n{'='*80}")
        logger.info(f"{scenario['name']}")
        logger.info(f"{scenario['description']}")
        logger.info('='*80)
        
        # 做出决策
        decision = engine.make_decision(scenario['state'])
        
        # 显示结果
        logger.info(f"\n📍 当前状态:")
        logger.info(f"  时间: {scenario['state'].time}h")
        logger.info(f"  平均水位: {np.mean(scenario['state'].levels):.2f}m")
        logger.info(f"  平均流量: {np.mean(scenario['state'].flows):.2f} m³/s")
        logger.info(f"  平均需求: {np.mean(scenario['state'].demands):.2f} m³/s")
        if scenario['state'].alerts:
            logger.info(f"  ⚠️  告警: {', '.join(scenario['state'].alerts)}")
        
        logger.info(f"\n🎯 场景识别:")
        logger.info(f"  ├─ 场景: {decision.scenario.scenario_name}")
        logger.info(f"  ├─ 置信度: {decision.scenario.confidence:.1%}")
        logger.info(f"  ├─ 类别: {decision.scenario.category}")
        logger.info(f"  └─ 优先级: {decision.scenario.priority}")
        
        logger.info(f"\n💡 智能策略:")
        logger.info(f"  ├─ 策略: {decision.strategy.name}")
        logger.info(f"  ├─ MPC时域: {decision.mpc_config['horizon']}步")
        logger.info(f"  ├─ 更新频率: {decision.mpc_config['dt']/60:.0f}分钟")
        logger.info(f"  └─ 前馈补偿: {'启用' if decision.mpc_config['use_feedforward'] else '禁用'}")
        
        # 风险评估
        risk_emoji = {'low': '✅', 'medium': '⚠️', 'high': '🔶', 'critical': '🚨'}
        logger.info(f"\n{risk_emoji[decision.risk_level]} 风险评估: {decision.risk_level.upper()}")
        if decision.risk_factors:
            for factor in decision.risk_factors[:3]:
                logger.info(f"  • {factor}")
        
        # 立即行动（显示前3条）
        logger.info(f"\n🚀 立即行动:")
        for action in decision.immediate_actions[:3]:
            logger.info(f"  {action}")
        
        # 监控要点
        logger.info(f"\n🔍 监控要点:")
        for point in decision.monitoring_points[:3]:
            logger.info(f"  {point}")
        
        # 暂停（方便查看）
        if i < len(scenarios) - 1:
            input(f"\n{'─'*80}\n按回车键继续下一个场景...\n{'─'*80}\n")
    
    # 总结
    logger.info(f"\n{'='*80}")
    logger.info("📊 演示总结")
    logger.info('='*80)
    
    logger.info(f"\n✓ 共演示 {len(scenarios)} 个场景")
    logger.info(f"✓ 所有场景均成功识别并给出决策")
    logger.info(f"✓ 系统能够:")
    logger.info(f"  • 自动识别从正常到紧急的各种场景")
    logger.info(f"  • 根据场景动态选择最优控制策略")
    logger.info(f"  • 实时评估风险并给出应对建议")
    logger.info(f"  • 提供决策推理过程，可解释性强")
    
    logger.info(f"\n🎯 关键特性:")
    logger.info(f"  ✓ 规则引擎 + 特征提取 混合识别")
    logger.info(f"  ✓ 知识库驱动的策略选择")
    logger.info(f"  ✓ 多维度风险评估")
    logger.info(f"  ✓ 场景自适应参数调整")
    logger.info(f"  ✓ 完整的决策推理链")
    
    logger.info("\n" + "="*80)
    logger.info("🎉 Phase 3 核心功能演示完成！")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        run_integrated_demo()
    except KeyboardInterrupt:
        logger.info("\n\n演示被用户中断")
    except Exception as e:
        logger.info(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

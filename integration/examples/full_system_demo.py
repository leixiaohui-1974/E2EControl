"""
完整系统演示
Phase 2 (分布式MPC) + Phase 3 (智能决策) 集成
"""

import sys
import logging

logger = logging.getLogger(__name__)
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

from phase2.models.cascaded_system import CascadedCanalSystem
from integration.controllers.adaptive_mpc import AdaptiveMPCController, AdaptiveControlConfig


def generate_complex_scenario(total_hours: int = 120):
    """
    生成复杂场景序列
    
    包括：正常运行、高峰、低谷、洪峰预警、洪峰来袭、洪后恢复等
    """
    scenario_plan = []
    
    # 第1天：正常运行 (0-24h)
    for h in range(24):
        hour_of_day = h % 24
        
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
            # 高峰期
            demand = 10.0 + np.random.normal(0, 0.5)
            weather = None
            alerts = []
        elif 0 <= hour_of_day <= 5 or hour_of_day >= 22:
            # 低谷期
            demand = 3.0 + np.random.normal(0, 0.3)
            weather = None
            alerts = []
        else:
            # 正常期
            demand = 6.0 + np.random.normal(0, 0.4)
            weather = None
            alerts = []
        
        scenario_plan.append({
            'demand': demand,
            'weather': weather,
            'alerts': alerts,
            'description': f"Day 1, Hour {h}: Normal operation"
        })
    
    # 第2天：开始出现降雨预警 (24-48h)
    for h in range(24, 48):
        hour_of_day = h % 24
        demand = 6.0 + np.random.normal(0, 0.4)
        
        if h >= 36:
            # 开始有降雨预报
            weather = {'rainfall_forecast': [10.0, 15.0, 20.0]}
            alerts = ['降雨预报', '注意水位']
            description = f"Day 2, Hour {h-24}: Rain forecast"
        else:
            weather = None
            alerts = []
            description = f"Day 2, Hour {h-24}: Normal"
        
        scenario_plan.append({
            'demand': demand,
            'weather': weather,
            'alerts': alerts,
            'description': description
        })
    
    # 第3天：洪峰来临 (48-72h)
    for h in range(48, 72):
        hour_of_day = h % 24
        
        # 洪峰曲线
        flood_intensity = 0
        if 48 <= h < 54:
            # 洪前准备期
            flood_intensity = (h - 48) / 6 * 0.3
            weather = {'rainfall': 10.0 + (h-48)*2, 'rainfall_forecast': [20.0, 25.0, 30.0]}
            alerts = ['洪水预警', '准备泄洪']
            description = f"Day 3, Hour {h-48}: Pre-flood preparation"
        elif 54 <= h < 60:
            # 洪峰期
            flood_intensity = 0.3 + (h - 54) / 6 * 0.7
            weather = {'rainfall': 25.0 + (h-54)*2}
            alerts = ['洪峰到达', '全力泄洪', '水位警戒']
            description = f"Day 3, Hour {h-48}: FLOOD PEAK"
        elif 60 <= h < 66:
            # 洪峰消退
            flood_intensity = 1.0 - (h - 60) / 6 * 0.5
            weather = {'rainfall': 35.0 - (h-60)*3}
            alerts = ['洪峰消退', '持续泄洪']
            description = f"Day 3, Hour {h-48}: Flood receding"
        else:
            # 洪后
            flood_intensity = 0.5 - (h - 66) / 6 * 0.3
            weather = {'rainfall': 10.0}
            alerts = ['洪后恢复']
            description = f"Day 3, Hour {h-48}: Post-flood recovery"
        
        demand = 5.0 + np.random.normal(0, 0.3)
        
        scenario_plan.append({
            'demand': demand,
            'weather': weather,
            'alerts': alerts,
            'flood_intensity': flood_intensity,
            'description': description
        })
    
    # 第4-5天：恢复正常 (72-120h)
    for h in range(72, total_hours):
        hour_of_day = h % 24
        
        if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
            demand = 9.0 + np.random.normal(0, 0.5)
        elif 0 <= hour_of_day <= 5 or hour_of_day >= 22:
            demand = 3.5 + np.random.normal(0, 0.3)
        else:
            demand = 6.0 + np.random.normal(0, 0.4)
        
        weather = None
        alerts = []
        description = f"Day {h//24 + 1}, Hour {h%24}: Back to normal"
        
        scenario_plan.append({
            'demand': demand,
            'weather': weather,
            'alerts': alerts,
            'description': description
        })
    
    return scenario_plan


def run_full_system_demo():
    """运行完整系统演示"""
    
    logger.info("="*80)
    logger.info(" "*25 + "完整系统集成演示")
    logger.info(" "*20 + "Phase 2 + Phase 3 智能控制")
    logger.info("="*80)
    
    # 配置
    num_pools = 3
    total_hours = 120
    dt = 3600.0
    
    logger.info(f"\n📋 仿真配置:")
    logger.info(f"  池数量: {num_pools}")
    logger.info(f"  仿真时长: {total_hours}小时 (5天)")
    logger.info(f"  时间步长: {dt/3600:.1f}小时")
    
    # 创建系统
    logger.info(f"\n🚀 初始化系统...")
    system = CascadedCanalSystem(
        num_pools=num_pools,
        pool_area=10000.0,
        dt=dt
    )
    
    # 创建自适应控制器
    controller = AdaptiveMPCController(
        num_pools=num_pools,
        horizon=10,
        dt=dt,
        config=AdaptiveControlConfig(
            enable_scenario_recognition=True,
            enable_adaptive_weights=True,
            enable_feedforward=True,
            enable_risk_adjustment=True
        )
    )
    
    logger.info(f"  ✓ 水网系统已创建")
    logger.info(f"  ✓ 自适应MPC控制器已创建")
    logger.info(f"  ✓ 智能决策引擎已启用")
    
    # 生成场景
    logger.info(f"\n📝 生成复杂场景序列...")
    scenario_plan = generate_complex_scenario(total_hours)
    logger.info(f"  ✓ {total_hours}小时场景已生成")
    logger.info(f"  ✓ 包含：正常→高峰→预警→洪峰→恢复")
    
    # 记录数据
    history = {
        'time': [],
        'levels': [[] for _ in range(num_pools)],
        'flows': [[] for _ in range(num_pools + 1)],
        'demands': [],
        'scenarios': [],
        'strategies': [],
        'risk_levels': [],
        'control_performance': [],
        'flood_intensity': []
    }
    
    # 仿真循环
    logger.info(f"\n🎬 开始仿真...")
    logger.info(f"{'─'*80}")
    
    start_time = time.time()
    
    for step in range(total_hours):
        # 获取当前状态
        state = system.get_system_state()
        current_levels = state['levels']
        current_flows = state['flows']
        
        # 获取场景信息
        scenario_info = scenario_plan[step]
        current_demand = scenario_info['demand']
        weather = scenario_info['weather']
        alerts = scenario_info['alerts']
        
        # 计算控制动作
        control_actions, debug_info = controller.compute_control(
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=[current_demand] * num_pools,
            time=step,
            weather=weather,
            alerts=alerts
        )
        
        # 提取控制流量
        gate_flows = [control_actions[0][0]]  # 第一个池的入流
        for q_in, q_out in control_actions:
            gate_flows.append(q_out)
        
        # 应用控制动作
        system.step(gate_flows, demand=current_demand)
        
        # 记录数据
        history['time'].append(step)
        for i, level in enumerate(current_levels):
            history['levels'][i].append(level)
        for i, flow in enumerate(gate_flows):
            history['flows'][i].append(flow)
        history['demands'].append(current_demand)
        history['flood_intensity'].append(scenario_info.get('flood_intensity', 0))
        
        if 'scenario' in debug_info:
            history['scenarios'].append(debug_info['scenario']['scenario_name'])
            history['risk_levels'].append(debug_info['scenario']['risk_level'])
        else:
            history['scenarios'].append('unknown')
            history['risk_levels'].append('low')
        
        if 'mpc_config' in debug_info:
            history['strategies'].append(debug_info['mpc_config'].get('strategy_id', 'default'))
        else:
            history['strategies'].append('default')
        
        # 每24小时输出一次
        if (step + 1) % 24 == 0:
            day = (step + 1) // 24
            avg_level = np.mean(current_levels)
            avg_flow = np.mean(gate_flows)
            current_scenario = history['scenarios'][-1]
            current_risk = history['risk_levels'][-1]
            
            logger.info(f"Day {day} 完成 | "
                  f"水位: {avg_level:.2f}m | "
                  f"流量: {avg_flow:.2f}m³/s | "
                  f"场景: {current_scenario:20s} | "
                  f"风险: {current_risk:8s}")
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"{'─'*80}")
    logger.info(f"✅ 仿真完成!")
    logger.info(f"  总耗时: {elapsed_time:.2f}秒")
    logger.info(f"  平均步时: {elapsed_time/total_hours*1000:.1f}ms/步")
    
    # 统计信息
    logger.info(f"\n{'='*80}")
    logger.info("📊 系统统计")
    logger.info('='*80)
    
    stats = controller.get_statistics()
    logger.info(f"\n控制器统计:")
    logger.info(f"  总步数: {stats['total_steps']}")
    logger.info(f"  场景切换: {stats['scenario_switches']}次 ({stats['scenario_switch_rate']:.1%})")
    logger.info(f"  策略切换: {stats['strategy_switches']}次 ({stats['strategy_switch_rate']:.1%})")
    logger.info(f"  风险升级: {stats['risk_escalations']}次 ({stats['risk_escalation_rate']:.1%})")
    
    summary = controller.get_history_summary()
    logger.info(f"\n场景分布:")
    for scenario, count in sorted(summary['scenario_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
        percentage = count / summary['total_steps'] * 100
        logger.info(f"  {scenario:25s}: {count:3d}次 ({percentage:5.1f}%)")
    
    logger.info(f"\n风险分布:")
    for risk, count in sorted(summary['risk_distribution'].items(),
                             key=lambda x: ['low', 'medium', 'high', 'critical'].index(x[0])):
        percentage = count / summary['total_steps'] * 100
        logger.info(f"  {risk:10s}: {count:3d}次 ({percentage:5.1f}%)")
    
    logger.info(f"\nADMM性能:")
    logger.info(f"  平均迭代次数: {summary['avg_iterations']:.1f}")
    logger.info(f"  收敛率: {summary['convergence_rate']:.1%}")
    logger.info(f"  平均求解时间: {summary['avg_solve_time']*1000:.1f}ms")
    
    # 控制性能评估
    logger.info(f"\n控制性能:")
    all_levels = np.concatenate([history['levels'][i] for i in range(num_pools)])
    level_target = 3.0
    rmse = np.sqrt(np.mean((all_levels - level_target)**2))
    max_deviation = np.max(np.abs(all_levels - level_target))
    
    logger.info(f"  水位RMSE: {rmse:.4f}m")
    logger.info(f"  最大偏差: {max_deviation:.4f}m")
    
    # 检查约束违反
    violations = np.sum((all_levels < 1.0) | (all_levels > 5.0))
    logger.info(f"  约束违反: {violations}次 ({violations/len(all_levels)*100:.2f}%)")
    
    # 可视化
    logger.info(f"\n🎨 生成可视化...")
    visualize_results(history, num_pools, total_hours)
    logger.info(f"  ✓ 图表已保存")
    
    logger.info(f"\n{'='*80}")
    logger.info("🎉 完整系统演示完成！")
    logger.info('='*80)
    
    return history, controller


def visualize_results(history, num_pools, total_hours):
    """可视化结果"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 颜色映射
    risk_colors = {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
    
    # 1. 水位曲线
    ax1 = plt.subplot(4, 1, 1)
    for i in range(num_pools):
        ax1.plot(history['time'], history['levels'][i], 
                label=f'Pool {i+1}', linewidth=1.5)
    ax1.axhline(y=3.0, color='green', linestyle='--', alpha=0.5, label='Target')
    ax1.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label='Max')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Min')
    ax1.set_ylabel('Water Level (m)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', ncol=num_pools+3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_hours)
    ax1.set_title('Complete System Performance (Phase 2 + Phase 3 Integration)', 
                  fontsize=13, fontweight='bold')
    
    # 添加洪峰高亮
    for i, intensity in enumerate(history['flood_intensity']):
        if intensity > 0:
            ax1.axvspan(i, i+1, alpha=intensity*0.2, color='blue')
    
    # 2. 流量曲线
    ax2 = plt.subplot(4, 1, 2)
    for i in range(num_pools + 1):
        ax2.plot(history['time'], history['flows'][i], 
                label=f'Gate {i}', linewidth=1.5, alpha=0.8)
    ax2.plot(history['time'], history['demands'], 
            'k--', label='Demand', linewidth=2, alpha=0.6)
    ax2.set_ylabel('Flow (m³/s)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', ncol=num_pools+2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_hours)
    
    # 3. 场景识别
    ax3 = plt.subplot(4, 1, 3)
    
    # 统计场景
    from collections import Counter
    scenario_counts = Counter(history['scenarios'])
    unique_scenarios = list(scenario_counts.keys())
    scenario_to_idx = {s: i for i, s in enumerate(unique_scenarios)}
    
    # 绘制场景条形图
    scenario_indices = [scenario_to_idx[s] for s in history['scenarios']]
    colors_scenario = plt.cm.Set3(np.linspace(0, 1, len(unique_scenarios)))
    
    for i, scenario in enumerate(unique_scenarios):
        scenario_mask = np.array(history['scenarios']) == scenario
        time_points = np.array(history['time'])[scenario_mask]
        for t in time_points:
            ax3.add_patch(Rectangle((t, 0), 1, 1, 
                                   facecolor=colors_scenario[i], alpha=0.7))
    
    ax3.set_ylabel('Scenario', fontsize=11, fontweight='bold')
    ax3.set_yticks([])
    ax3.set_xlim(0, total_hours)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 图例
    handles = [plt.Rectangle((0,0),1,1, facecolor=colors_scenario[i], alpha=0.7) 
              for i in range(len(unique_scenarios))]
    ax3.legend(handles, unique_scenarios, loc='center left', 
              bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # 4. 风险等级
    ax4 = plt.subplot(4, 1, 4)
    
    risk_to_val = {'low': 0.2, 'medium': 0.4, 'high': 0.7, 'critical': 1.0}
    risk_values = [risk_to_val[r] for r in history['risk_levels']]
    risk_color_list = [risk_colors[r] for r in history['risk_levels']]
    
    for i, (rv, rc) in enumerate(zip(risk_values, risk_color_list)):
        ax4.add_patch(Rectangle((i, 0), 1, rv, facecolor=rc, alpha=0.6))
    
    ax4.set_ylabel('Risk Level', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, total_hours)
    ax4.set_ylim(0, 1.0)
    ax4.set_yticks([0.2, 0.4, 0.7, 1.0])
    ax4.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/integration_demo_result.png', dpi=150, bbox_inches='tight')
    logger.info(f"  📊 图表已保存: integration_demo_result.png")
    
    plt.close()


if __name__ == "__main__":
    try:
        history, controller = run_full_system_demo()
    except Exception as e:
        logger.info(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

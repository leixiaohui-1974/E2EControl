"""
性能对比分析
对比：
1. 传统固定MPC vs 自适应智能MPC
2. 无场景识别 vs 有场景识别
3. 固定策略 vs 动态策略
"""

import sys
import logging

logger = logging.getLogger(__name__)
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List

from phase2.models.cascaded_system import CascadedCanalSystem
from phase2.controllers.improved_admm import ImprovedDistributedMPC
from integration.controllers.adaptive_mpc import AdaptiveMPCController, AdaptiveControlConfig


def run_comparison_test(scenario_plan: List[Dict], test_name: str):
    """
    运行对比测试
    
    Args:
        scenario_plan: 场景计划
        test_name: 测试名称
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"测试: {test_name}")
    logger.info('='*80)
    
    num_pools = 3
    dt = 3600.0
    total_steps = len(scenario_plan)
    
    results = {}
    
    # 测试1: 传统固定MPC
    logger.info(f"\n[1/3] 运行传统固定MPC...")
    results['traditional'] = run_traditional_mpc(
        num_pools, dt, scenario_plan
    )
    
    # 测试2: 自适应MPC（无场景识别）
    logger.info(f"\n[2/3] 运行自适应MPC（无场景识别）...")
    results['adaptive_no_scene'] = run_adaptive_mpc(
        num_pools, dt, scenario_plan, 
        enable_scenario=False
    )
    
    # 测试3: 完整智能MPC（有场景识别）
    logger.info(f"\n[3/3] 运行完整智能MPC（有场景识别）...")
    results['intelligent'] = run_adaptive_mpc(
        num_pools, dt, scenario_plan,
        enable_scenario=True
    )
    
    # 分析和对比
    logger.info(f"\n{'='*80}")
    logger.info("性能对比分析")
    logger.info('='*80)
    
    comparison = compare_results(results, num_pools)
    
    # 可视化
    visualize_comparison(results, comparison, test_name)
    
    return results, comparison


def run_traditional_mpc(num_pools, dt, scenario_plan):
    """运行传统固定MPC"""
    system = CascadedCanalSystem(num_pools=num_pools, pool_area=10000.0, dt=dt)
    controller = ImprovedDistributedMPC(num_pools=num_pools, horizon=10, dt=dt)
    
    history = {
        'levels': [[] for _ in range(num_pools)],
        'flows': [[] for _ in range(num_pools + 1)],
        'demands': [],
        'solve_times': [],
        'iterations': []
    }
    
    start_time = time.time()
    
    for step, scenario in enumerate(scenario_plan):
        state = system.get_system_state()
        current_levels = state['levels']
        current_flows = state['flows']
        demand = scenario['demand']
        
        # 固定MPC控制
        control_actions, admm_info = controller.solve(
            current_levels=current_levels,
            q_in_prevs=current_flows[:-1],
            q_out_forecasts=[[demand] * controller.horizon] * num_pools
        )
        
        gate_flows = [control_actions[0][0]]
        for q_in, q_out in control_actions:
            gate_flows.append(q_out)
        
        system.step(gate_flows, demand=demand)
        
        # 记录
        for i, level in enumerate(current_levels):
            history['levels'][i].append(level)
        for i, flow in enumerate(gate_flows):
            history['flows'][i].append(flow)
        history['demands'].append(demand)
        history['solve_times'].append(admm_info.get('solve_time', 0))
        history['iterations'].append(admm_info.get('iterations', 0))
    
    elapsed = time.time() - start_time
    
    logger.info(f"  完成! 耗时: {elapsed:.2f}秒")
    logger.info(f"  平均步时: {elapsed/len(scenario_plan)*1000:.1f}ms")
    
    return history


def run_adaptive_mpc(num_pools, dt, scenario_plan, enable_scenario=True):
    """运行自适应MPC"""
    system = CascadedCanalSystem(num_pools=num_pools, pool_area=10000.0, dt=dt)
    controller = AdaptiveMPCController(
        num_pools=num_pools,
        horizon=10,
        dt=dt,
        config=AdaptiveControlConfig(
            enable_scenario_recognition=enable_scenario,
            enable_adaptive_weights=True,
            enable_feedforward=enable_scenario,
            enable_risk_adjustment=enable_scenario
        )
    )
    
    history = {
        'levels': [[] for _ in range(num_pools)],
        'flows': [[] for _ in range(num_pools + 1)],
        'demands': [],
        'solve_times': [],
        'iterations': [],
        'scenarios': [] if enable_scenario else None,
        'risk_levels': [] if enable_scenario else None
    }
    
    start_time = time.time()
    
    for step, scenario in enumerate(scenario_plan):
        state = system.get_system_state()
        current_levels = state['levels']
        current_flows = state['flows']
        demand = scenario['demand']
        
        # 自适应MPC控制
        control_actions, debug_info = controller.compute_control(
            current_levels=current_levels,
            current_flows=current_flows,
            current_demands=[demand] * num_pools,
            time=step,
            weather=scenario.get('weather'),
            alerts=scenario.get('alerts')
        )
        
        gate_flows = [control_actions[0][0]]
        for q_in, q_out in control_actions:
            gate_flows.append(q_out)
        
        system.step(gate_flows, demand=demand)
        
        # 记录
        for i, level in enumerate(current_levels):
            history['levels'][i].append(level)
        for i, flow in enumerate(gate_flows):
            history['flows'][i].append(flow)
        history['demands'].append(demand)
        
        if 'admm' in debug_info:
            history['solve_times'].append(debug_info['admm'].get('solve_time', 0))
            history['iterations'].append(debug_info['admm'].get('iterations', 0))
        
        if enable_scenario and 'scenario' in debug_info:
            history['scenarios'].append(debug_info['scenario']['scenario_name'])
            history['risk_levels'].append(debug_info['scenario']['risk_level'])
    
    elapsed = time.time() - start_time
    
    logger.info(f"  完成! 耗时: {elapsed:.2f}秒")
    logger.info(f"  平均步时: {elapsed/len(scenario_plan)*1000:.1f}ms")
    
    if enable_scenario:
        stats = controller.get_statistics()
        logger.info(f"  场景切换: {stats['scenario_switches']}次")
        logger.info(f"  策略切换: {stats['strategy_switches']}次")
    
    return history


def compare_results(results: Dict, num_pools: int) -> Dict:
    """对比分析结果"""
    comparison = {}
    
    target_level = 3.0
    
    for name, history in results.items():
        all_levels = np.concatenate([history['levels'][i] for i in range(num_pools)])
        
        # 控制性能
        rmse = np.sqrt(np.mean((all_levels - target_level)**2))
        max_deviation = np.max(np.abs(all_levels - target_level))
        std_deviation = np.std(all_levels)
        
        # 约束违反
        violations = np.sum((all_levels < 1.0) | (all_levels > 5.0))
        violation_rate = violations / len(all_levels) * 100
        
        # 流量平滑性
        flow_changes = []
        for i in range(num_pools + 1):
            flows = np.array(history['flows'][i])
            flow_changes.extend(np.abs(np.diff(flows)))
        avg_flow_change = np.mean(flow_changes)
        max_flow_change = np.max(flow_changes)
        
        # 计算性能
        avg_solve_time = np.mean(history['solve_times']) if history['solve_times'] else 0
        avg_iterations = np.mean(history['iterations']) if history['iterations'] else 0
        
        # 供水保证率
        demands = np.array(history['demands'])
        outflows = np.array(history['flows'][-1])
        delivery_ratio = outflows / (demands + 1e-6)
        avg_delivery = np.mean(delivery_ratio)
        min_delivery = np.min(delivery_ratio)
        
        comparison[name] = {
            'rmse': rmse,
            'max_deviation': max_deviation,
            'std_deviation': std_deviation,
            'violations': violations,
            'violation_rate': violation_rate,
            'avg_flow_change': avg_flow_change,
            'max_flow_change': max_flow_change,
            'avg_solve_time': avg_solve_time,
            'avg_iterations': avg_iterations,
            'avg_delivery': avg_delivery,
            'min_delivery': min_delivery
        }
    
    # 打印对比表格
    logger.info(f"\n{'指标':<25} | {'传统MPC':>12} | {'自适应(无识别)':>15} | {'智能MPC':>12} | {'改进':>10}")
    logger.info('─' * 90)
    
    trad = comparison['traditional']
    adap = comparison['adaptive_no_scene']
    intl = comparison['intelligent']
    
    logger.info(f"{'水位RMSE (m)':<25} | {trad['rmse']:>12.4f} | {adap['rmse']:>15.4f} | {intl['rmse']:>12.4f} | {(1-intl['rmse']/trad['rmse'])*100:>9.1f}%")
    logger.info(f"{'最大偏差 (m)':<25} | {trad['max_deviation']:>12.4f} | {adap['max_deviation']:>15.4f} | {intl['max_deviation']:>12.4f} | {(1-intl['max_deviation']/trad['max_deviation'])*100:>9.1f}%")
    logger.info(f"{'约束违反次数':<25} | {trad['violations']:>12d} | {adap['violations']:>15d} | {intl['violations']:>12d} | {(1-intl['violations']/max(1,trad['violations']))*100:>9.1f}%")
    logger.info(f"{'平均流量变化 (m³/s)':<25} | {trad['avg_flow_change']:>12.4f} | {adap['avg_flow_change']:>15.4f} | {intl['avg_flow_change']:>12.4f} | {(1-intl['avg_flow_change']/trad['avg_flow_change'])*100:>9.1f}%")
    logger.info(f"{'平均供水保证率':<25} | {trad['avg_delivery']:>12.2%} | {adap['avg_delivery']:>15.2%} | {intl['avg_delivery']:>12.2%} | {(intl['avg_delivery']/trad['avg_delivery']-1)*100:>9.1f}%")
    logger.info(f"{'平均求解时间 (ms)':<25} | {trad['avg_solve_time']*1000:>12.1f} | {adap['avg_solve_time']*1000:>15.1f} | {intl['avg_solve_time']*1000:>12.1f} | {(1-intl['avg_solve_time']/trad['avg_solve_time'])*100:>9.1f}%")
    logger.info(f"{'平均迭代次数':<25} | {trad['avg_iterations']:>12.1f} | {adap['avg_iterations']:>15.1f} | {intl['avg_iterations']:>12.1f} | {(1-intl['avg_iterations']/trad['avg_iterations'])*100:>9.1f}%")
    
    return comparison


def visualize_comparison(results, comparison, test_name):
    """可视化对比结果"""
    fig = plt.figure(figsize=(16, 10))
    
    colors = {
        'traditional': 'blue',
        'adaptive_no_scene': 'orange',
        'intelligent': 'green'
    }
    
    labels = {
        'traditional': 'Traditional MPC',
        'adaptive_no_scene': 'Adaptive MPC (No Recognition)',
        'intelligent': 'Intelligent MPC (Full)'
    }
    
    # 1. 水位对比
    ax1 = plt.subplot(3, 2, 1)
    for name, history in results.items():
        avg_levels = np.mean([history['levels'][i] for i in range(3)], axis=0)
        ax1.plot(avg_levels, label=labels[name], 
                color=colors[name], linewidth=1.5, alpha=0.8)
    ax1.axhline(y=3.0, color='black', linestyle='--', alpha=0.3, label='Target')
    ax1.set_ylabel('Avg Water Level (m)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Water Level Comparison', fontweight='bold')
    
    # 2. 流量对比
    ax2 = plt.subplot(3, 2, 2)
    for name, history in results.items():
        avg_flows = np.mean([history['flows'][i] for i in range(4)], axis=0)
        ax2.plot(avg_flows, label=labels[name],
                color=colors[name], linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Avg Flow (m³/s)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Flow Comparison', fontweight='bold')
    
    # 3. 性能指标对比（柱状图）
    ax3 = plt.subplot(3, 2, 3)
    metrics = ['rmse', 'max_deviation', 'avg_flow_change']
    metric_labels = ['RMSE\n(m)', 'Max Dev\n(m)', 'Avg Flow\nChange']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (name, comp) in enumerate(comparison.items()):
        values = [comp[m] for m in metrics]
        ax3.bar(x + i*width, values, width, label=labels[name],
               color=colors[name], alpha=0.8)
    
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_title('Performance Metrics', fontweight='bold')
    
    # 4. 供水保证率对比
    ax4 = plt.subplot(3, 2, 4)
    metrics2 = ['avg_delivery', 'min_delivery']
    metric_labels2 = ['Avg Delivery\nRate', 'Min Delivery\nRate']
    x2 = np.arange(len(metrics2))
    
    for i, (name, comp) in enumerate(comparison.items()):
        values = [comp[m] for m in metrics2]
        ax4.bar(x2 + i*width, values, width, label=labels[name],
               color=colors[name], alpha=0.8)
    
    ax4.set_ylabel('Rate', fontweight='bold')
    ax4.set_xticks(x2 + width)
    ax4.set_xticklabels(metric_labels2)
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_title('Water Delivery Performance', fontweight='bold')
    
    # 5. 计算性能对比
    ax5 = plt.subplot(3, 2, 5)
    metrics3 = ['avg_solve_time', 'avg_iterations']
    metric_labels3 = ['Avg Solve Time\n(s)', 'Avg Iterations']
    x3 = np.arange(len(metrics3))
    
    for i, (name, comp) in enumerate(comparison.items()):
        values = [comp[m] for m in metrics3]
        ax5.bar(x3 + i*width, values, width, label=labels[name],
               color=colors[name], alpha=0.8)
    
    ax5.set_ylabel('Value', fontweight='bold')
    ax5.set_xticks(x3 + width)
    ax5.set_xticklabels(metric_labels3)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_title('Computational Performance', fontweight='bold')
    
    # 6. 改进百分比雷达图
    ax6 = plt.subplot(3, 2, 6, projection='polar')
    
    # 计算智能MPC相对传统MPC的改进
    trad = comparison['traditional']
    intl = comparison['intelligent']
    
    improvements = [
        (1 - intl['rmse'] / trad['rmse']) * 100,
        (1 - intl['max_deviation'] / trad['max_deviation']) * 100,
        (1 - intl['avg_flow_change'] / trad['avg_flow_change']) * 100,
        (intl['avg_delivery'] / trad['avg_delivery'] - 1) * 100,
        (1 - intl['violations'] / max(1, trad['violations'])) * 100
    ]
    
    categories = ['RMSE', 'Max Dev', 'Flow\nChange', 'Delivery', 'Violations']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    improvements += [improvements[0]]
    angles += [angles[0]]
    
    ax6.plot(angles, improvements, 'o-', linewidth=2, color='green')
    ax6.fill(angles, improvements, alpha=0.25, color='green')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, max(improvements) * 1.2)
    ax6.set_title('Improvement vs Traditional (%)', fontweight='bold', pad=20)
    ax6.grid(True)
    
    plt.suptitle(f'Performance Comparison: {test_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'/workspace/performance_comparison_{test_name.replace(" ", "_")}.png', 
               dpi=150, bbox_inches='tight')
    logger.info(f"\n  📊 对比图表已保存: performance_comparison_{test_name.replace(' ', '_')}.png")
    
    plt.close()


if __name__ == "__main__":
    logger.info("="*80)
    logger.info(" "*25 + "性能对比分析")
    logger.info("="*80)
    
    # 生成测试场景（简化版，48小时）
    logger.info(f"\n生成测试场景...")
    
    scenario_plan = []
    for h in range(48):
        hour_of_day = h % 24
        
        # 场景设计
        if 20 <= h < 30:
            # 洪峰场景
            demand = 5.0 + np.random.normal(0, 0.3)
            weather = {'rainfall': 20.0 + (h-20)*2}
            alerts = ['洪水预警'] if h < 25 else ['洪峰到达']
        elif 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
            # 高峰
            demand = 10.0 + np.random.normal(0, 0.5)
            weather = None
            alerts = []
        else:
            # 正常
            demand = 6.0 + np.random.normal(0, 0.4)
            weather = None
            alerts = []
        
        scenario_plan.append({
            'demand': demand,
            'weather': weather,
            'alerts': alerts
        })
    
    logger.info(f"  ✓ {len(scenario_plan)}小时场景已生成")
    
    # 运行对比测试
    results, comparison = run_comparison_test(scenario_plan, "48h_Complex_Scenario")
    
    logger.info(f"\n{'='*80}")
    logger.info("对比分析完成！")
    logger.info('='*80)

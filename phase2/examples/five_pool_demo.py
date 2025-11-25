"""
五池级联系统演示
测试大规模协同控制
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from models.cascaded_system import CascadedCanalSystem
from controllers.improved_admm import ImprovedDistributedMPC, ADMMParameters
from controllers.multi_objective_mpc import MultiObjectiveMPC
from topology.network_topology import create_simple_cascade
import time


def run_five_pool_simulation(total_hours: int = 50):
    """运行五池级联系统仿真"""
    
    print("="*70)
    print(" "*20 + "五池级联系统仿真演示")
    print("="*70)
    
    # 1. 初始化系统
    print("\n1. 初始化系统...")
    num_pools = 5
    system = CascadedCanalSystem(num_pools=num_pools, pool_area=10000.0, dt=3600.0)
    
    # 使用改进的ADMM控制器
    admm_params = ADMMParameters(
        rho=1.5,
        alpha=1.6,
        adaptive_rho=True,
        max_iterations=20,
        tolerance=1e-3
    )
    controller = ImprovedDistributedMPC(num_pools=num_pools, horizon=10, params=admm_params)
    
    # 创建拓扑
    topology = create_simple_cascade(num_pools=num_pools)
    print(f"   渠池数量: {num_pools}")
    print(f"   闸门数量: {len(system.gates)}")
    print(f"   MPC时域: {controller.horizon}")
    print(f"   ADMM参数: rho={admm_params.rho}, alpha={admm_params.alpha}")
    
    # 可视化拓扑
    print("\n2. 生成网络拓扑图...")
    topology.visualize('five_pool_topology.png', figsize=(16, 10))
    
    # 3. 生成复杂需求场景
    print("\n3. 生成复杂需求场景...")
    np.random.seed(42)
    base_demand = 5.0
    demands = np.zeros(total_hours + 20)
    
    for t in range(len(demands)):
        # 基础需求（日周期）
        hour = t % 24
        if 6 <= hour <= 10:
            # 早高峰
            demands[t] = base_demand + 3.0 + np.random.normal(0, 0.3)
        elif 17 <= hour <= 21:
            # 晚高峰
            demands[t] = base_demand + 2.5 + np.random.normal(0, 0.3)
        elif 0 <= hour <= 5:
            # 夜间低谷
            demands[t] = base_demand - 2.0 + np.random.normal(0, 0.2)
        else:
            # 正常
            demands[t] = base_demand + np.random.normal(0, 0.4)
        
        demands[t] = max(0.5, demands[t])  # 确保非负
    
    # 添加突发事件
    demands[15:20] += 4.0  # 突发高需求
    demands[35:40] -= 2.0  # 维护期低需求
    
    print(f"   基础需求: {base_demand:.2f} m³/s")
    print(f"   需求范围: [{np.min(demands):.2f}, {np.max(demands):.2f}] m³/s")
    
    # 4. 仿真循环
    print(f"\n4. 运行仿真 ({total_hours}小时)...")
    
    history = {
        'time': [],
        'pool_levels': [[] for _ in range(num_pools)],
        'pool_inflows': [[] for _ in range(num_pools)],
        'pool_outflows': [[] for _ in range(num_pools)],
        'demand': [],
        'admm_iterations': [],
        'admm_solve_time': [],
        'admm_converged': []
    }
    
    total_solve_time = 0
    total_iterations = 0
    
    for t in range(total_hours):
        # 获取当前状态
        current_levels = [pool.level for pool in system.pools]
        q_in_prevs = [pool.inflow_history[-1] for pool in system.pools]
        
        # 预测未来需求
        demand_forecast = demands[t:t+controller.horizon].tolist()
        q_out_forecasts = [demand_forecast] * num_pools
        
        # MPC求解
        solutions, info = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        
        # 提取控制动作
        control_actions = []
        for q_in, q_out in solutions:
            control_actions.append(q_in / 20.0)  # 归一化
        control_actions.append(demands[t] / 20.0)  # 最后一个闸门
        
        # 执行一步仿真
        state = system.step(control_actions, demand=demands[t])
        
        # 记录数据
        history['time'].append(t)
        for i, pool_state in enumerate(state['pools']):
            history['pool_levels'][i].append(pool_state['level'])
            history['pool_inflows'][i].append(pool_state['inflow'])
            history['pool_outflows'][i].append(pool_state['outflow'])
        history['demand'].append(demands[t])
        history['admm_iterations'].append(info['iterations'])
        history['admm_solve_time'].append(info['solve_time'])
        history['admm_converged'].append(info['converged'])
        
        total_solve_time += info['solve_time']
        total_iterations += info['iterations']
        
        # 进度显示
        if (t + 1) % 10 == 0:
            avg_iter = total_iterations / (t + 1)
            avg_time = total_solve_time / (t + 1) * 1000  # ms
            print(f"   进度: {t+1}/{total_hours}h | "
                  f"平均迭代: {avg_iter:.1f} | "
                  f"平均时间: {avg_time:.1f}ms", end='\r')
    
    print(f"\n   仿真完成！")
    
    # 5. 分析结果
    print("\n5. 性能分析...")
    print("\n   5.1 水位跟踪性能:")
    for i in range(num_pools):
        levels = np.array(history['pool_levels'][i])
        target = 3.0
        rmse = np.sqrt(np.mean((levels - target)**2))
        max_dev = np.max(np.abs(levels - target))
        std = np.std(levels)
        print(f"      池{i}: RMSE={rmse:.4f}m, 最大偏差={max_dev:.4f}m, 标准差={std:.4f}m")
    
    print("\n   5.2 流量协调性能:")
    for i in range(num_pools - 1):
        q_out_i = np.array(history['pool_outflows'][i])
        q_in_i1 = np.array(history['pool_inflows'][i + 1])
        balance_error = np.mean(np.abs(q_out_i - q_in_i1))
        max_error = np.max(np.abs(q_out_i - q_in_i1))
        print(f"      池{i}→池{i+1}: 平均误差={balance_error:.4f} m³/s, "
              f"最大误差={max_error:.4f} m³/s")
    
    print("\n   5.3 ADMM求解性能:")
    avg_iterations = np.mean(history['admm_iterations'])
    max_iterations = np.max(history['admm_iterations'])
    avg_solve_time = np.mean(history['admm_solve_time']) * 1000  # ms
    max_solve_time = np.max(history['admm_solve_time']) * 1000
    convergence_rate = np.mean(history['admm_converged']) * 100
    
    print(f"      平均迭代次数: {avg_iterations:.1f}")
    print(f"      最大迭代次数: {max_iterations}")
    print(f"      平均求解时间: {avg_solve_time:.2f} ms")
    print(f"      最大求解时间: {max_solve_time:.2f} ms")
    print(f"      收敛率: {convergence_rate:.1f}%")
    
    print("\n   5.4 供水保证率:")
    demands_array = np.array(history['demand'])
    delivered = np.array(history['pool_outflows'][-1])
    delivery_error = np.abs(demands_array - delivered)
    guarantee_rate = np.mean(delivery_error < 0.5) * 100  # 误差<0.5认为满足
    print(f"      供水保证率: {guarantee_rate:.1f}%")
    print(f"      平均偏差: {np.mean(delivery_error):.4f} m³/s")
    print(f"      最大偏差: {np.max(delivery_error):.4f} m³/s")
    
    # 6. 可视化
    print("\n6. 生成可视化...")
    visualize_five_pool_results(history, num_pools)
    
    print("\n" + "="*70)
    print("演示完成！结果已保存")
    print("="*70)
    
    return history


def visualize_five_pool_results(history, num_pools):
    """可视化五池系统结果"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 颜色方案
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 子图1: 水位
    ax1 = plt.subplot(3, 2, 1)
    for i in range(num_pools):
        ax1.plot(history['time'], history['pool_levels'][i], 
                label=f'池{i}', linewidth=2, color=colors[i], alpha=0.8)
    ax1.axhline(y=3.0, color='black', linestyle='--', label='目标', alpha=0.5)
    ax1.set_ylabel('水位 (m)', fontsize=11)
    ax1.set_title('水位控制', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 流量
    ax2 = plt.subplot(3, 2, 2)
    for i in range(num_pools):
        ax2.plot(history['time'], history['pool_inflows'][i],
                label=f'池{i}入流', linewidth=1.5, color=colors[i], alpha=0.7)
    ax2.plot(history['time'], history['demand'], 
            label='需求', linewidth=2, color='black', linestyle=':', alpha=0.6)
    ax2.set_ylabel('流量 (m³/s)', fontsize=11)
    ax2.set_title('流量分布', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: ADMM迭代次数
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(history['time'], history['admm_iterations'], 
            linewidth=2, color='#e74c3c', alpha=0.7)
    ax3.axhline(y=np.mean(history['admm_iterations']), 
               color='blue', linestyle='--', label=f'平均={np.mean(history["admm_iterations"]):.1f}')
    ax3.set_ylabel('迭代次数', fontsize=11)
    ax3.set_title('ADMM收敛速度', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 求解时间
    ax4 = plt.subplot(3, 2, 4)
    solve_times_ms = np.array(history['admm_solve_time']) * 1000
    ax4.plot(history['time'], solve_times_ms, 
            linewidth=2, color='#9b59b6', alpha=0.7)
    ax4.axhline(y=np.mean(solve_times_ms), 
               color='green', linestyle='--', label=f'平均={np.mean(solve_times_ms):.1f}ms')
    ax4.axhline(y=500, color='red', linestyle=':', label='目标<500ms', alpha=0.5)
    ax4.set_ylabel('求解时间 (ms)', fontsize=11)
    ax4.set_title('计算性能', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 水位偏差
    ax5 = plt.subplot(3, 2, 5)
    for i in range(num_pools):
        deviations = np.abs(np.array(history['pool_levels'][i]) - 3.0)
        ax5.plot(history['time'], deviations, 
                label=f'池{i}', linewidth=1.5, color=colors[i], alpha=0.7)
    ax5.set_xlabel('时间 (小时)', fontsize=11)
    ax5.set_ylabel('水位偏差 (m)', fontsize=11)
    ax5.set_title('水位偏差分析', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 供水偏差
    ax6 = plt.subplot(3, 2, 6)
    delivery_error = np.abs(np.array(history['demand']) - 
                           np.array(history['pool_outflows'][-1]))
    ax6.plot(history['time'], delivery_error, 
            linewidth=2, color='#e74c3c', alpha=0.7)
    ax6.axhline(y=0.5, color='orange', linestyle='--', 
               label='可接受阈值=0.5 m³/s', alpha=0.6)
    ax6.set_xlabel('时间 (小时)', fontsize=11)
    ax6.set_ylabel('供水偏差 (m³/s)', fontsize=11)
    ax6.set_title('供水保证分析', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('five_pool_results.png', dpi=150, bbox_inches='tight')
    print("   图表已保存: five_pool_results.png")
    plt.close()


if __name__ == "__main__":
    try:
        history = run_five_pool_simulation(total_hours=50)
    except KeyboardInterrupt:
        print("\n\n仿真被用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()

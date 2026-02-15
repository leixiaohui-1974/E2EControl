"""
三池级联系统演示
展示分布式MPC控制效果
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

try:
    from models.cascaded_system import CascadedCanalSystem
    from controllers.distributed_mpc import DistributedMPCController
except ImportError:
    from phase2.models.cascaded_system import CascadedCanalSystem
    from phase2.controllers.distributed_mpc import DistributedMPCController


def run_cascaded_simulation(total_hours: int = 50):
    """运行级联系统仿真"""
    
    logger.info("="*70)
    logger.info(" "*20 + "三池级联系统仿真演示")
    logger.info("="*70)
    
    # 1. 初始化系统
    logger.info("\n1. 初始化系统...")
    system = CascadedCanalSystem(num_pools=3, pool_area=10000.0, dt=3600.0)
    controller = DistributedMPCController(num_pools=3, horizon=10, dt=3600.0)
    
    logger.info("   渠池数量: %d", system.num_pools)
    logger.info("   闸门数量: %d", len(system.gates))
    logger.info("   MPC时域: %d", controller.horizon)
    
    # 2. 生成需求场景
    logger.info("\n2. 生成需求场景...")
    np.random.seed(42)
    base_demand = 5.0
    demands = base_demand + np.random.normal(0, 0.5, total_hours + 20)
    
    # 模拟需求变化
    demands[10:20] += 2.0  # 高峰期
    demands[30:40] -= 1.0  # 低谷期
    
    logger.info("   基础需求: %.2f m³/s", base_demand)
    logger.info("   需求波动: ±0.50 m³/s")
    
    # 3. 仿真循环
    logger.info("3. 运行仿真 (%d小时)...", total_hours)
    
    history = {
        'time': [],
        'pool_levels': [[] for _ in range(system.num_pools)],
        'pool_inflows': [[] for _ in range(system.num_pools)],
        'pool_outflows': [[] for _ in range(system.num_pools)],
        'demand': []
    }
    
    for t in range(total_hours):
        # 获取当前状态
        current_levels = [pool.level for pool in system.pools]
        q_in_prevs = [pool.inflow_history[-1] for pool in system.pools]
        
        # 预测未来需求
        demand_forecast = demands[t:t+controller.horizon].tolist()
        q_out_forecasts = [demand_forecast] * system.num_pools
        
        # MPC求解
        solutions = controller.solve(current_levels, q_in_prevs, q_out_forecasts)
        
        # 提取控制动作（闸门开度）
        # 这里简化：根据流量计算开度
        control_actions = []
        for q_in, q_out in solutions:
            # 入流闸门开度
            control_actions.append(q_in / 20.0)  # 归一化到0-1
        # 最后一个闸门（系统出口）
        control_actions.append(demands[t] / 20.0)
        
        # 执行一步仿真
        state = system.step(control_actions, demand=demands[t])
        
        # 记录数据
        history['time'].append(t)
        for i, pool_state in enumerate(state['pools']):
            history['pool_levels'][i].append(pool_state['level'])
            history['pool_inflows'][i].append(pool_state['inflow'])
            history['pool_outflows'][i].append(pool_state['outflow'])
        history['demand'].append(demands[t])
        
        # 进度显示
        if (t + 1) % 10 == 0:
            logger.info("   进度: %d/%dh", t + 1, total_hours)

    logger.info("   仿真完成！")
    
    # 4. 分析结果
    logger.info("\n4. 分析结果...")
    
    for i in range(system.num_pools):
        levels = np.array(history['pool_levels'][i])
        target = 3.0
        rmse = np.sqrt(np.mean((levels - target)**2))
        max_dev = np.max(np.abs(levels - target))
        logger.info("   池%d: RMSE=%.4fm, 最大偏差=%.4fm", i, rmse, max_dev)
    
    # 检查流量平衡
    for i in range(system.num_pools - 1):
        q_out_i = np.array(history['pool_outflows'][i])
        q_in_i1 = np.array(history['pool_inflows'][i + 1])
        balance_error = np.mean(np.abs(q_out_i - q_in_i1))
        logger.info("   池%d→池%d 流量平衡误差: %.4f m³/s", i, i + 1, balance_error)
    
    # 5. 可视化
    logger.info("\n5. 生成可视化...")
    visualize_results(history, system.num_pools)
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！结果已保存为 cascaded_system_result.png")
    logger.info("="*70)


def visualize_results(history, num_pools):
    """可视化结果"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 子图1: 水位
    ax1 = axes[0]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i in range(num_pools):
        ax1.plot(history['time'], history['pool_levels'][i], 
                label=f'池{i}水位', linewidth=2, color=colors[i])
    
    ax1.axhline(y=3.0, color='black', linestyle='--', label='目标水位', alpha=0.5)
    ax1.set_ylabel('水位 (m)', fontsize=12)
    ax1.set_title('级联渠道系统 - 水位控制', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 流量
    ax2 = axes[1]
    
    for i in range(num_pools):
        ax2.plot(history['time'], history['pool_inflows'][i],
                label=f'池{i}入流', linewidth=2, color=colors[i], alpha=0.7)
    
    ax2.plot(history['time'], history['demand'], 
            label='需求', linewidth=2, color='black', linestyle=':', alpha=0.6)
    
    ax2.set_xlabel('时间 (小时)', fontsize=12)
    ax2.set_ylabel('流量 (m³/s)', fontsize=12)
    ax2.set_title('流量分布', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cascaded_system_result.png', dpi=150)
    logger.info("   图表已保存: cascaded_system_result.png")


if __name__ == "__main__":
    try:
        run_cascaded_simulation(total_hours=50)
    except KeyboardInterrupt:
        logger.info("\n\n仿真被用户中断")
    except Exception as e:
        logger.error("错误: %s", e)
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
分布式SIL框架演示

演示如何使用分布式SIL框架进行:
1. 稳态场景仿真
2. 阶跃响应测试
3. 洪水场景测试
4. 集合仿真(鲁棒性测试)

Usage:
    python demo_sil_simulation.py
"""

import sys
import os
import numpy as np
import logging
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from phase5.distributed_sil import (
    DistributedSILFramework,
    ScenarioGenerator,
)
from phase5.distributed_sil.core.sil_framework import SILConfig, SimulationMode
from phase5.distributed_sil.core.scenario_generator import ScenarioType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_steady_state():
    """演示稳态仿真"""
    print("\n" + "="*60)
    print("演示1: 稳态场景仿真")
    print("="*60)

    # 配置
    config = SILConfig(
        num_segments=20,           # 20个渠段
        num_gates=21,              # 21个闸门
        dt_reduced=900.0,          # 15分钟时间步
        simulation_mode=SimulationMode.HYBRID,
        high_fidelity_segments=[0, 5, 10, 15, 19],  # 关键闸站
    )

    # 创建框架
    framework = DistributedSILFramework(config=config)

    # 加载稳态场景
    scenario = framework.load_scenario(
        scenario_type=ScenarioType.STEADY_STATE,
        duration=7200.0,           # 2小时
        base_flow=300.0,           # 基准流量 300 m³/s
        base_level=4.0,            # 基准水位 4.0 m
    )

    print(f"场景ID: {scenario.scenario_id}")
    print(f"持续时间: {scenario.duration/3600:.1f} 小时")

    # 运行仿真
    def progress_callback(progress, result):
        if result.get("kpi"):
            score = result["kpi"].overall_score
            print(f"进度: {progress*100:.1f}%, 评分: {score:.3f}")

    result = framework.run_simulation(scenario, progress_callback)

    # 输出结果
    print(f"\n仿真结果:")
    print(f"  通过: {result.passed}")
    print(f"  最终评分: {result.final_kpi.overall_score:.3f}")
    if result.failure_reasons:
        print(f"  失败原因: {result.failure_reasons}")

    # 生成报告
    report = framework.generate_report()
    print(f"\n评估统计:")
    print(f"  通过率: {report['evaluation_report']['summary']['pass_rate']*100:.1f}%")

    return result


def demo_step_response():
    """演示阶跃响应测试"""
    print("\n" + "="*60)
    print("演示2: 阶跃响应测试")
    print("="*60)

    config = SILConfig(
        num_segments=10,
        num_gates=11,
        dt_reduced=300.0,          # 5分钟时间步 (更高分辨率)
        simulation_mode=SimulationMode.HYBRID,
        high_fidelity_segments=[0, 5, 9],
    )

    framework = DistributedSILFramework(config=config)

    # 阶跃场景
    scenario = framework.load_scenario(
        scenario_type=ScenarioType.STEP_CHANGE,
        duration=14400.0,          # 4小时
        base_flow=300.0,
        step_time=3600.0,          # 1小时后阶跃
        step_magnitude=30.0,       # 流量增加 30 m³/s
    )

    print(f"场景: 在1小时后流量从300增加到330 m³/s")

    result = framework.run_simulation(scenario)

    print(f"\n仿真结果:")
    print(f"  通过: {result.passed}")
    print(f"  最终评分: {result.final_kpi.overall_score:.3f}")

    # 分析控制性能
    report = framework.generate_report()
    ctrl_stats = report['controller_statistics']
    print(f"\n控制器统计:")
    print(f"  总指令数: {ctrl_stats['total_commands']}")
    print(f"  活跃控制器: {ctrl_stats['active_controllers']}")

    return result


def demo_flood_scenario():
    """演示洪水场景测试"""
    print("\n" + "="*60)
    print("演示3: 洪水场景测试")
    print("="*60)

    config = SILConfig(
        num_segments=15,
        num_gates=16,
        dt_reduced=600.0,          # 10分钟时间步
        simulation_mode=SimulationMode.HYBRID,
        high_fidelity_segments=[0, 7, 14],
    )

    framework = DistributedSILFramework(config=config)

    # 洪水场景
    scenario = framework.load_scenario(
        scenario_type=ScenarioType.FLOOD,
        duration=21600.0,          # 6小时
        base_flow=300.0,
        peak_factor=1.5,           # 峰值为基准的1.5倍
        peak_time=10800.0,         # 3小时达到峰值
    )

    print(f"场景: 洪水过程,峰值流量450 m³/s")

    result = framework.run_simulation(scenario)

    print(f"\n仿真结果:")
    print(f"  通过: {result.passed}")
    print(f"  最终评分: {result.final_kpi.overall_score:.3f}")
    print(f"  约束违背数: {result.final_kpi.constraint_violation_count}")

    return result


def demo_ensemble_simulation():
    """演示集合仿真(参数不确定性测试)"""
    print("\n" + "="*60)
    print("演示4: 集合仿真 (鲁棒性测试)")
    print("="*60)

    config = SILConfig(
        num_segments=10,
        num_gates=11,
        dt_reduced=900.0,
        simulation_mode=SimulationMode.REDUCED_ORDER_ONLY,  # 快速模式
        high_fidelity_segments=[],
    )

    framework = DistributedSILFramework(config=config)

    # 基础场景
    base_scenario = framework.load_scenario(
        scenario_type=ScenarioType.STEADY_STATE,
        duration=7200.0,
        base_flow=300.0,
    )

    print(f"运行5个集合成员...")

    # 运行集合仿真
    results = framework.run_ensemble_simulation(
        base_scenario=base_scenario,
        ensemble_size=5,
    )

    # 统计
    pass_count = sum(1 for r in results if r.passed)
    scores = [r.final_kpi.overall_score for r in results]

    print(f"\n集合仿真结果:")
    print(f"  通过数: {pass_count}/5")
    print(f"  平均评分: {np.mean(scores):.3f}")
    print(f"  评分范围: [{min(scores):.3f}, {max(scores):.3f}]")

    return results


def demo_manual_step_simulation():
    """演示手动逐步仿真"""
    print("\n" + "="*60)
    print("演示5: 手动逐步仿真")
    print("="*60)

    config = SILConfig(
        num_segments=5,
        num_gates=6,
        dt_reduced=900.0,
        simulation_mode=SimulationMode.HYBRID,
        high_fidelity_segments=[0, 2, 4],
    )

    framework = DistributedSILFramework(config=config)
    framework.reset(initial_level=4.0, initial_flow=300.0)

    print("执行10个仿真步...")

    # 手动执行10步
    for step in range(10):
        # 可以动态改变上游流量
        upstream_flow = 300.0 + 10.0 * np.sin(step * 0.5)

        result = framework.step(upstream_flow=upstream_flow)

        if step % 3 == 0:
            state_summary = {
                "time": result["time"],
                "upstream_flow": upstream_flow,
                "num_states": len(result["states"]),
            }
            print(f"  步骤 {step}: {state_summary}")

    # 获取最终报告
    report = framework.generate_report()
    print(f"\n仿真统计:")
    print(f"  总步数: {report['simulation_summary']['total_steps']}")
    print(f"  仿真时间: {report['simulation_summary']['total_time']/3600:.2f} 小时")


def main():
    """主函数"""
    print("="*60)
    print("南水北调中线 分布式SIL框架 演示")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行各演示
    try:
        demo_steady_state()
        demo_step_response()
        demo_flood_scenario()
        demo_ensemble_simulation()
        demo_manual_step_simulation()

        print("\n" + "="*60)
        print("所有演示完成!")
        print("="*60)

    except Exception as e:
        logger.error(f"演示出错: {e}")
        raise


if __name__ == "__main__":
    main()

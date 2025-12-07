#!/usr/bin/env python3
"""
南水北调中线全线全场景自主运行系统演示
Water Transfer Autonomous System Demo

本脚本演示系统的核心功能:
1. 闭环仿真环境创建
2. 场景注入与控制
3. 级联控制与上报
4. 性能分析与报告生成
"""

import sys
import time


def demo_basic_simulation():
    """基础仿真演示"""
    print("=" * 60)
    print("  演示1: 基础闭环仿真")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
    )
    from .visualization import print_simulation_summary

    # 创建仿真配置
    config = SimulationConfig(
        num_pools=20,           # 20个渠池
        dt=60.0,                # 1分钟步长
        total_duration=1800.0,  # 30分钟仿真
        enable_cascade=True,    # 启用级联控制
    )

    # 创建闭环仿真
    sim = ClosedLoopSimulation(config)

    print(f"\n创建仿真环境:")
    print(f"  - 渠池数量: {config.num_pools}")
    print(f"  - 仿真时长: {config.total_duration/60:.0f} 分钟")
    print(f"  - 级联控制: {'启用' if config.enable_cascade else '禁用'}")

    # 运行仿真
    print("\n运行仿真...")
    start = time.time()
    result = sim.run()
    elapsed = time.time() - start

    print(f"\n仿真完成! 耗时: {elapsed:.2f}秒")

    # 打印摘要
    print("\n" + "-" * 60)
    print_simulation_summary(result)

    return sim, result


def demo_scenario_injection():
    """场景注入演示"""
    print("\n" + "=" * 60)
    print("  演示2: 场景注入与响应")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
        ScenarioInjectionPlan,
        L1ScenarioType,
        L1ScenarioEvent,
        ScenarioSeverity,
    )
    from .visualization import ReportGenerator

    # 创建仿真
    config = SimulationConfig(
        num_pools=30,
        dt=60.0,
        total_duration=3600.0,  # 1小时
    )
    sim = ClosedLoopSimulation(config)

    # 创建场景注入计划
    plan = ScenarioInjectionPlan(plan_id="DEMO_PLAN")

    scenarios = [
        # 5分钟后: 池10水位快速上涨
        (300, 10, L1ScenarioType.L1_LEVEL_RAPID_RISE, ScenarioSeverity.MEDIUM),
        # 10分钟后: 池15污染检测
        (600, 15, L1ScenarioType.L1_POLLUTION_DETECTED, ScenarioSeverity.HIGH),
        # 15分钟后: 池20闸门卡住
        (900, 20, L1ScenarioType.L1_GATE_STUCK, ScenarioSeverity.HIGH),
        # 25分钟后: 池5紧急退水
        (1500, 5, L1ScenarioType.L1_DISCHARGE_EMERGENCY, ScenarioSeverity.CRITICAL),
    ]

    for inject_time, pool_id, sc_type, severity in scenarios:
        scenario = L1ScenarioEvent(
            event_id=f"DEMO_SC_{pool_id}",
            scenario_type=sc_type,
            pool_id=pool_id,
            severity=severity,
        )
        plan.add_scenario(inject_time, pool_id, scenario)

    sim.set_scenario_plan(plan)

    print(f"\n注入计划:")
    for inject_time, pool_id, sc_type, severity in scenarios:
        print(f"  {inject_time/60:.0f}分钟: 池{pool_id} - {sc_type.value}")

    # 运行仿真
    print("\n运行仿真...")
    result = sim.run()

    # 生成场景报告
    gen = ReportGenerator()
    report = gen.generate_scenario_report(sim)
    print("\n" + report.summary)

    return sim, result


def demo_performance_analysis():
    """性能分析演示"""
    print("\n" + "=" * 60)
    print("  演示3: 性能分析")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
    )
    from .visualization import ReportGenerator

    # 创建带扰动的仿真
    config = SimulationConfig(
        num_pools=40,
        dt=60.0,
        total_duration=2400.0,  # 40分钟
    )
    sim = ClosedLoopSimulation(config)

    # 随机场景
    plan = sim.scenario_injector.generate_random_plan(
        num_scenarios=8,
        duration=2000.0,
    )
    sim.set_scenario_plan(plan)

    print(f"\n生成随机场景: {len(plan.scenarios)} 个")

    # 运行
    print("运行仿真...")
    result = sim.run()

    # 性能分析报告
    gen = ReportGenerator()
    report = gen.generate_performance_report(sim.simulator)
    print("\n" + report.summary)

    return sim, result


def demo_cascade_control():
    """级联控制演示"""
    print("\n" + "=" * 60)
    print("  演示4: 级联控制与上报")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
        ScenarioInjectionPlan,
        L1ScenarioType,
        L1ScenarioEvent,
        ScenarioSeverity,
    )
    from .visualization import ReportGenerator

    # 创建仿真
    config = SimulationConfig(
        num_pools=25,
        dt=60.0,
        total_duration=1800.0,
        enable_cascade=True,
    )
    sim = ClosedLoopSimulation(config)

    # 注入严重场景触发级联上报
    plan = ScenarioInjectionPlan(plan_id="CASCADE_DEMO")

    # 注入多个严重场景
    critical_scenarios = [
        (60, 5, L1ScenarioType.L1_GATE_STUCK, ScenarioSeverity.CRITICAL),
        (120, 10, L1ScenarioType.L1_DISCHARGE_EMERGENCY, ScenarioSeverity.CRITICAL),
        (180, 15, L1ScenarioType.L1_POLLUTION_TRACKING, ScenarioSeverity.HIGH),
    ]

    for inject_time, pool_id, sc_type, severity in critical_scenarios:
        scenario = L1ScenarioEvent(
            event_id=f"CASC_{pool_id}",
            scenario_type=sc_type,
            pool_id=pool_id,
            severity=severity,
        )
        plan.add_scenario(inject_time, pool_id, scenario)

    sim.set_scenario_plan(plan)

    print(f"\n注入严重场景: {len(critical_scenarios)} 个")

    # 运行
    print("运行仿真...")
    result = sim.run()

    # 控制效果报告
    gen = ReportGenerator()
    report = gen.generate_control_report(sim)
    print("\n" + report.summary)

    print(f"\n级联控制统计:")
    print(f"  - 总上报次数: {result['control']['total_escalations']}")
    print(f"  - 总干预次数: {result['control']['total_interventions']}")

    return sim, result


def demo_dashboard():
    """系统仪表板演示"""
    print("\n" + "=" * 60)
    print("  演示5: 系统仪表板")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
    )
    from .visualization import print_dashboard

    # 创建仿真
    config = SimulationConfig(
        num_pools=50,
        dt=60.0,
        total_duration=1200.0,  # 20分钟
    )
    sim = ClosedLoopSimulation(config)

    # 添加一些场景
    plan = sim.scenario_injector.generate_random_plan(
        num_scenarios=5,
        duration=1000.0,
    )
    sim.set_scenario_plan(plan)

    # 运行
    print("运行仿真...")
    sim.run()

    # 显示仪表板
    print()
    print_dashboard(sim)

    return sim


def demo_full_report():
    """完整报告演示"""
    print("\n" + "=" * 60)
    print("  演示6: 综合报告生成")
    print("=" * 60)

    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
    )
    from .visualization import print_full_report

    # 创建完整仿真
    config = SimulationConfig(
        num_pools=30,
        dt=60.0,
        total_duration=3600.0,  # 1小时
    )
    sim = ClosedLoopSimulation(config)

    # 添加场景
    plan = sim.scenario_injector.generate_random_plan(
        num_scenarios=10,
        duration=3400.0,
    )
    sim.set_scenario_plan(plan)

    # 运行
    print("运行仿真...")
    result = sim.run()

    # 完整报告
    print("\n生成综合报告...\n")
    print_full_report(sim, result)

    return sim, result


def run_all_demos():
    """运行所有演示"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "南水北调中线全线全场景自主运行系统".center(42) + "║")
    print("║" + "系统功能演示".center(50) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    demos = [
        ("基础仿真", demo_basic_simulation),
        ("场景注入", demo_scenario_injection),
        ("性能分析", demo_performance_analysis),
        ("级联控制", demo_cascade_control),
        ("系统仪表板", demo_dashboard),
    ]

    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, "成功", result))
        except Exception as e:
            results.append((name, f"失败: {e}", None))

    # 总结
    print("\n" + "=" * 60)
    print("  演示总结")
    print("=" * 60)
    for name, status, _ in results:
        icon = "✓" if "成功" in status else "✗"
        print(f"  {icon} {name}: {status}")

    return results


def quick_demo():
    """快速演示 (用于测试)"""
    from . import (
        SimulationConfig,
        ClosedLoopSimulation,
    )
    from .visualization import print_simulation_summary

    config = SimulationConfig(
        num_pools=10,
        dt=60.0,
        total_duration=300.0,
    )
    sim = ClosedLoopSimulation(config)

    plan = sim.scenario_injector.generate_random_plan(
        num_scenarios=2,
        duration=250.0,
    )
    sim.set_scenario_plan(plan)

    result = sim.run()
    print_simulation_summary(result)

    return sim, result


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        demo_full_report()
    else:
        run_all_demos()

"""
优化调度与水量分配系统测试
Test suite for Optimization Scheduling and Water Allocation System
"""

from hydroe2e.phase5.water_transfer_system.optimization_scheduler import (
    OptimizationObjective, AllocationStrategy, ScheduleType, ConstraintType,
    WaterDemand, WaterSupply, OptimizationConstraint, ScheduleSlot, OptimizationResult,
    ObjectiveFunction, ConstraintHandler, MultiObjectiveOptimizer,
    WaterAllocator, ScheduleGenerator, OptimizationSchedulingSystem,
)


def test_data_types():
    """测试数据类型定义"""
    print("测试1: 数据类型定义...")

    # 优化目标
    assert OptimizationObjective.LEVEL_TRACKING.value > 0
    assert OptimizationObjective.WATER_DELIVERY.value > 0

    # 分配策略
    assert AllocationStrategy.PRIORITY.value > 0
    assert AllocationStrategy.PROPORTIONAL.value > 0

    # 调度类型
    assert ScheduleType.DAILY.value > 0
    assert ScheduleType.EMERGENCY.value > 0

    # 约束类型
    assert ConstraintType.LEVEL_BOUND.value > 0
    assert ConstraintType.FLOW_BOUND.value > 0

    print("  ✓ 数据类型定义正确")
    return True


def test_water_demand():
    """测试用水需求数据结构"""
    print("测试2: 用水需求数据结构...")

    demand = WaterDemand(
        demand_id="D001",
        location="outlet_1",
        pool_id=2,
        volume=10000.0,
        flow_rate=5.0,
        priority=2,
        start_time=8.0,
        end_time=20.0,
        flexibility=0.3,
        min_flow=2.0,
        category="agricultural",
    )

    assert demand.demand_id == "D001"
    assert demand.volume == 10000.0
    assert demand.priority == 2
    assert demand.flexibility == 0.3

    print("  ✓ 用水需求数据结构正确")
    return True


def test_objective_function():
    """测试目标函数"""
    print("测试3: 目标函数...")

    obj = ObjectiveFunction()

    # 设置目标
    obj.set_targets(
        levels={0: 2.5, 1: 2.5, 2: 2.5},
        flows={0: 10.0, 1: 10.0, 2: 10.0}
    )

    # 评估 - 完美状态
    state = {
        'levels': {0: 2.5, 1: 2.5, 2: 2.5},
        'flows': {0: 10.0, 1: 10.0, 2: 10.0},
        'flow_changes': {},
        'gate_movements': {},
        'deliveries': {},
        'demands': {},
        'storages': {},
        'level_limits': {0: (1.0, 4.0), 1: (1.0, 4.0), 2: (1.0, 4.0)},
    }

    cost = obj.evaluate(state)
    assert cost >= 0, "目标函数值应非负"

    # 评估 - 偏离状态
    state2 = state.copy()
    state2['levels'] = {0: 3.0, 1: 2.0, 2: 2.5}  # 偏离目标

    cost2 = obj.evaluate(state2)
    assert cost2 > cost, "偏离状态成本应更高"

    print("  ✓ 目标函数正确")
    return True


def test_constraint_handler():
    """测试约束处理器"""
    print("测试4: 约束处理器...")

    handler = ConstraintHandler()

    # 添加约束
    handler.add_constraint(OptimizationConstraint(
        constraint_id="C1",
        constraint_type=ConstraintType.LEVEL_BOUND,
        target="0",
        lower_bound=1.0,
        upper_bound=4.0,
    ))

    handler.add_constraint(OptimizationConstraint(
        constraint_id="C2",
        constraint_type=ConstraintType.FLOW_BOUND,
        target="0",
        lower_bound=0.0,
        upper_bound=100.0,
    ))

    # 可行状态
    state = {
        'levels': {0: 2.5},
        'flows': {0: 50.0},
    }

    feasible, violations = handler.check_feasibility(state)
    assert feasible, "应该是可行的"
    assert len(violations) == 0

    # 违反状态
    state2 = {
        'levels': {0: 5.0},  # 超出上界
        'flows': {0: 50.0},
    }

    feasible2, violations2 = handler.check_feasibility(state2)
    assert not feasible2, "应该不可行"
    assert len(violations2) > 0

    # 计算惩罚
    penalty = handler.compute_penalty(state2)
    assert penalty > 0, "违反应有惩罚"

    print("  ✓ 约束处理器正确")
    return True


def test_constraint_projection():
    """测试约束投影"""
    print("测试5: 约束投影...")

    handler = ConstraintHandler()

    handler.add_constraint(OptimizationConstraint(
        constraint_id="C1",
        constraint_type=ConstraintType.LEVEL_BOUND,
        target="0",
        lower_bound=1.0,
        upper_bound=4.0,
    ))

    # 超出边界的状态
    state = {'level_0': 5.0}

    projected = handler.project_to_feasible(state)
    assert projected['level_0'] <= 4.0, "应投影到上界内"

    # 低于下界
    state2 = {'level_0': 0.5}
    projected2 = handler.project_to_feasible(state2)
    assert projected2['level_0'] >= 1.0, "应投影到下界上"

    print("  ✓ 约束投影正确")
    return True


def test_multi_objective_optimizer():
    """测试多目标优化器"""
    print("测试6: 多目标优化器...")

    optimizer = MultiObjectiveOptimizer(num_pools=3)

    # 添加约束
    for i in range(3):
        optimizer.constraint_handler.add_constraint(OptimizationConstraint(
            constraint_id=f"LEVEL_{i}",
            constraint_type=ConstraintType.LEVEL_BOUND,
            target=str(i),
            lower_bound=1.0,
            upper_bound=4.0,
        ))

    # 设置目标
    optimizer.objective.set_targets(
        levels={0: 2.5, 1: 2.5, 2: 2.5},
        flows={0: 10.0, 1: 10.0, 2: 10.0}
    )

    # 初始状态
    initial_state = {
        'levels': {0: 2.0, 1: 3.0, 2: 2.5},
        'flows': {0: 10.0, 1: 10.0, 2: 10.0},
        'gates': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        'level_limits': {i: (1.0, 4.0) for i in range(3)},
    }

    # 运行优化
    result = optimizer.optimize(initial_state, horizon=12)

    assert result is not None
    assert result.iterations > 0
    assert result.computation_time >= 0
    assert len(result.schedule) > 0

    print("  ✓ 多目标优化器正确")
    return True


def test_water_allocator_priority():
    """测试优先级分配"""
    print("测试7: 优先级分配...")

    allocator = WaterAllocator()
    allocator.set_strategy(AllocationStrategy.PRIORITY)

    # 添加需求
    allocator.add_demand(WaterDemand(
        demand_id="D1",
        location="out1",
        pool_id=1,
        volume=1000.0,
        flow_rate=5.0,
        priority=1,  # 最高优先级
        start_time=0,
        end_time=24,
    ))

    allocator.add_demand(WaterDemand(
        demand_id="D2",
        location="out2",
        pool_id=2,
        volume=2000.0,
        flow_rate=10.0,
        priority=3,
        start_time=0,
        end_time=24,
    ))

    allocator.add_demand(WaterDemand(
        demand_id="D3",
        location="out3",
        pool_id=3,
        volume=1500.0,
        flow_rate=7.0,
        priority=2,
        start_time=0,
        end_time=24,
    ))

    # 分配有限水量
    allocations = allocator.allocate(2500.0, current_time=12.0)

    # 高优先级应优先满足
    assert allocations.get("D1", 0) == 1000.0, "最高优先级应完全满足"
    assert allocations.get("D3", 0) == 1500.0, "第二优先级应完全满足"
    assert allocations.get("D2", 0) == 0.0, "最低优先级应无分配"

    print("  ✓ 优先级分配正确")
    return True


def test_water_allocator_proportional():
    """测试比例分配"""
    print("测试8: 比例分配...")

    allocator = WaterAllocator()
    allocator.set_strategy(AllocationStrategy.PROPORTIONAL)

    allocator.add_demand(WaterDemand(
        demand_id="D1",
        location="out1",
        pool_id=1,
        volume=1000.0,
        flow_rate=5.0,
        priority=1,
        start_time=0,
        end_time=24,
    ))

    allocator.add_demand(WaterDemand(
        demand_id="D2",
        location="out2",
        pool_id=2,
        volume=2000.0,
        flow_rate=10.0,
        priority=1,
        start_time=0,
        end_time=24,
    ))

    # 分配50%水量
    total_demand = 3000.0
    allocations = allocator.allocate(1500.0, current_time=12.0)

    # 比例分配
    assert abs(allocations.get("D1", 0) - 500.0) < 1, "应按比例分配"
    assert abs(allocations.get("D2", 0) - 1000.0) < 1, "应按比例分配"

    # 满足率
    satisfaction = allocator.get_satisfaction_rate(allocations)
    assert abs(satisfaction - 0.5) < 0.01, "满足率应为50%"

    print("  ✓ 比例分配正确")
    return True


def test_allocation_summary():
    """测试分配汇总"""
    print("测试9: 分配汇总...")

    allocator = WaterAllocator()
    allocator.set_strategy(AllocationStrategy.PROPORTIONAL)

    allocator.add_demand(WaterDemand(
        demand_id="D1",
        location="out1",
        pool_id=1,
        volume=1000.0,
        flow_rate=5.0,
        priority=1,
        start_time=0,
        end_time=24,
        category="agricultural",
    ))

    allocator.add_demand(WaterDemand(
        demand_id="D2",
        location="out2",
        pool_id=2,
        volume=500.0,
        flow_rate=3.0,
        priority=2,
        start_time=0,
        end_time=24,
        category="industrial",
    ))

    allocations = allocator.allocate(1500.0, current_time=12.0)
    summary = allocator.get_allocation_summary(allocations)

    assert 'total_demand' in summary
    assert 'total_allocated' in summary
    assert 'satisfaction_rate' in summary
    assert 'by_category' in summary
    assert 'agricultural' in summary['by_category']

    print("  ✓ 分配汇总正确")
    return True


def test_schedule_generator():
    """测试调度生成器"""
    print("测试10: 调度生成器...")

    generator = ScheduleGenerator(num_pools=3)

    # 设置目标水位
    generator.target_levels = {0: 2.5, 1: 2.5, 2: 2.5}

    # 生成日调度
    demands = [
        WaterDemand(
            demand_id="D1",
            location="out1",
            pool_id=1,
            volume=5000.0,
            flow_rate=10.0,
            priority=1,
            start_time=8.0,
            end_time=18.0,
        )
    ]

    initial_levels = {0: 2.5, 1: 2.5, 2: 2.5}
    upstream_forecast = [50.0] * 24

    schedule = generator.generate_daily_schedule(
        demands, initial_levels, upstream_forecast
    )

    assert len(schedule) == 24, "日调度应有24个时段"
    assert all(isinstance(s, ScheduleSlot) for s in schedule)
    assert schedule[0].slot_id == "H00"
    assert schedule[23].slot_id == "H23"

    print("  ✓ 调度生成器正确")
    return True


def test_emergency_schedule():
    """测试应急调度"""
    print("测试11: 应急调度...")

    generator = ScheduleGenerator(num_pools=5)

    current_levels = {i: 3.0 for i in range(5)}

    # 洪水应急
    flood_schedule = generator.generate_emergency_schedule(
        'flood', current_levels, affected_pools=[1, 2]
    )
    assert len(flood_schedule) > 0
    assert 'FLOOD' in flood_schedule[0].slot_id

    # 干旱应急
    drought_schedule = generator.generate_emergency_schedule(
        'drought', current_levels, affected_pools=[]
    )
    assert len(drought_schedule) > 0
    assert 'DROUGHT' in drought_schedule[0].slot_id

    # 污染应急
    pollution_schedule = generator.generate_emergency_schedule(
        'pollution', current_levels, affected_pools=[2]
    )
    assert len(pollution_schedule) > 0
    assert pollution_schedule[0].gate_positions.get(2, 0.5) == 0.0  # 关闭污染区闸门

    print("  ✓ 应急调度正确")
    return True


def test_optimization_system():
    """测试综合优化调度系统"""
    print("测试12: 综合优化调度系统...")

    system = OptimizationSchedulingSystem(num_pools=3)

    # 配置目标权重
    system.configure_objectives({
        OptimizationObjective.LEVEL_TRACKING: 1.0,
        OptimizationObjective.WATER_DELIVERY: 2.0,
    })

    # 添加约束
    for i in range(3):
        system.add_constraint(OptimizationConstraint(
            constraint_id=f"LEVEL_{i}",
            constraint_type=ConstraintType.LEVEL_BOUND,
            target=str(i),
            lower_bound=1.0,
            upper_bound=4.0,
        ))

    # 添加需求
    system.add_demand(WaterDemand(
        demand_id="D1",
        location="out1",
        pool_id=1,
        volume=5000.0,
        flow_rate=5.0,
        priority=1,
        start_time=0,
        end_time=24,
    ))

    # 生成调度
    initial_state = {
        'levels': {0: 2.5, 1: 2.5, 2: 2.5},
        'flows': {0: 10.0, 1: 10.0, 2: 10.0},
        'gates': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        'current_time': 0,
        'level_limits': {i: (1.0, 4.0) for i in range(3)},
    }

    forecast = {
        'upstream': [50.0] * 24,
    }

    result = system.generate_schedule(initial_state, forecast)

    assert result is not None
    assert len(result.schedule) > 0

    # 获取当前时段
    slot = system.get_current_slot(12.0)
    assert slot is not None

    print("  ✓ 综合优化调度系统正确")
    return True


def test_schedule_evaluation():
    """测试调度评估"""
    print("测试13: 调度评估...")

    system = OptimizationSchedulingSystem(num_pools=3)

    # 生成简单调度
    initial_state = {
        'levels': {0: 2.5, 1: 2.5, 2: 2.5},
        'flows': {0: 10.0, 1: 10.0, 2: 10.0},
        'gates': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        'current_time': 0,
        'level_limits': {i: (1.0, 4.0) for i in range(3)},
    }

    result = system.generate_schedule(initial_state, {'upstream': [50.0] * 24})

    # 评估执行情况 - 完美跟踪
    actual_state = {
        'levels': {0: 2.5, 1: 2.5, 2: 2.5},
        'flows': {0: 10.0, 1: 10.0, 2: 10.0},
        'gates': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        'current_time': 12.0,
    }

    evaluation = system.evaluate_schedule(actual_state)

    assert 'slot_id' in evaluation
    assert 'level_errors' in evaluation
    assert 'schedule_adherence' in evaluation

    print("  ✓ 调度评估正确")
    return True


def test_allocation_strategies():
    """测试所有分配策略"""
    print("测试14: 所有分配策略...")

    strategies = [
        AllocationStrategy.PRIORITY,
        AllocationStrategy.PROPORTIONAL,
        AllocationStrategy.EQUAL,
        AllocationStrategy.DEMAND_BASED,
        AllocationStrategy.EFFICIENCY_BASED,
    ]

    for strategy in strategies:
        allocator = WaterAllocator()
        allocator.set_strategy(strategy)

        # 添加需求
        allocator.add_demand(WaterDemand(
            demand_id="D1",
            location="out1",
            pool_id=1,
            volume=1000.0,
            flow_rate=5.0,
            priority=1,
            start_time=0,
            end_time=24,
            flexibility=0.1,
            min_flow=2.0,
        ))

        allocator.add_demand(WaterDemand(
            demand_id="D2",
            location="out2",
            pool_id=2,
            volume=1000.0,
            flow_rate=5.0,
            priority=2,
            start_time=0,
            end_time=24,
            flexibility=0.5,
            min_flow=1.0,
        ))

        # 执行分配
        allocations = allocator.allocate(1500.0, current_time=12.0)

        # 验证分配总量不超过可用量
        total_allocated = sum(allocations.values())
        assert total_allocated <= 1500.0, f"策略 {strategy.name} 分配超额"

    print("  ✓ 所有分配策略正确")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("优化调度与水量分配系统测试")
    print("=" * 60)
    print()

    tests = [
        test_data_types,
        test_water_demand,
        test_objective_function,
        test_constraint_handler,
        test_constraint_projection,
        test_multi_objective_optimizer,
        test_water_allocator_priority,
        test_water_allocator_proportional,
        test_allocation_summary,
        test_schedule_generator,
        test_emergency_schedule,
        test_optimization_system,
        test_schedule_evaluation,
        test_allocation_strategies,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

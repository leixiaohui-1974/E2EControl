"""
南水北调中线全线全场景自主运行系统 - 完整集成测试
Full Integration Test Suite for Water Transfer Autonomous System

测试覆盖:
1. 全模块导入验证
2. 核心功能测试
3. 全场景仿真测试
4. 故障注入与恢复测试
5. 数据记录与回放测试
6. 优化调度测试
7. 性能压力测试
8. 可视化与报告测试
"""

import sys
import os
import time
import math
from datetime import datetime

# 添加正确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)


# ============================================================
# 第一部分: 全模块导入验证
# ============================================================

def test_all_module_imports():
    """测试所有模块导入"""
    print("\n" + "=" * 70)
    print("第一部分: 全模块导入验证 (21个核心模块)")
    print("=" * 70)

    modules = [
        ('core_types', ['PoolRole', 'ScenarioType', 'ScenarioSeverity']),
        ('physics_model', ['SNWDMiddleRouteModel', 'IDZModel', 'IDZParameters']),
        ('system_identification', ['SystemIdentifier', 'RecursiveLeastSquares']),
        ('orchestrator', ['GlobalOrchestrator', 'ScenarioRoleMatrix']),
        ('regional_coordinator', ['RegionalCoordinator', 'FeedforwardDecoupler']),
        ('enhanced_mpc', ['EnhancedParameterizedMPC', 'HotReconfigurableMPC']),
        ('scenario_generator', ['ScenarioGenerator', 'CompositeScenario']),
        ('adaptive_mpc', ['AdaptiveMPCSystem', 'AdaptiveMPCConfigurator']),
        ('batch_testing', ['BatchTestExecutor', 'TestSuite', 'TestCase']),
        ('local_pool_scenarios', ['L1ScenarioType', 'PollutionTracker']),
        ('l1_controller', ['L1Controller', 'L1ControllerManager']),
        ('l2_l1_coordinator', ['L2L1Coordinator', 'FullLineCoordinatorManager']),
        ('multi_layer_coordinator', ['MultiLayerCoordinator', 'IntelligentDecisionEngine']),
        ('cascade_control', ['CascadeControlSystem', 'ControlEffectEvaluator']),
        ('hydraulic_simulator', ['FullLineHydraulicSimulator', 'IDZDynamicModel']),
        ('integrated_simulation', ['ClosedLoopSimulation', 'ScenarioTestRunner']),
        ('advanced_simulation', ['SensorModel', 'ActuatorModel', 'DataAssimilator']),
        ('visualization', ['TextVisualizer', 'SimulationReport']),
        ('fault_tolerant_control', ['FaultTolerantSystem', 'EmergencyResponseSystem']),
        ('data_recorder', ['DataRecordingSystem', 'SimulationRecorderV2', 'DataAnalyzer']),
        ('optimization_scheduler', ['OptimizationSchedulingSystem', 'MultiObjectiveOptimizer']),
    ]

    passed = 0
    failed = 0

    for module_name, classes in modules:
        try:
            module = __import__(f'water_transfer_system.{module_name}', fromlist=classes)
            for cls_name in classes:
                getattr(module, cls_name)
            print(f"  ✓ {module_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            failed += 1

    print(f"\n模块导入验证: {passed}/{passed+failed} 通过")
    return passed, failed


# ============================================================
# 第二部分: 核心功能测试
# ============================================================

def test_core_functionality():
    """测试核心功能"""
    print("\n" + "=" * 70)
    print("第二部分: 核心功能测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: L1控制器
    print("\n测试2.1: L1现地控制器...")
    try:
        from water_transfer_system import L1Controller, L1PoolState

        controller = L1Controller(pool_id=5, num_pools=20)

        # 更新控制器状态
        controller.update_state(
            water_level=2.5,
            upstream_flow=12.0,
            downstream_flow=10.0,
            gate_position=0.5
        )

        # 执行控制步骤
        result = controller.control_step(dt=60.0)
        assert result is not None
        print(f"  ✓ L1控制器响应正常，状态: {controller.state.controller_state.value}")
        passed += 1
    except Exception as e:
        print(f"  ✗ L1控制器失败: {e}")
        failed += 1

    # 测试2: 全局编排器
    print("\n测试2.2: 全局编排器...")
    try:
        from water_transfer_system import GlobalOrchestrator, ScenarioType

        orchestrator = GlobalOrchestrator()
        assert orchestrator is not None
        assert orchestrator.current_scenario == ScenarioType.S1_NORMAL_PLAN
        print(f"  ✓ 全局编排器初始化完成，当前场景: {orchestrator.current_scenario.name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 全局编排器失败: {e}")
        failed += 1

    # 测试3: 多层协同
    print("\n测试2.3: 多层协同控制器...")
    try:
        from water_transfer_system import MultiLayerCoordinator

        coordinator = MultiLayerCoordinator(num_pools=20)
        assert coordinator is not None
        assert coordinator.l3_scheduler is not None
        print(f"  ✓ 多层协同控制器初始化完成")
        passed += 1
    except Exception as e:
        print(f"  ✗ 多层协同失败: {e}")
        failed += 1

    # 测试4: 场景识别器
    print("\n测试2.4: 场景识别器...")
    try:
        from water_transfer_system import ScenarioIdentifier, ScenarioFeatures

        identifier = ScenarioIdentifier()
        # ScenarioFeatures使用dataclass默认值，使用正确的字段名
        features = ScenarioFeatures(
            level_mean=4.0,
            level_std=0.1,
            level_trend=0.0,
            level_anomaly_count=0,
            flow_mean=300.0,
            flow_std=10.0,
            flow_trend=0.0,
            flow_imbalance=0.0,
            affected_pools=0,
        )
        result = identifier.identify_scenario(features, {})
        assert result is not None
        print(f"  ✓ 场景识别: {result.detected_type.name}, 置信度: {result.confidence:.2f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 场景识别失败: {e}")
        failed += 1

    print(f"\n核心功能测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第三部分: 全场景仿真测试
# ============================================================

def test_all_scenarios():
    """测试所有8大场景"""
    print("\n" + "=" * 70)
    print("第三部分: 全场景仿真测试 (8大场景)")
    print("=" * 70)

    passed = 0
    failed = 0

    try:
        from water_transfer_system import (
            ScenarioType, FullLineHydraulicSimulator, GlobalOrchestrator
        )

        # 所有场景类型
        scenarios = list(ScenarioType)

        # 创建仿真器和编排器
        simulator = FullLineHydraulicSimulator(num_pools=10)
        orchestrator = GlobalOrchestrator()

        for idx, scenario_type in enumerate(scenarios, 1):
            print(f"\n测试3.{idx}: {scenario_type.name}...")
            try:
                # 切换场景
                orchestrator.current_scenario = scenario_type

                # 运行几步仿真
                for step in range(5):
                    simulator.step()

                print(f"  ✓ {scenario_type.name} 仿真完成")
                passed += 1
            except Exception as e:
                print(f"  ✗ {scenario_type.name} 失败: {e}")
                failed += 1

    except Exception as e:
        print(f"  ✗ 场景测试初始化失败: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    print(f"\n全场景仿真测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第四部分: 故障注入与恢复测试
# ============================================================

def test_fault_injection_and_recovery():
    """测试故障注入与恢复"""
    print("\n" + "=" * 70)
    print("第四部分: 故障注入与恢复测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 故障检测系统
    print("\n测试4.1: 故障检测系统...")
    try:
        from water_transfer_system import FaultTolerantSystem

        system = FaultTolerantSystem(num_pools=5)

        # 正常运行
        for t in range(10):
            state = {f'pool_{i}_level': 2.5 + 0.1*math.sin(t*0.3+i) for i in range(5)}
            state.update({f'pool_{i}_flow': 10.0 + 0.5*math.sin(t*0.2+i) for i in range(5)})
            for i in range(6):
                state[f'gate_{i}_command'] = 0.5
                state[f'gate_{i}_position'] = 0.5
            system.process_step(state, float(t))

        # 注入卡死故障
        for t in range(10, 40):
            state = {f'pool_{i}_level': 2.5 for i in range(5)}
            state.update({f'pool_{i}_flow': 10.0 for i in range(5)})
            for i in range(6):
                state[f'gate_{i}_command'] = 0.5
                state[f'gate_{i}_position'] = 0.5
            system.process_step(state, float(t))

        status = system.get_system_status()
        assert status['active_faults'] > 0, "应检测到故障"
        print(f"  ✓ 检测到 {status['active_faults']} 个故障")
        passed += 1
    except Exception as e:
        print(f"  ✗ 故障检测失败: {e}")
        failed += 1

    # 测试2: 容错控制器
    print("\n测试4.2: 容错控制器...")
    try:
        from water_transfer_system import FaultTolerantController, ControlMode

        controller = FaultTolerantController()
        assert controller.current_mode == ControlMode.NORMAL
        print(f"  ✓ 容错控制器初始模式: {controller.current_mode.name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 容错控制器失败: {e}")
        failed += 1

    # 测试3: 应急响应系统
    print("\n测试4.3: 应急响应系统...")
    try:
        from water_transfer_system import EmergencyResponseSystem, EmergencyType

        system = EmergencyResponseSystem()

        # 测试报告应急事件
        for etype in list(EmergencyType)[:3]:  # 测试前3种
            event = system.report_emergency(
                event_type=etype,
                location="pool_5",
                description=f"{etype.name}测试",
                severity=etype,
                timestamp=0.0
            )
            if event:
                response = system.generate_response(event)
                assert response is not None

        print(f"  ✓ 应急响应系统正常")
        passed += 1
    except Exception as e:
        print(f"  ✗ 应急响应失败: {e}")
        failed += 1

    # 测试4: 规则引擎
    print("\n测试4.4: 运行规则引擎...")
    try:
        from water_transfer_system import OperatingRuleEngine

        engine = OperatingRuleEngine()
        assert engine is not None
        print(f"  ✓ 规则引擎初始化完成")
        passed += 1
    except Exception as e:
        print(f"  ✗ 规则引擎失败: {e}")
        failed += 1

    print(f"\n故障注入与恢复测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第五部分: 数据记录与回放测试
# ============================================================

def test_data_recording_and_playback():
    """测试数据记录与回放"""
    print("\n" + "=" * 70)
    print("第五部分: 数据记录与回放测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 数据记录
    print("\n测试5.1: 数据记录功能...")
    try:
        from water_transfer_system import DataRecordingSystem

        recorder = DataRecordingSystem()
        recorder.start()

        # 记录数据
        for t in range(100):
            pool_states = {i: {'level': 2.5 + 0.2*math.sin(t*0.1+i), 'flow': 10.0} for i in range(5)}
            gate_states = {i: {'position': 0.5, 'command': 0.5} for i in range(6)}
            recorder.record_simulation_step(float(t), pool_states, gate_states)

        recorder.stop(100.0)
        print(f"  ✓ 记录 100 步仿真数据")
        passed += 1
    except Exception as e:
        print(f"  ✗ 数据记录失败: {e}")
        failed += 1

    # 测试2: 数据回放
    print("\n测试5.2: 数据回放功能...")
    try:
        replayer = recorder.create_replayer()
        replayer.play()

        frames = 0
        while frames < 50:
            frame = replayer.step(dt=1.0)
            if frame is None:
                break
            frames += 1

        print(f"  ✓ 回放 {frames} 帧")
        passed += 1
    except Exception as e:
        print(f"  ✗ 数据回放失败: {e}")
        failed += 1

    # 测试3: 时序存储
    print("\n测试5.3: 时序存储引擎...")
    try:
        from water_transfer_system import TimeSeriesStorage, DataChannel, DataPoint

        storage = TimeSeriesStorage()
        for t in range(100):
            storage.add_data_point(DataPoint(
                channel=DataChannel.POOL_LEVEL,
                source_id="pool_0",
                timestamp=float(t),
                value=2.5 + 0.1*math.sin(t*0.1),
            ))

        stats = storage.get_statistics()
        assert stats['total_points'] == 100
        print(f"  ✓ 存储 {stats['total_points']} 个数据点")
        passed += 1
    except Exception as e:
        print(f"  ✗ 时序存储失败: {e}")
        failed += 1

    # 测试4: 数据分析器
    print("\n测试5.4: 数据分析器...")
    try:
        from water_transfer_system import DataAnalyzer

        analyzer = DataAnalyzer(storage)
        stats = analyzer.compute_statistics(
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            start_time=0.0,
            end_time=100.0
        )
        assert 'mean' in stats
        print(f"  ✓ 统计分析: 均值={stats['mean']:.3f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 数据分析失败: {e}")
        failed += 1

    print(f"\n数据记录与回放测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第六部分: 优化调度测试
# ============================================================

def test_optimization_and_scheduling():
    """测试优化调度功能"""
    print("\n" + "=" * 70)
    print("第六部分: 优化调度测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 多目标优化器
    print("\n测试6.1: 多目标优化器...")
    try:
        from water_transfer_system import MultiObjectiveOptimizer

        optimizer = MultiObjectiveOptimizer(num_pools=10)

        state = {
            'levels': {f'pool_{i}': 2.5 + 0.1*(i-5) for i in range(10)},
            'flows': {f'pool_{i}': 10.0 for i in range(10)},
            'gates': {f'gate_{i}': 0.5 for i in range(11)}
        }

        result = optimizer.optimize(state)
        assert result is not None
        print(f"  ✓ 优化完成: 目标值={result.objective_value:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 多目标优化失败: {e}")
        failed += 1

    # 测试2: 水量分配器
    print("\n测试6.2: 水量分配器...")
    try:
        from water_transfer_system import WaterAllocator, WaterDemand, AllocationStrategy

        allocator = WaterAllocator()

        # 添加需求
        for i in range(5):
            allocator.add_demand(WaterDemand(
                demand_id=f"city_{i}",
                location=f"pool_{i*2}",
                pool_id=i*2,
                volume=100.0 + i*20,
                flow_rate=5.0 + i,
                priority=i+1,
                start_time=0.0,
                end_time=86400.0,
            ))

        # 测试各种策略
        strategies = list(AllocationStrategy)
        for strategy in strategies:
            allocator.set_strategy(strategy)
            allocation = allocator.allocate(500.0, 0.0)

        print(f"  ✓ 测试 {len(strategies)} 种分配策略")
        passed += 1
    except Exception as e:
        print(f"  ✗ 水量分配失败: {e}")
        failed += 1

    # 测试3: 约束处理器
    print("\n测试6.3: 约束处理器...")
    try:
        from water_transfer_system import ConstraintHandler, OptimizationConstraint, ConstraintType

        handler = ConstraintHandler()

        # 添加约束 - target必须是可转换为int的值 (用于索引)
        handler.add_constraint(OptimizationConstraint(
            constraint_id="level_bound",
            constraint_type=ConstraintType.LEVEL_BOUND,
            target="0",  # 使用字符串格式的整数，因为target是str但会被int()转换
            lower_bound=1.0,
            upper_bound=4.0,
        ))

        state = {'levels': {0: 2.5}, 'flows': {}, 'gates': {}}
        feasible, violations = handler.check_feasibility(state)
        print(f"  ✓ 约束处理正常, 可行={feasible}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 约束处理失败: {e}")
        failed += 1

    # 测试4: 调度生成器
    print("\n测试6.4: 调度生成器...")
    try:
        from water_transfer_system import ScheduleGenerator

        generator = ScheduleGenerator(num_pools=10)
        # generate_daily_schedule返回List[ScheduleSlot]
        schedule = generator.generate_daily_schedule(
            demands=[],
            initial_levels={i: 2.5 for i in range(10)},  # 使用整数键
            upstream_forecast=[15.0] * 24
        )
        assert schedule is not None
        assert isinstance(schedule, list)
        print(f"  ✓ 生成日调度: {len(schedule)} 个时段")
        passed += 1
    except Exception as e:
        print(f"  ✗ 调度生成失败: {e}")
        failed += 1

    print(f"\n优化调度测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第七部分: 性能压力测试
# ============================================================

def test_performance_and_stress():
    """测试性能与压力"""
    print("\n" + "=" * 70)
    print("第七部分: 性能压力测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 大规模水力仿真
    print("\n测试7.1: 大规模水力仿真 (60渠池)...")
    try:
        from water_transfer_system import FullLineHydraulicSimulator

        start = time.time()
        simulator = FullLineHydraulicSimulator(num_pools=60)

        for step in range(100):
            simulator.step()  # 使用正确的方法名

        elapsed = time.time() - start
        print(f"  ✓ 60渠池×100步 耗时: {elapsed:.2f}秒 ({100/elapsed:.1f} 步/秒)")
        passed += 1
    except Exception as e:
        print(f"  ✗ 水力仿真失败: {e}")
        failed += 1

    # 测试2: L1控制器批量创建
    print("\n测试7.2: L1控制器批量测试 (60个)...")
    try:
        from water_transfer_system import L1Controller, L1PoolState

        start = time.time()
        controllers = []

        for i in range(60):
            controllers.append(L1Controller(pool_id=i))

        # 运行控制
        for step in range(50):
            for i, controller in enumerate(controllers):
                # 更新控制器状态
                controller.update_state(
                    water_level=2.5 + 0.1 * math.sin(step * 0.1),
                    upstream_flow=12.0,
                    downstream_flow=10.0
                )
                # 执行控制步骤
                controller.control_step(dt=60.0)

        elapsed = time.time() - start
        ops = 60 * 50
        print(f"  ✓ {ops} 次控制计算 耗时: {elapsed:.2f}秒 ({ops/elapsed:.0f} 次/秒)")
        passed += 1
    except Exception as e:
        print(f"  ✗ L1控制器批量测试失败: {e}")
        failed += 1

    # 测试3: 故障检测吞吐量
    print("\n测试7.3: 故障检测吞吐量...")
    try:
        from water_transfer_system import FaultTolerantSystem

        start = time.time()
        system = FaultTolerantSystem(num_pools=30)

        for t in range(200):
            state = {}
            for i in range(30):
                state[f'pool_{i}_level'] = 2.5 + 0.1*math.sin(t*0.1+i)
                state[f'pool_{i}_flow'] = 10.0
            for i in range(31):
                state[f'gate_{i}_command'] = 0.5
                state[f'gate_{i}_position'] = 0.5
            system.process_step(state, float(t))

        elapsed = time.time() - start
        print(f"  ✓ 30渠池×200步 故障检测 耗时: {elapsed:.2f}秒")
        passed += 1
    except Exception as e:
        print(f"  ✗ 故障检测失败: {e}")
        failed += 1

    # 测试4: 场景识别性能
    print("\n测试7.4: 场景识别性能...")
    try:
        from water_transfer_system import ScenarioIdentifier, ScenarioFeatures
        import numpy as np

        identifier = ScenarioIdentifier()
        start = time.time()

        for i in range(100):
            # ScenarioFeatures使用dataclass默认值
            features = ScenarioFeatures(
                level_mean=4.0 + np.random.randn() * 0.1,
                level_std=0.1 + np.random.rand() * 0.05,
                level_trend=np.random.randn() * 0.01,
                flow_mean=300.0 + np.random.randn() * 10,
                flow_std=10.0 + np.random.rand() * 5,
                flow_trend=np.random.randn() * 0.5,
            )
            identifier.identify_scenario(features, {})

        elapsed = time.time() - start
        print(f"  ✓ 100次场景识别 耗时: {elapsed:.3f}秒 ({100/elapsed:.0f} 次/秒)")
        passed += 1
    except Exception as e:
        print(f"  ✗ 场景识别失败: {e}")
        failed += 1

    print(f"\n性能压力测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第八部分: 可视化与报告测试
# ============================================================

def test_visualization_and_reporting():
    """测试可视化与报告生成"""
    print("\n" + "=" * 70)
    print("第八部分: 可视化与报告测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 文本可视化
    print("\n测试8.1: 文本可视化器...")
    try:
        from water_transfer_system import TextVisualizer

        # 使用静态方法
        chart = TextVisualizer.bar_chart(
            data={'pool_0': 2.5, 'pool_1': 2.8, 'pool_2': 2.3}
        )
        assert len(chart) > 0
        print(f"  ✓ 生成柱状图 ({len(chart)} 字符)")
        passed += 1
    except Exception as e:
        print(f"  ✗ 文本可视化失败: {e}")
        failed += 1

    # 测试2: 仿真报告生成
    print("\n测试8.2: 仿真报告生成器...")
    try:
        from water_transfer_system import ReportGenerator

        generator = ReportGenerator()
        report = generator.generate_simulation_summary({
            'config': {
                'num_pools': 20,
                'duration': 86400,
                'dt': 60
            },
            'execution': {
                'total_steps': 1440,
                'elapsed_seconds': 10.5,
                'steps_per_second': 137.0
            },
            'control': {
                'total_escalations': 5,
                'total_interventions': 12
            },
            'scenarios': {
                'injected': 3,
                'active_at_end': 0
            },
            'performance': {
                'avg_rmse': 0.05,
                'max_rmse': 0.15,
                'avg_mae': 0.03
            }
        })
        assert report is not None
        print(f"  ✓ 生成仿真报告: {report.report_id}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 仿真报告失败: {e}")
        failed += 1

    # 测试3: 进度条和指示器
    print("\n测试8.3: 进度条和水位指示器...")
    try:
        from water_transfer_system import TextVisualizer

        progress = TextVisualizer.progress_bar(75.0, 100.0)
        indicator = TextVisualizer.level_indicator(2.5, 0.5, 4.0)
        assert len(progress) > 0
        assert len(indicator) > 0
        print(f"  ✓ 进度条: {progress}")
        passed += 1
    except Exception as e:
        print(f"  ✗ 可视化组件失败: {e}")
        failed += 1

    print(f"\n可视化与报告测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 第九部分: 闭环集成测试
# ============================================================

def test_closed_loop_integration():
    """测试闭环集成"""
    print("\n" + "=" * 70)
    print("第九部分: 闭环集成测试")
    print("=" * 70)

    passed = 0
    failed = 0

    # 测试1: 水力仿真器完整运行
    print("\n测试9.1: 水力仿真器完整运行...")
    try:
        from water_transfer_system import FullLineHydraulicSimulator

        simulator = FullLineHydraulicSimulator(num_pools=20)

        # 设置上游入流
        simulator.set_upstream_inflow(60.0)

        # 运行仿真
        results = simulator.run(duration=3600.0)  # 1小时
        assert len(results) > 0

        print(f"  ✓ 完成 {len(results)} 步仿真")
        passed += 1
    except Exception as e:
        print(f"  ✗ 水力仿真器运行失败: {e}")
        failed += 1

    # 测试2: 级联控制系统
    print("\n测试9.2: 级联控制系统...")
    try:
        from water_transfer_system import CascadeControlSystem

        system = CascadeControlSystem(num_pools=10)
        assert system is not None
        print(f"  ✓ 级联控制系统初始化完成")
        passed += 1
    except Exception as e:
        print(f"  ✗ 级联控制系统失败: {e}")
        failed += 1

    # 测试3: 高级仿真层
    print("\n测试9.3: 高级仿真层...")
    try:
        from water_transfer_system import AdvancedSimulationLayer, FullLineHydraulicSimulator

        # AdvancedSimulationLayer需要一个simulator参数
        simulator = FullLineHydraulicSimulator(num_pools=10)
        layer = AdvancedSimulationLayer(simulator=simulator)
        assert layer is not None
        assert layer.num_pools == 10
        print(f"  ✓ 高级仿真层初始化完成 ({layer.num_pools} 渠池)")
        passed += 1
    except Exception as e:
        print(f"  ✗ 高级仿真层失败: {e}")
        failed += 1

    # 测试4: 物理模型
    print("\n测试9.4: IDZ物理模型...")
    try:
        from water_transfer_system.physics_model import IDZModel, IDZParameters

        # IDZParameters dataclass使用正确的参数名
        params = IDZParameters(
            tau=14400.0,      # 滞后时间 [s]
            A_s=100000.0,     # 蓄水面积 [m²]
            c_in=1.0,         # 入流增益
            c_out=1.0         # 出流增益
        )
        model = IDZModel(params=params, dt=900.0)
        assert model is not None
        print(f"  ✓ IDZ物理模型初始化完成 (tau={params.tau}s, A_s={params.A_s}m²)")
        passed += 1
    except Exception as e:
        print(f"  ✗ IDZ物理模型失败: {e}")
        failed += 1

    print(f"\n闭环集成测试: {passed} 通过, {failed} 失败")
    return passed, failed


# ============================================================
# 主测试入口
# ============================================================

def run_full_integration_tests():
    """运行完整集成测试"""
    print("╔" + "═" * 68 + "╗")
    print("║" + "南水北调中线全线全场景自主运行系统 - 完整集成测试".center(52) + "║")
    print("║" + "Water Transfer Autonomous System - Full Integration Test".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_failed = 0

    test_suites = [
        ("模块导入验证", test_all_module_imports),
        ("核心功能测试", test_core_functionality),
        ("全场景仿真测试", test_all_scenarios),
        ("故障注入恢复", test_fault_injection_and_recovery),
        ("数据记录回放", test_data_recording_and_playback),
        ("优化调度系统", test_optimization_and_scheduling),
        ("性能压力测试", test_performance_and_stress),
        ("可视化与报告", test_visualization_and_reporting),
        ("闭环集成测试", test_closed_loop_integration),
    ]

    results = []

    for name, test_func in test_suites:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
            results.append((name, passed, failed))
        except Exception as e:
            print(f"\n{name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, 0, 1))
            total_failed += 1

    # 打印汇总
    print("\n" + "═" * 70)
    print("测 试 汇 总 报 告")
    print("═" * 70)

    print(f"\n{'测试套件':<20} {'通过':<10} {'失败':<10} {'状态':<10}")
    print("-" * 50)

    for name, passed, failed in results:
        status = "✓ PASS" if failed == 0 else "✗ FAIL"
        print(f"{name:<20} {passed:<10} {failed:<10} {status:<10}")

    print("-" * 50)
    print(f"{'总计':<20} {total_passed:<10} {total_failed:<10}")

    # 计算通过率
    total = total_passed + total_failed
    success_rate = total_passed / total * 100 if total > 0 else 0

    # 最终结果
    print("\n" + "═" * 70)

    if total_failed == 0:
        print("█" * 70)
        print("██                                                                ██")
        print("██          全部测试通过！系统功能验证完成                        ██")
        print("██          ALL TESTS PASSED! System Validated                     ██")
        print("██                                                                ██")
        print("█" * 70)
    elif success_rate >= 90:
        print(f"测试通过率: {success_rate:.1f}% - 优秀")
        print("系统核心功能完整，少数测试需要调整")
    elif success_rate >= 80:
        print(f"测试通过率: {success_rate:.1f}% - 良好")
        print("系统核心功能基本完整")
    else:
        print(f"测试通过率: {success_rate:.1f}%")
        print(f"警告: {total_failed} 个测试失败，请检查日志")

    print("═" * 70)

    return total_passed, total_failed


if __name__ == "__main__":
    passed, failed = run_full_integration_tests()
    sys.exit(0 if failed == 0 else 1)

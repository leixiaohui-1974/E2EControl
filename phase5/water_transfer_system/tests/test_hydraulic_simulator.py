"""
高保真水力学仿真器测试
Tests for High-Fidelity Hydraulic Simulator
"""

import pytest
import numpy as np
from collections import deque

from ..hydraulic_simulator import (
    PhysicalConstants,
    PoolPhysicalParams,
    GateParams,
    PoolState,
    SimulationState,
    IDZDynamicModel,
    GateDynamicModel,
    FullLineHydraulicSimulator,
    SimulationRecord,
    SimulationRecorder,
    SimulationReplayer,
    PerformanceMetrics,
    PerformanceAnalyzer,
)
from ..local_pool_scenarios import (
    L1ScenarioType,
    L1ScenarioEvent,
)
from ..core_types import ScenarioSeverity


# ==============================================================================
# 物理常数测试
# ==============================================================================

class TestPhysicalConstants:
    """物理常数测试"""

    def test_gravity_value(self):
        """测试重力加速度"""
        assert PhysicalConstants.GRAVITY == 9.81

    def test_water_density(self):
        """测试水密度"""
        assert PhysicalConstants.WATER_DENSITY == 1000.0

    def test_manning_coefficient(self):
        """测试曼宁系数"""
        assert PhysicalConstants.MANNING_N == 0.014


# ==============================================================================
# 渠池物理参数测试
# ==============================================================================

class TestPoolPhysicalParams:
    """渠池物理参数测试"""

    def test_default_params(self):
        """测试默认参数"""
        params = PoolPhysicalParams(pool_id=0)
        assert params.pool_id == 0
        assert params.length == 5000.0
        assert params.bottom_width == 15.0
        assert params.side_slope == 2.0
        assert params.min_level == 0.5
        assert params.max_level == 6.0

    def test_custom_params(self):
        """测试自定义参数"""
        params = PoolPhysicalParams(
            pool_id=5,
            length=6000.0,
            bottom_width=20.0,
            delay_time=400.0,
        )
        assert params.length == 6000.0
        assert params.bottom_width == 20.0
        assert params.delay_time == 400.0

    def test_surface_area_calculation(self):
        """测试水面面积计算"""
        params = PoolPhysicalParams(pool_id=0, length=5000.0, bottom_width=15.0, side_slope=2.0)
        # 水位3m: top_width = 15 + 2*2*3 = 27m
        # area = 5000 * (15 + 27) / 2 = 5000 * 21 = 105000 m²
        area = params.get_surface_area(3.0)
        assert area == 105000.0

    def test_surface_area_zero_level(self):
        """测试零水位面积"""
        params = PoolPhysicalParams(pool_id=0, length=5000.0, bottom_width=15.0)
        area = params.get_surface_area(0.0)
        assert area == 5000.0 * 15.0

    def test_storage_calculation(self):
        """测试库容计算"""
        params = PoolPhysicalParams(pool_id=0, length=5000.0, bottom_width=15.0, side_slope=2.0)
        # 水位3m: cross_area = (15 + 27) * 3 / 2 = 63 m²
        # storage = 5000 * 63 = 315000 m³
        storage = params.get_storage(3.0)
        assert storage == 315000.0


# ==============================================================================
# 闸门参数测试
# ==============================================================================

class TestGateParams:
    """闸门参数测试"""

    def test_default_params(self):
        """测试默认参数"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1)
        assert params.gate_id == "GATE_0"
        assert params.width == 10.0
        assert params.max_opening == 3.0
        assert params.max_rate == 0.01
        assert params.discharge_coef == 0.6

    def test_custom_params(self):
        """测试自定义参数"""
        params = GateParams(
            gate_id="GATE_5",
            pool_upstream=5,
            pool_downstream=6,
            width=12.0,
            max_opening=4.0,
        )
        assert params.width == 12.0
        assert params.max_opening == 4.0


# ==============================================================================
# 渠池状态测试
# ==============================================================================

class TestPoolState:
    """渠池状态测试"""

    def test_default_state(self):
        """测试默认状态"""
        state = PoolState(pool_id=0)
        assert state.pool_id == 0
        assert state.water_level == 3.0
        assert state.inflow == 50.0
        assert state.outflow == 50.0
        assert state.quality_index == 1.0
        assert not state.has_anomaly

    def test_custom_state(self):
        """测试自定义状态"""
        state = PoolState(
            pool_id=5,
            water_level=2.5,
            inflow=60.0,
            outflow=55.0,
            quality_index=0.8,
            has_anomaly=True,
            anomaly_type="pollution",
        )
        assert state.water_level == 2.5
        assert state.has_anomaly
        assert state.anomaly_type == "pollution"


# ==============================================================================
# IDZ动态模型测试
# ==============================================================================

class TestIDZDynamicModel:
    """IDZ动态模型测试"""

    def test_initialization(self):
        """测试初始化"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)
        assert model.current_level == params.target_level
        assert model.current_inflow == 50.0

    def test_step_balanced_flow(self):
        """测试平衡流量下的仿真步"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)
        initial_level = model.current_level

        # 入流=出流时，水位应基本保持
        new_level = model.step(60.0, 50.0, 50.0)
        assert abs(new_level - initial_level) < 0.01

    def test_step_inflow_greater(self):
        """测试入流大于出流"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)
        initial_level = model.current_level

        # 入流>出流时，水位应上升
        new_level = model.step(60.0, 60.0, 40.0)
        assert new_level > initial_level

    def test_step_outflow_greater(self):
        """测试出流大于入流"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)
        initial_level = model.current_level

        # 出流>入流时，水位应下降
        new_level = model.step(60.0, 40.0, 60.0)
        assert new_level < initial_level

    def test_level_constraint_max(self):
        """测试最大水位约束"""
        params = PoolPhysicalParams(pool_id=0, max_level=6.0)
        model = IDZDynamicModel(params)
        model.current_level = 5.9

        # 大量入流不应超过最大水位
        for _ in range(100):
            model.step(60.0, 100.0, 20.0)

        assert model.current_level <= params.max_level

    def test_level_constraint_min(self):
        """测试最小水位约束"""
        params = PoolPhysicalParams(pool_id=0, min_level=0.5)
        model = IDZDynamicModel(params)
        model.current_level = 0.6

        # 大量出流不应低于最小水位
        for _ in range(100):
            model.step(60.0, 10.0, 80.0)

        assert model.current_level >= params.min_level

    def test_delayed_flow(self):
        """测试延迟流量获取"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)

        # 创建流量历史
        flow_history = deque([10.0, 20.0, 30.0, 40.0, 50.0], maxlen=100)

        delayed = model.get_delayed_flow(flow_history, 3)
        assert delayed == 30.0  # flow_history[-3] = 30.0

    def test_delayed_flow_empty(self):
        """测试空历史的延迟流量"""
        params = PoolPhysicalParams(pool_id=0)
        model = IDZDynamicModel(params)

        flow_history = deque(maxlen=100)
        delayed = model.get_delayed_flow(flow_history, 5)
        assert delayed == model.current_inflow


# ==============================================================================
# 闸门动态模型测试
# ==============================================================================

class TestGateDynamicModel:
    """闸门动态模型测试"""

    def test_initialization(self):
        """测试初始化"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1)
        model = GateDynamicModel(params)
        assert model.current_opening == params.current_opening
        assert not model.is_moving

    def test_set_target(self):
        """测试设置目标开度"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1, current_opening=1.5)
        model = GateDynamicModel(params)

        model.set_target(2.0)
        assert model.target_opening == 2.0
        assert model.is_moving
        assert model.move_direction == 1  # 开启方向

    def test_set_target_close(self):
        """测试设置关闭方向目标"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1, current_opening=2.0)
        model = GateDynamicModel(params)

        model.set_target(1.0)
        assert model.target_opening == 1.0
        assert model.move_direction == -1  # 关闭方向

    def test_set_target_constraint(self):
        """测试目标开度约束"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1, max_opening=3.0)
        model = GateDynamicModel(params)

        model.set_target(5.0)  # 超过最大值
        assert model.target_opening == 3.0

        model.set_target(-1.0)  # 负值
        assert model.target_opening == 0.0

    def test_step_not_moving(self):
        """测试静止状态步进"""
        params = GateParams(gate_id="GATE_0", pool_upstream=0, pool_downstream=1)
        model = GateDynamicModel(params)

        opening = model.step(60.0)
        assert opening == model.current_opening
        assert not model.is_moving

    def test_step_rate_limit(self):
        """测试速率限制"""
        params = GateParams(
            gate_id="GATE_0",
            pool_upstream=0,
            pool_downstream=1,
            current_opening=1.0,
            max_rate=0.01,  # 0.01 m/s
        )
        model = GateDynamicModel(params)
        model.set_target(3.0)  # 目标差2m

        # 60秒最多移动 0.01 * 60 = 0.6m
        opening = model.step(60.0)
        assert abs(opening - 1.6) < 0.01

    def test_step_reach_target(self):
        """测试到达目标"""
        params = GateParams(
            gate_id="GATE_0",
            pool_upstream=0,
            pool_downstream=1,
            current_opening=1.0,
            max_rate=0.1,
        )
        model = GateDynamicModel(params)
        model.set_target(1.1)  # 小目标差

        opening = model.step(60.0)
        assert opening == 1.1
        assert not model.is_moving

    def test_calculate_flow_free_outflow(self):
        """测试自由出流"""
        params = GateParams(
            gate_id="GATE_0",
            pool_upstream=0,
            pool_downstream=1,
            width=10.0,
            discharge_coef=0.6,
            current_opening=1.0,
        )
        model = GateDynamicModel(params)

        # 下游水位低于开度：自由出流
        flow = model.calculate_flow(3.0, 0.5)
        assert flow > 0

    def test_calculate_flow_submerged(self):
        """测试淹没出流"""
        params = GateParams(
            gate_id="GATE_0",
            pool_upstream=0,
            pool_downstream=1,
            width=10.0,
            current_opening=1.0,
        )
        model = GateDynamicModel(params)

        # 下游水位高于开度：淹没出流
        flow = model.calculate_flow(3.0, 2.5)
        assert flow > 0

    def test_calculate_flow_closed(self):
        """测试全关状态"""
        params = GateParams(
            gate_id="GATE_0",
            pool_upstream=0,
            pool_downstream=1,
            current_opening=0.0,
        )
        model = GateDynamicModel(params)

        flow = model.calculate_flow(3.0, 2.0)
        assert flow == 0.0


# ==============================================================================
# 全线水力学仿真器测试
# ==============================================================================

class TestFullLineHydraulicSimulator:
    """全线仿真器测试"""

    def test_initialization(self):
        """测试初始化"""
        sim = FullLineHydraulicSimulator(num_pools=10, dt=60.0)
        assert sim.num_pools == 10
        assert sim.dt == 60.0
        assert len(sim.pool_models) == 10
        assert len(sim.gate_models) == 10

    def test_pool_params_initialized(self):
        """测试渠池参数初始化"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        for i in range(5):
            assert i in sim.pool_params
            assert sim.pool_params[i].pool_id == i

    def test_gate_params_initialized(self):
        """测试闸门参数初始化"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        for i in range(5):
            gate_id = f"GATE_{i}"
            assert gate_id in sim.gate_params

    def test_initial_state(self):
        """测试初始状态"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        for i in range(5):
            assert i in sim.state.pool_states
            assert sim.state.pool_states[i].pool_id == i

    def test_set_gate_opening(self):
        """测试设置闸门开度"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.set_gate_opening(2, 2.5)
        assert sim.gate_models["GATE_2"].target_opening == 2.5

    def test_set_upstream_inflow(self):
        """测试设置上游来水"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.set_upstream_inflow(80.0)
        assert sim.upstream_inflow == 80.0

    def test_set_upstream_inflow_negative(self):
        """测试负入流"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.set_upstream_inflow(-10.0)
        assert sim.upstream_inflow == 0.0

    def test_step_basic(self):
        """测试基本仿真步"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        initial_time = sim.state.current_time

        states = sim.step()

        assert len(states) == 5
        assert sim.state.current_time == initial_time + sim.dt
        assert sim.state.step_count == 1

    def test_step_multiple(self):
        """测试多步仿真"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        for _ in range(10):
            sim.step()

        assert sim.state.step_count == 10

    def test_inject_scenario(self):
        """测试场景注入"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        scenario = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=2,
            severity=ScenarioSeverity.MEDIUM,
        )

        sim.inject_scenario(2, scenario)
        assert 2 in sim.active_scenarios
        assert sim.active_scenarios[2] == scenario

    def test_remove_scenario(self):
        """测试移除场景"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        scenario = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=2,
            severity=ScenarioSeverity.MEDIUM,
        )

        sim.inject_scenario(2, scenario)
        sim.remove_scenario(2)
        assert 2 not in sim.active_scenarios

    def test_scenario_effect_rapid_rise(self):
        """测试水位快速上涨场景效果"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        initial_level = sim.state.pool_states[2].water_level

        scenario = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=2,
            severity=ScenarioSeverity.MEDIUM,
        )
        sim.inject_scenario(2, scenario)

        # 运行多步
        for _ in range(20):
            sim.step()

        # 水位应该上升
        new_level = sim.state.pool_states[2].water_level
        assert new_level >= initial_level

    def test_scenario_effect_pollution(self):
        """测试污染场景效果"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        scenario = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=2,
            severity=ScenarioSeverity.HIGH,
        )
        sim.inject_scenario(2, scenario)
        sim.step()

        state = sim.state.pool_states[2]
        assert state.quality_index == 0.5
        assert state.has_anomaly
        assert state.anomaly_type == "pollution"

    def test_run_simulation(self):
        """测试运行仿真"""
        sim = FullLineHydraulicSimulator(num_pools=5, dt=60.0)

        history = sim.run(duration=600.0)  # 10分钟

        assert sim.state.step_count == 10
        assert len(history) >= 1

    def test_run_with_callback(self):
        """测试带回调的运行"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        callback_count = [0]

        def callback(state):
            callback_count[0] += 1

        sim.run(duration=300.0, callback=callback)

        assert callback_count[0] == 5

    def test_get_pool_state(self):
        """测试获取池状态"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        state = sim.get_pool_state(2)
        assert state is not None
        assert state.pool_id == 2

    def test_get_pool_state_invalid(self):
        """测试获取无效池状态"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        state = sim.get_pool_state(100)
        assert state is None

    def test_get_all_levels(self):
        """测试获取所有水位"""
        sim = FullLineHydraulicSimulator(num_pools=5)

        levels = sim.get_all_levels()
        assert len(levels) == 5
        for i in range(5):
            assert i in levels

    def test_get_all_flows(self):
        """测试获取所有流量"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.step()

        flows = sim.get_all_flows()
        assert len(flows) == 5


# ==============================================================================
# 仿真记录器测试
# ==============================================================================

class TestSimulationRecorder:
    """仿真记录器测试"""

    def test_start_recording(self):
        """测试开始记录"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        recorder = SimulationRecorder(sim)

        recorder.start_recording("TEST_REC_001")

        assert recorder.is_recording
        assert recorder.current_record is not None
        assert recorder.current_record.record_id == "TEST_REC_001"

    def test_stop_recording(self):
        """测试停止记录"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        recorder = SimulationRecorder(sim)

        recorder.start_recording("TEST_REC_001")
        sim.run(duration=300.0)
        record = recorder.stop_recording()

        assert not recorder.is_recording
        assert record is not None
        assert record.end_time > record.start_time

    def test_record_data(self):
        """测试记录数据"""
        sim = FullLineHydraulicSimulator(num_pools=5, dt=60.0)
        recorder = SimulationRecorder(sim)

        recorder.start_recording("TEST_REC_001")
        sim.run(duration=300.0)  # 5步
        record = recorder.stop_recording()

        assert len(record.timestamps) == 5
        assert len(record.level_data[0]) == 5

    def test_record_scenario_event(self):
        """测试记录场景事件"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        recorder = SimulationRecorder(sim)

        recorder.start_recording("TEST_REC_001")

        scenario = L1ScenarioEvent(
            event_id="TEST_001",
            scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
            pool_id=2,
            severity=ScenarioSeverity.HIGH,
        )
        recorder.record_scenario_event(2, scenario)

        assert len(recorder.current_record.scenario_events) == 1

    def test_record_control_action(self):
        """测试记录控制动作"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        recorder = SimulationRecorder(sim)

        recorder.start_recording("TEST_REC_001")
        recorder.record_control_action(2, "gate_opening", 2.0)

        assert len(recorder.current_record.control_actions) == 1
        assert recorder.current_record.control_actions[0]['pool_id'] == 2


# ==============================================================================
# 仿真回放器测试
# ==============================================================================

class TestSimulationReplayer:
    """仿真回放器测试"""

    def test_initialization(self):
        """测试初始化"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=300.0,
            dt=60.0,
            num_pools=5,
        )
        record.timestamps = [60.0, 120.0, 180.0]

        replayer = SimulationReplayer(record)
        assert replayer.current_index == 0

    def test_reset(self):
        """测试重置"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=300.0,
            dt=60.0,
            num_pools=5,
        )
        record.timestamps = [60.0, 120.0, 180.0]

        replayer = SimulationReplayer(record)
        replayer.current_index = 2
        replayer.reset()

        assert replayer.current_index == 0

    def test_get_state_at(self):
        """测试获取指定时间状态"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=300.0,
            dt=60.0,
            num_pools=2,
        )
        record.timestamps = [60.0, 120.0, 180.0]
        record.level_data = {0: [3.0, 3.1, 3.2], 1: [2.9, 2.8, 2.7]}
        record.flow_data = {0: [50.0, 51.0, 52.0], 1: [49.0, 48.0, 47.0]}
        record.gate_data = {"GATE_0": [1.5, 1.6, 1.7]}

        replayer = SimulationReplayer(record)
        state = replayer.get_state_at(120.0)

        assert state is not None
        assert state['time'] == 120.0
        assert state['levels'][0] == 3.1

    def test_get_state_at_empty(self):
        """测试空记录"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=0.0,
            dt=60.0,
            num_pools=2,
        )

        replayer = SimulationReplayer(record)
        state = replayer.get_state_at(60.0)

        assert state is None

    def test_step(self):
        """测试单步回放"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=300.0,
            dt=60.0,
            num_pools=2,
        )
        record.timestamps = [60.0, 120.0, 180.0]
        record.level_data = {0: [3.0, 3.1, 3.2], 1: [2.9, 2.8, 2.7]}
        record.flow_data = {0: [50.0, 51.0, 52.0], 1: [49.0, 48.0, 47.0]}
        record.gate_data = {"GATE_0": [1.5, 1.6, 1.7]}

        replayer = SimulationReplayer(record)

        state1 = replayer.step()
        assert state1['time'] == 60.0

        state2 = replayer.step()
        assert state2['time'] == 120.0

    def test_step_end(self):
        """测试回放结束"""
        record = SimulationRecord(
            record_id="TEST",
            start_time=0.0,
            end_time=180.0,
            dt=60.0,
            num_pools=2,
        )
        record.timestamps = [60.0, 120.0]
        record.level_data = {0: [3.0, 3.1], 1: [2.9, 2.8]}
        record.flow_data = {0: [50.0, 51.0], 1: [49.0, 48.0]}
        record.gate_data = {}

        replayer = SimulationReplayer(record)
        replayer.step()
        replayer.step()
        state = replayer.step()

        assert state is None


# ==============================================================================
# 性能指标测试
# ==============================================================================

class TestPerformanceMetrics:
    """性能指标测试"""

    def test_calculate_rmse(self):
        """测试RMSE计算"""
        levels = [3.1, 2.9, 3.0, 3.2, 2.8]
        rmse = PerformanceMetrics.calculate_level_rmse(3.0, levels)
        assert rmse > 0

    def test_calculate_rmse_perfect(self):
        """测试完美情况RMSE"""
        levels = [3.0, 3.0, 3.0, 3.0]
        rmse = PerformanceMetrics.calculate_level_rmse(3.0, levels)
        assert rmse == 0.0

    def test_calculate_rmse_empty(self):
        """测试空列表RMSE"""
        rmse = PerformanceMetrics.calculate_level_rmse(3.0, [])
        assert rmse == 0.0

    def test_calculate_mae(self):
        """测试MAE计算"""
        levels = [3.1, 2.9, 3.0, 3.2, 2.8]
        mae = PerformanceMetrics.calculate_level_mae(3.0, levels)
        assert mae > 0

    def test_calculate_mae_perfect(self):
        """测试完美情况MAE"""
        levels = [3.0, 3.0, 3.0]
        mae = PerformanceMetrics.calculate_level_mae(3.0, levels)
        assert mae == 0.0

    def test_calculate_flow_balance(self):
        """测试流量平衡度"""
        inflows = [50.0, 52.0, 48.0]
        outflows = [49.0, 51.0, 50.0]
        balance = PerformanceMetrics.calculate_flow_balance(inflows, outflows)
        assert 0 <= balance <= 1

    def test_calculate_flow_balance_perfect(self):
        """测试完美平衡"""
        inflows = [50.0, 50.0, 50.0]
        outflows = [50.0, 50.0, 50.0]
        balance = PerformanceMetrics.calculate_flow_balance(inflows, outflows)
        assert balance == 0.0

    def test_calculate_settling_time(self):
        """测试调节时间计算"""
        # 逐渐趋近目标
        levels = [3.5, 3.3, 3.15, 3.05, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        settling = PerformanceMetrics.calculate_settling_time(levels, 3.0, 0.1)
        assert settling is not None
        assert settling <= 5

    def test_calculate_settling_time_not_settled(self):
        """测试未稳定情况"""
        levels = [3.5, 3.6, 3.4, 3.7, 3.3]
        settling = PerformanceMetrics.calculate_settling_time(levels, 3.0, 0.1)
        assert settling is None

    def test_calculate_overshoot(self):
        """测试超调量计算"""
        levels = [3.0, 3.2, 3.5, 3.3, 3.1, 3.0]
        overshoot = PerformanceMetrics.calculate_overshoot(levels, 3.0)
        assert overshoot == 0.5

    def test_calculate_overshoot_empty(self):
        """测试空列表超调"""
        overshoot = PerformanceMetrics.calculate_overshoot([], 3.0)
        assert overshoot == 0.0


# ==============================================================================
# 性能分析器测试
# ==============================================================================

class TestPerformanceAnalyzer:
    """性能分析器测试"""

    def test_analyze_pool(self):
        """测试分析单个池"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.run(duration=600.0)

        analyzer = PerformanceAnalyzer(sim)
        metrics = analyzer.analyze_pool(0)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'overshoot' in metrics
        assert 'settling_time' in metrics

    def test_analyze_all(self):
        """测试分析所有池"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.run(duration=600.0)

        analyzer = PerformanceAnalyzer(sim)
        all_metrics = analyzer.analyze_all()

        assert len(all_metrics) == 5
        for i in range(5):
            assert i in all_metrics

    def test_generate_report(self):
        """测试生成报告"""
        sim = FullLineHydraulicSimulator(num_pools=5)
        sim.run(duration=600.0)

        analyzer = PerformanceAnalyzer(sim)
        report = analyzer.generate_report()

        assert 'pool_metrics' in report
        assert 'summary' in report
        assert 'simulation_info' in report

        assert 'avg_rmse' in report['summary']
        assert 'max_rmse' in report['summary']
        assert 'step_count' in report['simulation_info']


# ==============================================================================
# 集成测试
# ==============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_simulation_with_scenarios(self):
        """测试带场景的完整仿真"""
        sim = FullLineHydraulicSimulator(num_pools=10, dt=60.0)
        recorder = SimulationRecorder(sim)

        # 开始记录
        recorder.start_recording("INTEG_TEST_001")

        # 注入场景
        scenario = L1ScenarioEvent(
            event_id="INTEG_SC_001",
            scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
            pool_id=5,
            severity=ScenarioSeverity.MEDIUM,
        )
        sim.inject_scenario(5, scenario)
        recorder.record_scenario_event(5, scenario)

        # 调整闸门
        sim.set_gate_opening(4, 2.0)
        recorder.record_control_action(4, "gate_opening", 2.0)

        # 运行仿真
        sim.run(duration=600.0)

        # 停止记录
        record = recorder.stop_recording()

        # 分析性能
        analyzer = PerformanceAnalyzer(sim)
        report = analyzer.generate_report()

        # 验证
        assert len(record.timestamps) == 10
        assert len(record.scenario_events) == 1
        assert len(record.control_actions) == 1
        assert report['simulation_info']['step_count'] == 10

    def test_replay_simulation(self):
        """测试仿真回放"""
        # 原始仿真
        sim = FullLineHydraulicSimulator(num_pools=5, dt=60.0)
        recorder = SimulationRecorder(sim)
        recorder.start_recording("REPLAY_TEST")
        sim.run(duration=300.0)
        record = recorder.stop_recording()

        # 回放
        replayer = SimulationReplayer(record)

        states = []
        while True:
            state = replayer.step()
            if state is None:
                break
            states.append(state)

        assert len(states) == 5

        # 验证水位趋势
        for i in range(1, len(states)):
            assert states[i]['time'] > states[i-1]['time']

    def test_multi_scenario_simulation(self):
        """测试多场景仿真"""
        sim = FullLineHydraulicSimulator(num_pools=10, dt=60.0)

        # 注入多个场景
        scenarios = [
            L1ScenarioEvent(
                event_id="MS_001",
                scenario_type=L1ScenarioType.L1_POLLUTION_DETECTED,
                pool_id=2,
                severity=ScenarioSeverity.HIGH,
            ),
            L1ScenarioEvent(
                event_id="MS_002",
                scenario_type=L1ScenarioType.L1_LEVEL_RAPID_RISE,
                pool_id=5,
                severity=ScenarioSeverity.MEDIUM,
            ),
            L1ScenarioEvent(
                event_id="MS_003",
                scenario_type=L1ScenarioType.L1_GATE_STUCK,
                pool_id=8,
                severity=ScenarioSeverity.HIGH,
            ),
        ]

        for sc in scenarios:
            sim.inject_scenario(sc.pool_id, sc)

        assert len(sim.active_scenarios) == 3

        # 运行仿真
        sim.run(duration=600.0)

        # 验证污染池的水质
        state = sim.get_pool_state(2)
        assert state.has_anomaly
        assert state.quality_index < 1.0

    def test_long_duration_simulation(self):
        """测试长时间仿真"""
        sim = FullLineHydraulicSimulator(num_pools=20, dt=60.0)

        # 运行1小时仿真
        history = sim.run(duration=3600.0)

        assert sim.state.step_count == 60
        assert len(history) >= 6  # 每10步记录一次

        # 检查水位在合理范围
        for pool_id in range(20):
            level = sim.get_pool_state(pool_id).water_level
            assert 0.5 <= level <= 6.0

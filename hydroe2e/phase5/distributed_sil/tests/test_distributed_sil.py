"""
分布式SIL框架集成测试

测试:
1. 各模块独立功能
2. 模块间接口
3. 完整仿真流程
4. KPI评估
"""

import pytest
import numpy as np
from datetime import datetime

from hydroe2e.phase5.distributed_sil.core.scenario_generator import (
    ScenarioGenerator, ScenarioType, NoiseType, NoiseModel
)
from hydroe2e.phase5.distributed_sil.models.reduced_order_engine import ReducedOrderEngine
from hydroe2e.phase5.distributed_sil.models.segmented_high_fidelity import SegmentedHighFidelityModel
from hydroe2e.phase5.distributed_sil.interfaces.boundary_assimilator import (
    BoundaryAssimilator, FusionConfig, FusionMode
)
from hydroe2e.phase5.distributed_sil.core.controller_orchestrator import (
    ControllerOrchestrator, ControlLevel
)
from hydroe2e.phase5.distributed_sil.evaluators.sil_evaluator import (
    SILEvaluator, EvaluationResult
)
from hydroe2e.phase5.distributed_sil.core.sil_framework import (
    DistributedSILFramework, SILConfig, SimulationMode
)


class TestScenarioGenerator:
    """场景生成器测试"""

    def test_init(self):
        """测试初始化"""
        gen = ScenarioGenerator(num_segments=10, dt=900.0, seed=42)
        assert gen.num_segments == 10
        assert gen.dt == 900.0

    def test_generate_noise_white(self):
        """测试白噪声生成"""
        gen = ScenarioGenerator(seed=42)
        noise_model = NoiseModel(
            noise_type=NoiseType.WHITE,
            amplitude=0.1,
            mean=0.0,
        )
        noise = gen.generate_noise(noise_model, num_steps=100)
        assert noise.shape == (100, 1)
        assert np.abs(np.mean(noise)) < 0.05  # 均值接近0
        assert np.abs(np.std(noise) - 0.1) < 0.03  # 标准差接近幅度

    def test_generate_noise_ou(self):
        """测试OU过程噪声生成"""
        gen = ScenarioGenerator(seed=42)
        noise_model = NoiseModel(
            noise_type=NoiseType.ORNSTEIN_UHLENBECK,
            amplitude=0.1,
            mean=1.0,
            correlation_time=1000.0,
        )
        noise = gen.generate_noise(noise_model, num_steps=100)
        assert noise.shape == (100, 1)
        # OU过程应该均值回归到mean
        assert np.mean(noise[-20:]) > 0.5  # 后期应接近均值

    def test_generate_boundary_profile_steady(self):
        """测试稳态边界生成"""
        gen = ScenarioGenerator(dt=900.0)
        profile = gen.generate_boundary_profile(
            ScenarioType.STEADY_STATE,
            base_value=300.0,
            duration=3600.0,
        )
        assert len(profile) == 4  # 3600/900 = 4
        assert np.allclose(profile, 300.0)

    def test_generate_boundary_profile_step(self):
        """测试阶跃边界生成"""
        gen = ScenarioGenerator(dt=100.0)
        profile = gen.generate_boundary_profile(
            ScenarioType.STEP_CHANGE,
            base_value=300.0,
            duration=1000.0,
            step_time=500.0,
            step_magnitude=50.0,
        )
        assert len(profile) == 10
        assert profile[0] == 300.0
        assert profile[-1] == 350.0

    def test_create_scenario(self):
        """测试场景创建"""
        gen = ScenarioGenerator(num_segments=5, dt=900.0)
        scenario = gen.create_scenario(
            scenario_type=ScenarioType.STEADY_STATE,
            duration=86400.0,
            base_flow=300.0,
            base_level=4.0,
        )
        assert scenario.scenario_id is not None
        assert scenario.duration == 86400.0
        assert len(scenario.initial_states) == 5

    def test_parameter_ensemble(self):
        """测试参数集合生成"""
        gen = ScenarioGenerator()
        base_params = {"manning_n": 0.014, "discharge_coefficient": 0.6}
        ensemble = gen.generate_parameter_ensemble(base_params, ensemble_size=5)
        assert len(ensemble) == 5
        # 每个成员应该不同
        assert ensemble[0]["manning_n"] != ensemble[1]["manning_n"]


class TestReducedOrderEngine:
    """降阶引擎测试"""

    def test_init(self):
        """测试初始化"""
        engine = ReducedOrderEngine(num_pools=10, dt=900.0)
        assert engine.num_pools == 10
        assert engine.dt == 900.0
        assert len(engine.pool_states) == 10

    def test_reset(self):
        """测试重置"""
        engine = ReducedOrderEngine(num_pools=10)
        engine.reset(initial_level=4.0, initial_flow=300.0)
        assert engine.pool_states[0].water_level == 4.0
        assert engine.pool_states[0].inflow == 300.0

    def test_step(self):
        """测试单步仿真"""
        engine = ReducedOrderEngine(num_pools=5, dt=900.0)
        engine.reset(initial_level=4.0, initial_flow=300.0)

        result = engine.step(upstream_flow=300.0)
        assert "simulation_time" in result
        assert "pools" in result
        assert len(result["pools"]) == 5

    def test_mass_balance(self):
        """测试质量平衡"""
        engine = ReducedOrderEngine(num_pools=5, dt=900.0)
        engine.reset(initial_level=4.0, initial_flow=300.0)

        # Run several steps to allow transients to settle
        result = None
        for _ in range(20):
            result = engine.step(upstream_flow=300.0)

        mb = result["mass_balance"]
        # After settling, bound the mass balance error
        assert abs(mb["balance_error"]) < 5000.0

    def test_to_segment_states(self):
        """测试状态转换"""
        engine = ReducedOrderEngine(num_pools=5, dt=900.0)
        engine.reset()
        states = engine.to_segment_states()
        assert len(states) == 5
        assert "SEG_000" in states


class TestSegmentedHighFidelityModel:
    """高保真模型测试"""

    def test_init(self):
        """测试初始化"""
        model = SegmentedHighFidelityModel(
            segment_id="SEG_001",
            length=20000.0,
        )
        assert model.segment_id == "SEG_001"
        assert model.length == 20000.0

    def test_reset(self):
        """测试重置"""
        model = SegmentedHighFidelityModel(segment_id="SEG_001")
        model.reset(initial_level=4.0, initial_flow=300.0)
        assert model.h[0] == pytest.approx(4.0, rel=0.01)

    def test_step(self):
        """测试单步仿真"""
        model = SegmentedHighFidelityModel(segment_id="SEG_001")
        model.reset(initial_level=4.0, initial_flow=300.0)
        result = model.step()
        assert "time" in result
        assert "max_cfl" in result

    def test_get_state(self):
        """测试状态获取"""
        model = SegmentedHighFidelityModel(segment_id="SEG_001")
        model.reset()
        state = model.get_state()
        assert state.segment_id == "SEG_001"
        assert len(state.water_levels) > 0


class TestBoundaryAssimilator:
    """边界同化器测试"""

    def test_init(self):
        """测试初始化"""
        assim = BoundaryAssimilator(num_interfaces=10, dt=900.0)
        assert assim.num_interfaces == 10

    def test_assimilate_idz_only(self):
        """测试仅IDZ同化"""
        assim = BoundaryAssimilator(num_interfaces=5, dt=900.0)
        result = assim.assimilate(
            boundary_id="BND_000_001",
            idz_state={"level": 4.0, "flow": 300.0},
            fine_state=None,
        )
        assert result.estimated_level == pytest.approx(4.0, rel=0.01)
        assert result.estimated_flow == pytest.approx(300.0, rel=0.01)

    def test_assimilate_with_fine(self):
        """测试带高保真同化"""
        assim = BoundaryAssimilator(num_interfaces=5, dt=900.0)
        result = assim.assimilate(
            boundary_id="BND_000_001",
            idz_state={"level": 4.0, "flow": 300.0},
            fine_state={"level": 4.1, "flow": 305.0},
        )
        # 融合值应该在两者之间
        assert 4.0 <= result.estimated_level <= 4.1
        assert 300.0 <= result.estimated_flow <= 305.0

    def test_conservation_correction(self):
        """测试守恒纠偏"""
        assim = BoundaryAssimilator(num_interfaces=5, dt=900.0)
        assim.assimilate("BND_000_001", {"level": 4.0, "flow": 300.0})
        correction = assim.apply_conservation_correction("BND_000_001", volume_error=100.0)
        assert correction != 0


class TestControllerOrchestrator:
    """控制器编排器测试"""

    def test_init(self):
        """测试初始化"""
        orch = ControllerOrchestrator(num_gates=10)
        assert orch.num_gates == 10

    def test_create_default_controllers(self):
        """测试默认控制器创建"""
        orch = ControllerOrchestrator(num_gates=10)
        orch.create_default_controllers()
        assert len(orch.controllers) > 0

    def test_get_gate_openings(self):
        """测试闸门开度获取"""
        orch = ControllerOrchestrator(num_gates=5)
        openings = orch.get_gate_openings()
        assert len(openings) == 5
        assert all(0 <= o <= 1 for o in openings)


class TestSILEvaluator:
    """SIL评估器测试"""

    def test_init(self):
        """测试初始化"""
        evaluator = SILEvaluator(num_segments=10)
        assert evaluator.num_segments == 10

    def test_evaluate_empty(self):
        """测试空评估"""
        evaluator = SILEvaluator(num_segments=5)
        # 创建简单状态
        from hydroe2e.phase5.distributed_sil.interfaces.data_types import SegmentState, BoundaryCondition

        states = {}
        for i in range(5):
            seg_id = f"SEG_{i:03d}"
            states[seg_id] = SegmentState(
                segment_id=seg_id,
                timestamp=datetime.now(),
                water_levels=np.ones(20) * 4.0,
                flow_rates=np.ones(20) * 300.0,
                velocities=np.ones(20) * 1.0,
                upstream_level=4.0,
                downstream_level=3.99,
                upstream_flow=300.0,
                downstream_flow=300.0,
            )

        boundaries = {}
        for i in range(4):
            bnd_id = f"BND_{i:03d}_{i+1:03d}"
            boundaries[bnd_id] = BoundaryCondition(
                boundary_id=bnd_id,
                upstream_segment_id=f"SEG_{i:03d}",
                downstream_segment_id=f"SEG_{i+1:03d}",
                timestamp=datetime.now(),
                water_level=4.0,
                flow_rate=300.0,
            )

        kpi = evaluator.evaluate(states, boundaries, time=0.0)
        assert kpi is not None
        assert kpi.overall_score >= 0


class TestDistributedSILFramework:
    """完整框架集成测试"""

    def test_init(self):
        """测试初始化"""
        config = SILConfig(
            num_segments=10,
            num_gates=11,
            simulation_mode=SimulationMode.HYBRID,
            high_fidelity_segments=[0, 5, 9],
        )
        framework = DistributedSILFramework(config=config)
        assert framework.config.num_segments == 10
        assert len(framework.high_fidelity_models) == 3

    def test_reset(self):
        """测试重置"""
        config = SILConfig(num_segments=5, num_gates=6)
        framework = DistributedSILFramework(config=config)
        framework.reset(initial_level=4.0, initial_flow=300.0)
        assert framework.current_time == 0.0
        assert framework.step_count == 0

    def test_load_scenario(self):
        """测试场景加载"""
        config = SILConfig(num_segments=5, num_gates=6)
        framework = DistributedSILFramework(config=config)
        scenario = framework.load_scenario(
            scenario_type=ScenarioType.STEADY_STATE,
            duration=3600.0,
        )
        assert scenario is not None
        assert scenario.duration == 3600.0

    def test_single_step(self):
        """测试单步仿真"""
        config = SILConfig(
            num_segments=5,
            num_gates=6,
            simulation_mode=SimulationMode.REDUCED_ORDER_ONLY,
        )
        framework = DistributedSILFramework(config=config)
        framework.reset()

        result = framework.step(upstream_flow=300.0)
        assert "time" in result
        assert "states" in result
        assert len(result["states"]) == 5

    def test_short_simulation(self):
        """测试短时仿真"""
        config = SILConfig(
            num_segments=5,
            num_gates=6,
            dt_reduced=900.0,
            simulation_mode=SimulationMode.REDUCED_ORDER_ONLY,
        )
        framework = DistributedSILFramework(config=config)

        scenario = framework.load_scenario(
            scenario_type=ScenarioType.STEADY_STATE,
            duration=7200.0,  # 2小时
            base_flow=300.0,
        )

        result = framework.run_simulation(scenario)
        assert result is not None
        assert result.scenario_id == scenario.scenario_id

    def test_generate_report(self):
        """测试报告生成"""
        config = SILConfig(num_segments=5, num_gates=6)
        framework = DistributedSILFramework(config=config)
        framework.reset()
        framework.step(upstream_flow=300.0)

        report = framework.generate_report()
        assert "framework_config" in report
        assert "simulation_summary" in report


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 配置
        config = SILConfig(
            num_segments=10,
            num_gates=11,
            dt_reduced=900.0,
            simulation_mode=SimulationMode.HYBRID,
            high_fidelity_segments=[0, 5, 9],
        )

        # 2. 创建框架
        framework = DistributedSILFramework(config=config)

        # 3. 加载场景
        scenario = framework.load_scenario(
            scenario_type=ScenarioType.STEP_CHANGE,
            duration=7200.0,
            base_flow=300.0,
            step_magnitude=20.0,
        )

        # 4. 运行仿真
        result = framework.run_simulation(scenario)

        # 5. 验证结果
        assert result is not None
        assert len(result.state_history) > 0
        assert result.final_kpi is not None

        # 6. 生成报告
        report = framework.generate_report()
        assert report["simulation_summary"]["total_steps"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

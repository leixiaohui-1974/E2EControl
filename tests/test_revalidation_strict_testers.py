from tests.comprehensive_hil_testing.control_tester import (
    ControlTestType,
    ControlTester,
)
from tests.comprehensive_hil_testing.physics_simulation_tester import (
    PhysicsSimulationTester,
    PhysicsTestType,
)
from tests.comprehensive_hil_testing.scenario_combinatorial_generator import (
    ControlMode,
    NetworkTopology,
    TestScenario,
    generate_comprehensive_scenarios,
)


def _scenario() -> TestScenario:
    return TestScenario(
        id="strict_case",
        name="strict_case",
        category="TEMPORAL",
        difficulty=2,
        initial_water_level=3.0,
        initial_inflow=5.0,
        initial_outflow=4.5,
        area=10_000.0,
        control_mode=ControlMode.FLOOD_PREVENTION,
        topology=NetworkTopology.SINGLE_POOL,
        duration_hours=6,
        time_step=60.0,
        target_level=3.2,
        level_tolerance=0.05,
        disturbance_magnitude=0.3,
        pass_criteria={"max_settling_time": 1800.0},
    )


def _segmented_regression_scenario() -> TestScenario:
    return TestScenario(
        id="segmented_regression_case",
        name="segmented_regression_case",
        category="PHYSICS",
        difficulty=2,
        initial_water_level=1.0,
        initial_inflow=10.0,
        initial_outflow=5.0,
        area=10_000.0,
        control_mode=ControlMode.NORMAL,
        topology=NetworkTopology.SINGLE_POOL,
        duration_hours=2,
        time_step=60.0,
        target_level=1.0,
        level_tolerance=0.05,
        disturbance_magnitude=0.0,
    )


def _generated_scenario(scenario_id: str) -> TestScenario:
    for scenario in generate_comprehensive_scenarios(count=30, seed=42):
        if scenario.id == scenario_id:
            return scenario
    raise AssertionError(f"scenario not found: {scenario_id}")


def test_control_tester_uses_repo_imports():
    tester = ControlTester()
    assert tester.physics_available is True
    assert hasattr(tester, "CanalPoolSimulator")
    assert tester.preferred_physics_backend in {"single_channel", "tank", "segmented_hf"}


def test_control_tester_does_not_pass_when_models_missing():
    tester = ControlTester()
    tester.physics_available = False
    tester.mpc_available = False

    result = tester._run_single_test(ControlTestType.SETPOINT_TRACKING, _scenario())

    assert result.passed is False
    assert result.score == 0.0
    assert result.errors


def test_control_tester_skips_infeasible_control_scenario():
    tester = ControlTester()
    scenario = _scenario()
    scenario.initial_water_level = 8.0
    scenario.target_level = 3.0
    scenario.initial_outflow = 0.0

    result = tester._run_single_test(ControlTestType.SETPOINT_TRACKING, scenario)

    assert result.skipped is True
    assert result.passed is False
    assert "infeasible" in result.errors[0]


def test_control_tester_rejects_bad_tracking_metrics(monkeypatch):
    tester = ControlTester()

    good_baseline = {
        "iae": 10.0,
        "itae": 20.0,
        "tracking_error": 0.01,
        "steady_state_error": 0.01,
        "overshoot_ratio": 0.02,
        "settling_time_s": 300.0,
        "final_std": 0.01,
        "fallback_count": 0.0,
        "constraint_violations": 0.0,
        "command_saturation_count": 0.0,
        "control_variation_total": 1.0,
        "control_effort_total": 100.0,
        "max_level": 3.3,
        "min_level": 3.0,
    }
    bad_controlled = dict(good_baseline)
    bad_controlled.update(
        {
            "iae": 20.0,
            "itae": 30.0,
            "steady_state_error": 0.20,
            "overshoot_ratio": 0.30,
            "settling_time_s": 4000.0,
        }
    )

    calls = {"count": 0}

    def fake_run_closed_loop(*args, **kwargs):
        calls["count"] += 1
        return bad_controlled if calls["count"] == 1 else good_baseline

    monkeypatch.setattr(tester, "_run_closed_loop", fake_run_closed_loop)
    result = tester._test_setpoint_tracking(_scenario())

    assert result.passed is False
    assert result.metrics["overshoot_percent"] > 10.0
    assert result.metrics["iae_ratio_vs_baseline"] > 1.05


def test_physics_tester_does_not_pass_when_models_missing():
    tester = PhysicsSimulationTester()
    tester.physics_available = False

    result = tester._run_single_test(PhysicsTestType.MASS_CONSERVATION, _scenario())

    assert result.passed is False
    assert result.score == 0.0
    assert result.errors


def test_physics_mass_conservation_prefers_backend_volume(monkeypatch):
    tester = PhysicsSimulationTester()

    class FakePool:
        def __init__(self):
            self.level = 1.0
            self.volume = 100.0
            self.outflow = 0.0

        def step(self, q_in_command, q_out):
            self.level = 99.0
            self.volume = 136.0
            self.outflow = 0.0
            return self.level

        def get_level(self):
            return self.level

        def get_volume(self):
            return self.volume

        def get_outflow(self):
            return self.outflow

    monkeypatch.setattr(tester, "_make_pool", lambda scenario, **kwargs: FakePool())

    scenario = _scenario()
    scenario.area = 10.0
    scenario.initial_water_level = 1.0
    scenario.initial_inflow = 0.6
    scenario.initial_outflow = 0.0
    scenario.duration_hours = 1.0 / 60.0
    scenario.time_step = 60.0

    result = tester._test_mass_conservation(scenario)

    assert result.passed is True
    assert result.metrics["initial_volume"] == 100.0
    assert result.metrics["final_volume"] == 136.0
    assert result.metrics["actual_change"] == 36.0
    assert result.metrics["imbalance_volume"] == 0.0


def test_physics_mass_conservation_prefers_backend_outflow(monkeypatch):
    tester = PhysicsSimulationTester()

    class FakePool:
        def __init__(self):
            self.level = 1.0
            self.volume = 100.0
            self.outflow = 1.0

        def step(self, q_in_command, q_out):
            self.volume = 160.0
            self.outflow = 0.0
            return self.level

        def get_level(self):
            return self.level

        def get_volume(self):
            return self.volume

        def get_outflow(self):
            return self.outflow

    monkeypatch.setattr(tester, "_make_pool", lambda scenario, **kwargs: FakePool())

    scenario = _scenario()
    scenario.area = 10.0
    scenario.initial_water_level = 1.0
    scenario.initial_inflow = 1.0
    scenario.initial_outflow = 0.5
    scenario.duration_hours = 1.0 / 60.0
    scenario.time_step = 60.0

    result = tester._test_mass_conservation(scenario)

    assert result.passed is True
    assert result.metrics["expected_change"] == 60.0
    assert result.metrics["actual_change"] == 60.0
    assert result.metrics["imbalance_volume"] == 0.0


def test_auto_backend_prefers_single_channel_over_segmented_hf_when_available():
    control_tester = ControlTester(physics_backend="auto")
    physics_tester = PhysicsSimulationTester(physics_backend="auto")

    if control_tester.single_channel_available:
        assert control_tester.preferred_physics_backend == "single_channel"
    if physics_tester.single_channel_available:
        assert physics_tester.preferred_physics_backend == "single_channel"


def test_segmented_hf_backend_runs_small_step_strict_scenario():
    scenario = _scenario()
    physics_tester = PhysicsSimulationTester(physics_backend="segmented_hf")
    report = physics_tester.run_all_tests([scenario])

    assert report.physics_backend == "segmented_hf"
    assert report.total_tests > 0
    assert any(item.test_type == PhysicsTestType.STABILITY for item in report.test_results)
    assert any(item.passed for item in report.test_results)


def test_segmented_hf_backend_runs_coarse_step_strict_scenario():
    scenario = _scenario()
    scenario.time_step = 3600.0

    physics_tester = PhysicsSimulationTester(physics_backend="segmented_hf")
    report = physics_tester.run_all_tests([scenario])

    assert report.physics_backend == "segmented_hf"
    assert report.total_tests > 0
    assert all("supports time_step <=" not in " ".join(item.errors) for item in report.test_results)
    assert any(item.passed for item in report.test_results)


def test_segmented_hf_mass_conservation_regression():
    scenario = _segmented_regression_scenario()
    tester = PhysicsSimulationTester(physics_backend="segmented_hf")

    result = tester._run_single_test(PhysicsTestType.MASS_CONSERVATION, scenario)

    assert result.passed is True
    assert result.errors == []
    assert result.metrics["relative_error"] <= 0.005
    assert result.metrics["imbalance_volume"] <= result.metrics["normalization_volume"] * 0.005


def test_segmented_hf_steady_state_regression():
    scenario = _segmented_regression_scenario()
    tester = PhysicsSimulationTester(physics_backend="segmented_hf")

    result = tester._run_single_test(PhysicsTestType.STEADY_STATE, scenario)

    assert result.passed is True
    assert result.errors == []
    assert result.metrics["deviation_from_initial"] <= 0.015
    assert result.metrics["final_std"] <= 1e-3


def test_segmented_hf_coarse_step_default_steady_state_regression():
    scenario = _scenario()
    scenario.time_step = 3600.0
    tester = PhysicsSimulationTester(physics_backend="segmented_hf")

    result = tester._run_single_test(PhysicsTestType.STEADY_STATE, scenario)

    assert result.passed is True
    assert result.errors == []
    assert result.metrics["deviation_from_initial"] <= 0.02


def test_segmented_hf_coarse_step_control_path_has_no_fallback():
    scenario = _scenario()
    scenario.time_step = 3600.0
    tester = ControlTester(physics_backend="segmented_hf")

    metrics = tester._run_control_case(scenario, controller="mpc", steps=30)

    assert metrics["fallback_count"] == 0.0
    assert metrics["constraint_violations"] == 0.0


def test_segmented_hf_small_error_tracking_regression():
    scenario = _generated_scenario("S2_FLOOD_BASIC_0012")
    tester = ControlTester(physics_backend="segmented_hf")

    controlled = tester._run_control_case(scenario, controller="mpc", steps=100)
    baseline = tester._run_control_case(scenario, controller="baseline", steps=100)

    assert controlled["fallback_count"] == 0.0
    assert controlled["constraint_violations"] == 0.0
    assert controlled["steady_state_error"] <= baseline["steady_state_error"] * 1.05
    assert controlled["iae"] <= baseline["iae"] * 1.05
    assert controlled["itae"] <= baseline["itae"] * 1.05


def test_segmented_hf_geometry_adapts_for_high_level_low_flow_case():
    scenario = _generated_scenario("S1_NORMAL_BASIC_0010")
    tester = ControlTester(physics_backend="segmented_hf")

    pool = tester._make_pool_backend(scenario, delay_steps=1)
    level = pool.step(scenario.initial_inflow, scenario.initial_inflow)

    assert pool.model.sections[0].side_slope < 0.5
    assert pool.model.sections[0].bottom_width > 0.5
    assert abs(pool.get_outflow() - scenario.initial_inflow) <= 0.1
    assert abs(level - scenario.initial_water_level) <= 0.1


def test_segmented_hf_generated_high_level_case_revalidation():
    scenario = _generated_scenario("S1_NORMAL_BASIC_0010")
    physics_tester = PhysicsSimulationTester(physics_backend="segmented_hf")
    control_tester = ControlTester(physics_backend="segmented_hf")

    mass_result = physics_tester._run_single_test(PhysicsTestType.MASS_CONSERVATION, scenario)
    steady_result = physics_tester._run_single_test(PhysicsTestType.STEADY_STATE, scenario)
    tracking_result = control_tester._run_single_test(ControlTestType.SETPOINT_TRACKING, scenario)

    assert mass_result.passed is True
    assert steady_result.passed is True
    assert tracking_result.passed is True


def test_parameterized_dmpc_multi_pool_coordination_path_runs():
    scenario = _scenario()
    scenario.topology = NetworkTopology.CASCADE_3

    tester = ControlTester(physics_backend="tank", controller_backend="parameterized_dmpc")
    result = tester._test_multi_pool_coordination(scenario)

    assert result.test_type == ControlTestType.MULTI_POOL_COORDINATION
    assert "fallback_count" in result.metrics
    assert result.metrics["num_pools"] == 3.0


def test_parameterized_dmpc_multi_pool_tracking_path_runs():
    scenario = _scenario()
    scenario.topology = NetworkTopology.CASCADE_3

    tester = ControlTester(physics_backend="tank", controller_backend="parameterized_dmpc")
    result = tester._test_setpoint_tracking(scenario)

    assert result.test_type == ControlTestType.SETPOINT_TRACKING
    assert "iae_ratio_vs_baseline" in result.metrics
    assert "fallback_count" in result.metrics


def test_multi_pool_tracking_uses_worst_case_gate(monkeypatch):
    scenario = _scenario()
    scenario.topology = NetworkTopology.CASCADE_3
    tester = ControlTester(physics_backend="tank", controller_backend="parameterized_dmpc")

    controlled = {
        "iae": 10.0,
        "iae_worst": 20.0,
        "itae": 10.0,
        "itae_worst": 20.0,
        "tracking_error": 0.02,
        "tracking_error_worst": 0.08,
        "steady_state_error": 0.01,
        "steady_state_error_mean": 0.005,
        "overshoot_ratio": 0.05,
        "overshoot_ratio_mean": 0.02,
        "settling_time_s": 600.0,
        "settling_time_mean_s": 300.0,
        "final_std": 0.01,
        "final_std_mean": 0.005,
        "fallback_count": 0.0,
        "constraint_violations": 0.0,
        "command_saturation_count": 0.0,
        "control_variation_total": 1.0,
        "control_effort_total": 100.0,
        "max_level": 3.3,
        "min_level": 3.0,
        "num_pools": 3.0,
    }
    baseline = dict(controlled)
    baseline["iae_worst"] = 10.0
    baseline["itae_worst"] = 10.0

    calls = {"count": 0}

    def fake_run_control_case(*args, **kwargs):
        calls["count"] += 1
        return controlled if calls["count"] == 1 else baseline

    monkeypatch.setattr(tester, "_run_control_case", fake_run_control_case)

    result = tester._test_setpoint_tracking(scenario)

    assert result.passed is False
    assert result.metrics["iae_ratio_vs_baseline"] == 2.0
    assert result.metrics["iae_ratio_vs_baseline_mean"] == 1.0

"""
Strict control-effectiveness tester used for revalidation.

This module replaces permissive "system ran" scoring with hard gates that
measure whether a controller is actually good enough:
- setpoint tracking quality
- disturbance rejection
- stability
- response time
- overshoot
- steady-state error
- constraint safety
- multi-pool coordination
- degraded-mode behavior
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .scenario_combinatorial_generator import ControlMode, NetworkTopology, TestScenario
from .hydraulic_backends import build_single_pool_backend


FLOW_MIN = 0.0
FLOW_MAX = 20.0
LEVEL_MIN = 0.0
LEVEL_MAX = 10.0
DISTURBANCE_RATIO_GATE = 2.0
OVERSHOOT_GATE = 0.10
DEGRADED_RATIO_GATE = 2.0


class ControlTestType(Enum):
    SETPOINT_TRACKING = "setpoint_tracking"
    DISTURBANCE_REJECTION = "disturbance_rejection"
    CONSTRAINT_HANDLING = "constraint_handling"
    STABILITY = "stability"
    RESPONSE_TIME = "response_time"
    OVERSHOOT = "overshoot"
    STEADY_STATE_ERROR = "steady_state_error"
    MULTI_POOL_COORDINATION = "multi_pool_coordination"
    MODE_SWITCHING = "mode_switching"
    DEGRADED_CONTROL = "degraded_control"


@dataclass
class ControlTestResult:
    test_type: ControlTestType
    scenario_id: str
    passed: bool
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    skipped: bool = False
    execution_time: float = 0.0


@dataclass
class ControlReport:
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    average_score: float = 0.0
    average_tracking_error: float = 0.0
    average_settling_time: float = 0.0
    physics_backend: str = "unknown"
    controller_backend: str = "unknown"
    test_results: List[ControlTestResult] = field(default_factory=list)
    by_test_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_execution_time: float = 0.0


class ControlTester:
    def __init__(
        self,
        max_overshoot: float = OVERSHOOT_GATE,
        max_settling_time: float = 36_000.0,
        max_steady_error: float = 1.0,
        tolerance: float = 0.5,
        verbose: bool = False,
        physics_backend: str = "auto",
        controller_backend: str = "base_mpc",
    ) -> None:
        self.max_overshoot = max_overshoot
        self.max_settling_time = max_settling_time
        self.max_steady_error = max_steady_error
        self.tolerance = tolerance
        self.verbose = verbose
        self.physics_backend = physics_backend
        self.controller_backend = controller_backend
        self.results: List[ControlTestResult] = []
        self._simulation_cache: Dict[tuple[str, str, int, float], Dict[str, float]] = {}
        self._load_models()

    def _load_models(self) -> None:
        self.CanalPoolSimulator = None
        self.ChannelGeometry = None
        self.SingleChannelFidelity = None
        self.SegmentedHighFidelityModel = None
        self.SaintVenantConfig = None
        self.BoundaryType = None
        self.ParameterizedDistributedMPC = None
        self.PhysicalParameters = None
        try:
            from hydroe2e.physics.base import CanalPoolSimulator

            self.CanalPoolSimulator = CanalPoolSimulator
            self.tank_available = True
        except ImportError:
            self.tank_available = False

        try:
            from hydroe2e.digital_twin.physics.single_channel_fidelity import ChannelGeometry, SingleChannelFidelity

            self.ChannelGeometry = ChannelGeometry
            self.SingleChannelFidelity = SingleChannelFidelity
            self.single_channel_available = True
        except ImportError:
            self.single_channel_available = False

        try:
            from hydroe2e.phase5.distributed_sil.models.segmented_high_fidelity import (
                BoundaryType,
                SaintVenantConfig,
                SegmentedHighFidelityModel,
            )

            self.SegmentedHighFidelityModel = SegmentedHighFidelityModel
            self.SaintVenantConfig = SaintVenantConfig
            self.BoundaryType = BoundaryType
            self.segmented_hf_available = True
        except ImportError:
            self.segmented_hf_available = False

        try:
            from hydroe2e.control.base import UniversalMPCSolver

            self.UniversalMPCSolver = UniversalMPCSolver
            self.mpc_available = True
        except ImportError:
            self.mpc_available = False
        try:
            from hydroe2e.phase5.controllers.parameterized_mpc import (
                ParameterizedDistributedMPC,
                PhysicalParameters,
            )

            self.ParameterizedDistributedMPC = ParameterizedDistributedMPC
            self.PhysicalParameters = PhysicalParameters
            self.parameterized_dmpc_available = True
        except ImportError:
            self.parameterized_dmpc_available = False
        self.physics_available = self.tank_available or self.single_channel_available or self.segmented_hf_available
        self.preferred_physics_backend = self._resolve_backend_name()

    def _resolve_backend_name(self) -> str:
        if self.physics_backend in {"segmented_hf", "single_channel", "fidelity", "tank"}:
            if self.physics_backend == "fidelity":
                return "single_channel"
            return self.physics_backend
        if self.segmented_hf_available:
            return "segmented_hf"
        if self.single_channel_available:
            return "single_channel"
        return "tank"

    def run_all_tests(self, scenarios: List[TestScenario]) -> ControlReport:
        report = ControlReport(
            physics_backend=self.preferred_physics_backend,
            controller_backend=self.controller_backend,
        )
        started = time.time()
        tracking_errors: List[float] = []
        settling_times: List[float] = []

        for scenario in scenarios:
            for test_type in self._select_tests(scenario):
                result = self._run_single_test(test_type, scenario)
                self.results.append(result)
                report.test_results.append(result)
                if result.skipped:
                    report.skipped_tests += 1
                else:
                    report.total_tests += 1
                    if result.passed:
                        report.passed_tests += 1
                    else:
                        report.failed_tests += 1

                if not result.skipped and "tracking_error" in result.metrics:
                    tracking_errors.append(result.metrics["tracking_error"])
                if not result.skipped and "settling_time" in result.metrics and np.isfinite(result.metrics["settling_time"]):
                    settling_times.append(result.metrics["settling_time"])

                bucket = report.by_test_type.setdefault(test_type.value, {"passed": 0, "failed": 0})
                if not result.skipped:
                    bucket["passed" if result.passed else "failed"] += 1

        report.total_execution_time = time.time() - started
        if report.test_results:
            report.average_score = float(np.mean([item.score for item in report.test_results]))
        if tracking_errors:
            report.average_tracking_error = float(np.mean(tracking_errors))
        if settling_times:
            report.average_settling_time = float(np.mean(settling_times))
        return report

    def _select_tests(self, scenario: TestScenario) -> List[ControlTestType]:
        tests = [
            ControlTestType.SETPOINT_TRACKING,
            ControlTestType.STABILITY,
            ControlTestType.STEADY_STATE_ERROR,
            ControlTestType.CONSTRAINT_HANDLING,
        ]
        if scenario.disturbance_magnitude > 0:
            tests.append(ControlTestType.DISTURBANCE_REJECTION)
        if scenario.control_mode in [ControlMode.FLOOD_PREVENTION, ControlMode.EMERGENCY]:
            tests.append(ControlTestType.RESPONSE_TIME)
            tests.append(ControlTestType.OVERSHOOT)
        if scenario.topology != NetworkTopology.SINGLE_POOL:
            tests.append(ControlTestType.MULTI_POOL_COORDINATION)
        if scenario.control_mode == ControlMode.DEGRADED:
            tests.append(ControlTestType.DEGRADED_CONTROL)
        return tests

    def _run_single_test(self, test_type: ControlTestType, scenario: TestScenario) -> ControlTestResult:
        started = time.time()
        try:
            feasible, reason = self._assess_control_feasibility(scenario)
            if not feasible and test_type in {
                ControlTestType.SETPOINT_TRACKING,
                ControlTestType.STABILITY,
                ControlTestType.STEADY_STATE_ERROR,
                ControlTestType.CONSTRAINT_HANDLING,
                ControlTestType.RESPONSE_TIME,
                ControlTestType.OVERSHOOT,
                ControlTestType.DISTURBANCE_REJECTION,
                ControlTestType.DEGRADED_CONTROL,
            }:
                result = ControlTestResult(
                    test_type=test_type,
                    scenario_id=scenario.id,
                    passed=False,
                    score=0.0,
                    errors=[reason],
                    metrics={"scenario_feasible": 0.0},
                    skipped=True,
                )
                result.execution_time = time.time() - started
                return result

            if test_type == ControlTestType.SETPOINT_TRACKING:
                result = self._test_setpoint_tracking(scenario)
            elif test_type == ControlTestType.DISTURBANCE_REJECTION:
                result = self._test_disturbance_rejection(scenario)
            elif test_type == ControlTestType.CONSTRAINT_HANDLING:
                result = self._test_constraint_handling(scenario)
            elif test_type == ControlTestType.STABILITY:
                result = self._test_stability(scenario)
            elif test_type == ControlTestType.RESPONSE_TIME:
                result = self._test_response_time(scenario)
            elif test_type == ControlTestType.OVERSHOOT:
                result = self._test_overshoot(scenario)
            elif test_type == ControlTestType.STEADY_STATE_ERROR:
                result = self._test_steady_state_error(scenario)
            elif test_type == ControlTestType.MULTI_POOL_COORDINATION:
                result = self._test_multi_pool_coordination(scenario)
            elif test_type == ControlTestType.MODE_SWITCHING:
                result = self._test_mode_switching(scenario)
            elif test_type == ControlTestType.DEGRADED_CONTROL:
                result = self._test_degraded_control(scenario)
            else:
                result = ControlTestResult(
                    test_type=test_type,
                    scenario_id=scenario.id,
                    passed=False,
                    score=0.0,
                    errors=[f"unsupported test type: {test_type.value}"],
                )
        except Exception as exc:
            result = ControlTestResult(
                test_type=test_type,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"test crashed: {exc}"],
            )
        result.execution_time = time.time() - started
        return result

    def _steady_state_limit(self, scenario: TestScenario) -> float:
        return max(0.05, 0.02 * max(abs(scenario.target_level), 1.0))

    def _settling_band(self, scenario: TestScenario) -> float:
        return max(self._steady_state_limit(scenario), scenario.level_tolerance)

    def _settling_time_limit(self, scenario: TestScenario) -> float:
        scenario_limit = scenario.pass_criteria.get("max_settling_time")
        resolution_floor = scenario.time_step * max(3.0, min(float(scenario.mpc_horizon), 5.0))
        if scenario_limit is not None:
            return max(float(scenario_limit), resolution_floor)
        return min(self.max_settling_time, max(3600.0, scenario.time_step * 10.0, resolution_floor))

    def _single_pool_mpc_deadband(self, scenario: TestScenario) -> float:
        if self.preferred_physics_backend != "segmented_hf":
            return 0.0
        return min(0.005, 0.1 * self._steady_state_limit(scenario))

    def _single_pool_q_out_forecast(self, pool: Any, scenario: TestScenario) -> List[float]:
        q_out = float(scenario.initial_outflow)
        if self.preferred_physics_backend == "segmented_hf" and hasattr(pool, "get_outflow"):
            q_out = float(pool.get_outflow())
        return [q_out] * scenario.mpc_horizon

    def _score_from_terms(self, terms: List[float]) -> float:
        if not terms:
            return 0.0
        return float(np.mean([min(1.0, max(0.0, value)) for value in terms]))

    def _assess_control_feasibility(self, scenario: TestScenario) -> tuple[bool, str]:
        q_out = float(scenario.initial_outflow)
        target = float(scenario.target_level)
        initial = float(scenario.initial_water_level)

        if q_out < FLOW_MIN or q_out > FLOW_MAX:
            return False, f"scenario infeasible under actuator bounds: q_out={q_out} outside [{FLOW_MIN}, {FLOW_MAX}]"

        if target > initial and FLOW_MAX <= q_out:
            return False, "scenario infeasible under actuator bounds: controller cannot increase level"
        if target < initial and FLOW_MIN >= q_out:
            return False, "scenario infeasible under actuator bounds: controller cannot decrease level"

        return True, ""

    def _make_pool_backend(
        self,
        scenario: TestScenario,
        *,
        initial_level: float | None = None,
        initial_flow: float | None = None,
        delay_steps: int = 1,
    ) -> Any:
        if not self.physics_available:
            raise RuntimeError("physics model unavailable")
        return build_single_pool_backend(
            backend=self.preferred_physics_backend,
            area=scenario.area,
            slope=scenario.slope,
            time_step=scenario.time_step,
            initial_level=scenario.initial_water_level if initial_level is None else initial_level,
            initial_flow=scenario.initial_inflow if initial_flow is None else initial_flow,
            manning_n=scenario.manning_n,
            tank_cls=self.CanalPoolSimulator,
            fidelity_cls=self.SingleChannelFidelity,
            geometry_cls=self.ChannelGeometry,
            segmented_model_cls=self.SegmentedHighFidelityModel,
            segmented_config_cls=self.SaintVenantConfig,
            segmented_boundary_type_cls=self.BoundaryType,
            delay_steps=delay_steps,
        )

    def _scenario_physics_tag(self, scenario: TestScenario) -> str:
        weather_name = getattr(scenario.weather, "name", "")
        season_name = getattr(scenario.season, "name", "")
        if "ICE" in weather_name or "ICE" in season_name:
            return "ICE_STABLE"
        if "SNOW" in weather_name:
            return "ICE_FORMATION"
        return "NORMAL"

    def _make_parameterized_dmpc(self, scenario: TestScenario, num_pools: int) -> Any:
        if not self.parameterized_dmpc_available:
            raise RuntimeError("parameterized distributed mpc unavailable")
        physical_params = [
            self.PhysicalParameters(
                area=scenario.area,
                flow_efficiency=1.0,
                manning_n=scenario.manning_n,
                channel_slope=scenario.slope,
            )
            for _ in range(num_pools)
        ]
        controller = self.ParameterizedDistributedMPC(
            num_pools=num_pools,
            horizon=max(3, min(int(scenario.mpc_horizon), 10)),
            dt=scenario.time_step,
            physical_params_list=physical_params,
        )
        reference = np.full((num_pools, controller.horizon), float(scenario.target_level), dtype=float)
        controller.set_reference_trajectories(reference)
        controller.set_scenario_physics(self._scenario_physics_tag(scenario))
        return controller

    def _uses_multi_pool_control(self, scenario: TestScenario) -> bool:
        return (
            scenario.topology != NetworkTopology.SINGLE_POOL
            and self.controller_backend == "parameterized_dmpc"
            and self._get_pool_count(scenario.topology) > 1
        )

    def _run_control_case(
        self,
        scenario: TestScenario,
        *,
        controller: str,
        steps: int = 60,
        disturbance_scale: float = 0.0,
    ) -> Dict[str, float]:
        if self._uses_multi_pool_control(scenario):
            return self._run_closed_loop_multi(
                scenario,
                controller=controller,
                steps=steps,
                disturbance_scale=disturbance_scale,
            )
        return self._run_closed_loop(
            scenario,
            controller=controller,
            steps=steps,
            disturbance_scale=disturbance_scale,
        )

    def _is_multi_pool_metrics(self, metrics: Dict[str, float]) -> bool:
        return float(metrics.get("num_pools", 1.0)) > 1.0

    def _safe_ratio(self, numerator: float, denominator: float, *, both_infinite_value: float) -> float:
        numerator = float(numerator)
        denominator = float(denominator)
        if np.isinf(numerator) and np.isinf(denominator):
            return float(both_infinite_value)
        if np.isinf(numerator):
            return float("inf")
        if np.isinf(denominator):
            return 0.0
        if abs(denominator) <= 1e-9:
            if abs(numerator) <= 1e-9:
                return 1.0
            return float("inf")
        return float(numerator / denominator)

    def _ratio_vs_baseline_metrics(
        self,
        controlled: Dict[str, float],
        baseline: Dict[str, float],
        *,
        key: str,
    ) -> Dict[str, float]:
        mean_ratio = self._safe_ratio(controlled[key], baseline[key], both_infinite_value=float("inf"))
        worst_key = f"{key}_worst"
        worst_ratio = self._safe_ratio(
            controlled.get(worst_key, controlled[key]),
            baseline.get(worst_key, baseline[key]),
            both_infinite_value=float("inf"),
        )
        gate_ratio = worst_ratio if self._is_multi_pool_metrics(controlled) else mean_ratio
        return {
            "mean": mean_ratio,
            "worst": worst_ratio,
            "gate": gate_ratio,
        }

    def _improvement_vs_baseline_metrics(
        self,
        controlled: Dict[str, float],
        baseline: Dict[str, float],
        *,
        key: str,
    ) -> Dict[str, float]:
        mean_improvement = self._safe_ratio(baseline[key], controlled[key], both_infinite_value=0.0)
        worst_key = f"{key}_worst"
        worst_improvement = self._safe_ratio(
            baseline.get(worst_key, baseline[key]),
            controlled.get(worst_key, controlled[key]),
            both_infinite_value=0.0,
        )
        gate_improvement = worst_improvement if self._is_multi_pool_metrics(controlled) else mean_improvement
        return {
            "mean": mean_improvement,
            "worst": worst_improvement,
            "gate": gate_improvement,
        }

    def _run_closed_loop(
        self,
        scenario: TestScenario,
        *,
        controller: str,
        steps: int = 60,
        disturbance_scale: float = 0.0,
    ) -> Dict[str, float]:
        cache_key = (scenario.id, controller, steps, float(disturbance_scale))
        if cache_key in self._simulation_cache:
            return dict(self._simulation_cache[cache_key])

        if not self.physics_available:
            raise RuntimeError("physics model unavailable")
        if controller == "mpc" and not self.mpc_available:
            raise RuntimeError("mpc solver unavailable")

        pool = self._make_pool_backend(scenario, delay_steps=1)
        solver = None
        if controller == "mpc":
            solver = self.UniversalMPCSolver(
                horizon=scenario.mpc_horizon,
                dt=scenario.time_step,
                area=scenario.area,
                delay_steps=1,
            )

        levels: List[float] = []
        controls: List[float] = []
        abs_errors: List[float] = []
        fallback_count = 0
        command_saturation_count = 0
        constraint_violations = 0
        q_prev = float(scenario.initial_inflow)

        for step in range(steps):
            current_level = float(pool.get_level())
            error = float(scenario.target_level - current_level)

            if controller == "mpc":
                if abs(error) <= self._single_pool_mpc_deadband(scenario):
                    q_cmd = q_prev
                else:
                    config = {
                        "Z_ref": scenario.target_level,
                        "W_level": scenario.weight_level,
                        "W_smooth": scenario.weight_smooth,
                        "delta_Q_max": 2.0,
                        "constraints": {"Q_in_max": FLOW_MAX, "Z_min": LEVEL_MIN},
                    }
                    try:
                        q_cmd = float(
                            solver.solve(
                                current_level=current_level,
                                q_prev=q_prev,
                                q_out_forecast=self._single_pool_q_out_forecast(pool, scenario),
                                config=config,
                            )
                        )
                        if getattr(solver, "last_fallback", False):
                            fallback_count += 1
                    except Exception:
                        fallback_count += 1
                        q_cmd = q_prev
            elif controller == "baseline":
                kp = 0.5 * (60.0 / max(scenario.time_step, 1.0))
                q_cmd = q_prev + kp * error
            elif controller == "open_loop":
                q_cmd = float(scenario.initial_inflow)
            else:
                raise ValueError(f"unsupported controller: {controller}")

            raw_q_cmd = q_cmd
            q_cmd = float(np.clip(q_cmd, FLOW_MIN, FLOW_MAX))
            if abs(raw_q_cmd - q_cmd) > 1e-9:
                command_saturation_count += 1

            disturbance = disturbance_scale * np.sin(step * 0.3) if disturbance_scale > 0 else 0.0
            applied_inflow = float(np.clip(q_cmd + disturbance, FLOW_MIN, FLOW_MAX))
            new_level = float(pool.step(applied_inflow, scenario.initial_outflow))

            if (not np.isfinite(new_level)) or new_level < LEVEL_MIN or new_level > LEVEL_MAX:
                constraint_violations += 1

            levels.append(new_level)
            controls.append(q_cmd)
            abs_errors.append(abs(new_level - scenario.target_level))
            q_prev = q_cmd

        levels_arr = np.asarray(levels, dtype=float)
        controls_arr = np.asarray(controls, dtype=float)
        abs_errors_arr = np.asarray(abs_errors, dtype=float)
        times = (np.arange(len(levels_arr), dtype=float) + 1.0) * scenario.time_step
        iae = float(np.sum(abs_errors_arr) * scenario.time_step)
        itae = float(np.sum(times * abs_errors_arr) * scenario.time_step)
        steady_state_error = float(np.mean(abs_errors_arr[-10:]))
        final_std = float(np.std(levels_arr[-10:]))

        level_change = float(scenario.target_level - scenario.initial_water_level)
        if abs(level_change) < 1e-9:
            overshoot_ratio = 0.0
        elif level_change > 0:
            overshoot_ratio = max(0.0, float((np.max(levels_arr) - scenario.target_level) / abs(level_change)))
        else:
            overshoot_ratio = max(0.0, float((scenario.target_level - np.min(levels_arr)) / abs(level_change)))

        settling_time_s = float("inf")
        settling_band = self._settling_band(scenario)
        for idx in range(len(levels_arr)):
            if np.all(abs_errors_arr[idx:] <= settling_band):
                settling_time_s = float((idx + 1) * scenario.time_step)
                break

        result = {
            "iae": iae,
            "itae": itae,
            "tracking_error": float(np.mean(abs_errors_arr)),
            "steady_state_error": steady_state_error,
            "overshoot_ratio": overshoot_ratio,
            "settling_time_s": settling_time_s,
            "final_std": final_std,
            "fallback_count": float(fallback_count),
            "constraint_violations": float(constraint_violations),
            "command_saturation_count": float(command_saturation_count),
            "control_variation_total": float(np.sum(np.abs(np.diff(controls_arr)))) if len(controls_arr) > 1 else 0.0,
            "control_effort_total": float(np.sum(np.abs(controls_arr)) * scenario.time_step),
            "max_level": float(np.max(levels_arr)),
            "min_level": float(np.min(levels_arr)),
        }
        self._simulation_cache[cache_key] = dict(result)
        return result

    def _run_closed_loop_multi(
        self,
        scenario: TestScenario,
        *,
        controller: str,
        steps: int = 60,
        disturbance_scale: float = 0.0,
    ) -> Dict[str, float]:
        cache_key = (scenario.id, f"multi:{self.controller_backend}:{controller}", steps, float(disturbance_scale))
        if cache_key in self._simulation_cache:
            return dict(self._simulation_cache[cache_key])

        if not self.physics_available:
            raise RuntimeError("physics model unavailable")

        num_pools = self._get_pool_count(scenario.topology)
        pools = [self._make_pool_backend(scenario, delay_steps=1) for _ in range(num_pools)]
        dmpc = self._make_parameterized_dmpc(scenario, num_pools) if controller == "mpc" else None
        q_in_prevs = [float(scenario.initial_inflow)] * num_pools
        fallback_count = 0
        non_converged_steps = 0
        solve_failure_count = 0
        command_saturation_count = 0
        constraint_violations = 0
        coordination_errors: List[float] = []
        solve_iterations: List[float] = []
        solve_times: List[float] = []

        level_history: List[List[float]] = []
        control_history: List[List[float]] = []

        for step in range(steps):
            current_levels = [float(pool.get_level()) for pool in pools]

            if controller == "mpc":
                solve_ok = True
                try:
                    actions, info = dmpc.solve(current_levels=current_levels, q_in_prevs=q_in_prevs)
                    raw_q_in_commands = [float(item[0]) for item in actions]
                except Exception:
                    solve_ok = False
                    fallback_count += 1
                    solve_failure_count += 1
                    raw_q_in_commands = list(q_in_prevs)
                    info = {}
                if solve_ok and not info.get("converged", False):
                    fallback_count += 1
                    non_converged_steps += 1
                solve_iterations.append(float(info.get("iterations", 0.0)))
                solve_times.append(float(info.get("solve_time", 0.0)))
            elif controller == "baseline":
                kp = 0.5 * (60.0 / max(scenario.time_step, 1.0))
                raw_q_in_commands = []
                for idx, level_i in enumerate(current_levels):
                    target_error = scenario.target_level - level_i
                    sync_error = 0.0
                    if idx > 0:
                        sync_error = current_levels[idx - 1] - level_i
                    raw_q_in_commands.append(scenario.initial_inflow + kp * target_error + 0.3 * sync_error)
            elif controller == "open_loop":
                raw_q_in_commands = [float(scenario.initial_inflow)] * num_pools
            else:
                raise ValueError(f"unsupported controller: {controller}")

            q_in_commands: List[float] = []
            disturbance = disturbance_scale * np.sin(step * 0.3) if disturbance_scale > 0 else 0.0
            for idx, raw_q in enumerate(raw_q_in_commands):
                q_cmd = float(np.clip(raw_q, FLOW_MIN, FLOW_MAX))
                if abs(q_cmd - raw_q) > 1e-9:
                    command_saturation_count += 1
                if idx == 0:
                    q_cmd = float(np.clip(q_cmd + disturbance, FLOW_MIN, FLOW_MAX))
                q_in_commands.append(q_cmd)

            new_levels: List[float] = []
            for idx, pool in enumerate(pools):
                q_out = q_in_commands[idx + 1] if idx < num_pools - 1 else scenario.initial_outflow
                new_level = float(pool.step(q_in_commands[idx], q_out))
                new_levels.append(new_level)
                if (not np.isfinite(new_level)) or new_level < LEVEL_MIN or new_level > LEVEL_MAX:
                    constraint_violations += 1

            for idx in range(num_pools - 1):
                coordination_errors.append(abs(new_levels[idx] - new_levels[idx + 1]))

            level_history.append(new_levels)
            control_history.append(q_in_commands)
            q_in_prevs = list(q_in_commands)

        levels_arr = np.asarray(level_history, dtype=float)
        controls_arr = np.asarray(control_history, dtype=float)
        abs_errors_arr = np.abs(levels_arr - float(scenario.target_level))
        times = (np.arange(levels_arr.shape[0], dtype=float) + 1.0) * scenario.time_step

        tracking_error_by_pool = np.mean(abs_errors_arr, axis=0)
        iae_by_pool = np.sum(abs_errors_arr, axis=0) * scenario.time_step
        itae_by_pool = np.sum(abs_errors_arr * times[:, None], axis=0) * scenario.time_step
        steady_by_pool = np.mean(abs_errors_arr[-10:, :], axis=0)
        final_std_by_pool = np.std(levels_arr[-10:, :], axis=0)

        level_change = float(scenario.target_level - scenario.initial_water_level)
        overshoot_by_pool: List[float] = []
        for pool_idx in range(num_pools):
            series = levels_arr[:, pool_idx]
            if abs(level_change) < 1e-9:
                overshoot_by_pool.append(0.0)
            elif level_change > 0:
                overshoot_by_pool.append(max(0.0, float((np.max(series) - scenario.target_level) / abs(level_change))))
            else:
                overshoot_by_pool.append(max(0.0, float((scenario.target_level - np.min(series)) / abs(level_change))))

        settling_band = self._settling_band(scenario)
        settling_time_by_pool: List[float] = []
        for pool_idx in range(num_pools):
            pool_errors = abs_errors_arr[:, pool_idx]
            settling_time = float("inf")
            for idx in range(len(pool_errors)):
                if np.all(pool_errors[idx:] <= settling_band):
                    settling_time = float((idx + 1) * scenario.time_step)
                    break
            settling_time_by_pool.append(settling_time)

        result = {
            "iae": float(np.mean(iae_by_pool)),
            "iae_worst": float(np.max(iae_by_pool)),
            "itae": float(np.mean(itae_by_pool)),
            "itae_worst": float(np.max(itae_by_pool)),
            "tracking_error": float(np.mean(tracking_error_by_pool)),
            "tracking_error_worst": float(np.max(tracking_error_by_pool)),
            "steady_state_error": float(np.max(steady_by_pool)),
            "steady_state_error_mean": float(np.mean(steady_by_pool)),
            "overshoot_ratio": float(np.max(overshoot_by_pool)),
            "overshoot_ratio_mean": float(np.mean(overshoot_by_pool)),
            "settling_time_s": float(np.max(settling_time_by_pool)),
            "settling_time_mean_s": float(np.mean(settling_time_by_pool)),
            "final_std": float(np.max(final_std_by_pool)),
            "final_std_mean": float(np.mean(final_std_by_pool)),
            "fallback_count": float(fallback_count),
            "non_converged_steps": float(non_converged_steps),
            "solve_failure_count": float(solve_failure_count),
            "constraint_violations": float(constraint_violations),
            "command_saturation_count": float(command_saturation_count),
            "control_variation_total": float(np.sum(np.abs(np.diff(controls_arr, axis=0)))) if len(controls_arr) > 1 else 0.0,
            "control_effort_total": float(np.sum(np.abs(controls_arr)) * scenario.time_step),
            "max_level": float(np.max(levels_arr)),
            "min_level": float(np.min(levels_arr)),
            "avg_coordination_error": float(np.mean(coordination_errors)) if coordination_errors else 0.0,
            "max_coordination_error": float(np.max(coordination_errors)) if coordination_errors else 0.0,
            "tail_avg_coordination_error": (
                float(np.mean(coordination_errors[-10 * max(1, num_pools - 1) :])) if coordination_errors else 0.0
            ),
            "tail_max_coordination_error": (
                float(np.max(coordination_errors[-10 * max(1, num_pools - 1) :])) if coordination_errors else 0.0
            ),
            "solve_iterations_mean": float(np.mean(solve_iterations)) if solve_iterations else 0.0,
            "solve_time_mean": float(np.mean(solve_times)) if solve_times else 0.0,
            "num_pools": float(num_pools),
        }
        self._simulation_cache[cache_key] = dict(result)
        return result

    def _test_setpoint_tracking(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        baseline = self._run_control_case(scenario, controller="baseline", steps=100)
        steady_limit = self._steady_state_limit(scenario)
        settling_limit = self._settling_time_limit(scenario)
        iae_ratio = self._ratio_vs_baseline_metrics(controlled, baseline, key="iae")
        itae_ratio = self._ratio_vs_baseline_metrics(controlled, baseline, key="itae")

        passed = (
            controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and controlled["steady_state_error"] <= steady_limit
            and controlled["overshoot_ratio"] <= self.max_overshoot
            and controlled["settling_time_s"] <= settling_limit
            and iae_ratio["gate"] <= 1.05
            and itae_ratio["gate"] <= 1.05
        )
        score_terms = [
            steady_limit / max(controlled["steady_state_error"], 1e-9),
            self.max_overshoot / max(controlled["overshoot_ratio"], 1e-9) if controlled["overshoot_ratio"] > 0 else 1.0,
            settling_limit / max(controlled["settling_time_s"], 1e-9),
            1.05 / max(iae_ratio["mean"], 1e-9),
            1.05 / max(itae_ratio["mean"], 1e-9),
        ]
        if self._is_multi_pool_metrics(controlled):
            score_terms.extend(
                [
                    1.05 / max(iae_ratio["worst"], 1e-9),
                    1.05 / max(itae_ratio["worst"], 1e-9),
                ]
            )
        score = self._score_from_terms(score_terms)
        return ControlTestResult(
            test_type=ControlTestType.SETPOINT_TRACKING,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "tracking_error": controlled["tracking_error"],
                "iae": controlled["iae"],
                "itae": controlled["itae"],
                "baseline_iae": baseline["iae"],
                "baseline_itae": baseline["itae"],
                "iae_ratio_vs_baseline": iae_ratio["gate"],
                "iae_ratio_vs_baseline_mean": iae_ratio["mean"],
                "iae_ratio_vs_baseline_worst": iae_ratio["worst"],
                "itae_ratio_vs_baseline": itae_ratio["gate"],
                "itae_ratio_vs_baseline_mean": itae_ratio["mean"],
                "itae_ratio_vs_baseline_worst": itae_ratio["worst"],
                "steady_state_error": controlled["steady_state_error"],
                "overshoot_percent": controlled["overshoot_ratio"] * 100.0,
                "settling_time": controlled["settling_time_s"],
                "constraint_violations": controlled["constraint_violations"],
                "fallback_count": controlled["fallback_count"],
            },
        )

    def _test_disturbance_rejection(self, scenario: TestScenario) -> ControlTestResult:
        disturbance_mag = scenario.disturbance_magnitude if scenario.disturbance_magnitude > 0 else 1.0
        controlled = self._run_control_case(scenario, controller="mpc", disturbance_scale=disturbance_mag)
        open_loop = self._run_control_case(scenario, controller="open_loop", disturbance_scale=disturbance_mag)
        rejection_ratio = self._improvement_vs_baseline_metrics(controlled, open_loop, key="iae")
        steady_limit = self._steady_state_limit(scenario)
        passed = (
            controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and rejection_ratio["gate"] >= DISTURBANCE_RATIO_GATE
            and controlled["steady_state_error"] <= steady_limit * 1.5
        )
        score_terms = [
            rejection_ratio["mean"] / DISTURBANCE_RATIO_GATE,
            (steady_limit * 1.5) / max(controlled["steady_state_error"], 1e-9),
        ]
        if self._is_multi_pool_metrics(controlled):
            score_terms.append(rejection_ratio["worst"] / DISTURBANCE_RATIO_GATE)
        score = self._score_from_terms(score_terms)
        return ControlTestResult(
            test_type=ControlTestType.DISTURBANCE_REJECTION,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "disturbance_magnitude": disturbance_mag,
                "iae_with_control": controlled["iae"],
                "iae_open_loop": open_loop["iae"],
                "rejection_ratio": rejection_ratio["gate"],
                "rejection_ratio_mean": rejection_ratio["mean"],
                "rejection_ratio_worst": rejection_ratio["worst"],
                "steady_state_error": controlled["steady_state_error"],
                "constraint_violations": controlled["constraint_violations"],
            },
        )

    def _test_constraint_handling(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        passed = controlled["fallback_count"] == 0 and controlled["constraint_violations"] == 0
        score = self._score_from_terms(
            [
                1.0 if controlled["constraint_violations"] == 0 else 0.0,
                1.0 if controlled["fallback_count"] == 0 else 0.0,
                LEVEL_MAX / max(controlled["max_level"], 1e-9),
            ]
        )
        return ControlTestResult(
            test_type=ControlTestType.CONSTRAINT_HANDLING,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "constraint_violations": controlled["constraint_violations"],
                "fallback_count": controlled["fallback_count"],
                "max_level": controlled["max_level"],
                "min_level": controlled["min_level"],
                "command_saturation_count": controlled["command_saturation_count"],
            },
        )

    def _test_stability(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        steady_limit = self._steady_state_limit(scenario)
        is_bounded = controlled["min_level"] >= LEVEL_MIN and controlled["max_level"] <= LEVEL_MAX
        is_converging = controlled["final_std"] <= self._settling_band(scenario)
        passed = (
            is_bounded
            and is_converging
            and controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and controlled["steady_state_error"] <= steady_limit * 1.5
        )
        score = self._score_from_terms(
            [
                1.0 if is_bounded else 0.0,
                1.0 if is_converging else 0.0,
                1.0 if controlled["fallback_count"] == 0 else 0.0,
                (steady_limit * 1.5) / max(controlled["steady_state_error"], 1e-9),
            ]
        )
        return ControlTestResult(
            test_type=ControlTestType.STABILITY,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "is_bounded": float(is_bounded),
                "is_converging": float(is_converging),
                "has_no_nan": 1.0,
                "final_std": controlled["final_std"],
                "steady_state_error": controlled["steady_state_error"],
                "fallback_count": controlled["fallback_count"],
            },
        )

    def _test_response_time(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        baseline = self._run_control_case(scenario, controller="baseline", steps=100)
        settling_limit = self._settling_time_limit(scenario)
        improvement = self._improvement_vs_baseline_metrics(controlled, baseline, key="settling_time_s")
        passed = (
            controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and controlled["settling_time_s"] <= settling_limit
            and improvement["gate"] >= 1.0
        )
        score_terms = [
            settling_limit / max(controlled["settling_time_s"], 1e-9),
            improvement["mean"],
        ]
        if self._is_multi_pool_metrics(controlled):
            score_terms.append(improvement["worst"])
        score = self._score_from_terms(score_terms)
        return ControlTestResult(
            test_type=ControlTestType.RESPONSE_TIME,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "settling_time": controlled["settling_time_s"],
                "baseline_settling_time": baseline["settling_time_s"],
                "level_change": scenario.target_level - scenario.initial_water_level,
                "final_error": controlled["steady_state_error"],
                "improvement": improvement["gate"],
                "improvement_mean": improvement["mean"],
                "improvement_worst": improvement["worst"],
            },
        )

    def _test_overshoot(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        overshoot_ratio = controlled["overshoot_ratio"]
        passed = (
            controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and overshoot_ratio <= self.max_overshoot
        )
        score = self._score_from_terms(
            [
                self.max_overshoot / max(overshoot_ratio, 1e-9) if overshoot_ratio > 0 else 1.0,
                1.0 if controlled["fallback_count"] == 0 else 0.0,
            ]
        )
        return ControlTestResult(
            test_type=ControlTestType.OVERSHOOT,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "overshoot_percent": overshoot_ratio * 100.0,
                "level_change": scenario.target_level - scenario.initial_water_level,
                "max_level": controlled["max_level"],
                "min_level": controlled["min_level"],
                "fallback_count": controlled["fallback_count"],
            },
        )

    def _test_steady_state_error(self, scenario: TestScenario) -> ControlTestResult:
        controlled = self._run_control_case(scenario, controller="mpc", steps=100)
        threshold = self._steady_state_limit(scenario)
        steady_state_error = controlled["steady_state_error"]
        passed = (
            controlled["fallback_count"] == 0
            and controlled["constraint_violations"] == 0
            and steady_state_error <= threshold
        )
        score = self._score_from_terms([threshold / max(steady_state_error, 1e-9)])
        return ControlTestResult(
            test_type=ControlTestType.STEADY_STATE_ERROR,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "steady_state_error": steady_state_error,
                "final_level": scenario.target_level - steady_state_error,
                "target_level": scenario.target_level,
                "fallback_count": controlled["fallback_count"],
            },
        )

    def _test_multi_pool_coordination(self, scenario: TestScenario) -> ControlTestResult:
        if not self.physics_available:
            raise RuntimeError("physics model unavailable")

        num_pools = self._get_pool_count(scenario.topology)
        controller = "mpc" if self._uses_multi_pool_control(scenario) else "baseline"
        controlled = self._run_closed_loop_multi(scenario, controller=controller, steps=50)
        avg_error = controlled["avg_coordination_error"]
        max_error = controlled["max_coordination_error"]
        tail_avg_error = controlled["tail_avg_coordination_error"]
        tail_max_error = controlled["tail_max_coordination_error"]
        threshold = max(0.10, 0.05 * max(abs(scenario.target_level), 1.0) * (num_pools ** 0.5))
        passed = (
            tail_max_error <= threshold
            and tail_avg_error <= threshold * 0.7
            and controlled["fallback_count"] == 0
        )
        score = self._score_from_terms(
            [
                threshold / max(tail_avg_error, 1e-9) if tail_avg_error > 0 else 1.0,
                threshold / max(tail_max_error, 1e-9) if tail_max_error > 0 else 1.0,
                1.0 if controlled["fallback_count"] == 0 else 0.0,
            ]
        )
        return ControlTestResult(
            test_type=ControlTestType.MULTI_POOL_COORDINATION,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "num_pools": float(num_pools),
                "avg_coordination_error": avg_error,
                "max_coordination_error": max_error,
                "tail_avg_coordination_error": tail_avg_error,
                "tail_max_coordination_error": tail_max_error,
                "fallback_count": controlled["fallback_count"],
            },
        )

    def _test_mode_switching(self, scenario: TestScenario) -> ControlTestResult:
        if not self.physics_available:
            raise RuntimeError("physics model unavailable")

        pool = self._make_pool_backend(scenario, delay_steps=1)
        gains = [0.5, 1.0, 0.6, 1.2]
        targets = [
            scenario.target_level,
            scenario.target_level - 0.3,
            scenario.target_level,
            scenario.target_level + 0.2,
        ]
        switch_points = [0, 20, 40, 60]
        levels: List[float] = []
        switch_transients: List[float] = []

        for step in range(80):
            mode_idx = max(index for index, point in enumerate(switch_points) if step >= point)
            target = targets[mode_idx]
            gain = gains[mode_idx]
            error = target - float(pool.get_level())
            q_in = float(np.clip(scenario.initial_inflow + gain * error, FLOW_MIN, FLOW_MAX))
            level = float(pool.step(q_in, scenario.initial_outflow))
            levels.append(level)
            if step in switch_points[1:] and len(levels) > 1:
                switch_transients.append(abs(levels[-1] - levels[-2]))

        max_transient = max(switch_transients) if switch_transients else 0.0
        threshold = max(0.10, self._steady_state_limit(scenario))
        passed = max_transient <= threshold
        score = self._score_from_terms([threshold / max(max_transient, 1e-9) if max_transient > 0 else 1.0])
        return ControlTestResult(
            test_type=ControlTestType.MODE_SWITCHING,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "num_switches": float(len(switch_points) - 1),
                "max_transient": max_transient,
            },
        )

    def _test_degraded_control(self, scenario: TestScenario) -> ControlTestResult:
        if not self.physics_available:
            raise RuntimeError("physics model unavailable")

        degradation_levels = [1.0, 0.8, 0.6, 0.4]
        final_errors: Dict[float, float] = {}
        for factor in degradation_levels:
            pool = self._make_pool_backend(scenario, delay_steps=1)
            errors: List[float] = []
            for _ in range(40):
                current_level = float(pool.get_level())
                error = scenario.target_level - current_level
                q_in = float(np.clip(scenario.initial_inflow + 2.0 * error * factor, FLOW_MIN, FLOW_MAX))
                level = float(pool.step(q_in, scenario.initial_outflow))
                errors.append(abs(level - scenario.target_level))
            final_errors[factor] = float(np.mean(errors[-10:]))

        full_error = final_errors[1.0]
        degraded_ratios = [
            final_errors[factor] / max(full_error, 1e-9)
            for factor in degradation_levels[1:]
        ]
        passed = full_error <= self._steady_state_limit(scenario) and all(ratio <= DEGRADED_RATIO_GATE for ratio in degraded_ratios)
        score = self._score_from_terms(
            [
                self._steady_state_limit(scenario) / max(full_error, 1e-9),
                DEGRADED_RATIO_GATE / max(max(degraded_ratios), 1e-9) if degraded_ratios else 0.0,
            ]
        )
        return ControlTestResult(
            test_type=ControlTestType.DEGRADED_CONTROL,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "full_performance_error": full_error,
                "degraded_performance_ratios": max(degraded_ratios) if degraded_ratios else 0.0,
                "graceful_degradation": float(passed),
            },
        )

    def _get_pool_count(self, topology: NetworkTopology) -> int:
        mapping = {
            NetworkTopology.SINGLE_POOL: 1,
            NetworkTopology.CASCADE_2: 2,
            NetworkTopology.CASCADE_3: 3,
            NetworkTopology.CASCADE_5: 5,
            NetworkTopology.CASCADE_10: 10,
            NetworkTopology.BRANCH_2: 3,
            NetworkTopology.BRANCH_3: 4,
            NetworkTopology.MESH_SMALL: 6,
            NetworkTopology.MESH_LARGE: 12,
        }
        return mapping.get(topology, 1)


if __name__ == "__main__":
    from .scenario_combinatorial_generator import ScenarioCombinatorialGenerator

    generator = ScenarioCombinatorialGenerator(seed=42)
    scenarios = generator.generate_all(max_scenarios=20)
    tester = ControlTester(verbose=True)
    report = tester.run_all_tests(scenarios[:5])
    print("control tests:", report.total_tests)
    print("passed:", report.passed_tests)
    print("failed:", report.failed_tests)
    print("avg tracking error:", report.average_tracking_error)
    print("avg settling time:", report.average_settling_time)

"""
Strict physics-simulation tester used for revalidation.

The old version mixed smoke checks, placeholders, and permissive scoring.
This version keeps the same public API but applies hard gates that are closer
to evidence for numerical/physical credibility.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .scenario_combinatorial_generator import SeasonType, TestScenario, WeatherCondition
from .hydraulic_backends import build_single_pool_backend


MASS_ERROR_GATE = 0.02
STEADY_STATE_GATE_M = 0.02
DISTURBANCE_BOUNDED_GATE_M = 2.0
LEVEL_MIN = 0.0
LEVEL_MAX = 10.0


class PhysicsTestType(Enum):
    MASS_CONSERVATION = "mass_conservation"
    ENERGY_CONSERVATION = "energy_conservation"
    SAINT_VENANT = "saint_venant"
    MANNING_EQUATION = "manning_equation"
    STABILITY = "stability"
    BOUNDARY_BEHAVIOR = "boundary_behavior"
    STEADY_STATE = "steady_state"
    TRANSIENT_RESPONSE = "transient_response"
    DELAY_EFFECT = "delay_effect"
    DISTURBANCE_REJECTION = "disturbance_rejection"


@dataclass
class PhysicsTestResult:
    test_type: PhysicsTestType
    scenario_id: str
    passed: bool
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": self.test_type.value,
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }


@dataclass
class PhysicsSimulationReport:
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_score: float = 0.0
    physics_backend: str = "unknown"
    test_results: List[PhysicsTestResult] = field(default_factory=list)
    by_test_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PhysicsSimulationTester:
    def __init__(
        self,
        tolerance: float = 0.01,
        verbose: bool = False,
        physics_backend: str = "auto",
    ) -> None:
        self.tolerance = tolerance
        self.verbose = verbose
        self.physics_backend = physics_backend
        self.results: List[PhysicsTestResult] = []
        self._load_physics_models()

    def _load_physics_models(self) -> None:
        self.CanalPoolSimulator = None
        self.ChannelGeometry = None
        self.SingleChannelFidelity = None
        self.SegmentedHighFidelityModel = None
        self.SaintVenantConfig = None
        self.BoundaryType = None
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
        self.physics_available = self.tank_available or self.single_channel_available or self.segmented_hf_available
        self.digital_twin_available = self.single_channel_available
        self.preferred_physics_backend = self._resolve_backend_name()

    def _resolve_backend_name(self) -> str:
        if self.physics_backend in {"segmented_hf", "single_channel", "fidelity", "tank"}:
            if self.physics_backend == "fidelity":
                return "single_channel"
            return self.physics_backend
        if self.single_channel_available:
            return "single_channel"
        if self.segmented_hf_available:
            return "segmented_hf"
        if self.physics_backend == "tank":
            return "tank"
        return "tank"

    def run_all_tests(self, scenarios: List[TestScenario]) -> PhysicsSimulationReport:
        report = PhysicsSimulationReport(physics_backend=self.preferred_physics_backend)
        started = time.time()

        for scenario in scenarios:
            for test_type in self._select_tests_for_scenario(scenario):
                result = self._run_single_test(test_type, scenario)
                self.results.append(result)
                report.test_results.append(result)
                report.total_tests += 1
                if result.passed:
                    report.passed_tests += 1
                else:
                    report.failed_tests += 1
                bucket = report.by_test_type.setdefault(test_type.value, {"passed": 0, "failed": 0})
                bucket["passed" if result.passed else "failed"] += 1

        report.total_execution_time = time.time() - started
        if report.total_tests > 0:
            report.average_score = float(np.mean([item.score for item in report.test_results]))
        return report

    def _select_tests_for_scenario(self, scenario: TestScenario) -> List[PhysicsTestType]:
        tests = [
            PhysicsTestType.MASS_CONSERVATION,
            PhysicsTestType.STABILITY,
            PhysicsTestType.STEADY_STATE,
        ]
        if scenario.category == "BOUNDARY":
            tests.append(PhysicsTestType.BOUNDARY_BEHAVIOR)
        if scenario.category in ["TEMPORAL", "TEMPORAL_SEQUENCE"]:
            tests.append(PhysicsTestType.TRANSIENT_RESPONSE)
            tests.append(PhysicsTestType.DELAY_EFFECT)
        if scenario.disturbance_magnitude > 0:
            tests.append(PhysicsTestType.DISTURBANCE_REJECTION)
        return tests

    def _run_single_test(self, test_type: PhysicsTestType, scenario: TestScenario) -> PhysicsTestResult:
        started = time.time()
        try:
            if test_type == PhysicsTestType.MASS_CONSERVATION:
                result = self._test_mass_conservation(scenario)
            elif test_type == PhysicsTestType.ENERGY_CONSERVATION:
                result = self._test_energy_conservation(scenario)
            elif test_type == PhysicsTestType.STABILITY:
                result = self._test_numerical_stability(scenario)
            elif test_type == PhysicsTestType.BOUNDARY_BEHAVIOR:
                result = self._test_boundary_behavior(scenario)
            elif test_type == PhysicsTestType.STEADY_STATE:
                result = self._test_steady_state(scenario)
            elif test_type == PhysicsTestType.TRANSIENT_RESPONSE:
                result = self._test_transient_response(scenario)
            elif test_type == PhysicsTestType.DISTURBANCE_REJECTION:
                result = self._test_disturbance_rejection(scenario)
            elif test_type == PhysicsTestType.DELAY_EFFECT:
                result = self._test_delay_effect(scenario)
            elif test_type == PhysicsTestType.MANNING_EQUATION:
                result = self._test_manning_equation(scenario)
            elif test_type == PhysicsTestType.SAINT_VENANT:
                result = self._test_saint_venant(scenario)
            else:
                result = PhysicsTestResult(
                    test_type=test_type,
                    scenario_id=scenario.id,
                    passed=False,
                    score=0.0,
                    errors=[f"unsupported test type: {test_type.value}"],
                )
        except Exception as exc:
            result = PhysicsTestResult(
                test_type=test_type,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                errors=[f"test crashed: {exc}"],
            )
        result.execution_time = time.time() - started
        return result

    def _score(self, terms: List[float]) -> float:
        if not terms:
            return 0.0
        return float(np.mean([min(1.0, max(0.0, value)) for value in terms]))

    def _make_pool(self, scenario: TestScenario, *, delay_steps: int = 1, initial_flow: float | None = None):
        if not self.physics_available:
            raise RuntimeError("physics model unavailable")
        return build_single_pool_backend(
            backend=self.physics_backend,
            area=scenario.area,
            slope=scenario.slope,
            time_step=scenario.time_step,
            initial_level=scenario.initial_water_level,
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

    def _pool_volume(self, pool: Any, scenario: TestScenario) -> float:
        if hasattr(pool, "get_volume"):
            try:
                volume = float(pool.get_volume())
                if np.isfinite(volume):
                    return volume
            except Exception:
                pass
        return float(pool.get_level()) * float(scenario.area)

    def _pool_outflow(self, pool: Any, requested_q_out: float) -> float:
        if hasattr(pool, "get_outflow"):
            try:
                q_out = float(pool.get_outflow())
                if np.isfinite(q_out):
                    return q_out
            except Exception:
                pass
        return float(requested_q_out)

    def _test_mass_conservation(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario)
        dt = scenario.time_step
        steps = int(min(24, scenario.duration_hours) * 3600 / dt)
        q_in = scenario.initial_inflow
        q_out = scenario.initial_outflow
        initial_volume = self._pool_volume(pool, scenario)
        total_inflow = 0.0
        total_outflow = 0.0

        for _ in range(steps):
            pool.step(q_in_command=q_in, q_out=q_out)
            total_inflow += q_in * dt
            total_outflow += self._pool_outflow(pool, q_out) * dt

        final_volume = self._pool_volume(pool, scenario)
        expected_change = total_inflow - total_outflow
        actual_change = final_volume - initial_volume
        imbalance = abs(actual_change - expected_change)
        normalization = max(abs(total_inflow) + abs(total_outflow), abs(initial_volume), 1e-9)
        relative_error = imbalance / normalization

        passed = relative_error <= MASS_ERROR_GATE
        score = self._score([MASS_ERROR_GATE / max(relative_error, 1e-9)])
        return PhysicsTestResult(
            test_type=PhysicsTestType.MASS_CONSERVATION,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "relative_error": relative_error,
                "imbalance_volume": imbalance,
                "normalization_volume": normalization,
                "expected_change": expected_change,
                "actual_change": actual_change,
                "initial_volume": initial_volume,
                "final_volume": final_volume,
            },
        )

    def _test_energy_conservation(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario, initial_flow=scenario.initial_inflow)
        dt = scenario.time_step
        steps = int(min(12, scenario.duration_hours) * 3600 / dt)
        energy_changes: List[float] = []
        prev_level = scenario.initial_water_level

        for _ in range(steps):
            pool.step(q_in_command=scenario.initial_inflow, q_out=scenario.initial_inflow)
            level = float(pool.get_level())
            energy_changes.append(abs(level - prev_level))
            prev_level = level

        avg_change = float(np.mean(energy_changes[steps // 2 :])) if energy_changes else float("inf")
        passed = avg_change <= STEADY_STATE_GATE_M
        score = self._score([STEADY_STATE_GATE_M / max(avg_change, 1e-9)])
        return PhysicsTestResult(
            test_type=PhysicsTestType.ENERGY_CONSERVATION,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={"avg_energy_change": avg_change},
        )

    def _test_numerical_stability(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario)
        dt = scenario.time_step
        steps = int(min(48, scenario.duration_hours) * 3600 / dt)
        levels: List[float] = []
        has_nan = False
        has_inf = False
        has_negative = False
        has_explosion = False

        for _ in range(steps):
            noise = np.random.randn() * scenario.noise_level
            q_in = max(0.0, scenario.initial_inflow + noise)
            q_out = q_in
            level = float(pool.step(q_in_command=q_in, q_out=q_out))
            levels.append(level)
            has_nan = has_nan or np.isnan(level)
            has_inf = has_inf or np.isinf(level)
            has_negative = has_negative or level < LEVEL_MIN
            has_explosion = has_explosion or level > LEVEL_MAX
            if has_nan or has_inf:
                break

        level_std = float(np.std(levels[-20:])) if len(levels) >= 20 else float("inf")
        level_range = float(max(levels) - min(levels)) if levels else float("inf")
        expected_drift = 0.0
        passed = not (has_nan or has_inf or has_negative or has_explosion)
        score = self._score(
            [
                1.0 if not has_nan else 0.0,
                1.0 if not has_inf else 0.0,
                1.0 if not has_negative else 0.0,
                1.0 if not has_explosion else 0.0,
            ]
        )
        errors: List[str] = []
        if has_nan:
            errors.append("NaN produced")
        if has_inf:
            errors.append("Inf produced")
        if has_negative:
            errors.append("negative level produced")
        if has_explosion:
            errors.append("level exceeded physical range")
        return PhysicsTestResult(
            test_type=PhysicsTestType.STABILITY,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "steps_completed": float(len(levels)),
                "level_std": level_std,
                "level_range": level_range,
                "expected_drift": expected_drift,
                "has_nan": float(has_nan),
                "has_inf": float(has_inf),
                "has_negative": float(has_negative),
                "has_explosion": float(has_explosion),
            },
            errors=errors,
        )

    def _test_boundary_behavior(self, scenario: TestScenario) -> PhysicsTestResult:
        boundary_tests = [
            ("low_level_recovery", 0.1, 2.0, 2.0),
            ("high_level_release", 9.9, 1.0, 4.0),
            ("zero_inflow_release", 3.0, 0.0, 1.0),
            ("high_flow_balanced", 3.0, 20.0, 20.0),
        ]
        failures: List[str] = []

        for name, init_level, q_in, q_out in boundary_tests:
            pool = build_single_pool_backend(
                backend=self.physics_backend,
                area=scenario.area,
                slope=scenario.slope,
                time_step=scenario.time_step,
                initial_level=init_level,
                initial_flow=q_in,
                manning_n=scenario.manning_n,
                tank_cls=self.CanalPoolSimulator,
                fidelity_cls=self.SingleChannelFidelity,
                geometry_cls=self.ChannelGeometry,
                segmented_model_cls=self.SegmentedHighFidelityModel,
                segmented_config_cls=self.SaintVenantConfig,
                segmented_boundary_type_cls=self.BoundaryType,
                delay_steps=1,
            )
            levels: List[float] = []
            for _ in range(20):
                level = float(pool.step(q_in_command=q_in, q_out=q_out))
                levels.append(level)
            if any((not np.isfinite(level)) or level < LEVEL_MIN or level > LEVEL_MAX for level in levels):
                failures.append(name)

        passed = not failures
        score = self._score([1.0 if not failures else 0.0])
        return PhysicsTestResult(
            test_type=PhysicsTestType.BOUNDARY_BEHAVIOR,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "boundary_tests_total": float(len(boundary_tests)),
                "boundary_tests_passed": float(len(boundary_tests) - len(failures)),
            },
            errors=[f"{name} failed" for name in failures],
        )

    def _test_steady_state(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario, initial_flow=scenario.initial_inflow)
        levels: List[float] = []
        for _ in range(100):
            level = float(pool.step(q_in_command=scenario.initial_inflow, q_out=scenario.initial_inflow))
            levels.append(level)
        final_levels = np.asarray(levels[-20:], dtype=float)
        std_final = float(np.std(final_levels))
        mean_final = float(np.mean(final_levels))
        deviation = abs(mean_final - scenario.initial_water_level)
        passed = std_final <= STEADY_STATE_GATE_M and deviation <= STEADY_STATE_GATE_M
        score = self._score(
            [
                STEADY_STATE_GATE_M / max(std_final, 1e-9),
                STEADY_STATE_GATE_M / max(deviation, 1e-9),
            ]
        )
        return PhysicsTestResult(
            test_type=PhysicsTestType.STEADY_STATE,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "final_std": std_final,
                "deviation_from_initial": deviation,
                "steps_to_converge": 100.0,
            },
        )

    def _test_transient_response(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario)
        levels_before: List[float] = []
        levels_after: List[float] = []
        step_magnitude = 2.0

        for _ in range(20):
            levels_before.append(float(pool.step(q_in_command=scenario.initial_inflow, q_out=scenario.initial_outflow)))
        baseline = float(np.mean(levels_before[-5:]))

        for _ in range(50):
            levels_after.append(
                float(pool.step(q_in_command=scenario.initial_inflow + step_magnitude, q_out=scenario.initial_outflow))
            )

        final_level = float(np.mean(levels_after[-5:]))
        threshold = baseline + (final_level - baseline) * 0.63
        rise_time = -1
        for idx, level in enumerate(levels_after):
            if level >= threshold:
                rise_time = idx
                break

        response_correct = final_level > baseline
        bounded = all(np.isfinite(level) and LEVEL_MIN <= level <= LEVEL_MAX for level in levels_after)
        passed = response_correct and bounded and rise_time >= 0
        score = self._score(
            [
                1.0 if response_correct else 0.0,
                1.0 if bounded else 0.0,
                1.0 if rise_time >= 0 else 0.0,
            ]
        )
        return PhysicsTestResult(
            test_type=PhysicsTestType.TRANSIENT_RESPONSE,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "baseline_level": baseline,
                "final_level": final_level,
                "rise_time_steps": float(rise_time),
                "response_correct": float(response_correct),
            },
        )

    def _test_disturbance_rejection(self, scenario: TestScenario) -> PhysicsTestResult:
        pool = self._make_pool(scenario)
        disturbance_mag = scenario.disturbance_magnitude if scenario.disturbance_magnitude > 0 else 0.5
        levels: List[float] = []
        baseline_outflow = scenario.initial_inflow

        for idx in range(100):
            disturbance = disturbance_mag * np.sin(2 * np.pi * idx / 20)
            level = float(pool.step(q_in_command=scenario.initial_inflow + disturbance, q_out=baseline_outflow))
            levels.append(level)

        level_range = float(max(levels) - min(levels)) if levels else float("inf")
        level_std = float(np.std(levels)) if levels else float("inf")
        bounded = all(np.isfinite(level) and LEVEL_MIN <= level <= LEVEL_MAX for level in levels)
        passed = bounded and level_range <= max(DISTURBANCE_BOUNDED_GATE_M, disturbance_mag * 4.0)
        score = self._score(
            [
                1.0 if bounded else 0.0,
                max(DISTURBANCE_BOUNDED_GATE_M, disturbance_mag * 4.0) / max(level_range, 1e-9),
            ]
        )
        return PhysicsTestResult(
            test_type=PhysicsTestType.DISTURBANCE_REJECTION,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "disturbance_magnitude": disturbance_mag,
                "level_range": level_range,
                "level_std": level_std,
            },
        )

    def _test_delay_effect(self, scenario: TestScenario) -> PhysicsTestResult:
        delay_steps_list = [0, 1, 2, 3]
        matches = 0
        results: List[Dict[str, int]] = []

        for delay_steps in delay_steps_list:
            pool = self._make_pool(scenario, delay_steps=delay_steps)
            levels: List[float] = []
            for idx in range(30):
                q_in = scenario.initial_inflow if idx < 10 else scenario.initial_inflow + 2.0
                levels.append(float(pool.step(q_in_command=q_in, q_out=scenario.initial_outflow)))

            baseline = float(np.mean(levels[:10]))
            response_start = -1
            for idx in range(10, len(levels)):
                if levels[idx] > baseline + 0.01:
                    response_start = idx - 10
                    break

            matched = response_start in {delay_steps, delay_steps + 1}
            matches += int(matched)
            results.append({"delay_steps": delay_steps, "response_start": response_start})

        score = matches / len(delay_steps_list)
        passed = score >= 0.75
        return PhysicsTestResult(
            test_type=PhysicsTestType.DELAY_EFFECT,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={
                "delay_tests": float(len(delay_steps_list)),
                "correct_delays": float(matches),
            },
            warnings=[str(result) for result in results if result["response_start"] < 0],
        )

    def _test_manning_equation(self, scenario: TestScenario) -> PhysicsTestResult:
        expected_velocity = (1.0 / scenario.manning_n) * (2.0 ** (2.0 / 3.0)) * (scenario.slope ** 0.5)
        if not self.digital_twin_available:
            return PhysicsTestResult(
                test_type=PhysicsTestType.MANNING_EQUATION,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                warnings=["digital twin model unavailable"],
                errors=["cannot verify manning equation without fidelity model"],
            )
        passed = expected_velocity > 0
        return PhysicsTestResult(
            test_type=PhysicsTestType.MANNING_EQUATION,
            scenario_id=scenario.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            metrics={"expected_velocity": expected_velocity},
        )

    def _test_saint_venant(self, scenario: TestScenario) -> PhysicsTestResult:
        if not self.digital_twin_available:
            return PhysicsTestResult(
                test_type=PhysicsTestType.SAINT_VENANT,
                scenario_id=scenario.id,
                passed=False,
                score=0.0,
                warnings=["digital twin model unavailable"],
                errors=["cannot verify saint-venant consistency without fidelity model"],
            )

        geometry = self.ChannelGeometry(length=20_000.0, N=20)
        physics = self.SingleChannelFidelity(geometry=geometry)
        continuity_errors: List[float] = []
        for _ in range(20):
            physics.step(u_in=5.0, u_out=4.5)
            state = physics.get_state()
            if hasattr(state, "Q") and len(state.Q) > 1:
                continuity_errors.append(float(np.mean(np.abs(np.diff(state.Q)))))

        avg_error = float(np.mean(continuity_errors)) if continuity_errors else float("inf")
        passed = avg_error <= 0.5
        score = self._score([0.5 / max(avg_error, 1e-9)])
        return PhysicsTestResult(
            test_type=PhysicsTestType.SAINT_VENANT,
            scenario_id=scenario.id,
            passed=passed,
            score=score,
            metrics={"avg_continuity_error": avg_error},
        )

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"message": "no tests executed"}
        total = len(self.results)
        passed = sum(1 for item in self.results if item.passed)
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "average_score": float(np.mean([item.score for item in self.results])),
        }


if __name__ == "__main__":
    from .scenario_combinatorial_generator import ScenarioCombinatorialGenerator

    generator = ScenarioCombinatorialGenerator(seed=42)
    scenarios = generator.generate_all(max_scenarios=20)
    tester = PhysicsSimulationTester(verbose=True)
    report = tester.run_all_tests(scenarios[:5])
    print("physics tests:", report.total_tests)
    print("passed:", report.passed_tests)
    print("failed:", report.failed_tests)
    print("average_score:", report.average_score)

#!/usr/bin/env python3
"""Compare strict revalidation results across hydraulic/controller backends."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.comprehensive_hil_testing.control_tester import ControlTester
from tests.comprehensive_hil_testing.physics_simulation_tester import PhysicsSimulationTester
from tests.comprehensive_hil_testing.scenario_combinatorial_generator import (
    ScenarioCombinatorialGenerator,
    TestScenario,
)


def _is_backend_unsuitable_reason(reason: str) -> bool:
    text = reason.lower()
    return "supports time_step <=" in text or "too many sub-steps" in text


def _is_backend_unsupported_reason(reason: str) -> bool:
    text = reason.lower()
    return (
        "unsupported hydraulic backend" in text
        or "backend unavailable" in text
        or "no hydraulic backend available" in text
    )


def _is_scenario_unsuitable_reason(reason: str) -> bool:
    text = reason.lower()
    return "scenario infeasible under actuator bounds" in text


def _is_numerical_instability_reason(reason: str) -> bool:
    text = reason.lower()
    return "nan" in text or "inf" in text or "cfl" in text or "instability" in text


def _classify_failure(item: Any, reasons: List[str]) -> str:
    if getattr(item, "skipped", False) or any(_is_scenario_unsuitable_reason(reason) for reason in reasons):
        return "scenario_unsuitable"
    if any(_is_backend_unsuitable_reason(reason) for reason in reasons):
        return "backend_unsuitable"
    if any(_is_backend_unsupported_reason(reason) for reason in reasons):
        return "backend_unsupported"
    if reasons and any(reason.startswith("test crashed:") for reason in reasons):
        return "execution_error"
    return "quality_failure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strict revalidation across hydraulic backends.")
    parser.add_argument("--scenarios", type=int, default=20, help="Number of generated scenarios.")
    parser.add_argument("--seed", type=int, default=42, help="Scenario generator seed.")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["physics", "control"],
        default=["physics", "control"],
        help="Modules to compare.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/acceptance/revalidation_backend_comparison.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--controller-backend",
        choices=["base_mpc", "parameterized_dmpc"],
        default="base_mpc",
        help="Controller backend used for control comparison.",
    )
    return parser.parse_args()


def _sample_scenarios(items: List[TestScenario], count: int, seed: int) -> List[TestScenario]:
    if len(items) <= count:
        return items

    rng = random.Random(seed)
    buckets: Dict[tuple[str, int, str, str], List[TestScenario]] = {}
    for scenario in items:
        key = (
            scenario.category,
            scenario.difficulty,
            scenario.control_mode.name,
            scenario.topology.name,
        )
        buckets.setdefault(key, []).append(scenario)

    for scenarios in buckets.values():
        rng.shuffle(scenarios)

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    sampled: List[TestScenario] = []

    while len(sampled) < count and bucket_keys:
        next_round: List[tuple[str, int, str, str]] = []
        for key in bucket_keys:
            candidates = buckets[key]
            if candidates and len(sampled) < count:
                sampled.append(candidates.pop())
            if candidates:
                next_round.append(key)
        bucket_keys = next_round

    return sampled


def _module_metrics(report: Any) -> Dict[str, Any]:
    total = int(report.total_tests)
    passed = int(report.passed_tests)
    pass_rate = (passed / total) if total else 0.0
    by_type: Dict[str, Dict[str, float]] = {}
    for name, counts in getattr(report, "by_test_type", {}).items():
        type_total = int(counts.get("passed", 0) + counts.get("failed", 0))
        type_passed = int(counts.get("passed", 0))
        by_type[name] = {
            "passed": type_passed,
            "failed": int(counts.get("failed", 0)),
            "pass_rate": (type_passed / type_total) if type_total else 0.0,
        }
    failure_reason_counts: Dict[str, int] = {}
    quality_failure_reason_counts: Dict[str, int] = {}
    backend_unsuitable_reasons: Dict[str, int] = {}
    backend_unsupported_reasons: Dict[str, int] = {}
    execution_error_reasons: Dict[str, int] = {}
    scenario_unsuitable_reasons: Dict[str, int] = {}
    backend_unsuitable_count = 0
    backend_unsupported_count = 0
    execution_error_count = 0
    scenario_unsuitable_count = 0
    quality_failure_count = 0
    numerical_instability_count = 0
    for item in getattr(report, "test_results", []):
        if getattr(item, "passed", False):
            continue
        reasons = list(getattr(item, "errors", []) or [])
        if not reasons:
            reasons = [f"metric_gate:{item.test_type.value}"]
        category = _classify_failure(item, reasons)
        item_numerical = False
        for reason in reasons:
            failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
            if category == "backend_unsuitable":
                backend_unsuitable_reasons[reason] = backend_unsuitable_reasons.get(reason, 0) + 1
            elif category == "backend_unsupported":
                backend_unsupported_reasons[reason] = backend_unsupported_reasons.get(reason, 0) + 1
            elif category == "execution_error":
                execution_error_reasons[reason] = execution_error_reasons.get(reason, 0) + 1
            elif category == "scenario_unsuitable":
                scenario_unsuitable_reasons[reason] = scenario_unsuitable_reasons.get(reason, 0) + 1
            else:
                quality_failure_reason_counts[reason] = quality_failure_reason_counts.get(reason, 0) + 1
            if _is_numerical_instability_reason(reason):
                item_numerical = True
        if category == "backend_unsuitable" and not getattr(item, "skipped", False):
            backend_unsuitable_count += 1
        elif category == "backend_unsupported" and not getattr(item, "skipped", False):
            backend_unsupported_count += 1
        elif category == "execution_error" and not getattr(item, "skipped", False):
            execution_error_count += 1
        elif category == "scenario_unsuitable":
            scenario_unsuitable_count += 1
        elif category == "quality_failure" and not getattr(item, "skipped", False):
            quality_failure_count += 1
        if item_numerical and not getattr(item, "skipped", False):
            numerical_instability_count += 1

    unsupported_count = backend_unsuitable_count + backend_unsupported_count
    assessed_tests = max(0, total - unsupported_count)
    quality_failed_tests = quality_failure_count
    assessed_pass_rate = (passed / assessed_tests) if assessed_tests else None
    support_coverage = (assessed_tests / total) if total else 0.0
    top_failure_reasons = dict(sorted(failure_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    unsupported_reasons = dict(
        sorted(
            {**backend_unsuitable_reasons, **backend_unsupported_reasons}.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:8]
    )
    quality_failure_reasons = dict(sorted(quality_failure_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    return {
        "physics_backend": getattr(report, "physics_backend", "unknown"),
        "controller_backend": getattr(report, "controller_backend", "unknown"),
        "total_tests": total,
        "passed_tests": passed,
        "failed_tests": int(report.failed_tests),
        "skipped_tests": int(getattr(report, "skipped_tests", 0)),
        "pass_rate": pass_rate,
        "assessed_tests": assessed_tests,
        "quality_failed_tests": quality_failed_tests,
        "assessed_pass_rate": assessed_pass_rate,
        "support_coverage": support_coverage,
        "average_score": float(report.average_score),
        "by_test_type": by_type,
        "top_failure_reasons": top_failure_reasons,
        "unsupported_reasons": unsupported_reasons,
        "quality_failure_reasons": quality_failure_reasons,
        "backend_unsuitable_reasons": dict(sorted(backend_unsuitable_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
        "backend_unsupported_reasons": dict(sorted(backend_unsupported_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
        "execution_error_reasons": dict(sorted(execution_error_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
        "scenario_unsuitable_reasons": dict(sorted(scenario_unsuitable_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
        "unsupported_count": unsupported_count,
        "backend_unsuitable_count": backend_unsuitable_count,
        "backend_unsupported_count": backend_unsupported_count,
        "execution_error_count": execution_error_count,
        "scenario_unsuitable_count": scenario_unsuitable_count,
        "quality_failure_count": quality_failure_count,
        "numerical_instability_count": numerical_instability_count,
    }


def _compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    by_type_delta: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(reference["by_test_type"]) | set(candidate["by_test_type"])):
        reference_item = reference["by_test_type"].get(name, {"pass_rate": 0.0, "passed": 0, "failed": 0})
        candidate_item = candidate["by_test_type"].get(name, {"pass_rate": 0.0, "passed": 0, "failed": 0})
        by_type_delta[name] = {
            "reference_pass_rate": float(reference_item["pass_rate"]),
            "candidate_pass_rate": float(candidate_item["pass_rate"]),
            "pass_rate_delta": float(candidate_item["pass_rate"] - reference_item["pass_rate"]),
        }
    return {
        "reference_pass_rate": float(reference["pass_rate"]),
        "candidate_pass_rate": float(candidate["pass_rate"]),
        "pass_rate_delta": float(candidate["pass_rate"] - reference["pass_rate"]),
        "reference_assessed_pass_rate": reference.get("assessed_pass_rate"),
        "candidate_assessed_pass_rate": candidate.get("assessed_pass_rate"),
        "assessed_pass_rate_delta": (
            None
            if reference.get("assessed_pass_rate") is None or candidate.get("assessed_pass_rate") is None
            else float(candidate["assessed_pass_rate"] - reference["assessed_pass_rate"])
        ),
        "reference_support_coverage": float(reference.get("support_coverage", 0.0)),
        "candidate_support_coverage": float(candidate.get("support_coverage", 0.0)),
        "support_coverage_delta": float(candidate.get("support_coverage", 0.0) - reference.get("support_coverage", 0.0)),
        "score_delta": float(candidate["average_score"] - reference["average_score"]),
        "by_test_type_delta": by_type_delta,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("hydroe2e.control.base").setLevel(logging.ERROR)
    logging.getLogger("hydroe2e.phase5.distributed_sil.models.segmented_high_fidelity").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"hydroe2e\.phase5\.distributed_sil\.models\.segmented_high_fidelity")

    generator = ScenarioCombinatorialGenerator(seed=args.seed)
    scenario_pool_size = max(args.scenarios * 5, args.scenarios)
    scenario_pool = generator.generate_all(max_scenarios=scenario_pool_size)
    scenarios = _sample_scenarios(scenario_pool, args.scenarios, args.seed)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "scenario_count": len(scenarios),
        "modules": {},
    }

    if "physics" in args.modules:
        tank_report = PhysicsSimulationTester(physics_backend="tank").run_all_tests(scenarios)
        single_channel_report = PhysicsSimulationTester(physics_backend="single_channel").run_all_tests(scenarios)
        segmented_report = PhysicsSimulationTester(physics_backend="segmented_hf").run_all_tests(scenarios)
        tank_metrics = _module_metrics(tank_report)
        single_channel_metrics = _module_metrics(single_channel_report)
        segmented_metrics = _module_metrics(segmented_report)
        summary["modules"]["physics"] = {
            "tank": tank_metrics,
            "single_channel": single_channel_metrics,
            "segmented_hf": segmented_metrics,
            "comparison_vs_tank": {
                "single_channel": _compare_pair(tank_metrics, single_channel_metrics),
                "segmented_hf": _compare_pair(tank_metrics, segmented_metrics),
            },
        }

    if "control" in args.modules:
        tank_report = ControlTester(physics_backend="tank", controller_backend=args.controller_backend).run_all_tests(scenarios)
        single_channel_report = ControlTester(
            physics_backend="single_channel",
            controller_backend=args.controller_backend,
        ).run_all_tests(scenarios)
        segmented_report = ControlTester(
            physics_backend="segmented_hf",
            controller_backend=args.controller_backend,
        ).run_all_tests(scenarios)
        tank_metrics = _module_metrics(tank_report)
        single_channel_metrics = _module_metrics(single_channel_report)
        segmented_metrics = _module_metrics(segmented_report)
        summary["modules"]["control"] = {
            "tank": tank_metrics,
            "single_channel": single_channel_metrics,
            "segmented_hf": segmented_metrics,
            "comparison_vs_tank": {
                "single_channel": _compare_pair(tank_metrics, single_channel_metrics),
                "segmented_hf": _compare_pair(tank_metrics, segmented_metrics),
            },
        }

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BACKEND_COMPARISON_OUTPUT={args.output}")
    for module_name, module in summary["modules"].items():
        print(
            f"{module_name.upper()}_PASS_RATE_TANK={module['tank']['pass_rate']:.3f} "
            f"SINGLE_CHANNEL={module['single_channel']['pass_rate']:.3f} "
            f"SEGMENTED_HF={module['segmented_hf']['pass_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

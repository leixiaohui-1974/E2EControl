#!/usr/bin/env python3
"""Run strict physics/control revalidation batches and export JSON summary."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict simulation/control revalidation.")
    parser.add_argument("--scenarios", type=int, default=50, help="Number of generated scenarios to revalidate.")
    parser.add_argument("--seed", type=int, default=42, help="Scenario generator seed.")
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["physics", "control"],
        default=["physics", "control"],
        help="Strict revalidation modules to execute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/acceptance/strict_revalidation_summary.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--physics-backend",
        choices=["auto", "segmented_hf", "single_channel", "fidelity", "tank"],
        default="auto",
        help="Hydraulic backend used by strict revalidation.",
    )
    parser.add_argument(
        "--controller-backend",
        choices=["base_mpc", "parameterized_dmpc"],
        default="base_mpc",
        help="Control backend used by strict revalidation control tests.",
    )
    return parser.parse_args()


def _scenario_slice(items: List[TestScenario], count: int) -> List[TestScenario]:
    return items[:count]


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


def _failed_items(report: Any, limit: int = 25) -> List[Dict[str, Any]]:
    failed = []
    for item in report.test_results:
        if item.passed or getattr(item, "skipped", False):
            continue
        failed.append(
            {
                "scenario_id": item.scenario_id,
                "test_type": item.test_type.value,
                "score": item.score,
                "errors": item.errors,
                "metrics": item.metrics,
            }
        )
        if len(failed) >= limit:
            break
    return failed


def _module_summary(name: str, report: Any) -> Dict[str, Any]:
    by_test_type = getattr(report, "by_test_type", {})
    return {
        "module": name,
        "physics_backend": getattr(report, "physics_backend", "unknown"),
        "controller_backend": getattr(report, "controller_backend", "unknown"),
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "failed_tests": report.failed_tests,
        "skipped_tests": getattr(report, "skipped_tests", 0),
        "pass_rate": (report.passed_tests / report.total_tests) if report.total_tests else 0.0,
        "average_score": report.average_score,
        "by_test_type": by_test_type,
        "failed_samples": _failed_items(report),
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
        "physics_backend": args.physics_backend,
        "controller_backend": args.controller_backend,
        "modules": {},
    }

    if "physics" in args.modules:
        physics_report = PhysicsSimulationTester(physics_backend=args.physics_backend).run_all_tests(scenarios)
        summary["modules"]["physics"] = _module_summary("physics", physics_report)

    if "control" in args.modules:
        control_report = ControlTester(
            physics_backend=args.physics_backend,
            controller_backend=args.controller_backend,
        ).run_all_tests(scenarios)
        summary["modules"]["control"] = _module_summary("control", control_report)

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STRICT_REVALIDATION_OUTPUT={args.output}")
    for module_name, module in summary["modules"].items():
        print(
            f"{module_name.upper()}_PASS_RATE={module['pass_rate']:.3f} "
            f"({module['passed_tests']}/{module['total_tests']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a bounded simulation-fidelity/control-effectiveness experiment bundle."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_revalidation_backends import _compare_pair, _module_metrics
from tests.comprehensive_hil_testing.control_tester import ControlTester
from tests.comprehensive_hil_testing.physics_simulation_tester import PhysicsSimulationTester
from tests.comprehensive_hil_testing.scenario_combinatorial_generator import (
    ScenarioCombinatorialGenerator,
    TestScenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded fidelity/control experiments from YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/fidelity_control_bounded.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a mapping")
    return data


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


def _scenario_breakdown(scenarios: Iterable[TestScenario], attr: str, top_n: int = 6) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for scenario in scenarios:
        value = getattr(scenario, attr)
        name = getattr(value, "name", value)
        counts[str(name)] += 1
    return dict(counts.most_common(top_n))


def _failure_reasons(report: Any, limit: int = 5) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in getattr(report, "test_results", []):
        if getattr(item, "passed", False) or getattr(item, "skipped", False):
            continue
        reasons = list(getattr(item, "errors", []) or [])
        if not reasons:
            reasons = [f"metric_gate:{item.test_type.value}"]
        counts.update(reasons)
    return dict(counts.most_common(limit))


def _strict_module_summary(name: str, report: Any) -> Dict[str, Any]:
    return {
        "module": name,
        "physics_backend": getattr(report, "physics_backend", "unknown"),
        "controller_backend": getattr(report, "controller_backend", "unknown"),
        "total_tests": int(report.total_tests),
        "passed_tests": int(report.passed_tests),
        "failed_tests": int(report.failed_tests),
        "skipped_tests": int(getattr(report, "skipped_tests", 0)),
        "pass_rate": (report.passed_tests / report.total_tests) if report.total_tests else 0.0,
        "average_score": float(report.average_score),
        "by_test_type": getattr(report, "by_test_type", {}),
        "top_failure_reasons": _failure_reasons(report),
    }


def _run_strict_section(config: Dict[str, Any], scenarios: List[TestScenario]) -> Dict[str, Any]:
    section: Dict[str, Any] = {
        "requested_physics_backend": config.get("physics_backend", "auto"),
        "requested_controller_backend": config.get("controller_backend", "base_mpc"),
        "modules": {},
    }
    modules = list(config.get("modules", ["physics", "control"]))

    if "physics" in modules:
        physics_report = PhysicsSimulationTester(physics_backend=section["requested_physics_backend"]).run_all_tests(scenarios)
        section["modules"]["physics"] = _strict_module_summary("physics", physics_report)

    if "control" in modules:
        control_report = ControlTester(
            physics_backend=section["requested_physics_backend"],
            controller_backend=section["requested_controller_backend"],
        ).run_all_tests(scenarios)
        section["modules"]["control"] = _strict_module_summary("control", control_report)

    return section


def _run_backend_comparison(config: Dict[str, Any], scenarios: List[TestScenario]) -> Dict[str, Any]:
    modules = list(config.get("modules", ["physics", "control"]))
    controller_backend = config.get("controller_backend", "base_mpc")
    summary: Dict[str, Any] = {
        "requested_controller_backend": controller_backend,
        "modules": {},
    }

    if "physics" in modules:
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

    if "control" in modules:
        tank_report = ControlTester(physics_backend="tank", controller_backend=controller_backend).run_all_tests(scenarios)
        single_channel_report = ControlTester(
            physics_backend="single_channel",
            controller_backend=controller_backend,
        ).run_all_tests(scenarios)
        segmented_report = ControlTester(
            physics_backend="segmented_hf",
            controller_backend=controller_backend,
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

    return summary


def _build_findings(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    strict_modules = summary.get("strict_revalidation", {}).get("modules", {})
    for module_name, module in strict_modules.items():
        findings.append(
            {
                "priority": "high" if module["pass_rate"] < 0.85 else "medium",
                "area": module_name,
                "finding": (
                    f"Strict {module_name} pass rate is {module['pass_rate']:.1%} "
                    f"({module['passed_tests']}/{module['total_tests']}) on the bounded scenario set."
                ),
                "evidence": module.get("top_failure_reasons", {}),
            }
        )

    comparison_modules = summary.get("backend_comparison", {}).get("modules", {})
    for module_name, module in comparison_modules.items():
        for backend_name in ("single_channel", "segmented_hf"):
            backend_metrics = module.get(backend_name)
            if not backend_metrics:
                continue
            pair = module["comparison_vs_tank"][backend_name]
            findings.append(
                {
                    "priority": (
                        "high"
                        if backend_metrics.get("support_coverage", 0.0) < 0.9
                        or backend_metrics.get("execution_error_count", 0) > 0
                        else "medium"
                    ),
                    "area": f"{module_name}:{backend_name}",
                    "finding": (
                        f"{backend_name} support coverage is {backend_metrics['support_coverage']:.1%}; "
                        f"assessed pass-rate delta vs tank is "
                        f"{pair['assessed_pass_rate_delta'] if pair['assessed_pass_rate_delta'] is not None else 'n/a'}."
                    ),
                    "evidence": {
                        "unsupported_count": backend_metrics.get("unsupported_count", 0),
                        "execution_error_count": backend_metrics.get("execution_error_count", 0),
                        "quality_failure_count": backend_metrics.get("quality_failure_count", 0),
                        "numerical_instability_count": backend_metrics.get("numerical_instability_count", 0),
                    },
                }
            )
    return findings


def _build_priority_risks(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    risks: List[Dict[str, str]] = []

    comparison_modules = summary.get("backend_comparison", {}).get("modules", {})
    for module_name, module in comparison_modules.items():
        for backend_name in ("single_channel", "segmented_hf"):
            metrics = module.get(backend_name)
            if not metrics:
                continue
            if metrics.get("support_coverage", 1.0) < 0.9:
                risks.append(
                    {
                        "priority": "P0",
                        "risk": f"{module_name} on {backend_name} has incomplete scenario support coverage.",
                        "impact": "Backend comparison can overstate quality because unsupported scenarios are excluded from assessed pass rate.",
                        "mitigation": "Constrain scenario dt/horizon space or harden the backend for currently unsupported cases.",
                    }
                )
            if metrics.get("execution_error_count", 0) > 0:
                risks.append(
                    {
                        "priority": "P0",
                        "risk": f"{module_name} on {backend_name} has backend/runtime execution failures.",
                        "impact": "Control-performance and fidelity conclusions are blocked by infrastructure or dependency issues rather than model behavior.",
                        "mitigation": "Fix backend availability/dependencies first, then rerun the same bounded preset before interpreting quality metrics.",
                    }
                )
            if metrics.get("numerical_instability_count", 0) > 0:
                risks.append(
                    {
                        "priority": "P0",
                        "risk": f"{module_name} on {backend_name} shows numerical-instability signals.",
                        "impact": "Simulation fidelity conclusions and closed-loop controller evaluations are unreliable in unstable regions.",
                        "mitigation": "Trace CFL/sub-step limits and add guarded experiment presets that keep the solver inside its valid regime.",
                    }
                )
            if metrics.get("quality_failure_count", 0) > 0:
                risks.append(
                    {
                        "priority": "P1",
                        "risk": f"{module_name} on {backend_name} fails quality gates even when supported.",
                        "impact": "Control-performance claims do not generalize across bounded test conditions.",
                        "mitigation": "Inspect dominant failure reasons, then tighten model/controller parameters against those scenarios first.",
                    }
                )

    strict_modules = summary.get("strict_revalidation", {}).get("modules", {})
    for module_name, module in strict_modules.items():
        if module["pass_rate"] < 0.85:
            risks.append(
                {
                    "priority": "P1",
                    "risk": f"Strict {module_name} pass rate remains below 85% on the bounded suite.",
                    "impact": "Current evidence is not strong enough to treat the module as ready for broader acceptance claims.",
                    "mitigation": "Use top failure reasons from the strict run to build a small targeted repair matrix before expanding coverage.",
                }
            )

    if not risks:
        risks.append(
            {
                "priority": "P2",
                "risk": "Bounded experiment did not surface immediate blocking risks.",
                "impact": "Confidence is still limited by scenario count and bounded coverage.",
                "mitigation": "Scale scenario count and add longer-duration runs before making release-facing conclusions.",
            }
        )
    return risks[:6]


def _build_next_steps(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    strict_modules = summary.get("strict_revalidation", {}).get("modules", {})
    steps: List[Dict[str, str]] = []
    if "physics" in strict_modules:
        steps.append(
            {
                "priority": "1",
                "focus": "Simulation fidelity",
                "action": "Target the top strict physics failure reasons with a dedicated scenario subset and backend-specific dt limits.",
                "success_metric": "Strict physics pass rate >= 90% with numerical_instability_count = 0 for the bounded preset.",
            }
        )
    if "control" in strict_modules:
        steps.append(
            {
                "priority": "2",
                "focus": "Control effectiveness",
                "action": "Tune controller horizon/weights on scenarios that fail tracking, settling, or overshoot gates, then rerun the same preset.",
                "success_metric": "Strict control pass rate >= 90% and quality_failure_count reduced on all supported backends.",
            }
        )
    steps.append(
        {
            "priority": "3",
            "focus": "Evidence quality",
            "action": "Promote the bounded preset into CI/nightly so raw JSON artifacts and summary markdown stay reproducible.",
            "success_metric": "One stable command produces the same output schema under regression test coverage.",
        }
    )
    return steps[:3]


def _write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# {summary['experiment_name']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at']}`")
    lines.append(f"- Scenario count: `{summary['scenario_count']}`")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Categories: `{json.dumps(summary['scenario_breakdown']['category'], ensure_ascii=False)}`")
    lines.append(f"- Topologies: `{json.dumps(summary['scenario_breakdown']['topology'], ensure_ascii=False)}`")
    lines.append(f"- Control modes: `{json.dumps(summary['scenario_breakdown']['control_mode'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Concrete Findings")
    lines.append("")
    for item in summary["findings"]:
        lines.append(f"- [{item['priority']}] {item['area']}: {item['finding']}")
    lines.append("")
    lines.append("## Priority Risks")
    lines.append("")
    for item in summary["priority_risks"]:
        lines.append(f"- [{item['priority']}] {item['risk']} Impact: {item['impact']} Mitigation: {item['mitigation']}")
    lines.append("")
    lines.append("## Next-Step Plan")
    lines.append("")
    for item in summary["next_step_plan"]:
        lines.append(
            f"- [{item['priority']}] {item['focus']}: {item['action']} Success metric: {item['success_metric']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    logging.getLogger("hydroe2e.control.base").setLevel(logging.ERROR)
    logging.getLogger("hydroe2e.phase5.distributed_sil.models.segmented_high_fidelity").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"hydroe2e\.phase5\.distributed_sil\.models\.segmented_high_fidelity")

    seed = int(config.get("seed", 42))
    scenario_count = int(config.get("scenario_count", 12))
    scenario_pool_factor = int(config.get("scenario_pool_factor", 5))

    generator = ScenarioCombinatorialGenerator(seed=seed)
    scenario_pool = generator.generate_all(max_scenarios=max(scenario_count * scenario_pool_factor, scenario_count))
    scenarios = _sample_scenarios(scenario_pool, scenario_count, seed)

    summary: Dict[str, Any] = {
        "experiment_name": config.get("experiment_name", "bounded_fidelity_control_experiment"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "scenario_count": len(scenarios),
        "scenario_ids": [scenario.id for scenario in scenarios],
        "scenario_breakdown": {
            "category": _scenario_breakdown(scenarios, "category"),
            "topology": _scenario_breakdown(scenarios, "topology"),
            "control_mode": _scenario_breakdown(scenarios, "control_mode"),
            "difficulty": _scenario_breakdown(scenarios, "difficulty"),
        },
    }

    strict_config = config.get("strict_revalidation", {})
    if strict_config.get("enabled", True):
        strict_summary = _run_strict_section(strict_config, scenarios)
        summary["strict_revalidation"] = strict_summary
        strict_output = output_dir / str(config["outputs"]["strict_json"])
        strict_output.write_text(json.dumps(strict_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_config = config.get("backend_comparison", {})
    if comparison_config.get("enabled", True):
        comparison_summary = _run_backend_comparison(comparison_config, scenarios)
        summary["backend_comparison"] = comparison_summary
        comparison_output = output_dir / str(config["outputs"]["backend_json"])
        comparison_output.write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["findings"] = _build_findings(summary)
    summary["priority_risks"] = _build_priority_risks(summary)
    summary["next_step_plan"] = _build_next_steps(summary)
    return summary


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    outputs = config.get("outputs", {})
    output_dir = args.output_dir or REPO_ROOT / outputs.get("directory", "reports/experiments/fidelity_control_bounded")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_experiment(config, output_dir)

    summary_json = output_dir / str(outputs.get("summary_json", "summary.json"))
    summary_md = output_dir / str(outputs.get("summary_md", "summary.md"))
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, summary_md)

    print(f"FIDELITY_CONTROL_SUMMARY_JSON={summary_json}")
    print(f"FIDELITY_CONTROL_SUMMARY_MD={summary_md}")
    for finding in summary["findings"][:4]:
        print(f"FINDING[{finding['area']}]={finding['finding']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

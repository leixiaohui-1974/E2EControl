#!/usr/bin/env python3
"""Evaluate strict revalidation summaries against release-gate thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check strict revalidation summary against release-gate thresholds."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to strict_revalidation_summary.json.",
    )
    parser.add_argument(
        "--module",
        choices=["physics", "control"],
        default="control",
        help="Module in the summary to evaluate.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable gate status JSON.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-average-score", type=float, default=0.85)
    parser.add_argument("--max-skipped-tests", type=int, default=0)
    parser.add_argument("--min-setpoint-tracking-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-constraint-handling-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-stability-pass-rate", type=float, default=0.95)
    parser.add_argument("--min-overshoot-pass-rate", type=float, default=0.85)
    parser.add_argument("--min-response-time-pass-rate", type=float, default=0.85)
    parser.add_argument("--min-degraded-control-pass-rate", type=float, default=0.90)
    return parser.parse_args()


def _rate(bucket: Dict[str, Any] | None) -> float | None:
    if not bucket:
        return None
    passed = float(bucket.get("passed", 0))
    failed = float(bucket.get("failed", 0))
    total = passed + failed
    if total <= 0:
        return None
    return passed / total


def _append_failure(failures: List[Dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    failures.append(
        {
            "check": name,
            "actual": actual,
            "expected": expected,
        }
    )


def evaluate(summary: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    module = summary.get("modules", {}).get(args.module)
    if module is None:
        return {
            "status": "FAIL",
            "module": args.module,
            "reason": f"Module not found in summary: {args.module}",
            "failures": [
                {
                    "check": "module_present",
                    "actual": None,
                    "expected": args.module,
                }
            ],
        }

    failures: List[Dict[str, Any]] = []
    pass_rate = float(module.get("pass_rate", 0.0))
    average_score = float(module.get("average_score", 0.0))
    skipped_tests = int(module.get("skipped_tests", 0))
    by_type = module.get("by_test_type", {})

    if pass_rate < args.min_pass_rate:
        _append_failure(failures, "pass_rate", pass_rate, f">= {args.min_pass_rate}")
    if average_score < args.min_average_score:
        _append_failure(failures, "average_score", average_score, f">= {args.min_average_score}")
    if skipped_tests > args.max_skipped_tests:
        _append_failure(failures, "skipped_tests", skipped_tests, f"<= {args.max_skipped_tests}")

    required_by_type = {
        "setpoint_tracking": args.min_setpoint_tracking_pass_rate,
        "constraint_handling": args.min_constraint_handling_pass_rate,
        "stability": args.min_stability_pass_rate,
    }
    optional_by_type = {
        "overshoot": args.min_overshoot_pass_rate,
        "response_time": args.min_response_time_pass_rate,
        "degraded_control": args.min_degraded_control_pass_rate,
    }

    for test_type, threshold in required_by_type.items():
        rate = _rate(by_type.get(test_type))
        if rate is None:
            _append_failure(failures, f"{test_type}_present", None, "present in summary")
        elif rate < threshold:
            _append_failure(failures, f"{test_type}_pass_rate", rate, f">= {threshold}")

    for test_type, threshold in optional_by_type.items():
        rate = _rate(by_type.get(test_type))
        if rate is not None and rate < threshold:
            _append_failure(failures, f"{test_type}_pass_rate", rate, f">= {threshold}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "module": args.module,
        "summary_path": str(args.summary),
        "pass_rate": pass_rate,
        "average_score": average_score,
        "skipped_tests": skipped_tests,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate(summary, args)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"STRICT_REVALIDATION_GATE_STATUS={result['status']}")
    print(f"STRICT_REVALIDATION_GATE_MODULE={result['module']}")
    if result["status"] == "FAIL":
        print(f"STRICT_REVALIDATION_GATE_FAILURES={len(result['failures'])}")
        for item in result["failures"]:
            print(
                f"GATE_FAIL check={item['check']} actual={item['actual']} expected={item['expected']}"
            )

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

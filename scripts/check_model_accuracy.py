#!/usr/bin/env python3
"""Minimal reproducible model-accuracy gate built on scripts/verify_solvers.py."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydroe2e.hydraulics.solvers import (
    ChannelParams,
    GodunuvHLL,
    LaxWendroff,
    MacCormack,
    TVDMUSCL,
    create_all_solvers,
)


BASELINE_SOLVER_CLASSES = [
    LaxWendroff,
    MacCormack,
    GodunuvHLL,
    TVDMUSCL,
]

DEFAULT_REFERENCE_PACK = (
    PROJECT_ROOT
    / "tests"
    / "reference_data"
    / "model_accuracy_reference_pack.json"
)

DEFAULT_THRESHOLDS = {
    "mass_error_pct": 5.0,
    "steady_error_pct": 3.0,
    "wave_arrival_error_pct": 30.0,
    "cross_spread_pct": 3.0,
    "reference_gauge_rmse_m": 0.03,
    "reference_profile_rmse_m": 0.01,
}


def _load_verify_module() -> Any:
    # verify_solvers has module-level prints; suppress those during import.
    verify_path = Path(__file__).with_name("verify_solvers.py")
    spec = importlib.util.spec_from_file_location("verify_solvers", verify_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load verify_solvers from {verify_path}")

    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        sys.modules["verify_solvers"] = module
        spec.loader.exec_module(module)
    return module


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types for JSON output."""
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]

    # numpy types expose `.item()` for scalar conversion.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # numpy arrays expose `.tolist()`.
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal model-accuracy gate (V1/V2/V4 by default) "
            "using existing verify_solvers checks."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Use a shorter scenario duration for faster checks "
            "(keeps same physics/check definitions)."
        ),
    )
    parser.add_argument(
        "--total-seconds",
        type=float,
        default=None,
        help="Override simulation total seconds (default comes from verify_solvers).",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=51,
        help="Number of spatial grid points when creating solvers (default: 51).",
    )
    parser.add_argument(
        "--all-solvers",
        action="store_true",
        help="Create all available solvers from hydroe2e.hydraulics.solvers.",
    )
    parser.add_argument(
        "--solver-filter",
        action="append",
        default=[],
        help=(
            "Only keep solvers whose name contains this substring. "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--include-wave-speed",
        action="store_true",
        help="Deprecated alias. Wave-speed is mandatory by default.",
    )
    parser.add_argument(
        "--skip-wave-speed",
        action="store_true",
        help="Skip V3 wave-speed from mandatory checks.",
    )
    parser.add_argument(
        "--require-cross-consistency",
        action="store_true",
        help="Require V5 cross-consistency to pass as part of overall pass/fail.",
    )
    parser.add_argument(
        "--skip-reference-comparison",
        action="store_true",
        help="Skip comparison against the tracked golden reference pack.",
    )
    parser.add_argument(
        "--reference-pack",
        type=Path,
        default=DEFAULT_REFERENCE_PACK,
        help="Path to the tracked golden-reference pack JSON.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable summary JSON.",
    )
    parser.add_argument(
        "--max-mass-error-pct",
        type=float,
        default=DEFAULT_THRESHOLDS["mass_error_pct"],
        help="Maximum allowed V1 mass-conservation relative error percentage.",
    )
    parser.add_argument(
        "--max-steady-error-pct",
        type=float,
        default=DEFAULT_THRESHOLDS["steady_error_pct"],
        help="Maximum allowed V2 steady-state relative error percentage.",
    )
    parser.add_argument(
        "--max-wave-arrival-error-pct",
        type=float,
        default=DEFAULT_THRESHOLDS["wave_arrival_error_pct"],
        help="Maximum allowed V3 wave-arrival relative error percentage.",
    )
    parser.add_argument(
        "--max-cross-spread-pct",
        type=float,
        default=DEFAULT_THRESHOLDS["cross_spread_pct"],
        help="Maximum allowed V5 cross-solver spread percentage.",
    )
    parser.add_argument(
        "--max-reference-gauge-rmse-m",
        type=float,
        default=DEFAULT_THRESHOLDS["reference_gauge_rmse_m"],
        help="Maximum allowed gauge RMSE against golden reference (meters).",
    )
    parser.add_argument(
        "--max-reference-profile-rmse-m",
        type=float,
        default=DEFAULT_THRESHOLDS["reference_profile_rmse_m"],
        help="Maximum allowed final-profile RMSE against golden reference (meters).",
    )
    return parser.parse_args()


def _configure_scenario(vs: Any, args: argparse.Namespace) -> None:
    if args.total_seconds is not None:
        vs.TOTAL = float(args.total_seconds)
    elif args.quick:
        # Ensure enough post-step samples for steady-state averaging (-100 points).
        vs.TOTAL = 2400.0

    vs.N_STEPS = int(vs.TOTAL / vs.DT)


def _build_solvers(vs: Any, args: argparse.Namespace) -> list[Any]:
    params = ChannelParams(vs.L, vs.W, vs.S0, vs.n_m, args.grid_points)
    if args.all_solvers:
        solvers = create_all_solvers(params)
    else:
        solvers = [cls(params) for cls in BASELINE_SOLVER_CLASSES]

    if args.solver_filter:
        patterns = [p.lower() for p in args.solver_filter]
        solvers = [
            s
            for s in solvers
            if any(pattern in s.name.lower() for pattern in patterns)
        ]
    return solvers


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clone_check(check: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(check)
    cloned["legacy_pass"] = bool(check.get("passed", False))
    return cloned


def _apply_gate(
    check: dict[str, Any],
    passed: bool,
    threshold_label: str,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> dict[str, Any]:
    gated = _clone_check(check)
    gated["passed"] = bool(passed)
    gated["gate_pass"] = bool(passed)
    gated["gate_threshold"] = threshold_label
    if metric_name is not None:
        gated["gate_metric_name"] = metric_name
    if metric_value is not None:
        gated["gate_metric_value"] = round(metric_value, 4)
    return gated


def _evaluate_mass_gate(check: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rel_error = _safe_float(check.get("rel_error_pct"))
    passed = rel_error is not None and rel_error <= args.max_mass_error_pct
    gated = _apply_gate(
        check,
        passed=passed,
        threshold_label=f"<= {args.max_mass_error_pct:.2f}%",
        metric_name="rel_error_pct",
        metric_value=rel_error,
    )
    if rel_error is None:
        gated["detail"] = f"{check.get('detail', '')} | strict gate缺少rel_error_pct".strip(" |")
    return gated


def _evaluate_steady_gate(check: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rel_error = _safe_float(check.get("rel_error_pct"))
    passed = rel_error is not None and rel_error <= args.max_steady_error_pct
    gated = _apply_gate(
        check,
        passed=passed,
        threshold_label=f"<= {args.max_steady_error_pct:.2f}%",
        metric_name="rel_error_pct",
        metric_value=rel_error,
    )
    if rel_error is None:
        gated["detail"] = f"{check.get('detail', '')} | strict gate缺少rel_error_pct".strip(" |")
    return gated


def _evaluate_wave_gate(check: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    arrival = _safe_float(check.get("arrival_time_s"))
    reference = _safe_float(check.get("t_dynamic_s"))
    rel_error = None
    if arrival is not None and reference not in (None, 0.0):
        rel_error = abs(arrival - reference) / reference * 100.0

    passed = rel_error is not None and rel_error <= args.max_wave_arrival_error_pct
    gated = _apply_gate(
        check,
        passed=passed,
        threshold_label=f"<= {args.max_wave_arrival_error_pct:.2f}% vs t_dynamic",
        metric_name="arrival_rel_error_pct_vs_t_dynamic",
        metric_value=rel_error,
    )
    if rel_error is None:
        gated["detail"] = f"{check.get('detail', '')} | strict gate缺少有效波速数据".strip(" |")
    return gated


def _evaluate_physical_gate(
    check: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    issues = list(check.get("issues", []))
    for series_name in ("h_up", "h_mid", "h_down"):
        series = data.get(series_name, [])
        for value in series:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                issues.append(f"{series_name}存在非数值")
                break
            if not math.isfinite(numeric):
                issues.append(f"{series_name}存在NaN/Inf")
                break
        if any(issue.startswith(series_name) for issue in issues):
            continue

        if any(float(value) < 0.0 for value in series):
            issues.append(f"{series_name}出现负水深")
        if any(float(value) > 10.0 for value in series):
            issues.append(f"{series_name}水深异常偏大")

    gated = _apply_gate(
        check,
        passed=len(issues) == 0,
        threshold_label="no NaN/Inf, no negative depth, no blow-up",
        metric_name="issue_count",
        metric_value=float(len(issues)),
    )
    gated["issues"] = issues
    gated["detail"] = "; ".join(issues) if issues else "全部通过"
    return gated


def _collect_checks(
    vs: Any,
    solver_name: str,
    data: dict[str, Any],
    solver_obj: Any,
    include_v3: bool,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    raw_checks: dict[str, dict[str, Any]] = {
        "V1": _to_builtin(vs.verify_mass_conservation(solver_name, data, solver_obj)),
        "V2": _to_builtin(vs.verify_steady_state(solver_name, data)),
        "V4": _to_builtin(vs.verify_physical_reasonableness(solver_name, data)),
    }
    if include_v3:
        raw_checks["V3"] = _to_builtin(vs.verify_wave_speed(solver_name, data))

    checks = {
        "V1": _evaluate_mass_gate(raw_checks["V1"], args),
        "V2": _evaluate_steady_gate(raw_checks["V2"], args),
        "V4": _evaluate_physical_gate(raw_checks["V4"], data),
    }
    if include_v3:
        checks["V3"] = _evaluate_wave_gate(raw_checks["V3"], args)
    return checks


def _evaluate_cross_consistency(
    raw_v5: dict[str, Any],
    solvers_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spread_pct = _safe_float(raw_v5.get("spread_pct"))
    if solvers_count < 2:
        gated = _apply_gate(
            raw_v5,
            passed=False,
            threshold_label=f"<= {args.max_cross_spread_pct:.2f}% with >=2 solvers",
        )
        gated["detail"] = "至少需要2个基线求解器才能进行V5互验"
        return gated

    passed = spread_pct is not None and spread_pct <= args.max_cross_spread_pct
    gated = _apply_gate(
        raw_v5,
        passed=passed,
        threshold_label=f"<= {args.max_cross_spread_pct:.2f}%",
        metric_name="spread_pct",
        metric_value=spread_pct,
    )
    if spread_pct is None:
        gated["detail"] = f"{raw_v5.get('detail', '')} | strict gate缺少spread_pct".strip(" |")
    return gated


def _load_reference_pack(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select_reference_scenario(
    reference_pack: dict[str, Any] | None,
    vs: Any,
) -> dict[str, Any] | None:
    if not reference_pack:
        return None

    for scenario in reference_pack.get("scenarios", []):
        cfg = scenario.get("scenario", {})
        if (
            abs(float(cfg.get("L", -1.0)) - float(vs.L)) < 1e-9
            and abs(float(cfg.get("W", -1.0)) - float(vs.W)) < 1e-9
            and abs(float(cfg.get("S0", -1.0)) - float(vs.S0)) < 1e-12
            and abs(float(cfg.get("n_m", -1.0)) - float(vs.n_m)) < 1e-12
            and abs(float(cfg.get("h0", -1.0)) - float(vs.h0)) < 1e-9
            and abs(float(cfg.get("Q0", -1.0)) - float(vs.Q0)) < 1e-9
            and abs(float(cfg.get("Q_STEP", -1.0)) - float(vs.Q_STEP)) < 1e-9
            and abs(float(cfg.get("DT", -1.0)) - float(vs.DT)) < 1e-9
            and abs(float(cfg.get("STEP_T", -1.0)) - float(vs.STEP_T)) < 1e-9
            and abs(float(cfg.get("TOTAL", -1.0)) - float(vs.TOTAL)) < 1e-9
        ):
            return scenario
    return None


def _rmse(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - reference) ** 2)))


def _evaluate_reference_check(
    solver_name: str,
    data: dict[str, Any],
    solver_obj: Any,
    reference_scenario: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gauges = reference_scenario.get("gauges", {})
    gauge_rmses: dict[str, float] = {}
    for gauge_name in ("h_up", "h_mid", "h_down"):
        actual_series = np.asarray(data.get(gauge_name, []), dtype=float)
        reference_series = np.asarray(gauges.get(gauge_name, []), dtype=float)
        if actual_series.size == 0 or reference_series.size == 0:
            gauge_rmses[gauge_name] = float("inf")
            continue
        if actual_series.shape != reference_series.shape:
            gauge_rmses[gauge_name] = float("inf")
            continue
        gauge_rmses[gauge_name] = _rmse(actual_series, reference_series)

    reference_profile = reference_scenario.get("final_profile", {})
    ref_x = np.asarray(reference_profile.get("x", []), dtype=float)
    ref_h = np.asarray(reference_profile.get("h", []), dtype=float)
    solver_x = np.linspace(0.0, float(solver_obj.p.length), int(solver_obj.N))
    solver_h = np.asarray(solver_obj.get_h_profile(), dtype=float)
    if ref_x.size == 0 or ref_h.size == 0 or solver_h.size == 0:
        profile_rmse = float("inf")
    else:
        interpolated = np.interp(solver_x, ref_x, ref_h)
        profile_rmse = _rmse(solver_h, interpolated)

    max_gauge_rmse = max(gauge_rmses.values()) if gauge_rmses else float("inf")
    passed = (
        math.isfinite(max_gauge_rmse)
        and math.isfinite(profile_rmse)
        and max_gauge_rmse <= args.max_reference_gauge_rmse_m
        and profile_rmse <= args.max_reference_profile_rmse_m
    )
    detail = (
        f"max gauge RMSE={max_gauge_rmse:.4f}m, "
        f"profile RMSE={profile_rmse:.4f}m vs {reference_scenario.get('scenario_id', 'reference')}"
    )
    return {
        "test": "VREF 黄金真值比对",
        "passed": passed,
        "legacy_pass": passed,
        "gate_pass": passed,
        "gate_threshold": (
            f"gauge<={args.max_reference_gauge_rmse_m:.3f}m, "
            f"profile<={args.max_reference_profile_rmse_m:.3f}m"
        ),
        "gate_metric_name": "max_gauge_rmse_m/profile_rmse_m",
        "gauge_rmse_m": {k: round(v, 6) for k, v in gauge_rmses.items()},
        "max_gauge_rmse_m": round(max_gauge_rmse, 6),
        "profile_rmse_m": round(profile_rmse, 6),
        "reference_scenario_id": reference_scenario.get("scenario_id"),
        "reference_solver": reference_scenario.get("reference_solver", {}),
        "detail": detail,
        "solver": solver_name,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    vs = _load_verify_module()
    _configure_scenario(vs, args)
    solvers = _build_solvers(vs, args)
    reference_pack = None if args.skip_reference_comparison else _load_reference_pack(args.reference_pack)
    reference_scenario = _select_reference_scenario(reference_pack, vs)

    include_v3 = not args.skip_wave_speed
    mandatory_checks = ["V1", "V2", "V4"]
    if include_v3:
        mandatory_checks.insert(2, "V3")
    reference_comparison_enabled = not args.skip_reference_comparison
    if reference_comparison_enabled:
        mandatory_checks.append("VREF")

    reference_error: str | None = None
    if reference_comparison_enabled and reference_pack is None:
        reference_error = f"Reference pack not found: {args.reference_pack}"
    elif reference_comparison_enabled and reference_scenario is None:
        reference_error = (
            "Reference pack does not contain a scenario matching the selected "
            "check_model_accuracy configuration."
        )

    if not solvers:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overall_pass": False,
            "status": "FAIL",
            "reason": "No solvers selected after filtering.",
            "gate_rule": {
                "mandatory_checks": mandatory_checks,
                "all_selected_solvers_must_pass": True,
                "require_v5_cross_consistency": args.require_cross_consistency,
                "reference_pack": str(args.reference_pack) if reference_comparison_enabled else None,
                "reference_scenario_id": (
                    reference_scenario.get("scenario_id") if reference_scenario else None
                ),
            },
            "solvers": [],
            "failed_items": [],
        }

    if reference_error is not None:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "overall_pass": False,
            "status": "FAIL",
            "reason": reference_error,
            "scenario": {
                "L": vs.L,
                "W": vs.W,
                "S0": vs.S0,
                "n_m": vs.n_m,
                "h0": vs.h0,
                "Q0": vs.Q0,
                "Q_STEP": vs.Q_STEP,
                "DT": vs.DT,
                "STEP_T": vs.STEP_T,
                "TOTAL": vs.TOTAL,
                "N_STEPS": vs.N_STEPS,
            },
            "gate_rule": {
                "mandatory_checks": mandatory_checks,
                "all_selected_solvers_must_pass": True,
                "require_v5_cross_consistency": args.require_cross_consistency,
                "reference_pack": str(args.reference_pack),
                "reference_scenario_id": None,
                "strict_thresholds": {
                    "V1_mass_error_pct_max": args.max_mass_error_pct,
                    "V2_steady_error_pct_max": args.max_steady_error_pct,
                    "V3_wave_arrival_error_pct_max": args.max_wave_arrival_error_pct,
                    "V5_cross_spread_pct_max": args.max_cross_spread_pct,
                    "VREF_gauge_rmse_m_max": args.max_reference_gauge_rmse_m,
                    "VREF_profile_rmse_m_max": args.max_reference_profile_rmse_m,
                },
            },
            "solvers": [],
            "failed_items": [
                {
                    "solver": "GLOBAL",
                    "check": "VREF",
                    "detail": reference_error,
                }
            ],
            "cross_consistency_v5": None,
        }

    q_step_fn = lambda t: vs.Q_STEP if t >= vs.STEP_T else vs.Q0
    solver_summaries: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    all_data: dict[str, dict[str, Any]] = {}

    for solver in solvers:
        t0 = time.time()
        data = vs.run_solver(solver, q_step_fn)
        elapsed_ms = round((time.time() - t0) * 1000.0, 1)

        all_data[solver.name] = data
        checks = _collect_checks(vs, solver.name, data, solver, include_v3, args)
        if reference_comparison_enabled:
            checks["VREF"] = _evaluate_reference_check(
                solver.name,
                data,
                solver,
                reference_scenario,
                args,
            )
        mandatory_pass = all(bool(checks[c]["passed"]) for c in mandatory_checks)

        for check_id in mandatory_checks:
            check = checks[check_id]
            if not check["passed"]:
                failed_items.append(
                    {
                        "solver": solver.name,
                        "check": check_id,
                        "detail": str(check.get("detail", "")),
                    }
                )

        solver_summaries.append(
            {
                "solver": solver.name,
                "runtime_ms": elapsed_ms,
                "mandatory_pass": mandatory_pass,
                "checks": checks,
            }
        )

    raw_v5 = _to_builtin(vs.verify_cross_consistency(all_data))
    v5 = _evaluate_cross_consistency(raw_v5, len(solvers), args)
    mandatory_ok = len(failed_items) == 0
    v5_ok = bool(v5.get("passed", True))
    if args.require_cross_consistency and not v5_ok:
        failed_items.append(
            {
                "solver": "GLOBAL",
                "check": "V5",
                "detail": str(v5.get("detail", "")),
            }
        )
    overall_pass = mandatory_ok and (v5_ok or not args.require_cross_consistency)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_pass": overall_pass,
        "status": "PASS" if overall_pass else "FAIL",
        "scenario": {
            "L": vs.L,
            "W": vs.W,
            "S0": vs.S0,
            "n_m": vs.n_m,
            "h0": vs.h0,
            "Q0": vs.Q0,
            "Q_STEP": vs.Q_STEP,
            "DT": vs.DT,
            "STEP_T": vs.STEP_T,
            "TOTAL": vs.TOTAL,
            "N_STEPS": vs.N_STEPS,
        },
        "gate_rule": {
            "mandatory_checks": mandatory_checks,
            "all_selected_solvers_must_pass": True,
            "require_v5_cross_consistency": args.require_cross_consistency,
            "reference_pack": str(args.reference_pack) if reference_comparison_enabled else None,
            "reference_scenario_id": (
                reference_scenario.get("scenario_id") if reference_scenario else None
            ),
            "strict_thresholds": {
                "V1_mass_error_pct_max": args.max_mass_error_pct,
                "V2_steady_error_pct_max": args.max_steady_error_pct,
                "V3_wave_arrival_error_pct_max": args.max_wave_arrival_error_pct,
                "V5_cross_spread_pct_max": args.max_cross_spread_pct,
                "VREF_gauge_rmse_m_max": args.max_reference_gauge_rmse_m,
                "VREF_profile_rmse_m_max": args.max_reference_profile_rmse_m,
            },
        },
        "solvers": solver_summaries,
        "failed_items": failed_items,
        "cross_consistency_v5": v5,
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    gate_rule = summary.get("gate_rule", {})
    mandatory_checks = gate_rule.get("mandatory_checks", [])
    print("MODEL_ACCURACY_GATE")
    print(f"MANDATORY_CHECKS={','.join(mandatory_checks)}")

    for solver in summary.get("solvers", []):
        checks = solver.get("checks", {})
        compact = ", ".join(
            f"{k}={'PASS' if checks.get(k, {}).get('passed') else 'FAIL'}"
            for k in mandatory_checks
        )
        status = "PASS" if solver.get("mandatory_pass") else "FAIL"
        print(f"SOLVER={solver['solver']} STATUS={status} CHECKS=[{compact}]")

    if summary.get("cross_consistency_v5"):
        v5 = summary["cross_consistency_v5"]
        print(f"V5_CROSS_CONSISTENCY={'PASS' if v5.get('passed') else 'FAIL'}")

    print(f"FAILED_ITEMS={len(summary.get('failed_items', []))}")
    print(f"MODEL_ACCURACY_STATUS={summary.get('status', 'FAIL')}")


def main() -> int:
    args = _parse_args()
    summary = run_gate(args)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"MODEL_ACCURACY_SUMMARY_JSON={args.json_out}")

    _print_summary(summary)
    return 0 if summary.get("overall_pass") else 1


if __name__ == "__main__":
    sys.exit(main())

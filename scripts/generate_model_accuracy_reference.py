#!/usr/bin/env python3
"""Generate tracked golden-reference scenarios for model-accuracy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from hydroe2e.hydraulics.solvers import ChannelParams, TVDMUSCL, manning_Q


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "reference_data"
    / "model_accuracy_reference_pack.json"
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "reference_data"
    / "model_accuracy_reference_manifest.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the golden reference pack used by check_model_accuracy.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path for the golden reference pack.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Output JSON path for the stable manifest companion.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=201,
        help="Spatial grid size for the golden reference solver.",
    )
    return parser.parse_args()


def _canonical_base_scenario() -> dict[str, float]:
    params = ChannelParams(length=5000.0, width=10.0, slope=0.0005, manning_n=0.025, n_nodes=201)
    h0 = 2.0
    q0 = manning_Q(params.width, h0, params.manning_n, params.slope)
    return {
        "L": params.length,
        "W": params.width,
        "S0": params.slope,
        "n_m": params.manning_n,
        "h0": h0,
        "Q0": q0,
        "Q_STEP": q0 + 3.0,
        "DT": 10.0,
        "STEP_T": 600.0,
    }


def _run_reference_scenario(
    scenario_id: str,
    total_seconds: float,
    grid_points: int,
) -> dict[str, object]:
    base = _canonical_base_scenario()
    params = ChannelParams(
        length=base["L"],
        width=base["W"],
        slope=base["S0"],
        manning_n=base["n_m"],
        n_nodes=grid_points,
    )
    solver = TVDMUSCL(params)
    solver.initialize(base["h0"], base["Q0"])
    n_steps = int(total_seconds / base["DT"])
    mid_idx = solver.N // 2
    down_idx = max(0, solver.N - 2)

    series = {
        "time": [],
        "h_up": [],
        "h_mid": [],
        "h_down": [],
    }

    started = time.time()
    for step in range(n_steps):
        current_time = step * base["DT"]
        q_in = base["Q_STEP"] if current_time >= base["STEP_T"] else base["Q0"]
        solver.advance(base["DT"], q_in, base["h0"])
        h = solver.get_h_profile()
        series["time"].append(current_time)
        series["h_up"].append(float(h[0]))
        series["h_mid"].append(float(h[mid_idx]))
        series["h_down"].append(float(h[down_idx]))

    final_x = np.linspace(0.0, params.length, params.n_nodes)
    final_h = solver.get_h_profile()

    scenario = {
        **base,
        "TOTAL": total_seconds,
        "N_STEPS": n_steps,
    }
    return {
        "scenario_id": scenario_id,
        "scenario": scenario,
        "reference_solver": {
            "name": solver.name,
            "grid_points": grid_points,
            "runtime_s": round(time.time() - started, 3),
        },
        "gauges": series,
        "final_profile": {
            "x": [float(x) for x in final_x],
            "h": [float(v) for v in final_h],
        },
        "summary": {
            "steady_upstream_depth_m": round(float(final_h[0]), 6),
            "steady_mid_depth_m": round(float(final_h[mid_idx]), 6),
            "steady_downstream_depth_m": round(float(final_h[down_idx]), 6),
        },
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    pack = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/generate_model_accuracy_reference.py",
        "reference_solver_family": "TVD-MUSCL",
        "scenarios": [
            _run_reference_scenario("canonical_step_response_quick", 2400.0, args.grid_points),
            _run_reference_scenario("canonical_step_response_full", 7200.0, args.grid_points),
        ],
    }

    args.output.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pack_bytes = args.output.read_bytes()
    manifest = {
        "schema_version": 1,
        "generator": "scripts/generate_model_accuracy_reference.py",
        "reference_pack": str(args.output.resolve().relative_to(repo_root)),
        "reference_pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
        "reference_solver_family": pack["reference_solver_family"],
        "scenario_ids": [scenario["scenario_id"] for scenario in pack["scenarios"]],
        "scenarios": [
            {
                "scenario_id": scenario["scenario_id"],
                "grid_points": scenario["reference_solver"]["grid_points"],
                "total_seconds": scenario["scenario"]["TOTAL"],
                "n_steps": scenario["scenario"]["N_STEPS"],
                "steady_upstream_depth_m": scenario["summary"]["steady_upstream_depth_m"],
                "steady_mid_depth_m": scenario["summary"]["steady_mid_depth_m"],
                "steady_downstream_depth_m": scenario["summary"]["steady_downstream_depth_m"],
            }
            for scenario in pack["scenarios"]
        ],
    }
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"MODEL_ACCURACY_REFERENCE_PACK={args.output}")
    print(f"MODEL_ACCURACY_REFERENCE_MANIFEST={args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

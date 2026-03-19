import json
import subprocess
import sys
from pathlib import Path


def test_model_accuracy_gate_default_gate_matches_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_model_accuracy.py"
    json_out = tmp_path / "model_accuracy_summary.json"

    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--solver-filter",
        "TVD-MUSCL",
        "--json-out",
        str(json_out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode in (0, 1), proc.stderr
    assert json_out.exists(), proc.stdout

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(summary.get("overall_pass"), bool)
    assert summary.get("status") in {"PASS", "FAIL"}
    assert summary.get("gate_rule", {}).get("mandatory_checks") == ["V1", "V2", "V3", "V4", "VREF"]
    assert summary.get("gate_rule", {}).get("reference_scenario_id") == "canonical_step_response_quick"
    assert summary.get("gate_rule", {}).get("reference_pack", "").endswith(
        "tests\\reference_data\\model_accuracy_reference_pack.json"
    )
    assert summary.get("gate_rule", {}).get("strict_thresholds") == {
        "V1_mass_error_pct_max": 5.0,
        "V2_steady_error_pct_max": 3.0,
        "V3_wave_arrival_error_pct_max": 30.0,
        "V5_cross_spread_pct_max": 3.0,
        "VREF_gauge_rmse_m_max": 0.03,
        "VREF_profile_rmse_m_max": 0.01,
    }
    assert summary.get("solvers"), "Expected at least one solver result."
    solver_summary = summary["solvers"][0]
    assert solver_summary["solver"].startswith("TVD-MUSCL")
    assert solver_summary["checks"]["V1"]["legacy_pass"] is True
    assert solver_summary["checks"]["V3"]["gate_metric_name"] == "arrival_rel_error_pct_vs_t_dynamic"
    assert isinstance(solver_summary["checks"]["V3"]["gate_metric_value"], float)
    assert solver_summary["checks"]["VREF"]["reference_scenario_id"] == "canonical_step_response_quick"
    assert solver_summary["checks"]["VREF"]["passed"] is True
    assert solver_summary["checks"]["VREF"]["max_gauge_rmse_m"] <= 0.03
    assert solver_summary["checks"]["VREF"]["profile_rmse_m"] <= 0.01

    expected_code = 0 if summary["overall_pass"] else 1
    assert proc.returncode == expected_code
    assert f"MODEL_ACCURACY_STATUS={summary['status']}" in proc.stdout


def test_model_accuracy_gate_fails_when_no_solver_matches(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_model_accuracy.py"
    json_out = tmp_path / "no_solver_summary.json"

    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--solver-filter",
        "definitely-not-a-real-solver",
        "--json-out",
        str(json_out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert json_out.exists(), proc.stdout

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert summary["overall_pass"] is False
    assert summary["status"] == "FAIL"
    assert summary["reason"] == "No solvers selected after filtering."
    assert summary["solvers"] == []


def test_model_accuracy_gate_fails_when_reference_pack_is_missing(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_model_accuracy.py"
    json_out = tmp_path / "missing_reference_summary.json"
    missing_pack = tmp_path / "does_not_exist_reference_pack.json"

    cmd = [
        sys.executable,
        str(script_path),
        "--quick",
        "--solver-filter",
        "TVD-MUSCL",
        "--reference-pack",
        str(missing_pack),
        "--json-out",
        str(json_out),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert json_out.exists(), proc.stdout

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert summary["overall_pass"] is False
    assert summary["status"] == "FAIL"
    assert summary["failed_items"] == [
        {
            "solver": "GLOBAL",
            "check": "VREF",
            "detail": f"Reference pack not found: {missing_pack}",
        }
    ]
    assert summary["reason"] == f"Reference pack not found: {missing_pack}"

import json
import subprocess
import sys
from pathlib import Path


def _write_summary(tmp_path: Path, *, control_overrides=None):
    control = {
        "pass_rate": 0.93,
        "average_score": 0.88,
        "skipped_tests": 0,
        "by_test_type": {
            "setpoint_tracking": {"passed": 9, "failed": 1},
            "constraint_handling": {"passed": 19, "failed": 1},
            "stability": {"passed": 19, "failed": 1},
            "overshoot": {"passed": 9, "failed": 1},
            "response_time": {"passed": 9, "failed": 1},
            "degraded_control": {"passed": 9, "failed": 1},
        },
    }
    if control_overrides:
        control.update(control_overrides)

    summary = {
        "modules": {
            "control": control,
        }
    }
    path = tmp_path / "strict_revalidation_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_strict_revalidation_gate_passes(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_strict_revalidation_gate.py"
    summary_path = _write_summary(tmp_path)
    json_out = tmp_path / "gate_status.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--summary",
            str(summary_path),
            "--module",
            "control",
            "--json-out",
            str(json_out),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "STRICT_REVALIDATION_GATE_STATUS=PASS" in proc.stdout
    status = json.loads(json_out.read_text(encoding="utf-8"))
    assert status["status"] == "PASS"


def test_strict_revalidation_gate_fails_on_skips_and_thresholds(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_strict_revalidation_gate.py"
    summary_path = _write_summary(
        tmp_path,
        control_overrides={
            "pass_rate": 0.82,
            "average_score": 0.80,
            "skipped_tests": 2,
            "by_test_type": {
                "setpoint_tracking": {"passed": 8, "failed": 2},
                "constraint_handling": {"passed": 17, "failed": 3},
                "stability": {"passed": 18, "failed": 2},
            },
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--summary",
            str(summary_path),
            "--module",
            "control",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "STRICT_REVALIDATION_GATE_STATUS=FAIL" in proc.stdout
    assert "GATE_FAIL check=pass_rate" in proc.stdout
    assert "GATE_FAIL check=skipped_tests" in proc.stdout


def test_strict_revalidation_gate_fails_when_module_missing(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "check_strict_revalidation_gate.py"
    summary_path = tmp_path / "strict_revalidation_summary.json"
    summary_path.write_text(json.dumps({"modules": {}}, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--summary",
            str(summary_path),
            "--module",
            "control",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "STRICT_REVALIDATION_GATE_STATUS=FAIL" in proc.stdout

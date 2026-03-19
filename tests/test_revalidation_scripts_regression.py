import json
import sys
from types import SimpleNamespace

from scripts import compare_revalidation_backends as compare_script
from scripts import run_strict_revalidation as strict_script


def test_run_strict_revalidation_output_schema_smoke(tmp_path, monkeypatch):
    output = tmp_path / "strict_revalidation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_strict_revalidation.py",
            "--scenarios",
            "1",
            "--modules",
            "physics",
            "--output",
            str(output),
        ],
    )

    assert strict_script.main() == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert {"generated_at", "seed", "scenario_count", "physics_backend", "controller_backend", "modules"} <= set(data)
    physics = data["modules"]["physics"]
    assert {
        "module",
        "physics_backend",
        "controller_backend",
        "total_tests",
        "passed_tests",
        "failed_tests",
        "skipped_tests",
        "pass_rate",
        "average_score",
        "by_test_type",
        "failed_samples",
    } <= set(physics)
    assert isinstance(physics["failed_samples"], list)


def test_run_strict_revalidation_requested_vs_effective_backend(tmp_path, monkeypatch):
    output = tmp_path / "strict_revalidation_auto.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_strict_revalidation.py",
            "--scenarios",
            "1",
            "--modules",
            "physics",
            "--physics-backend",
            "auto",
            "--output",
            str(output),
        ],
    )

    assert strict_script.main() == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["physics_backend"] == "auto"
    assert data["modules"]["physics"]["physics_backend"] in {"single_channel", "segmented_hf", "tank"}


def test_compare_backends_schema_and_unsupported_counter_presence(tmp_path, monkeypatch):
    output = tmp_path / "backend_compare.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_revalidation_backends.py",
            "--scenarios",
            "1",
            "--modules",
            "physics",
            "--output",
            str(output),
        ],
    )

    assert compare_script.main() == 0

    data = json.loads(output.read_text(encoding="utf-8"))
    module = data["modules"]["physics"]
    for backend in ("tank", "single_channel", "segmented_hf"):
        assert backend in module
        assert {
            "unsupported_count",
            "backend_unsuitable_count",
            "backend_unsupported_count",
            "execution_error_count",
            "scenario_unsuitable_count",
            "quality_failure_count",
            "numerical_instability_count",
            "by_test_type",
            "top_failure_reasons",
            "unsupported_reasons",
            "quality_failure_reasons",
            "backend_unsuitable_reasons",
            "backend_unsupported_reasons",
            "execution_error_reasons",
            "scenario_unsuitable_reasons",
            "assessed_tests",
            "assessed_pass_rate",
            "support_coverage",
        } <= set(module[backend])


def test_compare_module_metrics_tracks_unsupported_and_numerical_counts():
    fake_report = SimpleNamespace(
        total_tests=3,
        passed_tests=1,
        failed_tests=2,
        skipped_tests=1,
        average_score=0.25,
        physics_backend="segmented_hf",
        controller_backend="unknown",
        by_test_type={},
        test_results=[
            SimpleNamespace(
                passed=True,
                skipped=False,
                errors=[],
                test_type=SimpleNamespace(value="stability"),
            ),
            SimpleNamespace(
                passed=False,
                skipped=False,
                errors=["test crashed: segmented_hf backend currently supports time_step <= 60s"],
                test_type=SimpleNamespace(value="stability"),
            ),
            SimpleNamespace(
                passed=False,
                skipped=False,
                errors=["test crashed: segmented_hf backend requires too many sub-steps for this scenario", "cfl exceeded"],
                test_type=SimpleNamespace(value="stability"),
            ),
            SimpleNamespace(
                passed=False,
                skipped=True,
                errors=["scenario infeasible under actuator bounds: q_out=50 outside [0, 20]"],
                test_type=SimpleNamespace(value="setpoint_tracking"),
            ),
        ],
    )

    metrics = compare_script._module_metrics(fake_report)

    assert metrics["unsupported_count"] == 2
    assert metrics["backend_unsuitable_count"] == 2
    assert metrics["backend_unsupported_count"] == 0
    assert metrics["numerical_instability_count"] == 1
    assert metrics["scenario_unsuitable_count"] == 1
    assert metrics["quality_failure_count"] == 0
    assert metrics["execution_error_count"] == 0
    assert metrics["assessed_tests"] == 1
    assert metrics["assessed_pass_rate"] == 1.0

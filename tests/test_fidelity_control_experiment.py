import json
import sys

import yaml

from scripts import run_fidelity_control_experiment as experiment_script


def test_fidelity_control_experiment_outputs_summary_bundle(tmp_path, monkeypatch):
    output_dir = tmp_path / "experiment_outputs"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "test_bundle",
                "seed": 7,
                "scenario_count": 1,
                "scenario_pool_factor": 2,
                "strict_revalidation": {
                    "enabled": True,
                    "modules": ["physics"],
                    "physics_backend": "tank",
                    "controller_backend": "base_mpc",
                },
                "backend_comparison": {
                    "enabled": True,
                    "modules": ["physics"],
                    "controller_backend": "base_mpc",
                },
                "outputs": {
                    "directory": "ignored-by-test",
                    "strict_json": "strict.json",
                    "backend_json": "compare.json",
                    "summary_json": "summary.json",
                    "summary_md": "summary.md",
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fidelity_control_experiment.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert experiment_script.main() == 0

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "test_bundle"
    assert summary["scenario_count"] == 1
    assert {"strict_revalidation", "backend_comparison", "findings", "priority_risks", "next_step_plan"} <= set(summary)
    assert "physics" in summary["strict_revalidation"]["modules"]
    assert "physics" in summary["backend_comparison"]["modules"]
    assert summary["findings"]
    assert summary["priority_risks"]
    assert summary["next_step_plan"]

    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Concrete Findings" in markdown
    assert "## Priority Risks" in markdown
    assert "## Next-Step Plan" in markdown

#!/usr/bin/env python3
"""Lightweight CI guard for repository structure drift.

This check intentionally avoids expensive computation. It validates:
1) key files and test directories still exist;
2) pytest testpaths in pyproject.toml resolve to existing paths;
3) smoke scripts point to current in-repo locations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 local fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

REQUIRED_FILES = [
    REPO_ROOT / ".github/workflows/ci.yml",
    REPO_ROOT / "quick_test.sh",
    REPO_ROOT / "run_all_demos.sh",
    REPO_ROOT / "scripts/check_model_accuracy.py",
    REPO_ROOT / "scripts/generate_model_accuracy_reference.py",
    REPO_ROOT / "tests/reference_data/model_accuracy_reference_pack.json",
    REPO_ROOT / "tests/reference_data/model_accuracy_reference_manifest.json",
    REPO_ROOT / "hydroe2e/main.py",
    REPO_ROOT / "hydroe2e/phase4/self_healing/self_healing_system.py",
    REPO_ROOT / "hydroe2e/phase5/integrated_system.py",
]

FORBIDDEN_LEGACY_PATTERNS = [
    "python3 config_manager.py",
    "python3 logger.py",
    "python3 test_units.py",
    "python3 api.py",
    "python3 main.py",
    "python3 digital_twin/",
    "python3 phase4/",
    "python3 phase5/",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(REPO_ROOT)}")

    if not PYPROJECT.exists():
        errors.append("missing pyproject.toml")
    else:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        testpaths = (
            data.get("tool", {})
            .get("pytest", {})
            .get("ini_options", {})
            .get("testpaths", [])
        )
        if not testpaths:
            errors.append("tool.pytest.ini_options.testpaths is empty")
        else:
            for rel in testpaths:
                candidate = REPO_ROOT / rel
                if not candidate.exists():
                    errors.append(f"pytest testpath does not exist: {rel}")

    ci_text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    if '"3.9"' in ci_text or "'3.9'" in ci_text or '"3.10"' in ci_text or "'3.10'" in ci_text:
        errors.append("ci.yml still references Python < 3.11")

    quick_text = (REPO_ROOT / "quick_test.sh").read_text(encoding="utf-8")
    run_demo_text = (REPO_ROOT / "run_all_demos.sh").read_text(encoding="utf-8")
    for pattern in FORBIDDEN_LEGACY_PATTERNS:
        if pattern in quick_text:
            errors.append(f"quick_test.sh still contains legacy pattern: {pattern}")
        if pattern in run_demo_text:
            errors.append(f"run_all_demos.sh still contains legacy pattern: {pattern}")

    manifest_path = REPO_ROOT / "tests/reference_data/model_accuracy_reference_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"reference manifest is not valid JSON: {exc}")
        else:
            if manifest.get("schema_version") != 1:
                errors.append("reference manifest schema_version must be 1")
            pack_ref = manifest.get("reference_pack")
            if not pack_ref:
                errors.append("reference manifest missing reference_pack")
            else:
                pack_path = Path(pack_ref)
                if not pack_path.is_absolute():
                    pack_path = REPO_ROOT / pack_path
                if not pack_path.exists():
                    errors.append(f"reference manifest points to missing pack: {pack_ref}")
                else:
                    actual_sha = hashlib.sha256(pack_path.read_bytes()).hexdigest()
                    if manifest.get("reference_pack_sha256") != actual_sha:
                        errors.append("reference manifest sha256 does not match reference pack")
            if not manifest.get("scenario_ids"):
                errors.append("reference manifest scenario_ids is empty")

    if errors:
        for err in errors:
            fail(err)
        print(f"CI smoke check failed with {len(errors)} issue(s).")
        return 1

    print("CI smoke check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

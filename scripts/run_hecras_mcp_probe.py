#!/usr/bin/env python3
"""自动探测本机 HEC-RAS / HydroMind MCP 可用性并输出机器可读结论。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "acceptance" / "hecras_mcp_probe.json"
DEFAULT_CODEX_CONFIG = Path.home() / ".codex" / "config.toml"
CURATED_REPOS = [
    REPO_ROOT,
    Path("Z:/research/HydroMAS"),
    Path("Z:/research/HydroClaude"),
    Path("Z:/research/HydroLab"),
    Path("Z:/research/HydroDesign"),
    Path("Z:/research/HydroArena"),
    Path("Z:/research/HydroWriter"),
]
HEC_RAS_INSTALL_HINTS = [
    Path("C:/Program Files (x86)/HEC"),
    Path("C:/Program Files/HEC"),
    Path("C:/HEC"),
    Path("D:/HEC"),
]
HEC_RAS_EXE_NAMES = ("Ras.exe", "RAS.exe", "HEC-RAS.exe", "ras.exe")
MCP_HINTS = ("FastMCP", "mcp.server.fastmcp", "hydromind.engines", "mcp.run()")
HEC_RAS_HINTS = ("HEC-RAS", "hec_ras", "HECRAS", "RASController", "win32com.client")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe local HEC-RAS / HydroMind MCP readiness.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--require-runnable",
        action="store_true",
        help="Return non-zero when no runnable HEC-RAS MCP path is found.",
    )
    return parser.parse_args()


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if tomllib is not None:
        return tomllib.loads(path.read_text(encoding="utf-8"))

    # Python 3.10 without tomli fallback: extract the only field we need.
    data: Dict[str, Any] = {"mcp_servers": {}}
    current_server: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[mcp_servers.") and line.endswith("]"):
            current_server = line[len("[mcp_servers.") : -1]
            data["mcp_servers"][current_server] = {}
            continue
        if current_server and "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            data["mcp_servers"][current_server][key] = value.strip("\"")
    return data


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _find_first_existing(paths: List[Path]) -> List[str]:
    return [str(path) for path in paths if path.exists()]


def detect_codex_mcp_config(path: Path) -> Dict[str, Any]:
    data = _load_toml(path)
    mcp_servers = data.get("mcp_servers", {}) if isinstance(data, dict) else {}
    server_names = sorted(mcp_servers.keys()) if isinstance(mcp_servers, dict) else []
    hec_ras_servers = [name for name in server_names if "ras" in name.lower() or "hec" in name.lower()]
    return {
        "config_path": str(path),
        "config_exists": path.exists(),
        "registered_servers": server_names,
        "hec_ras_related_servers": hec_ras_servers,
    }


def detect_hec_ras_installation() -> Dict[str, Any]:
    existing_dirs = _find_first_existing(HEC_RAS_INSTALL_HINTS)
    discovered_executables: List[str] = []
    for base_dir in HEC_RAS_INSTALL_HINTS:
        if not base_dir.exists():
            continue
        for exe_name in HEC_RAS_EXE_NAMES:
            discovered_executables.extend(str(path) for path in base_dir.rglob(exe_name))
    discovered_executables = sorted(set(discovered_executables))
    return {
        "install_dirs": existing_dirs,
        "executables": discovered_executables,
        "present": bool(existing_dirs or discovered_executables),
    }


def detect_python_automation_capability() -> Dict[str, Any]:
    win32com_spec = importlib.util.find_spec("win32com.client")
    mcp_spec = importlib.util.find_spec("mcp.server.fastmcp")
    return {
        "pywin32_available": win32com_spec is not None,
        "fastmcp_available": mcp_spec is not None,
    }


def _classify_candidate(path: Path, text: str) -> Dict[str, Any]:
    lower = text.lower()
    return {
        "path": str(path),
        "launchable": "if __name__ == \"__main__\"" in text or "mcp.run()" in lower or "def main(" in lower,
        "has_mcp_hints": any(hint.lower() in lower for hint in MCP_HINTS),
        "hec_ras_related": any(hint.lower() in lower for hint in HEC_RAS_HINTS),
        "hydromind_entrypoint": "hydromind.engines" in lower,
    }


def detect_server_candidates() -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for repo in CURATED_REPOS:
        if not repo.exists():
            continue
        for path in repo.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if not any(part.lower().startswith("mcp") or part.lower().endswith("server") for part in path.parts):
                continue
            text = _safe_read_text(path)
            if not any(hint.lower() in text.lower() for hint in MCP_HINTS + HEC_RAS_HINTS):
                continue
            record = _classify_candidate(path, text)
            if record["has_mcp_hints"] or record["hec_ras_related"]:
                candidates.append(record)

    runnable_hec_ras = [item for item in candidates if item["launchable"] and item["hec_ras_related"]]
    runnable_general = [item for item in candidates if item["launchable"]]
    return {
        "candidates": candidates,
        "runnable_hec_ras_candidates": runnable_hec_ras,
        "runnable_general_candidates": runnable_general,
    }


def detect_hydroclaude_evidence_scaffold() -> Dict[str, Any]:
    case_root = Path("Z:/research/HydroClaude/validation_cases/engineering/hec_ras_steady_flow_example_3_1")
    expected_results = case_root / "expected_results.yaml"
    manifest = case_root / "evidence" / "MANIFEST.md"
    notes: List[str] = []
    disputed = False
    if expected_results.exists():
        payload = yaml.safe_load(expected_results.read_text(encoding="utf-8")) or {}
        disputed = str(payload.get("provenance_status", "")).strip().lower() == "disputed"
        notes = [str(item) for item in payload.get("notes", [])]
    return {
        "case_root": str(case_root),
        "present": case_root.exists(),
        "expected_results_path": str(expected_results),
        "manifest_path": str(manifest),
        "provenance_disputed": disputed,
        "notes": notes,
    }


def build_summary() -> Dict[str, Any]:
    codex_config = detect_codex_mcp_config(DEFAULT_CODEX_CONFIG)
    install_info = detect_hec_ras_installation()
    automation_info = detect_python_automation_capability()
    server_info = detect_server_candidates()
    evidence_info = detect_hydroclaude_evidence_scaffold()

    blockers: List[str] = []
    if not codex_config["hec_ras_related_servers"]:
        blockers.append("当前 Codex MCP 配置未注册任何 HEC-RAS / hec 相关服务。")
    if not install_info["present"]:
        blockers.append("本机未发现 HEC-RAS 安装目录或可执行文件。")
    if not server_info["runnable_hec_ras_candidates"]:
        blockers.append("代码库中未发现可直接启动的 HEC-RAS 相关 MCP 服务入口。")
    if evidence_info["present"] and evidence_info["provenance_disputed"]:
        blockers.append("HydroClaude 中仅存在 disputed provenance scaffold，缺少原始 HEC-RAS 工程/导出证据。")

    runnable = bool(
        codex_config["hec_ras_related_servers"]
        and install_info["present"]
        and server_info["runnable_hec_ras_candidates"]
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "codex_mcp_config": codex_config,
        "hec_ras_installation": install_info,
        "python_automation": automation_info,
        "server_candidates": server_info,
        "hydroclaude_evidence_scaffold": evidence_info,
        "runnable": runnable,
        "blockers": blockers,
    }


def main() -> int:
    args = parse_args()
    summary = build_summary()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"HECRAS_MCP_PROBE_JSON={args.json_out}")
    print(f"RUNNABLE={summary['runnable']}")
    if summary["blockers"]:
        print("BLOCKERS:")
        for blocker in summary["blockers"]:
            print(f"- {blocker}")

    if args.require_runnable and not summary["runnable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

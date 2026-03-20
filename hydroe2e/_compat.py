"""HydroMind contracts 兼容层 - hydromind-contracts 为可选依赖。"""
from __future__ import annotations

from importlib import metadata

try:
    from hydromind_contracts import (
        SimulatorProtocol,
        ControllerProtocol,
        DetectorProtocol,
        HydraulicSolverProtocol,
        ChannelConfigProtocol,
    )
    HAS_HYDROMIND = True
except ImportError:
    SimulatorProtocol = object
    ControllerProtocol = object
    DetectorProtocol = object
    HydraulicSolverProtocol = object
    ChannelConfigProtocol = object
    HAS_HYDROMIND = False


def get_compat_info() -> dict[str, object]:
    """返回当前 HydroMind 兼容层状态，便于集成自检。"""
    exported_protocols = [
        "SimulatorProtocol",
        "ControllerProtocol",
        "DetectorProtocol",
        "HydraulicSolverProtocol",
        "ChannelConfigProtocol",
    ]

    try:
        contracts_version = metadata.version("hydromind-contracts")
    except metadata.PackageNotFoundError:
        contracts_version = None

    return {
        "has_hydromind": HAS_HYDROMIND,
        "contracts_version": contracts_version,
        "protocols": exported_protocols,
        "protocol_module": (
            getattr(SimulatorProtocol, "__module__", "builtins")
            if HAS_HYDROMIND else None
        ),
    }

__all__ = [
    "HAS_HYDROMIND",
    "SimulatorProtocol",
    "ControllerProtocol",
    "DetectorProtocol",
    "HydraulicSolverProtocol",
    "ChannelConfigProtocol",
    "get_compat_info",
]

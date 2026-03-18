"""HydroMind contracts 兼容层 - hydromind-contracts 为可选依赖。"""
from __future__ import annotations

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

__all__ = [
    "HAS_HYDROMIND",
    "SimulatorProtocol",
    "ControllerProtocol",
    "DetectorProtocol",
    "HydraulicSolverProtocol",
    "ChannelConfigProtocol",
]

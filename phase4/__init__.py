"""Compatibility shim for legacy ``phase4`` imports."""

from __future__ import annotations

import sys
from importlib import import_module

_SUBMODULES = {
    "anomaly_detection": "hydroe2e.phase4.anomaly_detection",
    "fault_diagnosis": "hydroe2e.phase4.fault_diagnosis",
    "self_healing": "hydroe2e.phase4.self_healing",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SUBMODULES[name])
    sys.modules[f"{__name__}.{name}"] = module
    return module


for _name, _target in _SUBMODULES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", import_module(_target))

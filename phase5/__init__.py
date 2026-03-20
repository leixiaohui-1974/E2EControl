"""Compatibility shim for legacy ``phase5`` imports."""

from __future__ import annotations

import sys
from importlib import import_module

_BASE = import_module("hydroe2e.phase5")

__all__ = getattr(_BASE, "__all__", [])
__version__ = getattr(_BASE, "__version__", "0.0.0")

_SUBMODULES = {
    "autonomous_learning": "hydroe2e.phase5.autonomous_learning",
    "controllers": "hydroe2e.phase5.controllers",
    "data_interface": "hydroe2e.phase5.data_interface",
    "distributed_sil": "hydroe2e.phase5.distributed_sil",
    "hil_testing": "hydroe2e.phase5.hil_testing",
    "monitoring": "hydroe2e.phase5.monitoring",
    "self_healing": "hydroe2e.phase5.self_healing",
    "water_transfer_system": "hydroe2e.phase5.water_transfer_system",
    "web": "hydroe2e.phase5.web",
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        sys.modules[f"{__name__}.{name}"] = module
        return module
    return getattr(_BASE, name)


for _name, _target in _SUBMODULES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", import_module(_target))


"""
Phase 5 controllers.

Expose controller classes lazily so importing non-control Phase 5 modules does
not require optional optimization dependencies.
"""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "CentralizedScheduler": ("hydroe2e.phase5.controllers.centralized_scheduler", "CentralizedScheduler"),
    "NetworkTopology": ("hydroe2e.phase5.controllers.centralized_scheduler", "NetworkTopology"),
    "SchedulingMode": ("hydroe2e.phase5.controllers.centralized_scheduler", "SchedulingMode"),
    "SchedulingForecast": ("hydroe2e.phase5.controllers.centralized_scheduler", "SchedulingForecast"),
    "SchedulingResult": ("hydroe2e.phase5.controllers.centralized_scheduler", "SchedulingResult"),
    "SchedulerConfig": ("hydroe2e.phase5.controllers.centralized_scheduler", "SchedulerConfig"),
    "ParameterizedLocalMPC": ("hydroe2e.phase5.controllers.parameterized_mpc", "ParameterizedLocalMPC"),
    "ParameterizedDistributedMPC": ("hydroe2e.phase5.controllers.parameterized_mpc", "ParameterizedDistributedMPC"),
    "PhysicalParameters": ("hydroe2e.phase5.controllers.parameterized_mpc", "PhysicalParameters"),
    "ScenarioPhysics": ("hydroe2e.phase5.controllers.parameterized_mpc", "ScenarioPhysics"),
    "HierarchicalMPCController": ("hydroe2e.phase5.controllers.hierarchical_mpc", "HierarchicalMPCController"),
    "HierarchicalConfig": ("hydroe2e.phase5.controllers.hierarchical_mpc", "HierarchicalConfig"),
    "HierarchicalState": ("hydroe2e.phase5.controllers.hierarchical_mpc", "HierarchicalState"),
    "PhysicalScenario": ("hydroe2e.phase5.controllers.hierarchical_mpc", "PhysicalScenario"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

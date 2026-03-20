"""
HydroE2E - 智能水网端到端控制系统
==================================
HydroMind 生态卫星项目，支持独立运行。

Keep package import lightweight so physics-only tooling can run without
installing the full control/optimization stack.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "1.1.0"
__project__ = "HydroE2E"

_LAZY_EXPORTS = {
    "ConfigManager": ("hydroe2e.config_manager", "ConfigManager"),
    "get_config": ("hydroe2e.config_manager", "get_config"),
    "SemanticInterpreter": ("hydroe2e.brain", "SemanticInterpreter"),
    "EnhancedSemanticInterpreter": ("hydroe2e.brain_enhanced", "EnhancedSemanticInterpreter"),
    "SimulationManager": ("hydroe2e.simulation_manager", "SimulationManager"),
    "SimulationDatabase": ("hydroe2e.database", "SimulationDatabase"),
    "MonitoringSystem": ("hydroe2e.monitor", "MonitoringSystem"),
    "Alert": ("hydroe2e.monitor", "Alert"),
    "AlertLevel": ("hydroe2e.monitor", "AlertLevel"),
    "get_logger": ("hydroe2e.logger", "get_logger"),
    "setup_logging": ("hydroe2e.logger", "setup_logging"),
    "SmartPoolException": ("hydroe2e.exceptions", "SmartPoolException"),
    "ConfigurationError": ("hydroe2e.exceptions", "ConfigurationError"),
    "OptimizationError": ("hydroe2e.exceptions", "OptimizationError"),
    "PhysicsError": ("hydroe2e.exceptions", "PhysicsError"),
    "SemanticError": ("hydroe2e.exceptions", "SemanticError"),
    "ValidationError": ("hydroe2e.exceptions", "ValidationError"),
    "DatabaseError": ("hydroe2e.exceptions", "DatabaseError"),
    "MonitoringError": ("hydroe2e.exceptions", "MonitoringError"),
    "APIError": ("hydroe2e.exceptions", "APIError"),
    "NetworkError": ("hydroe2e.exceptions", "NetworkError"),
}

__all__ = ["__version__", "__project__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

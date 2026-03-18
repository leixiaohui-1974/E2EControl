"""
HydroE2E - 智能水网端到端控制系统
==================================
HydroMind 生态卫星项目，支持独立运行。

核心能力：
- MPC/DMPC 最优控制
- 语义指令解析
- 数字孪生仿真
- 异常检测与故障诊断
- 自愈控制
"""

__version__ = "1.1.0"
__project__ = "HydroE2E"

from hydroe2e.config_manager import ConfigManager, get_config
from hydroe2e.brain import SemanticInterpreter
from hydroe2e.brain_enhanced import EnhancedSemanticInterpreter
from hydroe2e.simulation_manager import SimulationManager
from hydroe2e.database import SimulationDatabase
from hydroe2e.monitor import MonitoringSystem, Alert, AlertLevel
from hydroe2e.logger import get_logger, setup_logging
from hydroe2e.exceptions import (
    SmartPoolException, ConfigurationError, OptimizationError,
    PhysicsError, SemanticError, ValidationError, DatabaseError,
    MonitoringError, APIError, NetworkError
)

__all__ = [
    "__version__", "__project__",
    "ConfigManager", "get_config",
    "SemanticInterpreter", "EnhancedSemanticInterpreter",
    "SimulationManager", "SimulationDatabase",
    "MonitoringSystem", "Alert", "AlertLevel",
    "get_logger", "setup_logging",
]

"""
接口模块 - 定义分布式SIL系统的数据接口
"""

from .data_types import (
    SegmentState,
    BoundaryCondition,
    ControlCommand,
    AssimilatedState,
    ModelError,
    KPIMetrics,
)
from .boundary_assimilator import BoundaryAssimilator

__all__ = [
    "SegmentState",
    "BoundaryCondition",
    "ControlCommand",
    "AssimilatedState",
    "ModelError",
    "KPIMetrics",
    "BoundaryAssimilator",
]

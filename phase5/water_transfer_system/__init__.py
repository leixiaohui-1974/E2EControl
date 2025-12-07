"""
南水北调中线全线全场景自主运行系统
Water Transfer Autonomous System for South-to-North Water Diversion Middle Route

基于动态角色协同的分层分布式MPC架构
Hierarchical Distributed MPC Architecture with Dynamic Role Coordination

核心特性:
1. 三层控制架构 (L3全局调度 + L2区域协调 + L1现地控制)
2. 全场景动态角色矩阵 (8大场景 × 6种角色)
3. 高保真物理建模 (IDZ模型 + 系统辨识)
4. 在线热重构MPC (支持场景切换)

模块结构:
- core_types: 核心数据结构和类型定义
- physics_model: 中线物理模型 (60+渠池)
- system_identification: 系统辨识 (IDZ参数)
- orchestrator: 全局编排器 (L3)
- regional_coordinator: 区域协调器 (L2)
- enhanced_mpc: 增强参数化MPC (支持热重构)
"""

from .core_types import (
    PoolRole,
    ScenarioType,
    ControlDirective,
    PoolTopology,
    CanalPoolConfig,
    SpecialStructure,
    StructureType,
    RegionConfig,
)

from .physics_model import (
    SNWDMiddleRouteModel,
    IDZModel,
    CanalPool,
    SpecialNode,
)

from .system_identification import (
    SystemIdentifier,
    IDZParameters,
)

from .orchestrator import (
    GlobalOrchestrator,
    ScenarioRoleMatrix,
)

from .regional_coordinator import (
    RegionalCoordinator,
    FeedforwardDecoupler,
)

from .enhanced_mpc import (
    EnhancedParameterizedMPC,
    HotReconfigurableMPC,
)

__all__ = [
    # Core Types
    'PoolRole',
    'ScenarioType',
    'ControlDirective',
    'PoolTopology',
    'CanalPoolConfig',
    'SpecialStructure',
    'StructureType',
    'RegionConfig',
    # Physics Model
    'SNWDMiddleRouteModel',
    'IDZModel',
    'CanalPool',
    'SpecialNode',
    # System Identification
    'SystemIdentifier',
    'IDZParameters',
    # Orchestrator
    'GlobalOrchestrator',
    'ScenarioRoleMatrix',
    # Regional Coordinator
    'RegionalCoordinator',
    'FeedforwardDecoupler',
    # Enhanced MPC
    'EnhancedParameterizedMPC',
    'HotReconfigurableMPC',
]

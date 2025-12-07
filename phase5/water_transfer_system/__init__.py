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
5. 场景自适应MPC (自动更新目标函数和约束)
6. 批量测试框架 (支持成千上万场景)
7. L1现地全场景 (污染溯源、边坡漂浮、退水等)
8. L1控制器与L2-L1层级协调
9. 多层协同控制器 (L1-L2-L3联动)
10. 智能决策引擎与场景组合
11. 级联控制与事件驱动上报系统

模块结构:
- core_types: 核心数据结构和类型定义
- physics_model: 中线物理模型 (60+渠池)
- system_identification: 系统辨识 (IDZ参数)
- orchestrator: 全局编排器 (L3)
- regional_coordinator: 区域协调器 (L2)
- enhanced_mpc: 增强参数化MPC (支持热重构)
- scenario_generator: 场景生成器 (支持成千上万种组合)
- adaptive_mpc: 场景自适应MPC (自动配置)
- batch_testing: 批量测试框架
- local_pool_scenarios: L1现地渠池全场景 (污染、边坡、退水)
- l1_controller: L1层现地控制器 (分钟级自主响应)
- l2_l1_coordinator: L2-L1层级协调器 (区域-现地协调)
- multi_layer_coordinator: 多层协同控制器 (L1-L2-L3联动)
- cascade_control: 级联控制与事件驱动上报 (失控检测与干预)
"""

from .core_types import (
    PoolRole,
    ScenarioType,
    ScenarioSeverity,
    ScenarioPhase,
    ControlDirective,
    ControlPlan,
    ScenarioEvent,
    PoolTopology,
    CanalPoolConfig,
    SpecialStructure,
    StructureType,
    RegionConfig,
)

from .physics_model import (
    SNWDMiddleRouteModel,
    IDZModel,
    IDZParameters,
    CanalPool,
    SpecialNode,
)

from .system_identification import (
    SystemIdentifier,
    CrossCorrelationAnalyzer,
    RecursiveLeastSquares,
)

from .orchestrator import (
    GlobalOrchestrator,
    ScenarioRoleMatrix,
)

from .regional_coordinator import (
    RegionalCoordinator,
    FeedforwardDecoupler,
    GlobalRegionalManager,
)

from .enhanced_mpc import (
    EnhancedParameterizedMPC,
    HotReconfigurableMPC,
    MPCWeights,
    MPCConstraints,
    MPCPhysics,
    RoleParameterMapper,
)

from .scenario_generator import (
    ScenarioGenerator,
    ScenarioValidator,
    ExtendedScenarioEvent,
    CompositeScenario,
    SeasonType,
    WeatherType,
    TimeOfDay,
    EvolutionPattern,
    RegionZone,
)

from .adaptive_mpc import (
    AdaptiveMPCSystem,
    AdaptiveMPCConfigurator,
    ScenarioIdentifier,
    ScenarioFeatures,
    ScenarioDetectionResult,
    AdaptiveMPCConfig,
)

from .batch_testing import (
    BatchTestExecutor,
    TestSuite,
    TestCase,
    TestResult,
    BatchTestResult,
    TestStatus,
    ReportGenerator,
    run_quick_test,
    run_comprehensive_test,
    run_exhaustive_test,
)

from .local_pool_scenarios import (
    L1ScenarioType,
    L1ActionType,
    L1ScenarioEvent,
    L1ActionCommand,
    PollutionTracker,
    SlopePanelMonitor,
    DischargeManager,
    DischargeType,
    L1ScenarioGenerator,
)

from .l1_controller import (
    L1Controller,
    L1ControllerManager,
    L1ControllerState,
    L1PoolState,
    L1ControlResult,
    L1ResponseStrategy,
)

from .l2_l1_coordinator import (
    L2L1Coordinator,
    FullLineCoordinatorManager,
    CoordinationType,
    CoordinationRequest,
    CoordinationResponse,
)

from .multi_layer_coordinator import (
    MultiLayerCoordinator,
    MultiLayerEvent,
    MultiLayerEventType,
    MultiLayerDecision,
    DecisionPriority,
    L3GlobalScheduler,
    ScenarioCombinationGenerator,
    IntelligentDecisionEngine,
)

from .cascade_control import (
    CascadeControlSystem,
    ControlEffectiveness,
    ControlEffectEvaluator,
    ControlMetrics,
    EscalationEvent,
    EscalationReason,
    InterventionDecision,
    InterventionType,
    UpperLayerInterventionDecider,
    ExtendedL1Scenarios,
)

__all__ = [
    # Core Types
    'PoolRole',
    'ScenarioType',
    'ScenarioSeverity',
    'ScenarioPhase',
    'ControlDirective',
    'ControlPlan',
    'ScenarioEvent',
    'PoolTopology',
    'CanalPoolConfig',
    'SpecialStructure',
    'StructureType',
    'RegionConfig',
    # Physics Model
    'SNWDMiddleRouteModel',
    'IDZModel',
    'IDZParameters',
    'CanalPool',
    'SpecialNode',
    # System Identification
    'SystemIdentifier',
    'CrossCorrelationAnalyzer',
    'RecursiveLeastSquares',
    # Orchestrator
    'GlobalOrchestrator',
    'ScenarioRoleMatrix',
    # Regional Coordinator
    'RegionalCoordinator',
    'FeedforwardDecoupler',
    'GlobalRegionalManager',
    # Enhanced MPC
    'EnhancedParameterizedMPC',
    'HotReconfigurableMPC',
    'MPCWeights',
    'MPCConstraints',
    'MPCPhysics',
    'RoleParameterMapper',
    # Scenario Generator
    'ScenarioGenerator',
    'ScenarioValidator',
    'ExtendedScenarioEvent',
    'CompositeScenario',
    'SeasonType',
    'WeatherType',
    'TimeOfDay',
    'EvolutionPattern',
    'RegionZone',
    # Adaptive MPC
    'AdaptiveMPCSystem',
    'AdaptiveMPCConfigurator',
    'ScenarioIdentifier',
    'ScenarioFeatures',
    'ScenarioDetectionResult',
    'AdaptiveMPCConfig',
    # Batch Testing
    'BatchTestExecutor',
    'TestSuite',
    'TestCase',
    'TestResult',
    'BatchTestResult',
    'TestStatus',
    'ReportGenerator',
    'run_quick_test',
    'run_comprehensive_test',
    'run_exhaustive_test',
    # L1 Local Pool Scenarios
    'L1ScenarioType',
    'L1ActionType',
    'L1ScenarioEvent',
    'L1ActionCommand',
    'PollutionTracker',
    'SlopePanelMonitor',
    'DischargeManager',
    'DischargeType',
    'L1ScenarioGenerator',
    # L1 Controller
    'L1Controller',
    'L1ControllerManager',
    'L1ControllerState',
    'L1PoolState',
    'L1ControlResult',
    'L1ResponseStrategy',
    # L2-L1 Coordinator
    'L2L1Coordinator',
    'FullLineCoordinatorManager',
    'CoordinationType',
    'CoordinationRequest',
    'CoordinationResponse',
    # Multi-Layer Coordinator
    'MultiLayerCoordinator',
    'MultiLayerEvent',
    'MultiLayerEventType',
    'MultiLayerDecision',
    'DecisionPriority',
    'L3GlobalScheduler',
    'ScenarioCombinationGenerator',
    'IntelligentDecisionEngine',
    # Cascade Control
    'CascadeControlSystem',
    'ControlEffectiveness',
    'ControlEffectEvaluator',
    'ControlMetrics',
    'EscalationEvent',
    'EscalationReason',
    'InterventionDecision',
    'InterventionType',
    'UpperLayerInterventionDecider',
    'ExtendedL1Scenarios',
]

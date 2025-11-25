"""
Phase 5: 系统集成与优化
System Integration and Optimization

Complete autonomous water network control system with:
- Phase 5.1: HIL Testing Framework
- Phase 5.2: Scenario Library (64 test conditions)
- Phase 5.3: Hierarchical MPC Architecture (L3+L2)
- Phase 5.4: L4 Self-Healing Capabilities
- Phase 5.5: Performance Monitoring & System Integration
- Phase 5.6: Web Dashboard for Real-time Monitoring
- Phase 5.7: Full Scenario Certification Testing (103 scenarios)
- Phase 5.8: Docker Containerization
- Phase 5.9: Real-time Data Interface (OPC-UA, Modbus, SCADA)
- Phase 5.10: L5 Autonomous Learning and Decision Making
"""

__version__ = "1.0.0"

# Phase 5.1 & 5.2: HIL Testing Framework and Scenario Library
from .hil_testing import (
    ScenarioGenerator,
    Scenario,
    Condition,
    ScenarioCategory,
    DifficultyLevel,
    AutonomousLevel,
    ConditionInjector,
    InjectionType,
    EvaluationEngine,
    TestResult,
    HILTestRunner,
    ReportGenerator
)

# Phase 5.3: Hierarchical MPC Architecture
from .controllers import (
    HierarchicalMPCController,
    HierarchicalConfig,
    HierarchicalState,
    CentralizedScheduler,
    ParameterizedLocalMPC,
    ParameterizedDistributedMPC
)

# Phase 5.4: L4 Self-Healing Capabilities
from .self_healing import (
    EnhancedDiagnosisEngine,
    ExtendedFaultType,
    FaultSeverity,
    FaultPattern,
    OptimizedIsolationStrategy,
    IsolationAction,
    PredictiveRecoveryEngine,
    RecoveryPlan,
    FaultLearningEngine,
    MacroCognitiveLayer,
    SystemHealthLevel,
    RiskLevel,
    StrategicAdvice
)

# Phase 5.5: Performance Monitoring and System Integration
from .monitoring import (
    PerformanceMonitor,
    MetricType,
    MetricLevel,
    PerformanceAlert,
    UnifiedControlSystem,
    SystemMode,
    SystemStatus,
    ControlCommand,
    SystemAPI,
    CommandType
)

# Phase 5.6: Web Dashboard
from .web import (
    WebDashboard,
    DashboardConfig
)

# Phase 5.7: Certification Testing
from .hil_testing import (
    CertificationRunner,
    CertificationReport,
    CertificationResult,
    CertLevel,
    LevelRequirement,
    ScenarioResult,
    CategorySummary,
    run_certification
)

# Phase 5.9: Real-time Data Interface
from .data_interface import (
    # OPC-UA Protocol
    OPCUAAdapter,
    OPCUANode,
    OPCUASubscription,
    OPCUADataPoint,
    OPCUAConnectionConfig,
    # Modbus Protocol
    ModbusAdapter,
    ModbusRegister,
    ModbusDeviceConfig,
    ModbusDataType,
    ModbusReadResult,
    # SCADA Integration
    SCADAInterface,
    SCADATag,
    SCADAAlarm,
    SCADACommand,
    SCADAConnectionStatus,
    # Data Replay
    DataReplayEngine,
    ReplaySession,
    ReplayConfig,
    TimeScaleMode,
    DataSource,
)

# Phase 5.10: L5 Autonomous Learning
from .autonomous_learning import (
    # Online Learning
    OnlineLearningEngine,
    LearningAlgorithm,
    ModelUpdate,
    LearningRate,
    AdaptiveOptimizer,
    # Experience Memory
    ExperienceMemory,
    Experience,
    ExperienceType,
    MemoryPriority,
    ReplayBuffer,
    # Knowledge Transfer
    KnowledgeTransferEngine,
    KnowledgeBase,
    Pattern,
    PatternType,
    TransferStrategy,
    # Autonomous Decision
    AutonomousDecisionEngine,
    DecisionContext,
    Decision,
    ConfidenceLevel,
    DecisionOutcome,
    L5Controller,
)

__all__ = [
    # Version
    '__version__',

    # HIL Testing (5.1, 5.2)
    'ScenarioGenerator',
    'Scenario',
    'Condition',
    'ScenarioCategory',
    'DifficultyLevel',
    'AutonomousLevel',
    'ConditionInjector',
    'InjectionType',
    'EvaluationEngine',
    'TestResult',
    'HILTestRunner',
    'ReportGenerator',

    # Hierarchical MPC (5.3)
    'HierarchicalMPCController',
    'HierarchicalConfig',
    'HierarchicalState',
    'CentralizedScheduler',
    'ParameterizedLocalMPC',
    'ParameterizedDistributedMPC',

    # Self-Healing (5.4)
    'EnhancedDiagnosisEngine',
    'ExtendedFaultType',
    'FaultSeverity',
    'FaultPattern',
    'OptimizedIsolationStrategy',
    'IsolationAction',
    'PredictiveRecoveryEngine',
    'RecoveryPlan',
    'FaultLearningEngine',
    'MacroCognitiveLayer',
    'SystemHealthLevel',
    'RiskLevel',
    'StrategicAdvice',

    # Monitoring & Integration (5.5)
    'PerformanceMonitor',
    'MetricType',
    'MetricLevel',
    'PerformanceAlert',
    'UnifiedControlSystem',
    'SystemMode',
    'SystemStatus',
    'ControlCommand',
    'SystemAPI',
    'CommandType',

    # Web Dashboard (5.6)
    'WebDashboard',
    'DashboardConfig',

    # Certification Testing (5.7)
    'CertificationRunner',
    'CertificationReport',
    'CertificationResult',
    'CertLevel',
    'LevelRequirement',
    'ScenarioResult',
    'CategorySummary',
    'run_certification',

    # Real-time Data Interface (5.9)
    'OPCUAAdapter',
    'OPCUANode',
    'OPCUASubscription',
    'OPCUADataPoint',
    'OPCUAConnectionConfig',
    'ModbusAdapter',
    'ModbusRegister',
    'ModbusDeviceConfig',
    'ModbusDataType',
    'ModbusReadResult',
    'SCADAInterface',
    'SCADATag',
    'SCADAAlarm',
    'SCADACommand',
    'SCADAConnectionStatus',
    'DataReplayEngine',
    'ReplaySession',
    'ReplayConfig',
    'TimeScaleMode',
    'DataSource',

    # L5 Autonomous Learning (5.10)
    'OnlineLearningEngine',
    'LearningAlgorithm',
    'ModelUpdate',
    'LearningRate',
    'AdaptiveOptimizer',
    'ExperienceMemory',
    'Experience',
    'ExperienceType',
    'MemoryPriority',
    'ReplayBuffer',
    'KnowledgeTransferEngine',
    'KnowledgeBase',
    'Pattern',
    'PatternType',
    'TransferStrategy',
    'AutonomousDecisionEngine',
    'DecisionContext',
    'Decision',
    'ConfidenceLevel',
    'DecisionOutcome',
    'L5Controller',
]

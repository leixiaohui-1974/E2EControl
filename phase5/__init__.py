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
    'DashboardConfig'
]

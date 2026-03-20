"""
Phase 5: 系统集成与优化
System Integration and Optimization

Keep Phase 5 imports lazy so physics-only tools and tests can run without
installing optional control dependencies such as cvxpy.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "1.0.0"

_LAZY_EXPORTS = {
    # Phase 5.1 & 5.2: HIL Testing Framework and Scenario Library
    "ScenarioGenerator": ("hydroe2e.phase5.hil_testing", "ScenarioGenerator"),
    "Scenario": ("hydroe2e.phase5.hil_testing", "Scenario"),
    "Condition": ("hydroe2e.phase5.hil_testing", "Condition"),
    "ScenarioCategory": ("hydroe2e.phase5.hil_testing", "ScenarioCategory"),
    "DifficultyLevel": ("hydroe2e.phase5.hil_testing", "DifficultyLevel"),
    "AutonomousLevel": ("hydroe2e.phase5.hil_testing", "AutonomousLevel"),
    "ConditionInjector": ("hydroe2e.phase5.hil_testing", "ConditionInjector"),
    "InjectionType": ("hydroe2e.phase5.hil_testing", "InjectionType"),
    "EvaluationEngine": ("hydroe2e.phase5.hil_testing", "EvaluationEngine"),
    "TestResult": ("hydroe2e.phase5.hil_testing", "TestResult"),
    "HILTestRunner": ("hydroe2e.phase5.hil_testing", "HILTestRunner"),
    "ReportGenerator": ("hydroe2e.phase5.hil_testing", "ReportGenerator"),

    # Phase 5.3: Hierarchical MPC Architecture
    "HierarchicalMPCController": ("hydroe2e.phase5.controllers", "HierarchicalMPCController"),
    "HierarchicalConfig": ("hydroe2e.phase5.controllers", "HierarchicalConfig"),
    "HierarchicalState": ("hydroe2e.phase5.controllers", "HierarchicalState"),
    "CentralizedScheduler": ("hydroe2e.phase5.controllers", "CentralizedScheduler"),
    "ParameterizedLocalMPC": ("hydroe2e.phase5.controllers", "ParameterizedLocalMPC"),
    "ParameterizedDistributedMPC": ("hydroe2e.phase5.controllers", "ParameterizedDistributedMPC"),

    # Phase 5.4: L4 Self-Healing Capabilities
    "EnhancedDiagnosisEngine": ("hydroe2e.phase5.self_healing", "EnhancedDiagnosisEngine"),
    "ExtendedFaultType": ("hydroe2e.phase5.self_healing", "ExtendedFaultType"),
    "FaultSeverity": ("hydroe2e.phase5.self_healing", "FaultSeverity"),
    "FaultPattern": ("hydroe2e.phase5.self_healing", "FaultPattern"),
    "OptimizedIsolationStrategy": ("hydroe2e.phase5.self_healing", "OptimizedIsolationStrategy"),
    "IsolationAction": ("hydroe2e.phase5.self_healing", "IsolationAction"),
    "PredictiveRecoveryEngine": ("hydroe2e.phase5.self_healing", "PredictiveRecoveryEngine"),
    "RecoveryPlan": ("hydroe2e.phase5.self_healing", "RecoveryPlan"),
    "FaultLearningEngine": ("hydroe2e.phase5.self_healing", "FaultLearningEngine"),
    "MacroCognitiveLayer": ("hydroe2e.phase5.self_healing", "MacroCognitiveLayer"),
    "SystemHealthLevel": ("hydroe2e.phase5.self_healing", "SystemHealthLevel"),
    "RiskLevel": ("hydroe2e.phase5.self_healing", "RiskLevel"),
    "StrategicAdvice": ("hydroe2e.phase5.self_healing", "StrategicAdvice"),

    # Phase 5.5: Performance Monitoring and System Integration
    "PerformanceMonitor": ("hydroe2e.phase5.monitoring", "PerformanceMonitor"),
    "MetricType": ("hydroe2e.phase5.monitoring", "MetricType"),
    "MetricLevel": ("hydroe2e.phase5.monitoring", "MetricLevel"),
    "PerformanceAlert": ("hydroe2e.phase5.monitoring", "PerformanceAlert"),
    "UnifiedControlSystem": ("hydroe2e.phase5.monitoring", "UnifiedControlSystem"),
    "SystemMode": ("hydroe2e.phase5.monitoring", "SystemMode"),
    "SystemStatus": ("hydroe2e.phase5.monitoring", "SystemStatus"),
    "ControlCommand": ("hydroe2e.phase5.monitoring", "ControlCommand"),
    "SystemAPI": ("hydroe2e.phase5.monitoring", "SystemAPI"),
    "CommandType": ("hydroe2e.phase5.monitoring", "CommandType"),

    # Phase 5.6: Web Dashboard
    "WebDashboard": ("hydroe2e.phase5.web", "WebDashboard"),
    "DashboardConfig": ("hydroe2e.phase5.web", "DashboardConfig"),

    # Phase 5.7: Certification Testing
    "CertificationRunner": ("hydroe2e.phase5.hil_testing", "CertificationRunner"),
    "CertificationReport": ("hydroe2e.phase5.hil_testing", "CertificationReport"),
    "CertificationResult": ("hydroe2e.phase5.hil_testing", "CertificationResult"),
    "CertLevel": ("hydroe2e.phase5.hil_testing", "CertLevel"),
    "LevelRequirement": ("hydroe2e.phase5.hil_testing", "LevelRequirement"),
    "ScenarioResult": ("hydroe2e.phase5.hil_testing", "ScenarioResult"),
    "CategorySummary": ("hydroe2e.phase5.hil_testing", "CategorySummary"),
    "run_certification": ("hydroe2e.phase5.hil_testing", "run_certification"),

    # Phase 5.9: Real-time Data Interface
    "OPCUAAdapter": ("hydroe2e.phase5.data_interface", "OPCUAAdapter"),
    "OPCUANode": ("hydroe2e.phase5.data_interface", "OPCUANode"),
    "OPCUASubscription": ("hydroe2e.phase5.data_interface", "OPCUASubscription"),
    "OPCUADataPoint": ("hydroe2e.phase5.data_interface", "OPCUADataPoint"),
    "OPCUAConnectionConfig": ("hydroe2e.phase5.data_interface", "OPCUAConnectionConfig"),
    "ModbusAdapter": ("hydroe2e.phase5.data_interface", "ModbusAdapter"),
    "ModbusRegister": ("hydroe2e.phase5.data_interface", "ModbusRegister"),
    "ModbusDeviceConfig": ("hydroe2e.phase5.data_interface", "ModbusDeviceConfig"),
    "ModbusDataType": ("hydroe2e.phase5.data_interface", "ModbusDataType"),
    "ModbusReadResult": ("hydroe2e.phase5.data_interface", "ModbusReadResult"),
    "SCADAInterface": ("hydroe2e.phase5.data_interface", "SCADAInterface"),
    "SCADATag": ("hydroe2e.phase5.data_interface", "SCADATag"),
    "SCADAAlarm": ("hydroe2e.phase5.data_interface", "SCADAAlarm"),
    "SCADACommand": ("hydroe2e.phase5.data_interface", "SCADACommand"),
    "SCADAConnectionStatus": ("hydroe2e.phase5.data_interface", "SCADAConnectionStatus"),
    "DataReplayEngine": ("hydroe2e.phase5.data_interface", "DataReplayEngine"),
    "ReplaySession": ("hydroe2e.phase5.data_interface", "ReplaySession"),
    "ReplayConfig": ("hydroe2e.phase5.data_interface", "ReplayConfig"),
    "TimeScaleMode": ("hydroe2e.phase5.data_interface", "TimeScaleMode"),
    "DataSource": ("hydroe2e.phase5.data_interface", "DataSource"),

    # Phase 5.10: L5 Autonomous Learning
    "OnlineLearningEngine": ("hydroe2e.phase5.autonomous_learning", "OnlineLearningEngine"),
    "LearningAlgorithm": ("hydroe2e.phase5.autonomous_learning", "LearningAlgorithm"),
    "ModelUpdate": ("hydroe2e.phase5.autonomous_learning", "ModelUpdate"),
    "LearningRate": ("hydroe2e.phase5.autonomous_learning", "LearningRate"),
    "AdaptiveOptimizer": ("hydroe2e.phase5.autonomous_learning", "AdaptiveOptimizer"),
    "ExperienceMemory": ("hydroe2e.phase5.autonomous_learning", "ExperienceMemory"),
    "Experience": ("hydroe2e.phase5.autonomous_learning", "Experience"),
    "ExperienceType": ("hydroe2e.phase5.autonomous_learning", "ExperienceType"),
    "MemoryPriority": ("hydroe2e.phase5.autonomous_learning", "MemoryPriority"),
    "ReplayBuffer": ("hydroe2e.phase5.autonomous_learning", "ReplayBuffer"),
    "KnowledgeTransferEngine": ("hydroe2e.phase5.autonomous_learning", "KnowledgeTransferEngine"),
    "KnowledgeBase": ("hydroe2e.phase5.autonomous_learning", "KnowledgeBase"),
    "Pattern": ("hydroe2e.phase5.autonomous_learning", "Pattern"),
    "PatternType": ("hydroe2e.phase5.autonomous_learning", "PatternType"),
    "TransferStrategy": ("hydroe2e.phase5.autonomous_learning", "TransferStrategy"),
    "AutonomousDecisionEngine": ("hydroe2e.phase5.autonomous_learning", "AutonomousDecisionEngine"),
    "DecisionContext": ("hydroe2e.phase5.autonomous_learning", "DecisionContext"),
    "Decision": ("hydroe2e.phase5.autonomous_learning", "Decision"),
    "ConfidenceLevel": ("hydroe2e.phase5.autonomous_learning", "ConfidenceLevel"),
    "DecisionOutcome": ("hydroe2e.phase5.autonomous_learning", "DecisionOutcome"),
    "L5Controller": ("hydroe2e.phase5.autonomous_learning", "L5Controller"),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

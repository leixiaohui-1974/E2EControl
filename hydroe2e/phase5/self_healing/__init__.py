"""
Phase 5.4 Self-Healing Enhancements - 自愈能力增强

实现L4级自愈能力:
1. 增强故障诊断 - 多故障类型、模式学习
2. 优化隔离策略 - 最小影响分析
3. 预测性恢复 - 快速恢复路径
4. 历史学习 - 从故障中学习
5. L3宏观认知 - 系统级健康评估和战略决策
"""

from .enhanced_diagnosis import (
    EnhancedDiagnosisEngine,
    ExtendedFaultType,
    FaultSeverity,
    FaultPattern,
    CompoundFault,
    CascadeFault,
    DiagnosisConfidence,
    EnhancedDiagnosisResult
)

from .optimized_isolation import (
    OptimizedIsolationStrategy,
    IsolationAction,
    IsolationScope,
    ImpactAnalysis,
    GracefulDegradationPath,
    ServiceContinuityPlan,
    IsolationPlan
)

from .predictive_recovery import (
    PredictiveRecoveryEngine,
    RecoveryPhase,
    RecoveryStrategy,
    RecoveryPriority,
    PredictionConfidence,
    FailureTrajectory,
    RecoveryResource,
    RecoveryAction,
    RecoveryTimeline,
    RecoveryPlan,
    RecoveryProgress,
    PreemptiveAction
)

from .fault_learning import (
    FaultLearningEngine,
    LearningMode,
    PatternStatus,
    ConfidenceLevel,
    FaultSignature,
    CausalRelation,
    RecoveryOutcome,
    LearningMetrics,
    DiagnosisRule,
    FeatureImportance
)

from .macro_cognitive_layer import (
    MacroCognitiveLayer,
    SystemHealthLevel,
    SeasonalPattern,
    RiskLevel,
    TrendDirection,
    EquipmentHealth,
    DegradationTrend,
    NetworkRiskAssessment,
    StrategicAdvice,
    MacroCognitiveState
)

__all__ = [
    # 增强诊断
    'EnhancedDiagnosisEngine',
    'ExtendedFaultType',
    'FaultSeverity',
    'FaultPattern',
    'CompoundFault',
    'CascadeFault',
    'DiagnosisConfidence',
    'EnhancedDiagnosisResult',

    # 优化隔离
    'OptimizedIsolationStrategy',
    'IsolationAction',
    'IsolationScope',
    'ImpactAnalysis',
    'GracefulDegradationPath',
    'ServiceContinuityPlan',
    'IsolationPlan',

    # 预测恢复
    'PredictiveRecoveryEngine',
    'RecoveryPhase',
    'RecoveryStrategy',
    'RecoveryPriority',
    'PredictionConfidence',
    'FailureTrajectory',
    'RecoveryResource',
    'RecoveryAction',
    'RecoveryTimeline',
    'RecoveryPlan',
    'RecoveryProgress',
    'PreemptiveAction',

    # 故障学习
    'FaultLearningEngine',
    'LearningMode',
    'PatternStatus',
    'ConfidenceLevel',
    'FaultSignature',
    'CausalRelation',
    'RecoveryOutcome',
    'LearningMetrics',
    'DiagnosisRule',
    'FeatureImportance',

    # L3宏观认知层
    'MacroCognitiveLayer',
    'SystemHealthLevel',
    'SeasonalPattern',
    'RiskLevel',
    'TrendDirection',
    'EquipmentHealth',
    'DegradationTrend',
    'NetworkRiskAssessment',
    'StrategicAdvice',
    'MacroCognitiveState',
]

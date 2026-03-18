# Phase 5.10: L5 Autonomous Learning Module
# L5自主学习模块 - 实现完全自主水网控制

from .online_learning import (
    OnlineLearningEngine,
    LearningAlgorithm,
    ModelUpdate,
    LearningRate,
    AdaptiveOptimizer,
)

from .experience_memory import (
    ExperienceMemory,
    Experience,
    ExperienceType,
    MemoryPriority,
    ReplayBuffer,
)

from .knowledge_transfer import (
    KnowledgeTransferEngine,
    KnowledgeBase,
    Pattern,
    PatternType,
    TransferStrategy,
)

from .autonomous_decision import (
    AutonomousDecisionEngine,
    DecisionContext,
    Decision,
    ConfidenceLevel,
    DecisionOutcome,
    L5Controller,
)

__all__ = [
    # Online Learning
    'OnlineLearningEngine',
    'LearningAlgorithm',
    'ModelUpdate',
    'LearningRate',
    'AdaptiveOptimizer',
    # Experience Memory
    'ExperienceMemory',
    'Experience',
    'ExperienceType',
    'MemoryPriority',
    'ReplayBuffer',
    # Knowledge Transfer
    'KnowledgeTransferEngine',
    'KnowledgeBase',
    'Pattern',
    'PatternType',
    'TransferStrategy',
    # Autonomous Decision
    'AutonomousDecisionEngine',
    'DecisionContext',
    'Decision',
    'ConfidenceLevel',
    'DecisionOutcome',
    'L5Controller',
]

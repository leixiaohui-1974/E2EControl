"""
L4级自主运行系统 (L4 Autonomous Operation System)
南水北调中线全线自主控制

自动驾驶等级类比:
- L0: 人工控制 - 系统仅提供显示
- L1: 辅助决策 - 系统生成建议，人工确认
- L2: 部分自动 - PID/MPC自动控制
- L3: 条件自动 - 神经网络控制，场景识别
- L4: 高度自动 - 端到端自主，仅异常时接管

本模块实现完整的L0-L4等级能力:
1. e2e_controller.py - 端到端自主控制器 (L4核心)
2. full_line_world_model.py - 全线统一世界模型
3. multi_level_controller.py - 多等级统一控制器 (L0-L4)
4. multi_agent_coordinator.py - 多智能体协同决策
5. continual_learner.py - 持续学习与在线优化
6. safety_boundary.py - 安全边界与人机接管
"""

# E2E自主控制器
from .e2e_controller import (
    E2EAutonomousController,
    AutonomousConfig,
    AutonomyLevel,
    PerceptionModule,
    PredictionModule,
    DecisionModule,
)

# 全线世界模型
from .full_line_world_model import (
    FullLineWorldModel,
    WorldModelConfig,
    SpatioTemporalEncoder,
    CascadePredictor,
    CascadeGraphEncoder,
)

# 多等级控制器
from .multi_level_controller import (
    MultiLevelController,
    L0ManualController,
    L1AdvisoryController,
    L2PartialAutoController,
    L3ConditionalAutoController,
    L4HighAutoController,
    LEVEL_CAPABILITIES,
    ControlMode,
)

# 多智能体协同
from .multi_agent_coordinator import (
    MultiAgentCoordinator,
    GateAgent,
    ConsensusProtocol,
    AgentConfig,
    Message,
    CommunicationType,
)

# 持续学习
from .continual_learner import (
    ContinualLearner,
    OnlineLearningBuffer,
    ExperienceReplay,
    EWCRegularizer,
    LearningConfig,
    Experience,
)

# 安全边界
from .safety_boundary import (
    SafetyBoundary,
    SafetyMonitor,
    HumanOverrideInterface,
    SafetyConstraints,
    SafetyLevel,
    SafetyEvent,
    OverrideReason,
)

__version__ = "1.0.0"
__author__ = "E2EControl Team"

"""
端到端闭环控制框架 (E2E Closed-Loop Control Framework)

整合 MBD、ODD、SIM、场景识别+自适应MAS、SIL、HIL 形成完整闭环

架构层次:
┌─────────────────────────────────────────────────────────────────┐
│                    ODD (Operational Design Domain)              │
│                    定义系统运行边界和约束                         │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    MBD (Model-Based Design)                     │
│               模型驱动的设计、验证、部署流程                      │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    场景识别 + 自适应MAS                          │
│              实时场景识别 → 多智能体自适应协调                    │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    SIL ←→ HIL 闭环验证                          │
│              软件在环 ←→ 硬件在环 双向验证                        │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    执行反馈 + 持续学习                           │
│              执行效果评估 → 模型更新 → 策略优化                   │
└────────────────────────────────────────────────────────────────┘

对于已建工程的核心价值:
1. MBD用于控制策略的V模型开发 (设计→仿真→测试→部署)
2. ODD定义自动控制的边界条件 (何时可自动、何时需人工)
3. 场景识别驱动不同控制策略的切换
4. 自适应MAS处理通信延迟和故障隔离
5. SIL/HIL在部署前验证新策略
6. 执行反馈持续校准数字孪生模型
"""

from .core.e2e_controller import E2EClosedLoopController, E2EConfig, AutonomyLevel
from .odd.operational_domain import OperationalDesignDomain
from .mbd.model_based_design import ModelBasedDesignManager, DevelopmentPhase
from .scenario.scenario_recognizer import ScenarioRecognizer, ScenarioType
from .mas.adaptive_coordinator import AdaptiveMASCoordinator, AgentRole
from .verification.sil_hil_bridge import SILHILBridge, VerificationMode
from .feedback.execution_feedback import ExecutionFeedbackManager, CalibrationMode

__version__ = "1.0.0"
__all__ = [
    # 核心控制器
    "E2EClosedLoopController",
    "E2EConfig",
    "AutonomyLevel",
    # ODD
    "OperationalDesignDomain",
    # MBD
    "ModelBasedDesignManager",
    "DevelopmentPhase",
    # 场景识别
    "ScenarioRecognizer",
    "ScenarioType",
    # MAS协调
    "AdaptiveMASCoordinator",
    "AgentRole",
    # SIL/HIL
    "SILHILBridge",
    "VerificationMode",
    # 执行反馈
    "ExecutionFeedbackManager",
    "CalibrationMode",
]

"""
AI Models for Generative Water Network World Model
生成式水网世界模型 - AI模型模块

本模块包含将 e2econtrol 从 L2 级自动化升级到 L3 级智能化所需的AI组件:

1. neural_physics_engine.py - 神经代理模型 (Neural Surrogate Model)
   - LSTM/GRU 网络捕捉水流时滞和记忆效应
   - PINN (Physics-Informed Neural Networks) 物理约束
   - 与原 PhysicsModel 接口兼容

2. scenario_vae.py - 生成式场景引擎 (ScenarioVAE)
   - CVAE (Conditional VAE) 条件生成
   - 时间序列生成器
   - 上游来水、下游需水、糙率变化、传感器噪声

3. water_canal_env.py - RL训练环境 (WaterCanalEnv)
   - Gymnasium API 兼容
   - "一闸两渠池" 拓扑
   - 生成式边界条件

4. deep_scenario_encoder.py - 场景识别与泛化层 (DeepScenarioEncoder)
   - 对比学习 (Contrastive Learning)
   - 向量数据库检索
   - 未知场景检测

5. hydrographnet/ - HydroGraphNet 集成模块
   - KAN层 (Kolmogorov-Arnold Network)
   - 物理守恒损失函数
   - 微观世界模型适配器

架构理念:
- 不推翻现有代码，采用"组件替换与增强"策略
- 保持与 Phase 5 工程架构的兼容性
- 支持从传统模型到神经模型的平滑过渡
"""

import warnings as _warnings

# These modules require PyTorch. Make them available when torch is installed
# but degrade gracefully (with a warning) when it is not.
try:
    from .neural_physics_engine import (
        NeuralPhysicsEngine,
        LSTMSurrogateModel,
        PINNLoss,
    )

    from .scenario_vae import (
        ScenarioVAE,
        ConditionalScenarioVAE,
        ScenarioLatentSpace,
    )

    from .water_canal_env import (
        WaterCanalEnv,
        OneGateTwoPoolsEnv,
    )

    from .deep_scenario_encoder import (
        DeepScenarioEncoder,
        ContrastiveLoss,
        ScenarioVectorDB,
    )

    # HydroGraphNet 集成模块
    from .hydrographnet import (
        # KAN Layers
        KANLinear,
        KolmogorovArnoldNetwork,
        FourierKAN,
        ChebyshevKAN,
        # Physics Loss
        GlobalMassConservationLoss,
        WaterBalanceLoss,
        HydroPhysicsLoss,
        # Micro World Adapter
        MicroWorldAdapter,
        GraphBuilder,
    )

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False
    _warnings.warn(
        "PyTorch is not installed. AI model components are unavailable. "
        "Install torch to enable them.",
        ImportWarning,
        stacklevel=2,
    )

__version__ = "1.0.0"
__author__ = "E2EControl Team"

"""
HydroGraphNet Integration Module
基于 acostacos/HydroGraphNet 的微观流态代理模块

本模块集成了 HydroGraphNet 的核心组件用于水网世界模型:

1. kan_layers.py - Kolmogorov-Arnold Network (KAN) 层
   - 基于傅里叶的函数逼近网络
   - 可用于场景识别、节点编码等任务

2. physics_loss.py - 物理守恒损失函数
   - 全局质量守恒损失
   - 连续性方程约束
   - 与现有 LSTM 模型配合使用

3. micro_world_adapter.py - 微观世界模型适配器
   - 将宏观边界条件转换为图结构输入
   - 调用 HydroGraphNet 进行推理
   - 输出关键节点的流速矢量场和压力分布

参考:
- HydroGraphNet: https://github.com/acostacos/HydroGraphNet
- MeshGraphNet: Pfaff et al., "Learning mesh-based simulation with graph networks"
- KAN: Kolmogorov-Arnold representation theorem
"""

from .kan_layers import (
    KANLinear,
    KolmogorovArnoldNetwork,
    FourierKAN,
    ChebyshevKAN,
)

from .physics_loss import (
    GlobalMassConservationLoss,
    ContinuityEquationLoss,
    WaterBalanceLoss,
    HydroPhysicsLoss,
)

from .micro_world_adapter import (
    MicroWorldAdapter,
    GraphBuilder,
    HydroGraphNetWrapper,
)

__version__ = "1.0.0"
__all__ = [
    # KAN Layers
    'KANLinear',
    'KolmogorovArnoldNetwork',
    'FourierKAN',
    'ChebyshevKAN',
    # Physics Loss
    'GlobalMassConservationLoss',
    'ContinuityEquationLoss',
    'WaterBalanceLoss',
    'HydroPhysicsLoss',
    # Adapter
    'MicroWorldAdapter',
    'GraphBuilder',
    'HydroGraphNetWrapper',
]

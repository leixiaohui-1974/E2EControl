"""
分布式SIL（软件在环）测试框架

针对南水北调中线工程"超长距离、串级闸站、强耦合"特点设计的
分层分布式控制SIL架构。

核心模块:
- ScenarioGenerator: 场景生成器，支持真值驱动与随机扰动
- ReducedOrderEngine: 全线降阶引擎，基于IDZ模型
- SegmentedHighFidelity: 分段高保真模型，Saint-Venant方程
- BoundaryAssimilator: 接口同化器，误差断路与状态估计
- ControllerOrchestrator: 控制器编排器，多层级调度
- SILEvaluator: 评估器，全线一致性KPI监控

设计原则:
1. 误差不沿全线自由传播
2. 分段+接口同化切断误差
3. 全线降阶+局部高保真避免漂移
4. 参数集合+漂移偏置建模正规化不确定性

Author: E2EControl Team
Version: 1.0.0
"""

from .core.sil_framework import DistributedSILFramework
from .core.scenario_generator import ScenarioGenerator
from .models.reduced_order_engine import ReducedOrderEngine
from .models.segmented_high_fidelity import SegmentedHighFidelityModel
from .interfaces.boundary_assimilator import BoundaryAssimilator
from .core.controller_orchestrator import ControllerOrchestrator
from .evaluators.sil_evaluator import SILEvaluator

__version__ = "1.0.0"
__all__ = [
    "DistributedSILFramework",
    "ScenarioGenerator",
    "ReducedOrderEngine",
    "SegmentedHighFidelityModel",
    "BoundaryAssimilator",
    "ControllerOrchestrator",
    "SILEvaluator",
]

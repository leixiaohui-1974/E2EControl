"""
全场景在环测试框架 (Comprehensive Hardware-in-the-Loop Testing Framework)

本框架实现成千上万种场景组合的研究型全功能测试，覆盖：
- 本体仿真 (Physics Simulation)
- 同步孪生 (Synchronized Digital Twin)
- 预测功能 (Prediction)
- 调度优化 (Scheduling Optimization)
- 控制功能 (Control)
- 异常检测与自愈 (Anomaly Detection & Self-Healing)

设计目标：
- 10,000+ 场景组合测试
- 全功能覆盖率 > 95%
- 自动化报告生成
- 并行测试支持

注意：
- 当前输出用于研究评估与回归对比，不等同于正式验收或生产认证结论。
"""

from .scenario_combinatorial_generator import (
    ScenarioCombinatorialGenerator,
    ScenarioSpace,
    TestScenario,
)
from .physics_simulation_tester import PhysicsSimulationTester
from .digital_twin_sync_tester import DigitalTwinSyncTester
from .prediction_tester import PredictionTester
from .scheduling_optimization_tester import SchedulingOptimizationTester
from .control_tester import ControlTester
from .anomaly_self_healing_tester import AnomalySelfHealingTester
from .hil_coordinator import HILTestCoordinator
from .report_generator import ComprehensiveReportGenerator

__all__ = [
    'ScenarioCombinatorialGenerator',
    'ScenarioSpace',
    'TestScenario',
    'PhysicsSimulationTester',
    'DigitalTwinSyncTester',
    'PredictionTester',
    'SchedulingOptimizationTester',
    'ControlTester',
    'AnomalySelfHealingTester',
    'HILTestCoordinator',
    'ComprehensiveReportGenerator',
]

__version__ = '1.0.0'

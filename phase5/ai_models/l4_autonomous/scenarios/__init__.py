"""
全场景定义模块 (Complete Scenario Definition)
南水北调中线全场景覆盖

场景分类:
1. 正常运行场景 (Normal Operation)
2. 需水调度场景 (Water Demand Scheduling)
3. 极端天气场景 (Extreme Weather)
4. 设备故障场景 (Equipment Failure)
5. 应急处置场景 (Emergency Response)
6. 维护检修场景 (Maintenance)
"""

from .scenario_definitions import (
    ScenarioCategory,
    Scenario,
    ScenarioLibrary,
    COMPLETE_SCENARIO_MATRIX,
)

from .scenario_generator import (
    ScenarioGenerator,
    DynamicScenarioBuilder,
)

from .scenario_validator import (
    ScenarioValidator,
    ValidationResult,
)

__all__ = [
    'ScenarioCategory',
    'Scenario',
    'ScenarioLibrary',
    'COMPLETE_SCENARIO_MATRIX',
    'ScenarioGenerator',
    'DynamicScenarioBuilder',
    'ScenarioValidator',
    'ValidationResult',
]

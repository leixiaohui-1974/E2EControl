"""
Phase 5: 在环测试框架 (Hardware-in-the-Loop Testing Framework)

实现水网系统自主运行等级(L0-L5)的测试与认证
"""

from .scenario_generator import (
    ScenarioGenerator, Scenario, Condition, Injection,
    ScenarioCategory, DifficultyLevel, AutonomousLevel, InitialState, PassCriteria as ScenarioPassCriteria
)
from .condition_injector import ConditionInjector, InjectionType, InjectionEvent
from .evaluation_engine import EvaluationEngine, TestResult, PassCriteria
from .test_runner import HILTestRunner
from .report_generator import ReportGenerator

__all__ = [
    'ScenarioGenerator',
    'Scenario',
    'Condition',
    'Injection',
    'ScenarioCategory',
    'DifficultyLevel',
    'AutonomousLevel',
    'InitialState',
    'ConditionInjector',
    'InjectionType',
    'InjectionEvent',
    'EvaluationEngine',
    'TestResult',
    'PassCriteria',
    'ScenarioPassCriteria',
    'HILTestRunner',
    'ReportGenerator'
]

__version__ = '1.0.0'

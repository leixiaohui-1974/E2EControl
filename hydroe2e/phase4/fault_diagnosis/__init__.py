"""
故障诊断模块

提供基于规则的故障诊断和综合诊断引擎。
"""

from .diagnosis_engine import (
    DiagnosisEngine,
    DiagnosisResult,
    DiagnosisRule,
    FaultCategory,
    FaultSeverity,
    quick_diagnose,
)
from .rule_based_diagnosis import (
    RuleBasedDiagnosisSystem,
    DiagnosisReport,
    FaultType,
)

__all__ = [
    "DiagnosisEngine",
    "DiagnosisResult",
    "DiagnosisRule",
    "FaultCategory",
    "FaultSeverity",
    "quick_diagnose",
    "RuleBasedDiagnosisSystem",
    "DiagnosisReport",
    "FaultType",
]

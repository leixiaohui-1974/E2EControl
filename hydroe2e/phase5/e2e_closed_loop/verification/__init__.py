"""验证模块 - SIL/HIL桥接"""
from .sil_hil_bridge import (
    SILHILBridge,
    VerificationMode,
    FaultInjection,
    VerificationResult,
    SILHILTestSuite,
)

__all__ = [
    "SILHILBridge",
    "VerificationMode",
    "FaultInjection",
    "VerificationResult",
    "SILHILTestSuite",
]

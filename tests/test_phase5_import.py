import importlib

import pytest


def test_phase5_package_import_is_lightweight():
    phase5 = importlib.import_module("hydroe2e.phase5")
    assert phase5.__version__ == "1.0.0"
    assert hasattr(phase5, "__all__")
    assert "CentralizedScheduler" in phase5.__all__


def test_phase5_integrated_system_import_when_optional_stack_available():
    pytest.importorskip("cvxpy")
    module = importlib.import_module("hydroe2e.phase5.integrated_system")
    assert hasattr(module, "IntegratedWaterNetworkSystem")

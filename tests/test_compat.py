"""
Tests for _compat.py (HydroMind contracts compatibility layer).

Supports both installed and uninstalled hydromind-contracts.
"""

import importlib
import pytest

from hydroe2e._compat import (
    HAS_HYDROMIND,
    SimulatorProtocol,
    ControllerProtocol,
    DetectorProtocol,
)
import hydroe2e._compat as compat_module


class TestHasHydroMind:
    def test_has_hydromind_is_bool(self):
        assert isinstance(HAS_HYDROMIND, bool)


class TestProtocols:
    """Protocols are real Protocol classes when contracts installed, else object."""

    def test_simulator_protocol_usable(self):
        assert SimulatorProtocol is not None

    def test_controller_protocol_usable(self):
        assert ControllerProtocol is not None

    def test_detector_protocol_usable(self):
        assert DetectorProtocol is not None

    def test_protocols_match_hydromind_state(self):
        if HAS_HYDROMIND:
            assert SimulatorProtocol is not object
        else:
            assert SimulatorProtocol is object


class TestAllExports:
    def test_all_exists(self):
        assert hasattr(compat_module, "__all__")

    def test_all_exports_accessible(self):
        for name in compat_module.__all__:
            assert hasattr(compat_module, name), f"{name} listed in __all__ but not found"

    def test_minimum_expected_names(self):
        expected_min = {"HAS_HYDROMIND", "SimulatorProtocol", "ControllerProtocol", "DetectorProtocol"}
        assert expected_min.issubset(set(compat_module.__all__))


class TestSubclassing:
    @pytest.mark.skipif(HAS_HYDROMIND, reason="Protocol classes can't be subclassed directly when real")
    def test_subclass_simulator(self):
        class MySimulator(SimulatorProtocol):
            pass
        obj = MySimulator()
        assert isinstance(obj, SimulatorProtocol)

    @pytest.mark.skipif(HAS_HYDROMIND, reason="Protocol classes can't be subclassed directly when real")
    def test_subclass_controller(self):
        class MyController(ControllerProtocol):
            pass
        obj = MyController()
        assert isinstance(obj, ControllerProtocol)


class TestModuleImport:
    def test_reimport(self):
        mod = importlib.reload(compat_module)
        assert hasattr(mod, "HAS_HYDROMIND")

    def test_hydraulic_protocol_available(self):
        """New protocol added for hydraulic solvers."""
        if HAS_HYDROMIND:
            assert hasattr(compat_module, "HydraulicSolverProtocol")
            assert hasattr(compat_module, "ChannelConfigProtocol")

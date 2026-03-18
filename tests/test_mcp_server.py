"""
Tests for mcp_servers/e2e_server.py (MCP tool server).

Covers:
- _create_server ImportError when mcp SDK is absent
- main() ImportError when mcp SDK is absent
- HAS_MCP flag
- Tool function logic via mocking (get_system_status, interpret_instruction,
  run_mpc_simulation, detect_anomaly, diagnose_fault)
"""

import importlib
from unittest.mock import patch, MagicMock

import pytest

import hydroe2e.mcp_servers.e2e_server as server_module
from hydroe2e.mcp_servers.e2e_server import _create_server, main


# ---------------------------------------------------------------------------
# HAS_MCP flag
# ---------------------------------------------------------------------------

class TestHasMCP:
    """MCP SDK is typically not installed in test environments."""

    def test_has_mcp_is_bool(self):
        assert isinstance(server_module.HAS_MCP, bool)

    def test_has_mcp_false_when_mcp_absent(self):
        """If this test runs without the mcp package, HAS_MCP should be False."""
        # If mcp IS installed the test is still valid; we just skip assertion.
        try:
            import mcp  # noqa: F401
            pytest.skip("mcp is installed; cannot verify False")
        except ImportError:
            assert server_module.HAS_MCP is False


# ---------------------------------------------------------------------------
# _create_server without mcp
# ---------------------------------------------------------------------------

class TestCreateServerWithoutMCP:
    """When mcp is not installed, _create_server must raise ImportError."""

    def test_raises_import_error(self):
        if server_module.HAS_MCP:
            pytest.skip("mcp is installed")
        with pytest.raises(ImportError, match="MCP SDK"):
            _create_server()


# ---------------------------------------------------------------------------
# main() without mcp
# ---------------------------------------------------------------------------

class TestMainWithoutMCP:

    def test_main_raises_import_error(self):
        if server_module.HAS_MCP:
            pytest.skip("mcp is installed")
        with pytest.raises(ImportError, match="MCP SDK"):
            main()


# ---------------------------------------------------------------------------
# _create_server with mocked FastMCP
# ---------------------------------------------------------------------------

class TestCreateServerWithMockedMCP:
    """Patch FastMCP so _create_server succeeds, then inspect registered tools."""

    @pytest.fixture(autouse=True)
    def _patch_fastmcp(self):
        """Provide a fake FastMCP that records decorated tool functions."""
        self.registered_tools = {}

        class _FakeMCP:
            def __init__(self_inner, name, **kwargs):
                self_inner.name = name

            def tool(self_inner):
                def decorator(fn):
                    self.registered_tools[fn.__name__] = fn
                    return fn
                return decorator

            def run(self_inner):
                pass

        with patch.object(server_module, "HAS_MCP", True), \
             patch.object(server_module, "FastMCP", _FakeMCP):
            self.server = _create_server()
            yield

    # -- server creation ---------------------------------------------------

    def test_server_created(self):
        assert self.server is not None

    def test_server_name(self):
        assert self.server.name == "HydroE2E"

    def test_five_tools_registered(self):
        assert len(self.registered_tools) == 6

    def test_expected_tool_names(self):
        expected = {
            "run_mpc_simulation",
            "interpret_instruction",
            "detect_anomaly",
            "diagnose_fault",
            "get_system_status",
            "generate_scenario_report",
        }
        # run_digital_twin may not exist if omitted from server; adjust set
        # to only what is actually defined in the source.
        registered = set(self.registered_tools.keys())
        # At minimum these five must be present (run_digital_twin is NOT in
        # e2e_server.py source; only the other five are).
        mandatory = {
            "run_mpc_simulation",
            "interpret_instruction",
            "detect_anomaly",
            "diagnose_fault",
            "get_system_status",
        }
        assert mandatory.issubset(registered)

    # -- get_system_status -------------------------------------------------

    def test_get_system_status_returns_dict(self):
        fn = self.registered_tools["get_system_status"]
        with patch("hydroe2e.mcp_servers.e2e_server.__import__", create=True):
            # The function does: from hydroe2e import __version__
            result = fn()
        assert isinstance(result, dict)

    def test_get_system_status_has_name(self):
        result = self.registered_tools["get_system_status"]()
        assert result["name"] == "HydroE2E"

    def test_get_system_status_has_version(self):
        result = self.registered_tools["get_system_status"]()
        assert "version" in result

    def test_get_system_status_has_status_running(self):
        result = self.registered_tools["get_system_status"]()
        assert result["status"] == "running"

    def test_get_system_status_capabilities(self):
        result = self.registered_tools["get_system_status"]()
        assert "mpc_control" in result["capabilities"]

    # -- interpret_instruction --------------------------------------------

    def test_interpret_instruction_success(self):
        mock_interpreter = MagicMock()
        mock_interpreter.interpret.return_value = {"action": "open_gate"}

        with patch(
            "hydroe2e.mcp_servers.e2e_server.EnhancedSemanticInterpreter",
            return_value=mock_interpreter,
            create=True,
        ):
            # Need to patch at the point of import inside the function
            import hydroe2e.brain_enhanced as _be
            with patch.object(
                _be, "EnhancedSemanticInterpreter", return_value=mock_interpreter,
            ):
                fn = self.registered_tools["interpret_instruction"]
                result = fn("打开闸门到50%")

        assert result["status"] == "success"
        assert result["interpretation"] == {"action": "open_gate"}

    # -- run_mpc_simulation ------------------------------------------------

    def test_run_mpc_simulation_success(self):
        mock_config = {"simulation": {"total_hours": 50, "time_step": 3600.0}}
        mock_manager = MagicMock()
        mock_manager.run.return_value = {"converged": True}

        with patch("hydroe2e.config_manager.get_config", return_value=mock_config), \
             patch("hydroe2e.simulation_manager.SimulationManager", return_value=mock_manager):
            fn = self.registered_tools["run_mpc_simulation"]
            result = fn("测试指令", total_hours=10, time_step=1800.0)

        assert result["status"] == "success"
        assert result["result"] == {"converged": True}
        assert mock_config["simulation"]["total_hours"] == 10

    # -- detect_anomaly ----------------------------------------------------

    def test_detect_anomaly_fallback_error(self):
        """When EnsembleDetector is unavailable, returns error dict."""
        fn = self.registered_tools["detect_anomaly"]
        with patch.dict("sys.modules", {"hydroe2e.phase4.anomaly_detection": None}):
            # Importing from a None module triggers ImportError
            result = fn(water_levels=[1.0, 2.0, 3.0])
        assert result["status"] == "error"
        assert "未就绪" in result["message"]

    def test_detect_anomaly_success(self):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [{"index": 2, "score": 0.95}]

        mock_module = MagicMock()
        mock_module.EnsembleDetector.return_value = mock_detector

        with patch.dict("sys.modules", {"hydroe2e.phase4.anomaly_detection": mock_module}):
            import numpy as np  # noqa: F401
            fn = self.registered_tools["detect_anomaly"]
            result = fn(water_levels=[1.0, 2.0, 99.0])

        assert result["status"] == "success"

    # -- diagnose_fault ----------------------------------------------------

    def test_diagnose_fault_fallback_error(self):
        """When DiagnosisEngine is unavailable, returns error dict."""
        fn = self.registered_tools["diagnose_fault"]
        with patch.dict("sys.modules", {"hydroe2e.phase4.fault_diagnosis": None}):
            result = fn(anomaly_type="level_spike")
        assert result["status"] == "error"
        assert "未就绪" in result["message"]

    def test_diagnose_fault_success(self):
        mock_engine = MagicMock()
        mock_engine.diagnose.return_value = {"fault": "sensor_drift", "confidence": 0.88}

        mock_module = MagicMock()
        mock_module.DiagnosisEngine.return_value = mock_engine

        with patch.dict("sys.modules", {"hydroe2e.phase4.fault_diagnosis": mock_module}):
            fn = self.registered_tools["diagnose_fault"]
            result = fn(anomaly_type="level_spike", sensor_data={"wl_01": 9.8})

        assert result["status"] == "success"
        assert result["diagnosis"]["fault"] == "sensor_drift"

    # -- main() with mocked server -----------------------------------------

    def test_main_calls_run(self):
        with patch.object(server_module, "HAS_MCP", True), \
             patch.object(server_module, "_create_server") as mock_create:
            mock_server = MagicMock()
            mock_create.return_value = mock_server
            main()
            mock_server.run.assert_called_once()

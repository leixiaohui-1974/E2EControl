"""
Tests for role.py (E2ERoleModule).

Covers:
- Default initialization and attributes
- Custom initialization with config
- get_tools() structure and content
- initialize() / shutdown() lifecycle
- Capabilities list
- Edge cases (multiple init, idempotent shutdown)
"""

import pytest

from hydroe2e.role import E2ERoleModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_role(**overrides):
    """Create an E2ERoleModule with optional field overrides."""
    return E2ERoleModule(**overrides)


# ---------------------------------------------------------------------------
# Default initialization
# ---------------------------------------------------------------------------

class TestE2ERoleModuleDefaults:
    """Verify default attribute values."""

    def test_default_name(self):
        role = _make_role()
        assert role.name == "e2e_controller"

    def test_default_version(self):
        role = _make_role()
        assert role.version == "1.1.0"

    def test_default_description(self):
        role = _make_role()
        assert role.description == "端到端智能水网控制系统"

    def test_default_not_initialized(self):
        role = _make_role()
        assert role._initialized is False

    def test_default_config_empty(self):
        role = _make_role()
        assert role._config == {}


# ---------------------------------------------------------------------------
# Custom initialization
# ---------------------------------------------------------------------------

class TestE2ERoleModuleCustom:
    """Verify custom field values are accepted."""

    def test_custom_name(self):
        role = _make_role(name="custom_controller")
        assert role.name == "custom_controller"

    def test_custom_version(self):
        role = _make_role(version="2.0.0")
        assert role.version == "2.0.0"


# ---------------------------------------------------------------------------
# get_tools()
# ---------------------------------------------------------------------------

class TestGetTools:
    """Verify the tool list returned by get_tools()."""

    def test_returns_list(self):
        tools = _make_role().get_tools()
        assert isinstance(tools, list)

    def test_tools_are_dicts(self):
        tools = _make_role().get_tools()
        for tool in tools:
            assert isinstance(tool, dict)

    def test_each_tool_has_name_and_description(self):
        tools = _make_role().get_tools()
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"

    def test_tool_count(self):
        tools = _make_role().get_tools()
        assert len(tools) == 7

    def test_expected_tool_names(self):
        names = {t["name"] for t in _make_role().get_tools()}
        expected = {
            "run_mpc_simulation",
            "interpret_instruction",
            "detect_anomaly",
            "diagnose_fault",
            "run_digital_twin",
            "get_system_status",
            "generate_scenario_report",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestCapabilities:
    """Verify the capabilities list."""

    def test_capabilities_is_list(self):
        role = _make_role()
        assert isinstance(role.capabilities, list)

    def test_capabilities_count(self):
        role = _make_role()
        assert len(role.capabilities) == 9

    def test_mpc_control_in_capabilities(self):
        role = _make_role()
        assert "mpc_control" in role.capabilities

    def test_anomaly_detection_in_capabilities(self):
        role = _make_role()
        assert "anomaly_detection" in role.capabilities


# ---------------------------------------------------------------------------
# Lifecycle: initialize / shutdown
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Verify initialize() and shutdown() behaviour."""

    def test_initialize_sets_flag(self):
        role = _make_role()
        role.initialize()
        assert role._initialized is True

    def test_initialize_with_config(self):
        role = _make_role()
        role.initialize(config={"key": "value"})
        assert role._initialized is True
        assert role._config["key"] == "value"

    def test_initialize_merges_config(self):
        role = _make_role()
        role.initialize(config={"a": 1})
        role.initialize(config={"b": 2})
        assert role._config == {"a": 1, "b": 2}

    def test_initialize_with_none_config(self):
        role = _make_role()
        role.initialize(config=None)
        assert role._initialized is True
        assert role._config == {}

    def test_shutdown_clears_initialized(self):
        role = _make_role()
        role.initialize()
        role.shutdown()
        assert role._initialized is False

    def test_shutdown_clears_config(self):
        role = _make_role()
        role.initialize(config={"x": 42})
        role.shutdown()
        assert role._config == {}

    def test_full_lifecycle(self):
        """init -> use -> shutdown -> verify clean state."""
        role = _make_role()
        role.initialize(config={"mode": "test"})
        assert role._initialized is True
        tools = role.get_tools()
        assert len(tools) > 0
        role.shutdown()
        assert role._initialized is False
        assert role._config == {}

    def test_multiple_init_calls_idempotent(self):
        role = _make_role()
        role.initialize()
        role.initialize()
        assert role._initialized is True

    def test_shutdown_without_init(self):
        """Shutdown on a fresh role should not raise."""
        role = _make_role()
        role.shutdown()
        assert role._initialized is False

"""
Phase 5.6 Web Dashboard Tests

Tests for:
1. DashboardConfig
2. WebDashboard initialization and lifecycle
3. DashboardHandler API endpoints
4. HTML dashboard serving
5. Command processing
"""

import pytest
import json
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO
from dataclasses import dataclass
from enum import Enum
from http.server import HTTPServer

from hydroe2e.phase5.web.web_dashboard import (
    WebDashboard,
    DashboardConfig,
    DashboardHandler
)


# ============================================================================
# Test Fixtures
# ============================================================================

class MockSystemMode(Enum):
    NORMAL = "normal"
    FLOOD_CONTROL = "flood_control"
    DROUGHT = "drought"
    EMERGENCY = "emergency"


@dataclass
class MockSystemStatus:
    """Mock system status for testing"""
    mode: MockSystemMode = MockSystemMode.NORMAL
    health: float = 0.95
    active_faults: int = 0
    active_alerts: int = 0
    control_active: bool = True
    uptime: timedelta = timedelta(hours=24)
    current_levels: list = None
    current_flows: list = None

    def __post_init__(self):
        if self.current_levels is None:
            self.current_levels = [3.0, 3.1, 2.9]
        if self.current_flows is None:
            self.current_flows = [1.5, 1.4, 1.6]


class MockMetricType(Enum):
    SOLVE_TIME = "solve_time"
    TRACKING_ERROR = "tracking_error"


class MockMetricLevel(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MockAlert:
    alert_id: str
    metric_type: MockMetricType
    level: MockMetricLevel
    message: str
    component: str
    current_value: float
    threshold: float
    timestamp: datetime


@dataclass
class MockComputationalMetrics:
    avg_solve_time: float = 0.05
    max_solve_time: float = 0.15
    avg_iterations: float = 3.5
    memory_usage_mb: float = 256.0
    cpu_utilization: float = 45.0


@dataclass
class MockControlMetrics:
    avg_tracking_error: float = 0.02
    max_tracking_error: float = 0.05
    rms_error: float = 0.025
    avg_control_effort: float = 1.2


class MockEventType(Enum):
    MODE_CHANGE = "mode_change"
    FAULT_DETECTED = "fault_detected"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class MockSystemEvent:
    event_id: str
    event_type: MockEventType
    source: str
    description: str
    timestamp: datetime


class MockPerformanceMonitor:
    def __init__(self):
        self.active_alerts = {}

    def get_computational_metrics(self):
        return MockComputationalMetrics()

    def get_control_metrics(self):
        return MockControlMetrics()

    def acknowledge_alert(self, alert_id, resolution=None):
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            return True
        return False


class MockSystem:
    """Mock control system for testing"""

    def __init__(self):
        self._performance_monitor = MockPerformanceMonitor()
        self.event_history = []
        self._commands = []

    def get_status(self):
        return MockSystemStatus()

    def get_dashboard_data(self):
        return {
            "mode": "normal",
            "health": 0.95,
            "timestamp": datetime.now().isoformat()
        }

    def submit_command(self, command):
        self._commands.append(command)


@pytest.fixture
def mock_system():
    """Create mock system"""
    return MockSystem()


@pytest.fixture
def dashboard_config():
    """Create default dashboard config"""
    return DashboardConfig(
        host="127.0.0.1",
        port=9999,  # Use high port for testing
        refresh_interval=5,
        enable_control=True
    )


@pytest.fixture
def dashboard(dashboard_config, mock_system):
    """Create web dashboard instance"""
    dash = WebDashboard(system=mock_system, config=dashboard_config)
    return dash


# ============================================================================
# DashboardConfig Tests
# ============================================================================

class TestDashboardConfig:
    """Tests for DashboardConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = DashboardConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.refresh_interval == 5
        assert config.enable_control is True
        assert config.max_history_points == 100

    def test_custom_config(self):
        """Test custom configuration"""
        config = DashboardConfig(
            host="localhost",
            port=3000,
            refresh_interval=10,
            enable_control=False,
            max_history_points=50
        )

        assert config.host == "localhost"
        assert config.port == 3000
        assert config.refresh_interval == 10
        assert config.enable_control is False
        assert config.max_history_points == 50


# ============================================================================
# WebDashboard Initialization Tests
# ============================================================================

class TestWebDashboardInitialization:
    """Tests for WebDashboard initialization"""

    def test_init_default(self):
        """Test initialization with defaults"""
        dashboard = WebDashboard()

        assert dashboard.config is not None
        assert dashboard._system is None
        assert dashboard._server is None
        assert dashboard._running is False

    def test_init_with_config(self, dashboard_config):
        """Test initialization with custom config"""
        dashboard = WebDashboard(config=dashboard_config)

        assert dashboard.config.port == 9999
        assert dashboard.config.host == "127.0.0.1"

    def test_init_with_system(self, mock_system):
        """Test initialization with system"""
        dashboard = WebDashboard(system=mock_system)

        assert dashboard._system is mock_system

    def test_set_system(self, dashboard, mock_system):
        """Test setting system after initialization"""
        new_system = MockSystem()
        dashboard.set_system(new_system)

        assert dashboard._system is new_system


# ============================================================================
# WebDashboard Lifecycle Tests
# ============================================================================

class TestWebDashboardLifecycle:
    """Tests for WebDashboard start/stop"""

    def test_start_stop(self, dashboard):
        """Test starting and stopping dashboard"""
        # Start server
        dashboard.start()
        time.sleep(0.1)  # Give server time to start

        assert dashboard.is_running() is True
        assert dashboard._server is not None
        assert dashboard._server_thread is not None

        # Stop server
        dashboard.stop()
        time.sleep(0.1)

        assert dashboard.is_running() is False

    def test_double_start(self, dashboard):
        """Test starting already running server"""
        dashboard.start()
        time.sleep(0.1)

        # Try to start again
        dashboard.start()  # Should not raise

        assert dashboard.is_running() is True

        dashboard.stop()

    def test_stop_not_running(self, dashboard):
        """Test stopping server that isn't running"""
        dashboard.stop()  # Should not raise
        assert dashboard.is_running() is False

    def test_get_url(self, dashboard):
        """Test getting dashboard URL"""
        url = dashboard.get_url()

        assert "127.0.0.1:9999" in url
        assert url.startswith("http://")


# ============================================================================
# DashboardHandler Tests (Mock HTTP)
# ============================================================================

class MockHTTPRequest:
    """Mock HTTP request for testing handler"""

    def __init__(self, path="/", method="GET", body=None):
        self.path = path
        self.method = method
        self.body = body or b""


class MockRFile:
    """Mock request body reader"""

    def __init__(self, data=b""):
        self.data = data

    def read(self, length):
        return self.data[:length]


class MockWFile:
    """Mock response writer"""

    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data


class TestDashboardHandler:
    """Tests for DashboardHandler"""

    def create_handler(self, path="/", method="GET", body=None, dashboard=None):
        """Create a mock handler for testing"""
        # Create handler without full HTTP setup
        handler = object.__new__(DashboardHandler)
        handler.path = path
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Content-Length": str(len(body) if body else 0)}
        handler.rfile = MockRFile(body.encode() if isinstance(body, str) else (body or b""))
        handler.wfile = MockWFile()
        handler._headers_buffer = []

        DashboardHandler.dashboard = dashboard

        return handler

    def test_get_status_no_system(self):
        """Test status endpoint without system"""
        handler = self.create_handler("/api/status")
        handler._set_headers = Mock()

        handler._api_get_status()

        response = json.loads(handler.wfile.data.decode())
        assert "error" in response

    def test_get_status_with_system(self, mock_system):
        """Test status endpoint with system"""
        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/status", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_status()

        response = json.loads(handler.wfile.data.decode())
        assert "mode" in response
        assert response["mode"] == "normal"
        assert "health" in response
        assert response["health"] == 0.95

    def test_get_metrics_no_system(self):
        """Test metrics endpoint without system"""
        handler = self.create_handler("/api/metrics")
        handler._set_headers = Mock()

        handler._api_get_metrics()

        response = json.loads(handler.wfile.data.decode())
        assert "error" in response

    def test_get_metrics_with_system(self, mock_system):
        """Test metrics endpoint with system"""
        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/metrics", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_metrics()

        response = json.loads(handler.wfile.data.decode())
        assert "computational" in response
        assert "control" in response
        assert response["computational"]["avg_solve_time_ms"] == 50.0  # 0.05 * 1000

    def test_get_alerts_empty(self, mock_system):
        """Test alerts endpoint with no alerts"""
        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/alerts", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_alerts()

        response = json.loads(handler.wfile.data.decode())
        assert "alerts" in response
        assert response["count"] == 0

    def test_get_alerts_with_alerts(self, mock_system):
        """Test alerts endpoint with alerts"""
        alert = MockAlert(
            alert_id="alert_001",
            metric_type=MockMetricType.SOLVE_TIME,
            level=MockMetricLevel.WARNING,
            message="Solve time exceeded",
            component="controller",
            current_value=0.2,
            threshold=0.1,
            timestamp=datetime.now()
        )
        mock_system._performance_monitor.active_alerts["alert_001"] = alert

        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/alerts", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_alerts()

        response = json.loads(handler.wfile.data.decode())
        assert response["count"] == 1
        assert response["alerts"][0]["id"] == "alert_001"

    def test_get_events_empty(self, mock_system):
        """Test events endpoint with no events"""
        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/events", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_events()

        response = json.loads(handler.wfile.data.decode())
        assert "events" in response
        assert len(response["events"]) == 0

    def test_get_events_with_events(self, mock_system):
        """Test events endpoint with events"""
        event = MockSystemEvent(
            event_id="evt_001",
            event_type=MockEventType.MODE_CHANGE,
            source="operator",
            description="Mode changed to flood control",
            timestamp=datetime.now()
        )
        mock_system.event_history.append(event)

        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/events", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_events()

        response = json.loads(handler.wfile.data.decode())
        assert len(response["events"]) == 1
        assert response["events"][0]["id"] == "evt_001"

    def test_get_dashboard_data(self, mock_system):
        """Test dashboard data endpoint"""
        dashboard = WebDashboard(system=mock_system)
        handler = self.create_handler("/api/dashboard", dashboard=dashboard)
        handler._set_headers = Mock()

        handler._api_get_dashboard()

        response = json.loads(handler.wfile.data.decode())
        assert "mode" in response
        assert "health" in response


# ============================================================================
# POST Endpoint Tests
# ============================================================================

class TestDashboardPostEndpoints:
    """Tests for POST endpoints"""

    def create_handler(self, path, body, dashboard):
        """Create handler for POST request"""
        handler = object.__new__(DashboardHandler)
        handler.path = path
        handler.requestline = f"POST {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = MockRFile(body.encode() if isinstance(body, str) else body)
        handler.wfile = MockWFile()
        handler._headers_buffer = []

        DashboardHandler.dashboard = dashboard

        return handler

    def test_post_mode_no_system(self):
        """Test mode endpoint without system"""
        dashboard = WebDashboard()
        body = json.dumps({"mode": "flood_control"})
        handler = self.create_handler("/api/mode", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_mode({"mode": "flood_control"})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is False
        assert "System not connected" in response["error"]

    def test_post_mode_missing_mode(self, mock_system):
        """Test mode endpoint with missing mode"""
        dashboard = WebDashboard(system=mock_system)
        body = json.dumps({})
        handler = self.create_handler("/api/mode", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_mode({})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is False
        assert "Missing mode" in response["error"]

    def test_post_reference_no_levels(self, mock_system):
        """Test reference endpoint with missing levels"""
        dashboard = WebDashboard(system=mock_system)
        body = json.dumps({})
        handler = self.create_handler("/api/reference", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_reference({})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is False
        assert "Missing levels" in response["error"]

    def test_post_acknowledge_success(self, mock_system):
        """Test acknowledge endpoint"""
        alert = MockAlert(
            alert_id="alert_001",
            metric_type=MockMetricType.SOLVE_TIME,
            level=MockMetricLevel.WARNING,
            message="Test alert",
            component="test",
            current_value=0.2,
            threshold=0.1,
            timestamp=datetime.now()
        )
        mock_system._performance_monitor.active_alerts["alert_001"] = alert

        dashboard = WebDashboard(system=mock_system)
        body = json.dumps({"alert_id": "alert_001"})
        handler = self.create_handler("/api/acknowledge", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_acknowledge({"alert_id": "alert_001"})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is True

    def test_post_acknowledge_missing_id(self, mock_system):
        """Test acknowledge endpoint without alert_id"""
        dashboard = WebDashboard(system=mock_system)
        body = json.dumps({})
        handler = self.create_handler("/api/acknowledge", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_acknowledge({})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is False
        assert "Missing alert_id" in response["error"]

    def test_post_command_disabled(self, mock_system):
        """Test command endpoint when control disabled"""
        config = DashboardConfig(enable_control=False)
        dashboard = WebDashboard(system=mock_system, config=config)
        body = json.dumps({"type": "test", "target": "system"})
        handler = self.create_handler("/api/command", body, dashboard)
        handler._set_headers = Mock()

        handler._api_post_command({"type": "test", "target": "system"})

        response = json.loads(handler.wfile.data.decode())
        assert response["success"] is False
        assert "Control disabled" in response["error"]


# ============================================================================
# HTML Dashboard Tests
# ============================================================================

class TestDashboardHTML:
    """Tests for HTML dashboard"""

    def test_get_dashboard_html(self):
        """Test HTML dashboard generation"""
        handler = object.__new__(DashboardHandler)
        html = handler._get_dashboard_html()

        # Check basic structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Check Chinese content
        assert "水网控制系统" in html
        assert "系统健康" in html
        assert "水位状态" in html

        # Check JavaScript functions
        assert "updateDashboard" in html
        assert "setMode" in html
        assert "emergencyStop" in html

        # Check API endpoint references (built dynamically in JS)
        assert "'/api/'" in html  # Base API path
        assert "'status'" in html
        assert "'metrics'" in html
        assert "'alerts'" in html

    def test_html_has_refresh_interval(self):
        """Test HTML contains refresh interval"""
        handler = object.__new__(DashboardHandler)
        html = handler._get_dashboard_html()

        assert "REFRESH_INTERVAL" in html
        assert "setInterval" in html

    def test_html_has_control_buttons(self):
        """Test HTML has control buttons"""
        handler = object.__new__(DashboardHandler)
        html = handler._get_dashboard_html()

        assert "正常模式" in html
        assert "防洪模式" in html
        assert "抗旱模式" in html
        assert "紧急停止" in html


# ============================================================================
# Static File Handling Tests
# ============================================================================

class TestStaticFileHandling:
    """Tests for static file serving"""

    def create_handler(self, path):
        """Create handler for testing"""
        handler = object.__new__(DashboardHandler)
        handler.path = path
        handler.wfile = MockWFile()
        handler._headers_buffer = []
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        return handler

    def test_serve_allowed_static(self):
        """Test serving allowed static files"""
        handler = self.create_handler("/static/dashboard.js")

        handler._serve_static("dashboard.js")

        # Should set headers and serve (empty content in test)
        handler.send_response.assert_called()

    def test_serve_disallowed_static(self):
        """Test blocking disallowed static files"""
        handler = self.create_handler("/static/malicious.php")
        handler._not_found = Mock()

        handler._serve_static("malicious.php")

        handler._not_found.assert_called_once()


# ============================================================================
# Route Handling Tests
# ============================================================================

class TestRouteHandling:
    """Tests for URL route handling"""

    def create_handler(self, path, dashboard=None):
        """Create handler with mocked methods"""
        handler = object.__new__(DashboardHandler)
        handler.path = path
        handler.wfile = MockWFile()
        handler.headers = {}

        # Mock response methods
        handler._serve_dashboard = Mock()
        handler._api_get_status = Mock()
        handler._api_get_metrics = Mock()
        handler._api_get_alerts = Mock()
        handler._api_get_events = Mock()
        handler._api_get_dashboard = Mock()
        handler._serve_static = Mock()
        handler._not_found = Mock()

        DashboardHandler.dashboard = dashboard

        return handler

    def test_route_index(self):
        """Test routing to index"""
        handler = self.create_handler("/")

        handler.do_GET()

        handler._serve_dashboard.assert_called_once()

    def test_route_index_html(self):
        """Test routing to index.html"""
        handler = self.create_handler("/index.html")

        handler.do_GET()

        handler._serve_dashboard.assert_called_once()

    def test_route_api_status(self):
        """Test routing to status API"""
        handler = self.create_handler("/api/status")

        handler.do_GET()

        handler._api_get_status.assert_called_once()

    def test_route_api_metrics(self):
        """Test routing to metrics API"""
        handler = self.create_handler("/api/metrics")

        handler.do_GET()

        handler._api_get_metrics.assert_called_once()

    def test_route_api_alerts(self):
        """Test routing to alerts API"""
        handler = self.create_handler("/api/alerts")

        handler.do_GET()

        handler._api_get_alerts.assert_called_once()

    def test_route_api_events(self):
        """Test routing to events API"""
        handler = self.create_handler("/api/events")

        handler.do_GET()

        handler._api_get_events.assert_called_once()

    def test_route_api_dashboard(self):
        """Test routing to dashboard API"""
        handler = self.create_handler("/api/dashboard")

        handler.do_GET()

        handler._api_get_dashboard.assert_called_once()

    def test_route_static(self):
        """Test routing to static files"""
        handler = self.create_handler("/static/test.js")

        handler.do_GET()

        handler._serve_static.assert_called_once_with("test.js")

    def test_route_not_found(self):
        """Test routing unknown paths"""
        handler = self.create_handler("/unknown/path")

        handler.do_GET()

        handler._not_found.assert_called_once()


# ============================================================================
# Integration Test
# ============================================================================

class TestDashboardIntegration:
    """Integration tests"""

    def test_full_lifecycle(self, mock_system):
        """Test full dashboard lifecycle"""
        config = DashboardConfig(host="127.0.0.1", port=19999)
        dashboard = WebDashboard(system=mock_system, config=config)

        # Start
        dashboard.start()
        time.sleep(0.2)
        assert dashboard.is_running()

        # Check URL
        url = dashboard.get_url()
        assert "19999" in url

        # Stop
        dashboard.stop()
        time.sleep(0.2)
        assert not dashboard.is_running()

    def test_dashboard_without_system(self):
        """Test dashboard can run without system (returns errors)"""
        config = DashboardConfig(host="127.0.0.1", port=19998)
        dashboard = WebDashboard(config=config)

        dashboard.start()
        time.sleep(0.1)

        assert dashboard.is_running()

        dashboard.stop()


# ============================================================================
# CORS Tests
# ============================================================================

class TestCORSHandling:
    """Tests for CORS handling"""

    def test_cors_headers(self):
        """Test CORS headers are set"""
        handler = object.__new__(DashboardHandler)
        handler.wfile = MockWFile()

        headers_sent = []

        def mock_send_header(key, value):
            headers_sent.append((key, value))

        handler.send_response = Mock()
        handler.send_header = mock_send_header
        handler.end_headers = Mock()

        handler._set_headers()

        # Check CORS headers
        header_dict = dict(headers_sent)
        assert header_dict.get("Access-Control-Allow-Origin") == "*"
        assert "GET" in header_dict.get("Access-Control-Allow-Methods", "")
        assert "POST" in header_dict.get("Access-Control-Allow-Methods", "")

    def test_options_request(self):
        """Test OPTIONS preflight request"""
        handler = object.__new__(DashboardHandler)
        handler._set_headers = Mock()

        handler.do_OPTIONS()

        handler._set_headers.assert_called_once()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

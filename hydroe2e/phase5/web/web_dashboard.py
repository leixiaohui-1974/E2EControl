"""
Web Dashboard Server for Water Network Control System

Provides a lightweight web interface for monitoring and control:
- Real-time status display
- Performance metrics visualization
- Alert management
- Command submission

Uses Python's built-in http.server with JSON API endpoints.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    refresh_interval: int = 5  # seconds
    enable_control: bool = True  # Allow control commands
    max_history_points: int = 100


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard"""

    # Class-level reference to dashboard instance
    dashboard = None

    def log_message(self, format, *args):
        """Override to use logging module"""
        logger.debug(f"[WebDashboard] {args[0]}")

    def _set_headers(self, content_type: str = "application/json", status: int = 200):
        """Set response headers"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._set_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/api/status":
            self._api_get_status()
        elif path == "/api/metrics":
            self._api_get_metrics()
        elif path == "/api/alerts":
            self._api_get_alerts()
        elif path == "/api/events":
            self._api_get_events()
        elif path == "/api/dashboard":
            self._api_get_dashboard()
        elif path.startswith("/static/"):
            self._serve_static(path[8:])
        else:
            self._not_found()

    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}

        if path == "/api/command":
            self._api_post_command(data)
        elif path == "/api/mode":
            self._api_post_mode(data)
        elif path == "/api/reference":
            self._api_post_reference(data)
        elif path == "/api/acknowledge":
            self._api_post_acknowledge(data)
        else:
            self._not_found()

    def _serve_dashboard(self):
        """Serve main dashboard HTML"""
        self._set_headers("text/html")
        html = self._get_dashboard_html()
        self.wfile.write(html.encode())

    def _serve_static(self, filename: str):
        """Serve static files"""
        # For security, only serve known files
        allowed = {"dashboard.js", "dashboard.css"}
        if filename not in allowed:
            self._not_found()
            return

        content_type = "text/javascript" if filename.endswith(".js") else "text/css"
        self._set_headers(content_type)
        # Would serve actual static files here
        self.wfile.write(b"")

    def _api_get_status(self):
        """Get system status"""
        self._set_headers()
        if self.dashboard and self.dashboard._system:
            status = self.dashboard._system.get_status()
            data = {
                "mode": status.mode.value,
                "health": status.health,
                "active_faults": status.active_faults,
                "active_alerts": status.active_alerts,
                "control_active": status.control_active,
                "uptime_seconds": status.uptime.total_seconds(),
                "levels": status.current_levels,
                "flows": status.current_flows,
                "timestamp": datetime.now().isoformat()
            }
        else:
            data = {"error": "System not connected"}

        self.wfile.write(json.dumps(data).encode())

    def _api_get_metrics(self):
        """Get performance metrics"""
        self._set_headers()
        if self.dashboard and self.dashboard._system and self.dashboard._system._performance_monitor:
            comp = self.dashboard._system._performance_monitor.get_computational_metrics()
            ctrl = self.dashboard._system._performance_monitor.get_control_metrics()
            data = {
                "computational": {
                    "avg_solve_time_ms": comp.avg_solve_time * 1000,
                    "max_solve_time_ms": comp.max_solve_time * 1000,
                    "avg_iterations": comp.avg_iterations,
                    "memory_mb": comp.memory_usage_mb,
                    "cpu_percent": comp.cpu_utilization
                },
                "control": {
                    "avg_tracking_error": ctrl.avg_tracking_error,
                    "max_tracking_error": ctrl.max_tracking_error,
                    "rms_error": ctrl.rms_error,
                    "avg_control_effort": ctrl.avg_control_effort
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            data = {"error": "Metrics not available"}

        self.wfile.write(json.dumps(data).encode())

    def _api_get_alerts(self):
        """Get active alerts"""
        self._set_headers()
        if self.dashboard and self.dashboard._system and self.dashboard._system._performance_monitor:
            alerts = list(self.dashboard._system._performance_monitor.active_alerts.values())
            data = {
                "alerts": [
                    {
                        "id": a.alert_id,
                        "type": a.metric_type.value,
                        "level": a.level.value,
                        "message": a.message,
                        "component": a.component,
                        "value": a.current_value,
                        "threshold": a.threshold,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in alerts
                ],
                "count": len(alerts)
            }
        else:
            data = {"alerts": [], "count": 0}

        self.wfile.write(json.dumps(data).encode())

    def _api_get_events(self):
        """Get recent events"""
        self._set_headers()
        if self.dashboard and self.dashboard._system:
            events = self.dashboard._system.event_history[-50:]
            data = {
                "events": [
                    {
                        "id": e.event_id,
                        "type": e.event_type.value,
                        "source": e.source,
                        "description": e.description,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in reversed(events)
                ]
            }
        else:
            data = {"events": []}

        self.wfile.write(json.dumps(data).encode())

    def _api_get_dashboard(self):
        """Get full dashboard data"""
        self._set_headers()
        if self.dashboard and self.dashboard._system:
            data = self.dashboard._system.get_dashboard_data()
        else:
            data = {"error": "System not connected"}

        self.wfile.write(json.dumps(data).encode())

    def _api_post_command(self, data: Dict):
        """Process command"""
        self._set_headers()
        if not self.dashboard or not self.dashboard._system:
            response = {"success": False, "error": "System not connected"}
        elif not self.dashboard.config.enable_control:
            response = {"success": False, "error": "Control disabled"}
        else:
            from ..monitoring.unified_system import ControlCommand
            cmd = ControlCommand(
                command_id=f"web_{datetime.now().timestamp()}",
                command_type=data.get("type", "unknown"),
                target=data.get("target", "system"),
                parameters=data.get("parameters", {}),
                timestamp=datetime.now(),
                source="web_dashboard"
            )
            self.dashboard._system.submit_command(cmd)
            response = {"success": True, "command_id": cmd.command_id}

        self.wfile.write(json.dumps(response).encode())

    def _api_post_mode(self, data: Dict):
        """Change system mode"""
        self._set_headers()
        mode = data.get("mode")
        if not mode:
            response = {"success": False, "error": "Missing mode"}
        elif not self.dashboard or not self.dashboard._system:
            response = {"success": False, "error": "System not connected"}
        else:
            from ..monitoring.unified_system import ControlCommand
            cmd = ControlCommand(
                command_id=f"web_{datetime.now().timestamp()}",
                command_type="set_mode",
                target="system",
                parameters={"mode": mode},
                timestamp=datetime.now(),
                source="web_dashboard"
            )
            self.dashboard._system.submit_command(cmd)
            response = {"success": True, "mode": mode}

        self.wfile.write(json.dumps(response).encode())

    def _api_post_reference(self, data: Dict):
        """Set reference levels"""
        self._set_headers()
        levels = data.get("levels")
        if not levels:
            response = {"success": False, "error": "Missing levels"}
        elif not self.dashboard or not self.dashboard._system:
            response = {"success": False, "error": "System not connected"}
        else:
            from ..monitoring.unified_system import ControlCommand
            cmd = ControlCommand(
                command_id=f"web_{datetime.now().timestamp()}",
                command_type="set_reference",
                target="controller",
                parameters={"levels": levels},
                timestamp=datetime.now(),
                source="web_dashboard"
            )
            self.dashboard._system.submit_command(cmd)
            response = {"success": True, "levels": levels}

        self.wfile.write(json.dumps(response).encode())

    def _api_post_acknowledge(self, data: Dict):
        """Acknowledge alert"""
        self._set_headers()
        alert_id = data.get("alert_id")
        if not alert_id:
            response = {"success": False, "error": "Missing alert_id"}
        elif not self.dashboard or not self.dashboard._system:
            response = {"success": False, "error": "System not connected"}
        elif not self.dashboard._system._performance_monitor:
            response = {"success": False, "error": "Monitor not available"}
        else:
            success = self.dashboard._system._performance_monitor.acknowledge_alert(
                alert_id, data.get("resolution")
            )
            response = {"success": success}

        self.wfile.write(json.dumps(response).encode())

    def _not_found(self):
        """Return 404"""
        self._set_headers(status=404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>水网控制系统仪表板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 { font-size: 1.5rem; color: #00d9ff; }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        .status-normal { background: #00c853; color: #000; }
        .status-warning { background: #ffc107; color: #000; }
        .status-critical { background: #ff5252; color: #fff; }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
        }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #0f3460;
        }
        .card h2 {
            font-size: 1rem;
            color: #888;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #0f3460;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #aaa; }
        .metric-value { font-weight: bold; color: #00d9ff; }
        .levels-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }
        .level-item {
            text-align: center;
            padding: 1rem;
            background: #0f3460;
            border-radius: 8px;
        }
        .level-value {
            font-size: 2rem;
            font-weight: bold;
            color: #00d9ff;
        }
        .level-label { color: #888; font-size: 0.9rem; }
        .health-bar {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .health-fill {
            height: 100%;
            background: linear-gradient(90deg, #00c853, #00d9ff);
            transition: width 0.3s;
        }
        .alert-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .alert-warning { background: rgba(255, 193, 7, 0.2); border-left: 3px solid #ffc107; }
        .alert-critical { background: rgba(255, 82, 82, 0.2); border-left: 3px solid #ff5252; }
        .event-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid #0f3460;
            font-size: 0.9rem;
        }
        .event-time { color: #666; font-size: 0.8rem; }
        .controls {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .btn-primary { background: #00d9ff; color: #000; }
        .btn-warning { background: #ffc107; color: #000; }
        .btn-danger { background: #ff5252; color: #fff; }
        .btn:hover { opacity: 0.8; transform: translateY(-1px); }
        .no-data { color: #666; text-align: center; padding: 2rem; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <div class="header">
        <h1>智能水网控制系统</h1>
        <div>
            <span id="mode-badge" class="status-badge status-normal">NORMAL</span>
            <span id="update-time" style="margin-left: 1rem; color: #666;"></span>
        </div>
    </div>

    <div class="dashboard">
        <!-- System Health -->
        <div class="card">
            <h2>系统健康</h2>
            <div class="metric">
                <span class="metric-label">健康度</span>
                <span id="health-value" class="metric-value">--</span>
            </div>
            <div class="health-bar">
                <div id="health-fill" class="health-fill" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">运行时间</span>
                <span id="uptime" class="metric-value">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">控制激活</span>
                <span id="control-active" class="metric-value">--</span>
            </div>
        </div>

        <!-- Water Levels -->
        <div class="card">
            <h2>水位状态</h2>
            <div id="levels-container" class="levels-grid">
                <div class="level-item">
                    <div class="level-value">--</div>
                    <div class="level-label">渠池 1</div>
                </div>
                <div class="level-item">
                    <div class="level-value">--</div>
                    <div class="level-label">渠池 2</div>
                </div>
                <div class="level-item">
                    <div class="level-value">--</div>
                    <div class="level-label">渠池 3</div>
                </div>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="card">
            <h2>性能指标</h2>
            <div class="metric">
                <span class="metric-label">平均求解时间</span>
                <span id="solve-time" class="metric-value">-- ms</span>
            </div>
            <div class="metric">
                <span class="metric-label">跟踪误差</span>
                <span id="tracking-error" class="metric-value">-- m</span>
            </div>
            <div class="metric">
                <span class="metric-label">RMS误差</span>
                <span id="rms-error" class="metric-value">-- m</span>
            </div>
            <div class="metric">
                <span class="metric-label">控制量</span>
                <span id="control-effort" class="metric-value">-- m³/s</span>
            </div>
        </div>

        <!-- Alerts -->
        <div class="card">
            <h2>告警 (<span id="alert-count">0</span>)</h2>
            <div id="alerts-container">
                <div class="no-data">暂无告警</div>
            </div>
        </div>

        <!-- Events -->
        <div class="card">
            <h2>最近事件</h2>
            <div id="events-container" style="max-height: 200px; overflow-y: auto;">
                <div class="no-data">暂无事件</div>
            </div>
        </div>

        <!-- Controls -->
        <div class="card">
            <h2>控制面板</h2>
            <div class="controls">
                <button class="btn btn-primary" onclick="setMode('normal')">正常模式</button>
                <button class="btn btn-warning" onclick="setMode('flood_control')">防洪模式</button>
                <button class="btn btn-warning" onclick="setMode('drought')">抗旱模式</button>
                <button class="btn btn-danger" onclick="emergencyStop()">紧急停止</button>
            </div>
            <div style="margin-top: 1rem;">
                <h3 style="font-size: 0.9rem; color: #888; margin-bottom: 0.5rem;">设置参考水位</h3>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <input type="number" id="ref-level" value="3.0" step="0.1" min="0" max="10"
                           style="width: 80px; padding: 0.5rem; border-radius: 4px; border: 1px solid #0f3460; background: #0f3460; color: #fff;">
                    <span style="color: #888;">m</span>
                    <button class="btn btn-primary" onclick="setReference()">应用</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const REFRESH_INTERVAL = 5000;
        let updateTimer = null;

        async function fetchData(endpoint) {
            try {
                const response = await fetch('/api/' + endpoint);
                return await response.json();
            } catch (e) {
                console.error('Fetch error:', e);
                return null;
            }
        }

        async function postData(endpoint, data) {
            try {
                const response = await fetch('/api/' + endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                return await response.json();
            } catch (e) {
                console.error('Post error:', e);
                return null;
            }
        }

        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            return h + '时 ' + m + '分';
        }

        function formatTime(isoString) {
            const d = new Date(isoString);
            return d.toLocaleTimeString('zh-CN');
        }

        async function updateDashboard() {
            // Update status
            const status = await fetchData('status');
            if (status && !status.error) {
                document.getElementById('mode-badge').textContent = status.mode.toUpperCase();
                document.getElementById('mode-badge').className = 'status-badge status-' +
                    (status.mode === 'emergency' ? 'critical' : status.mode === 'normal' ? 'normal' : 'warning');

                document.getElementById('health-value').textContent = (status.health * 100).toFixed(1) + '%';
                document.getElementById('health-fill').style.width = (status.health * 100) + '%';
                document.getElementById('uptime').textContent = formatUptime(status.uptime_seconds);
                document.getElementById('control-active').textContent = status.control_active ? '是' : '否';

                // Update levels
                const levelsHtml = status.levels.map((l, i) =>
                    '<div class="level-item">' +
                    '<div class="level-value">' + l.toFixed(2) + '</div>' +
                    '<div class="level-label">渠池 ' + (i + 1) + '</div>' +
                    '</div>'
                ).join('');
                document.getElementById('levels-container').innerHTML = levelsHtml;
            }

            // Update metrics
            const metrics = await fetchData('metrics');
            if (metrics && !metrics.error) {
                document.getElementById('solve-time').textContent =
                    metrics.computational.avg_solve_time_ms.toFixed(1) + ' ms';
                document.getElementById('tracking-error').textContent =
                    metrics.control.avg_tracking_error.toFixed(4) + ' m';
                document.getElementById('rms-error').textContent =
                    metrics.control.rms_error.toFixed(4) + ' m';
                document.getElementById('control-effort').textContent =
                    metrics.control.avg_control_effort.toFixed(2) + ' m³/s';
            }

            // Update alerts
            const alerts = await fetchData('alerts');
            if (alerts) {
                document.getElementById('alert-count').textContent = alerts.count;
                if (alerts.count === 0) {
                    document.getElementById('alerts-container').innerHTML =
                        '<div class="no-data">暂无告警</div>';
                } else {
                    const alertsHtml = alerts.alerts.slice(0, 5).map(a =>
                        '<div class="alert-item alert-' + a.level + '">' +
                        '<div><strong>' + a.type + '</strong><br>' +
                        '<small>' + a.message + '</small></div>' +
                        '<button class="btn btn-primary" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;" ' +
                        'onclick="acknowledgeAlert(\\'' + a.id + '\\')">确认</button>' +
                        '</div>'
                    ).join('');
                    document.getElementById('alerts-container').innerHTML = alertsHtml;
                }
            }

            // Update events
            const events = await fetchData('events');
            if (events && events.events.length > 0) {
                const eventsHtml = events.events.slice(0, 10).map(e =>
                    '<div class="event-item">' +
                    '<span class="event-time">' + formatTime(e.timestamp) + '</span> ' +
                    '<strong>' + e.type + '</strong> - ' + e.description +
                    '</div>'
                ).join('');
                document.getElementById('events-container').innerHTML = eventsHtml;
            }

            // Update timestamp
            document.getElementById('update-time').textContent = '更新: ' + new Date().toLocaleTimeString('zh-CN');
        }

        async function setMode(mode) {
            const result = await postData('mode', {mode: mode});
            if (result && result.success) {
                alert('模式切换命令已提交: ' + mode);
                updateDashboard();
            } else {
                alert('命令失败: ' + (result ? result.error : '网络错误'));
            }
        }

        async function emergencyStop() {
            if (!confirm('确认执行紧急停止？')) return;
            const result = await postData('command', {type: 'emergency_stop', target: 'system'});
            if (result && result.success) {
                alert('紧急停止命令已提交');
                updateDashboard();
            }
        }

        async function setReference() {
            const level = parseFloat(document.getElementById('ref-level').value);
            const numPools = 3; // Would get from status
            const levels = Array(numPools).fill(level);
            const result = await postData('reference', {levels: levels});
            if (result && result.success) {
                alert('参考水位已设置为 ' + level + ' m');
            }
        }

        async function acknowledgeAlert(alertId) {
            const result = await postData('acknowledge', {alert_id: alertId});
            if (result && result.success) {
                updateDashboard();
            }
        }

        // Start updates
        updateDashboard();
        updateTimer = setInterval(updateDashboard, REFRESH_INTERVAL);
    </script>
</body>
</html>'''


class WebDashboard:
    """
    Web Dashboard for Water Network Control System.

    Provides:
    - Real-time status monitoring
    - Performance visualization
    - Alert management
    - Control interface
    """

    def __init__(
        self,
        system=None,
        config: Optional[DashboardConfig] = None
    ):
        self.config = config or DashboardConfig()
        self._system = system
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"[WebDashboard] 仪表板初始化 (端口: {self.config.port})")

    def set_system(self, system):
        """Set the control system instance"""
        self._system = system

    def start(self):
        """Start the web server"""
        if self._running:
            logger.warning("[WebDashboard] 服务器已在运行")
            return

        # Set class-level reference
        DashboardHandler.dashboard = self

        self._server = HTTPServer(
            (self.config.host, self.config.port),
            DashboardHandler
        )

        self._server_thread = threading.Thread(target=self._run_server)
        self._server_thread.daemon = True
        self._server_thread.start()

        self._running = True
        logger.info(f"[WebDashboard] 服务器启动于 http://{self.config.host}:{self.config.port}")

    def _run_server(self):
        """Run server loop"""
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.error(f"[WebDashboard] 服务器错误: {e}")

    def stop(self):
        """Stop the web server"""
        if not self._running:
            return

        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None

        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None

        logger.info("[WebDashboard] 服务器已停止")

    def get_url(self) -> str:
        """Get dashboard URL"""
        return f"http://{self.config.host}:{self.config.port}"

    def is_running(self) -> bool:
        """Check if server is running"""
        return self._running

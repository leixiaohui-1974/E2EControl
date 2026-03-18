"""
Tests for api_client.py (SmartPoolAPIClient and APIError).

Covers:
- Initialization and URL normalization
- HTTP request handling with mocked responses
- All public API methods (health_check, interpret, run_simulation, etc.)
- Error handling and edge cases
- Timeout / wait_for_completion logic
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open, call

from hydroe2e.api_client import SmartPoolAPIClient, APIError


class TestSmartPoolAPIClientInit(unittest.TestCase):
    """Tests for client initialization."""

    def test_default_base_url(self):
        client = SmartPoolAPIClient()
        self.assertEqual(client.base_url, "http://localhost:5000")

    def test_custom_base_url(self):
        client = SmartPoolAPIClient("http://example.com:8080")
        self.assertEqual(client.base_url, "http://example.com:8080")

    def test_trailing_slash_stripped(self):
        client = SmartPoolAPIClient("http://example.com/")
        self.assertEqual(client.base_url, "http://example.com")

    def test_multiple_trailing_slashes_stripped(self):
        client = SmartPoolAPIClient("http://example.com///")
        self.assertEqual(client.base_url, "http://example.com")

    def test_default_timeout(self):
        client = SmartPoolAPIClient()
        self.assertEqual(client.timeout, 30)

    def test_custom_timeout(self):
        client = SmartPoolAPIClient(timeout=60)
        self.assertEqual(client.timeout, 60)

    def test_session_created(self):
        import requests
        client = SmartPoolAPIClient()
        self.assertIsInstance(client.session, requests.Session)


class TestSmartPoolAPIClientRequest(unittest.TestCase):
    """Tests for the internal _request method."""

    def setUp(self):
        self.client = SmartPoolAPIClient("http://localhost:5000")

    def test_request_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(self.client.session, 'request', return_value=mock_response):
            result = self.client._request('GET', '/health')

        self.assertEqual(result, {"status": "ok"})

    def test_request_builds_correct_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch.object(self.client.session, 'request', return_value=mock_response) as mock_req:
            self.client._request('POST', '/interpret', json={"instruction": "test"})
            mock_req.assert_called_once_with(
                'POST', 'http://localhost:5000/interpret',
                timeout=30, json={"instruction": "test"}
            )

    def test_request_raises_api_error_on_connection_error(self):
        import requests
        with patch.object(self.client.session, 'request',
                          side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(APIError) as ctx:
                self.client._request('GET', '/health')
            self.assertIn("请求失败", str(ctx.exception))

    def test_request_raises_api_error_on_http_error(self):
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        with patch.object(self.client.session, 'request', return_value=mock_response):
            with self.assertRaises(APIError):
                self.client._request('GET', '/nonexistent')

    def test_request_raises_api_error_on_timeout(self):
        import requests
        with patch.object(self.client.session, 'request',
                          side_effect=requests.exceptions.Timeout("timed out")):
            with self.assertRaises(APIError):
                self.client._request('GET', '/slow')


class TestHealthCheck(unittest.TestCase):
    """Tests for health_check and is_healthy."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_health_check_returns_response(self):
        with patch.object(self.client, '_request', return_value={"status": "healthy"}):
            result = self.client.health_check()
            self.assertEqual(result["status"], "healthy")

    def test_is_healthy_returns_true(self):
        with patch.object(self.client, 'health_check', return_value={"status": "healthy"}):
            self.assertTrue(self.client.is_healthy())

    def test_is_healthy_returns_false_on_unhealthy(self):
        with patch.object(self.client, 'health_check', return_value={"status": "degraded"}):
            self.assertFalse(self.client.is_healthy())

    def test_is_healthy_returns_false_on_exception(self):
        with patch.object(self.client, 'health_check', side_effect=APIError("down")):
            self.assertFalse(self.client.is_healthy())

    def test_is_healthy_returns_false_on_missing_status(self):
        with patch.object(self.client, 'health_check', return_value={}):
            self.assertFalse(self.client.is_healthy())


class TestInterpret(unittest.TestCase):
    """Tests for the interpret method."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_interpret_sends_instruction(self):
        expected = {"instruction": "test", "confidence": 0.9, "config": {"Z_ref": 3.0}}
        with patch.object(self.client, '_request', return_value=expected) as mock_req:
            result = self.client.interpret("test")
            mock_req.assert_called_once_with('POST', '/interpret', json={"instruction": "test"})
            self.assertEqual(result["confidence"], 0.9)


class TestRunSimulation(unittest.TestCase):
    """Tests for run_simulation."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_run_simulation_sync(self):
        with patch.object(self.client, '_request',
                          return_value={"success": True, "simulation_id": 42}):
            sim_id = self.client.run_simulation([[0, "test"]], async_mode=False)
            self.assertEqual(sim_id, 42)

    def test_run_simulation_async(self):
        with patch.object(self.client, '_request',
                          return_value={"success": True, "simulation_id": 7}) as mock_req:
            sim_id = self.client.run_simulation([[0, "test"]], async_mode=True)
            self.assertEqual(sim_id, 7)
            mock_req.assert_called_once_with(
                'POST', '/simulation/run',
                json={'script': [[0, "test"]], 'async': True}
            )

    def test_run_simulation_failure_raises_api_error(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "bad script"}):
            with self.assertRaises(APIError) as ctx:
                self.client.run_simulation([[0, "bad"]])
            self.assertIn("仿真启动失败", str(ctx.exception))


class TestGetHistory(unittest.TestCase):
    """Tests for get_history."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_get_history_success(self):
        history_data = [{"time": 0, "level": 3.0}, {"time": 1, "level": 3.1}]
        with patch.object(self.client, '_request',
                          return_value={"success": True, "history": history_data}):
            result = self.client.get_history(1)
            self.assertEqual(len(result), 2)

    def test_get_history_failure(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "not found"}):
            with self.assertRaises(APIError):
                self.client.get_history(999)


class TestGetAlerts(unittest.TestCase):
    """Tests for get_alerts."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_get_alerts_success(self):
        alerts = [{"level": "WARNING", "message": "high water"}]
        with patch.object(self.client, '_request',
                          return_value={"success": True, "alerts": alerts}):
            result = self.client.get_alerts(1)
            self.assertEqual(len(result), 1)

    def test_get_alerts_failure(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "sim not found"}):
            with self.assertRaises(APIError):
                self.client.get_alerts(999)


class TestGetReport(unittest.TestCase):
    """Tests for get_report."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_get_report_success(self):
        with patch.object(self.client, '_request',
                          return_value={"success": True, "report": "# Report\nOK"}):
            result = self.client.get_report(1)
            self.assertEqual(result, "# Report\nOK")

    def test_get_report_failure(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "no report"}):
            with self.assertRaises(APIError):
                self.client.get_report(1)


class TestDownloadResultImage(unittest.TestCase):
    """Tests for download_result_image."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_download_success(self):
        mock_response = MagicMock()
        mock_response.content = b'\x89PNG_FAKE_DATA'
        mock_response.raise_for_status = MagicMock()

        with patch.object(self.client.session, 'get', return_value=mock_response):
            with patch('builtins.open', mock_open()) as m:
                self.client.download_result_image(1, '/tmp/result.png')
                m.assert_called_once_with('/tmp/result.png', 'wb')
                m().write.assert_called_once_with(b'\x89PNG_FAKE_DATA')

    def test_download_network_error(self):
        import requests
        with patch.object(self.client.session, 'get',
                          side_effect=requests.exceptions.ConnectionError("fail")):
            with self.assertRaises(APIError):
                self.client.download_result_image(1, '/tmp/result.png')


class TestListSimulations(unittest.TestCase):
    """Tests for list_simulations."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_list_simulations_success(self):
        sims = [{"id": 1}, {"id": 2}]
        with patch.object(self.client, '_request',
                          return_value={"success": True, "simulations": sims}):
            result = self.client.list_simulations()
            self.assertEqual(len(result), 2)

    def test_list_simulations_failure(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "db error"}):
            with self.assertRaises(APIError):
                self.client.list_simulations()


class TestListScenarios(unittest.TestCase):
    """Tests for list_scenarios."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_list_scenarios_success(self):
        scenarios = [{"name": "flood"}, {"name": "drought"}]
        with patch.object(self.client, '_request',
                          return_value={"success": True, "scenarios": scenarios}):
            result = self.client.list_scenarios()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "flood")

    def test_list_scenarios_failure(self):
        with patch.object(self.client, '_request',
                          return_value={"success": False, "error": "n/a"}):
            with self.assertRaises(APIError):
                self.client.list_scenarios()


class TestWaitForCompletion(unittest.TestCase):
    """Tests for wait_for_completion."""

    def setUp(self):
        self.client = SmartPoolAPIClient()

    def test_immediate_completion(self):
        with patch.object(self.client, 'get_status',
                          return_value={"status": "completed", "result": "ok"}):
            result = self.client.wait_for_completion(1)
            self.assertEqual(result["status"], "completed")

    def test_completion_after_polling(self):
        statuses = [
            {"status": "running"},
            {"status": "running"},
            {"status": "completed"},
        ]
        with patch.object(self.client, 'get_status', side_effect=statuses):
            with patch('hydroe2e.api_client.time.sleep'):
                result = self.client.wait_for_completion(1, interval=1)
                self.assertEqual(result["status"], "completed")

    def test_timeout_raises_api_error(self):
        with patch.object(self.client, 'get_status',
                          return_value={"status": "running"}):
            with patch('hydroe2e.api_client.time.sleep'):
                with patch('hydroe2e.api_client.time.time', side_effect=[0, 0, 1000]):
                    with self.assertRaises(APIError) as ctx:
                        self.client.wait_for_completion(1, max_wait=10)
                    self.assertIn("等待超时", str(ctx.exception))

    def test_callback_invoked(self):
        statuses = [
            {"status": "running"},
            {"status": "completed"},
        ]
        callback = MagicMock()
        with patch.object(self.client, 'get_status', side_effect=statuses):
            with patch('hydroe2e.api_client.time.sleep'):
                self.client.wait_for_completion(1, interval=1, callback=callback)
        self.assertEqual(callback.call_count, 2)

    def test_failed_status_returns_immediately(self):
        with patch.object(self.client, 'get_status',
                          return_value={"status": "failed", "error": "crash"}):
            result = self.client.wait_for_completion(1)
            self.assertEqual(result["status"], "failed")


class TestClose(unittest.TestCase):
    """Tests for session close."""

    def test_close_calls_session_close(self):
        client = SmartPoolAPIClient()
        with patch.object(client.session, 'close') as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestAPIError(unittest.TestCase):
    """Tests for the APIError exception."""

    def test_api_error_is_exception(self):
        self.assertTrue(issubclass(APIError, Exception))

    def test_api_error_message(self):
        err = APIError("something went wrong")
        self.assertEqual(str(err), "something went wrong")


if __name__ == '__main__':
    unittest.main()

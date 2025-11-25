import unittest
import asyncio
import numpy as np
import requests
from playwright.async_api import async_playwright
import os
import time

# Helper function to check if the API server is ready
def is_server_ready(url, retries=5, delay=2):
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("API server is ready.")
                return True
        except requests.ConnectionError:
            print(f"Waiting for API server... (Attempt {i+1}/{retries})")
            time.sleep(delay)
    print("API server did not start in time.")
    return False

class APITests(unittest.TestCase):
    """API-level tests for the Smart Pool Agent."""
    BASE_URL = "http://127.0.0.1:5000"

    def test_a_single_pool_simulation(self):
        """Test running a single-pool simulation via API."""
        print("\n--- Testing API: Single-Pool Simulation ---")
        response = requests.post(
            f"{self.BASE_URL}/simulation/run",
            json={'instruction': '保持水位平稳', 'system_type': 'single'}
        )
        self.assertEqual(response.status_code, 202)
        sim_id = response.json()['simulation_id']

        # Poll until complete
        for _ in range(10):
            time.sleep(2)
            res = requests.get(f"{self.BASE_URL}/simulation/{sim_id}/history").json()
            if res['status'] == 'completed':
                self.assertIn('levels', res['history'])
                self.assertEqual(len(res['history']['levels']), 1)
                print("API Test for single-pool simulation PASSED.")
                return
        self.fail("Simulation did not complete in time.")

    def test_b_multi_pool_simulation(self):
        """Test running a multi-pool (cascaded) simulation via API."""
        print("\n--- Testing API: Multi-Pool Simulation ---")
        response = requests.post(
            f"{self.BASE_URL}/simulation/run",
            json={'instruction': '保持水位平稳', 'system_type': 'cascaded'}
        )
        self.assertEqual(response.status_code, 202)
        sim_id = response.json()['simulation_id']

        # Poll until complete
        for _ in range(10):
            time.sleep(2)
            res = requests.get(f"{self.BASE_URL}/simulation/{sim_id}/history").json()
            if res['status'] == 'completed':
                self.assertIn('levels', res['history'])
                self.assertEqual(len(res['history']['levels']), 3)
                print("API Test for multi-pool simulation PASSED.")
                return
        self.fail("Simulation did not complete in time.")


class WebUITests(unittest.TestCase):
    """Web UI tests using Playwright."""
    
    @classmethod
    def setUpClass(cls):
        os.makedirs('web_test_results', exist_ok=True)
        # We need to run this in a separate process or manage the event loop carefully
        # For simplicity, we'll use the async_to_sync wrapper if available or run it directly
        asyncio.run(cls.run_playwright_tests())

    @staticmethod
    async def run_playwright_tests():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            try:
                print("\n--- Running Web E2E Test for Verification ---")
                await page.goto("http://127.0.0.1:5000/")

                # Scenario 1: Run a simulation
                await page.click('.nav-link[data-view="view-simulation"]')
                await page.fill('#instruction-input', '运行一个多池级联仿真')
                await page.click('#run-sim-btn')
                await page.wait_for_selector('#run-sim-btn:not([disabled])', timeout=60000)
                print("Web test simulation complete.")
                await page.screenshot(path='web_test_results/web_simulation_result.png')

                # Scenario 2: Check History View
                await page.click('.nav-link[data-view="view-history"]')
                await page.wait_for_selector('#history-table tbody tr', timeout=5000)
                print("Web test history view loaded.")
                await page.screenshot(path='web_test_results/web_history_view.png')

            except Exception as e:
                await page.screenshot(path='web_test_results/error.png')
                raise e

            finally:
                await browser.close()

    def test_ui_scenarios(self):
        """Placeholder test that is effectively run by setUpClass."""
        # This test case will "pass" if the setUpClass method doesn't raise an exception.
        print("\n--- Web UI tests completed successfully (via setUpClass) ---")
        self.assertTrue(os.path.exists('web_test_results/web_simulation_result.png'))
        self.assertTrue(os.path.exists('web_test_results/web_history_view.png'))


if __name__ == '__main__':
    # Ensure server is ready before running tests
    if is_server_ready("http://127.0.0.1:5000/api"):
        # Run tests
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(APITests))
        suite.addTest(unittest.makeSuite(WebUITests))

        runner = unittest.TextTestRunner()
        result = runner.run(suite)

        # Exit with a non-zero status code if tests failed
        if not result.wasSuccessful():
            exit(1)
    else:
        exit(1)

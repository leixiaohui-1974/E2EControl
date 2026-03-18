import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def log(msg):
    print(f"[DEBUG] {msg}")

def test_frontend():
    log("Testing Frontend Serving...")
    try:
        res = requests.get(f"{BASE_URL}/", timeout=10)
        log(f"Index Page Status: {res.status_code}")
        if res.status_code == 200:
            log("Frontend is accessible.")
        else:
            log(f"Frontend failed: {res.text[:100]}")
    except Exception as e:
        log(f"Frontend Exception: {e}")

def test_single_pool_flow():
    log("Testing Single Pool Simulation Flow...")
    
    # 1. Run Simulation
    payload = {
        "script": None,
        "async": True,
        "system_type": "single"
    }
    try:
        res = requests.post(f"{BASE_URL}/simulation/run", json=payload, timeout=30)
        log(f"Run response: {res.status_code} - {res.text}")
        if res.status_code != 202:
            log("FAILED: Could not start simulation")
            return

        data = res.json()
        sim_id = data.get('simulation_id')
        log(f"Simulation ID: {sim_id}")

        # 2. Poll History
        for i in range(5):
            time.sleep(1)
            res_hist = requests.get(f"{BASE_URL}/simulation/{sim_id}/history", timeout=10)
            log(f"Poll {i+1}: History Status {res_hist.status_code}")

            if res_hist.status_code == 200:
                data = res_hist.json()
                log(f"  > Status: {data.get('status')}")
                log(f"  > Data Points: {data.get('count')}")

                if data.get('count', 0) > 0:
                    history = data.get('history', {})
                    levels = history.get('level', [])
                    log(f"  > Latest Level: {levels[-1] if levels else 'None'}")
                else:
                    log("  > WARNING: History is empty!")
            else:
                log(f"  > Error: {res_hist.text}")

        # 3. Stop
        log("Stopping simulation...")
        requests.post(f"{BASE_URL}/simulation/{sim_id}/stop", timeout=10)
        
    except Exception as e:
        log(f"EXCEPTION: {e}")

def test_cascaded_flow():
    log("\nTesting Cascaded Simulation Flow...")
    
    # 1. Run Simulation
    payload = {
        "script": None,
        "async": True,
        "system_type": "cascaded"
    }
    try:
        res = requests.post(f"{BASE_URL}/simulation/run", json=payload, timeout=30)
        log(f"Run response: {res.status_code} - {res.text}")
        if res.status_code != 202:
            log("FAILED: Could not start simulation")
            return

        data = res.json()
        sim_id = data.get('simulation_id')
        log(f"Simulation ID: {sim_id}")

        # 2. Poll History
        for i in range(5):
            time.sleep(1)
            res_hist = requests.get(f"{BASE_URL}/simulation/{sim_id}/history", timeout=10)
            log(f"Poll {i+1}: History Status {res_hist.status_code}")

            if res_hist.status_code == 200:
                data = res_hist.json()
                log(f"  > Status: {data.get('status')}")
                log(f"  > Data Points: {data.get('count')}")

                if data.get('count', 0) > 0:
                    history = data.get('history', {})
                    levels = history.get('levels', [])
                    flows = history.get('flows', [])
                    log(f"  > Latest Levels: {levels[-1] if levels else 'None'}")
                    log(f"  > Latest Flows: {flows[-1] if flows else 'None'}")
                else:
                    log("  > WARNING: History is empty!")
            else:
                log(f"  > Error: {res_hist.text}")

        # 3. Stop
        log("Stopping simulation...")
        requests.post(f"{BASE_URL}/simulation/{sim_id}/stop", timeout=10)

    except Exception as e:
        log(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_frontend()
    test_single_pool_flow()
    test_cascaded_flow()

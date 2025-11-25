import requests
import json
import time

url = "http://localhost:5000/simulation/run"
payload = {
    "async": True,
    "system_type": "cascaded"
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)
    
    data = response.json()
    if data.get('success'):
        sim_id = data['simulation_id']
        print(f"Simulation started with ID: {sim_id}")
        
        # Poll status
        status_url = f"http://localhost:5000/simulation/{sim_id}/status"
        while True:
            res = requests.get(status_url)
            status_data = res.json()
            print(f"Status: {status_data.get('status')}")
            if status_data.get('status') in ['completed', 'failed']:
                break
            time.sleep(1)
            
        # Get History
        if status_data.get('status') == 'completed':
            hist_url = f"http://localhost:5000/simulation/{sim_id}/history"
            hist_res = requests.get(hist_url)
            hist_data = hist_res.json()
            print("History retrieved successfully.")
            
            if 'scenarios' in hist_data['history']:
                print("Scenario Recognition Data Found:")
                print(json.dumps(hist_data['history']['scenarios'][0], indent=2))
                print("...")
                print(json.dumps(hist_data['history']['scenarios'][-1], indent=2))
            else:
                print("WARNING: No scenario data found in history.")
            
except Exception as e:
    print(f"Error: {e}")

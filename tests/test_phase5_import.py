
import sys
import os

try:
    from hydroe2e.phase5.integrated_system import IntegratedWaterNetworkSystem
    print("Successfully imported IntegratedWaterNetworkSystem")
    
    system = IntegratedWaterNetworkSystem(
        num_pools=3,
        enable_digital_twin=False,
        enable_self_healing=True,
        enable_anomaly_detection=True
    )
    print("Successfully instantiated IntegratedWaterNetworkSystem")
    
    status = system.get_system_status()
    print("System Status:", status)
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()


from physics import CanalPoolSimulator

def test_physics_simulator():
    """测试物理仿真器"""
    print("Running Physics Simulator Test...")
    pool = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0)
    
    # 测试场景1: 正常运行 (In > Out)
    print("\nScenario 1: Inflow (5.0) > Outflow (4.5)")
    for i in range(10):
        level = pool.step(q_in_command=5.0, q_out=4.5, disturbance=0.0)
        print(f"  Step {i+1}: Level={level:.4f}")
    
    scenario1_ok = 3.0 < level < 4.0
    print(f"  Result: Z={level:.4f}m (Expected 3.0 < Z < 4.0) -> {'✓' if scenario1_ok else '✗'}")
    
    # 测试场景2: 入流小于出流 (In < Out)
    print("\nScenario 2: Inflow (3.0) < Outflow (4.0)")
    for i in range(10):
        level = pool.step(q_in_command=3.0, q_out=4.0, disturbance=0.0)
        print(f"  Step {i+1}: Level={level:.4f}")
    
    scenario2_ok = 2.0 < level < 3.5
    print(f"  Result: Z={level:.4f}m (Expected 2.0 < Z < 3.5) -> {'✓' if scenario2_ok else '✗'}")
    
    # 测试场景3: 平衡状态 (In = Out)
    print("\nScenario 3: Balanced (In=4.0, Out=4.0)")
    pool2 = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0)
    for i in range(20):
        level = pool2.step(q_in_command=4.0, q_out=4.0, disturbance=0.0)
        print(f"  Step {i+1}: Level={level:.4f}")
    
    scenario3_ok = abs(level - 3.0) < 0.5
    print(f"  Result: Z={level:.4f}m (Expected |Z-3.0| < 0.5) -> {'✓' if scenario3_ok else '✗'}")

if __name__ == "__main__":
    test_physics_simulator()


from control import UniversalMPCSolver

def test_mpc_solver():
    """测试MPC求解器"""
    print("Running MPC Solver Test...")
    solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)
    
    config = {
        'Z_ref': 3.0,
        'W_level': 10.0,
        'W_smooth': 5.0,
        'delta_Q_max': 2.0,
        'constraints': {}
    }
    
    # 测试不同场景
    test_cases = [
        (3.0, "正常水位"),
        (1.0, "低水位"),
        (5.0, "高水位"),
        (9.0, "极限高水位")
    ]
    
    success_count = 0
    for level, desc in test_cases:
        try:
            u_in = solver.solve(
                current_level=level,
                q_prev=0.0,
                q_out_forecast=[3.0]*10,
                config=config
            )
            if 0 <= u_in <= 20:  # 合理范围
                success_count += 1
                print(f"    ✓ {desc} (Z={level}m) -> u_in={u_in:.2f} m³/s")
            else:
                print(f"    ⚠ {desc} (Z={level}m) -> u_in={u_in:.2f} (异常)")
        except Exception as e:
            print(f"    ✗ {desc} (Z={level}m) 求解失败: {e}")
    
    success_rate = success_count / len(test_cases)
    print(f"  MPC求解成功率: {success_rate*100:.1f}%")
    
    if success_rate >= 0.75:
        print("MPC Test Passed!")
    else:
        print("MPC Test Failed!")

if __name__ == "__main__":
    test_mpc_solver()

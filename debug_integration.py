
from phase5.integrated_system import IntegratedWaterNetworkSystem

def test_integrated_system():
    """测试集成系统"""
    print("Running Integration Test...")
    try:
        # 创建系统（禁用可选模块加快测试）
        system = IntegratedWaterNetworkSystem(
            num_pools=2,  # 减少池数加快测试
            enable_digital_twin=False,
            enable_self_healing=False,
            enable_anomaly_detection=False
        )
        
        # 简单场景
        scenario_script = [
            (0, "保持水位平稳，正常供水。"),
            (5, "收到暴雨预警，立刻降低水位腾出库容！安全第一！")
        ]
        
        # 运行短仿真
        print("Starting simulation...")
        history = system.run_simulation(
            scenario_script=scenario_script,
            total_steps=10,
            enable_faults=False
        )
        
        # 检查结果
        has_history = len(history['time']) > 0
        has_levels = len(history['levels'][0]) > 0
        
        print(f"    仿真步数: {len(history['time'])}")
        print(f"    数据完整: {'✓' if has_history and has_levels else '✗'}")
        
        if has_history and has_levels:
            print("Integration Test Passed!")
        else:
            print("Integration Test Failed: Missing data")
            
    except Exception as e:
        print(f"    ✗ 集成测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integrated_system()

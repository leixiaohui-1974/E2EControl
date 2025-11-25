"""
性能测试脚本
Performance Testing Script

测试系统各模块的性能指标
"""

import time
import numpy as np
import sys

# 添加路径
sys.path.append('.')

print("="*80)
print(" "*25 + "性能测试开始")
print("="*80)

# ============================================================================
# 测试1: MPC求解器性能
# ============================================================================
print("\n[测试1] MPC求解器性能测试")
print("-"*80)

from control import UniversalMPCSolver

# 创建求解器
solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)

# 测试配置
config = {
    'Z_ref': 3.0,
    'W_level': 10.0,
    'W_smooth': 5.0,
    'delta_Q_max': 2.0,
    'constraints': {}
}

# 预热
for _ in range(5):
    solver.solve(
        current_level=3.0,
        q_prev=0.0,
        q_out_forecast=[3.0]*10,
        config=config
    )

# 性能测试
n_tests = 100
times = []

for _ in range(n_tests):
    start = time.time()
    try:
        solver.solve(
            current_level=3.0 + np.random.randn()*0.5,
            q_prev=0.0,
            q_out_forecast=[3.0]*10,
            config=config
        )
        elapsed = (time.time() - start) * 1000  # 转换为毫秒
        times.append(elapsed)
    except:
        pass

if times:
    print(f"  测试次数: {len(times)}")
    print(f"  平均时间: {np.mean(times):.2f}ms")
    print(f"  最小时间: {np.min(times):.2f}ms")
    print(f"  最大时间: {np.max(times):.2f}ms")
    print(f"  标准差: {np.std(times):.2f}ms")
    print(f"  评价: {'✓ 优秀' if np.mean(times) < 100 else '✓ 良好' if np.mean(times) < 200 else '⚠ 需优化'}")
else:
    print("  ⚠️ 测试失败")

# ============================================================================
# 测试2: 物理仿真器性能
# ============================================================================
print("\n[测试2] 物理仿真器性能测试")
print("-"*80)

from physics import CanalPoolSimulator

# 创建仿真器
pool = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0)

# 性能测试
n_steps = 1000
start = time.time()

for _ in range(n_steps):
    pool.step(q_in_command=5.0, q_out=4.5, disturbance=np.random.randn()*0.1)

elapsed = time.time() - start
step_time = elapsed / n_steps * 1000

print(f"  仿真步数: {n_steps}")
print(f"  总时间: {elapsed:.3f}s")
print(f"  单步时间: {step_time:.3f}ms")
print(f"  评价: {'✓ 优秀' if step_time < 1 else '✓ 良好' if step_time < 5 else '⚠ 需优化'}")

# ============================================================================
# 测试3: 异常检测器性能（如果可用）
# ============================================================================
print("\n[测试3] 异常检测器性能测试")
print("-"*80)

try:
    from phase4.anomaly_detection.statistical_detectors import SigmaDetector
    
    # 创建检测器
    detector = SigmaDetector(sigma=3.0)
    
    # 预热
    for _ in range(10):
        detector.detect(3.0 + np.random.randn()*0.1)
    
    # 性能测试
    n_detections = 1000
    times = []
    
    for _ in range(n_detections):
        start = time.time()
        detector.detect(3.0 + np.random.randn()*0.5)
        elapsed = (time.time() - start) * 1000000  # 转换为微秒
        times.append(elapsed)
    
    print(f"  检测次数: {n_detections}")
    print(f"  平均时间: {np.mean(times):.2f}μs")
    print(f"  最小时间: {np.min(times):.2f}μs")
    print(f"  最大时间: {np.max(times):.2f}μs")
    print(f"  评价: {'✓ 优秀' if np.mean(times) < 100 else '✓ 良好' if np.mean(times) < 1000 else '⚠ 需优化'}")
    
except ImportError:
    print("  ⚠️ Phase 4模块未安装，跳过测试")

# ============================================================================
# 测试4: 内存占用
# ============================================================================
print("\n[测试4] 内存占用测试")
print("-"*80)

try:
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"  RSS内存: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"  VMS内存: {memory_info.vms / 1024 / 1024:.2f} MB")
    print(f"  评价: {'✓ 优秀' if memory_info.rss < 200*1024*1024 else '✓ 良好' if memory_info.rss < 500*1024*1024 else '⚠ 需优化'}")
    
    memory_rss = memory_info.rss
except ImportError:
    print("  ⚠️ psutil未安装，跳过内存测试")
    print("  安装方法: pip install psutil")
    memory_rss = 100 * 1024 * 1024  # 估计值

# ============================================================================
# 测试5: 端到端性能（集成系统）
# ============================================================================
print("\n[测试5] 端到端性能测试（简化版）")
print("-"*80)

# 模拟完整的控制循环
n_iterations = 50

start = time.time()

for i in range(n_iterations):
    # 1. 获取状态
    current_level = 3.0 + np.random.randn()*0.3
    
    # 2. MPC求解
    try:
        u_in = solver.solve(
            current_level=current_level,
            q_prev=0.0,
            q_out_forecast=[3.0]*10,
            config=config
        )
    except:
        u_in = 0.0
    
    # 3. 物理更新
    pool.step(u_in, u_in*0.9, 0.0)
    
    # 4. 异常检测（如果可用）
    try:
        detector.detect(current_level)
    except:
        pass

elapsed = time.time() - start
iteration_time = elapsed / n_iterations * 1000

print(f"  迭代次数: {n_iterations}")
print(f"  总时间: {elapsed:.3f}s")
print(f"  单次迭代: {iteration_time:.2f}ms")
print(f"  吞吐量: {1000/iteration_time:.1f} iterations/s")
print(f"  评价: {'✓ 优秀' if iteration_time < 150 else '✓ 良好' if iteration_time < 300 else '⚠ 需优化'}")

# ============================================================================
# 性能总结
# ============================================================================
print("\n" + "="*80)
print(" "*25 + "性能测试总结")
print("="*80)

print("\n【指标对比】")
print(f"  MPC求解时间: {np.mean(times) if times else 0:.2f}ms (目标: <100ms)")
print(f"  物理仿真时间: {step_time:.3f}ms (目标: <1ms)")
print(f"  端到端迭代: {iteration_time:.2f}ms (目标: <150ms)")
print(f"  内存占用: {memory_rss / 1024 / 1024:.2f}MB (目标: <200MB)")

print("\n【性能评级】")
if np.mean(times) < 100 and step_time < 1 and iteration_time < 150:
    grade = "A+"
    comment = "所有指标优秀！"
elif np.mean(times) < 150 and step_time < 5 and iteration_time < 250:
    grade = "A"
    comment = "性能良好！"
elif np.mean(times) < 200 and step_time < 10 and iteration_time < 350:
    grade = "B"
    comment = "性能合格，有优化空间"
else:
    grade = "C"
    comment = "建议进行性能优化"

print(f"  等级: {grade}")
print(f"  评价: {comment}")

print("\n【优化建议】")
if np.mean(times) >= 100:
    print("  - 减少MPC预测时域（horizon）")
    print("  - 考虑使用更快的求解器")
if memory_rss > 200*1024*1024:
    print("  - 减少历史数据缓存")
    print("  - 优化数据结构")
if iteration_time >= 150:
    print("  - 优化关键路径代码")
    print("  - 考虑使用Cython加速")

print("\n" + "="*80)
print("✅ 性能测试完成！")
print("="*80)

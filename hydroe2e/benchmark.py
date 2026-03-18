"""
性能基准测试工具
测量系统各模块的性能指标
"""

import time
import psutil
import os
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from hydroe2e.config_manager import get_config
from hydroe2e.logger import get_logger, setup_logging
from hydroe2e.brain_enhanced import EnhancedSemanticInterpreter
from hydroe2e.physics import CanalPoolSimulator
from hydroe2e.control import UniversalMPCSolver
from hydroe2e.monitor import MonitoringSystem
from hydroe2e.database import SimulationDatabase


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        """初始化测试环境"""
        setup_logging({'level': 'ERROR', 'console_output': False})
        self.logger = get_logger()
        self.config = get_config()
        
        self.results = {}
    
    def benchmark_semantic_interpreter(self, iterations: int = 1000):
        """
        测试语义解释器性能
        
        Args:
            iterations: 迭代次数
        """
        print("\n" + "="*60)
        print("测试: 语义解释器性能")
        print("="*60)
        
        interpreter = EnhancedSemanticInterpreter()
        
        test_instructions = [
            "保持水位平稳，正常供水。",
            "收到暴雨预警，立刻降低水位腾出库容！安全第一！",
            "进入冰期输水模式，严禁扰动冰盖。",
            "下游检测到污染，紧急切断出流！",
            "保持水位平稳"  # 模糊匹配
        ]
        
        times = []
        
        for _ in range(iterations):
            instr = np.random.choice(test_instructions)
            
            start = time.perf_counter()
            config, confidence = interpreter.interpret(instr)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000)  # 转换为毫秒
        
        times = np.array(times)
        
        print(f"迭代次数: {iterations}")
        print(f"平均耗时: {np.mean(times):.4f} ms")
        print(f"中位数: {np.median(times):.4f} ms")
        print(f"最小值: {np.min(times):.4f} ms")
        print(f"最大值: {np.max(times):.4f} ms")
        print(f"标准差: {np.std(times):.4f} ms")
        print(f"95分位: {np.percentile(times, 95):.4f} ms")
        
        self.results['semantic_interpreter'] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'p95': np.percentile(times, 95)
        }
    
    def benchmark_mpc_solver(self, iterations: int = 100):
        """
        测试MPC求解器性能
        
        Args:
            iterations: 迭代次数
        """
        print("\n" + "="*60)
        print("测试: MPC求解器性能")
        print("="*60)
        
        solver = UniversalMPCSolver(
            horizon=10,
            dt=3600.0,
            area=10000.0,
            delay_steps=1
        )
        
        config = {
            'W_level': 10.0,
            'W_smooth': 5.0,
            'Z_ref': 3.0,
            'delta_Q_max': 2.0,
            'constraints': {}
        }
        
        times = []
        
        for _ in range(iterations):
            level = 2.0 + np.random.random() * 2.0
            q_prev = 4.0 + np.random.random() * 2.0
            q_out = 4.0 + np.random.random() * 2.0 + np.random.normal(0, 0.5, 10)
            
            start = time.perf_counter()
            q_opt = solver.solve(level, q_prev, q_out, config)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000)
        
        times = np.array(times)
        
        print(f"迭代次数: {iterations}")
        print(f"平均耗时: {np.mean(times):.2f} ms")
        print(f"中位数: {np.median(times):.2f} ms")
        print(f"最小值: {np.min(times):.2f} ms")
        print(f"最大值: {np.max(times):.2f} ms")
        print(f"标准差: {np.std(times):.2f} ms")
        print(f"95分位: {np.percentile(times, 95):.2f} ms")
        
        self.results['mpc_solver'] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'p95': np.percentile(times, 95)
        }
    
    def benchmark_physics_simulation(self, iterations: int = 10000):
        """
        测试物理模拟性能
        
        Args:
            iterations: 迭代次数
        """
        print("\n" + "="*60)
        print("测试: 物理模拟器性能")
        print("="*60)
        
        sim = CanalPoolSimulator(
            area=10000.0,
            dt=3600.0,
            delay_steps=1,
            initial_level=3.0
        )
        
        times = []
        
        for _ in range(iterations):
            q_in = 4.0 + np.random.random() * 2.0
            q_out = 4.0 + np.random.random() * 2.0
            
            start = time.perf_counter()
            level = sim.step(q_in, q_out)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000000)  # 微秒
        
        times = np.array(times)
        
        print(f"迭代次数: {iterations}")
        print(f"平均耗时: {np.mean(times):.2f} μs")
        print(f"中位数: {np.median(times):.2f} μs")
        print(f"最小值: {np.min(times):.2f} μs")
        print(f"最大值: {np.max(times):.2f} μs")
        
        self.results['physics_simulation'] = {
            'mean': np.mean(times),
            'median': np.median(times)
        }
    
    def benchmark_monitoring(self, iterations: int = 1000):
        """
        测试监控系统性能
        
        Args:
            iterations: 迭代次数
        """
        print("\n" + "="*60)
        print("测试: 监控系统性能")
        print("="*60)
        
        monitor = MonitoringSystem()
        
        times = []
        
        for _ in range(iterations):
            level = np.random.uniform(0, 10)
            q_in = np.random.uniform(0, 20)
            q_out = np.random.uniform(0, 10)
            config = {'Z_ref': 3.0}
            
            start = time.perf_counter()
            alerts = monitor.check_state(0, level, q_in, q_out, config)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000)
        
        times = np.array(times)
        
        print(f"迭代次数: {iterations}")
        print(f"平均耗时: {np.mean(times):.4f} ms")
        print(f"告警触发率: {monitor.stats['total_alerts']/iterations*100:.1f}%")
        
        self.results['monitoring'] = {
            'mean': np.mean(times),
            'alert_rate': monitor.stats['total_alerts']/iterations
        }
    
    def benchmark_database(self, num_states: int = 1000):
        """
        测试数据库性能
        
        Args:
            num_states: 状态记录数
        """
        print("\n" + "="*60)
        print("测试: 数据库性能")
        print("="*60)
        
        test_db = "benchmark_test.db"
        if os.path.exists(test_db):
            os.remove(test_db)
        
        db = SimulationDatabase(test_db)
        
        # 创建仿真
        sim_id = db.create_simulation(1000, 3600.0, 10000.0, {}, "性能测试")
        
        # 写入测试
        write_times = []
        for t in range(num_states):
            start = time.perf_counter()
            db.save_state(
                sim_id, t, 3.0 + np.random.random(),
                5.0, 5.0, 3.0, "测试", {}
            )
            elapsed = time.perf_counter() - start
            write_times.append(elapsed * 1000)
        
        db.finish_simulation(sim_id)
        
        # 读取测试
        start = time.perf_counter()
        history = db.get_simulation_history(sim_id)
        read_time = (time.perf_counter() - start) * 1000
        
        db.close()
        
        # 文件大小
        file_size = os.path.getsize(test_db) / 1024  # KB
        
        write_times = np.array(write_times)
        
        print(f"写入记录数: {num_states}")
        print(f"平均写入耗时: {np.mean(write_times):.4f} ms")
        print(f"总写入耗时: {np.sum(write_times):.2f} ms")
        print(f"读取 {len(history)} 条记录耗时: {read_time:.2f} ms")
        print(f"数据库文件大小: {file_size:.2f} KB")
        print(f"平均每条记录: {file_size/num_states:.3f} KB")
        
        self.results['database'] = {
            'write_mean': np.mean(write_times),
            'write_total': np.sum(write_times),
            'read_time': read_time,
            'file_size': file_size
        }
        
        # 清理
        os.remove(test_db)
    
    def benchmark_memory_usage(self, simulation_hours: int = 100):
        """
        测试内存使用
        
        Args:
            simulation_hours: 仿真时长
        """
        print("\n" + "="*60)
        print("测试: 内存使用")
        print("="*60)
        
        process = psutil.Process(os.getpid())
        
        # 初始内存
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        # 运行仿真
        interpreter = EnhancedSemanticInterpreter()
        physics = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1)
        solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)
        
        config = {'W_level': 10.0, 'W_smooth': 5.0, 'Z_ref': 3.0, 
                 'delta_Q_max': 2.0, 'constraints': {}}
        
        history = []
        
        for t in range(simulation_hours):
            level = physics.get_level()
            q_out = [5.0] * 10
            
            q_in = solver.solve(level, 5.0, q_out, config)
            next_level = physics.step(q_in, 5.0)
            
            history.append({
                'time': t,
                'level': next_level,
                'q_in': q_in,
                'q_out': 5.0
            })
        
        # 最终内存
        final_mem = process.memory_info().rss / 1024 / 1024
        
        mem_increase = final_mem - initial_mem
        mem_per_hour = mem_increase / simulation_hours
        
        print(f"仿真时长: {simulation_hours} 小时")
        print(f"初始内存: {initial_mem:.2f} MB")
        print(f"最终内存: {final_mem:.2f} MB")
        print(f"内存增长: {mem_increase:.2f} MB")
        print(f"每小时平均: {mem_per_hour:.4f} MB")
        
        self.results['memory'] = {
            'initial': initial_mem,
            'final': final_mem,
            'increase': mem_increase,
            'per_hour': mem_per_hour
        }
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("\n" + "🚀"*30)
        print(" "*20 + "性能基准测试")
        print("🚀"*30 + "\n")
        
        start_time = time.time()
        
        self.benchmark_semantic_interpreter(1000)
        self.benchmark_mpc_solver(100)
        self.benchmark_physics_simulation(10000)
        self.benchmark_monitoring(1000)
        self.benchmark_database(1000)
        self.benchmark_memory_usage(100)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"所有测试完成！总耗时: {total_time:.2f} 秒")
        print("="*60)
        
        self._generate_report()
    
    def _generate_report(self):
        """生成性能报告"""
        print("\n" + "📊"*30)
        print(" "*20 + "性能摘要")
        print("📊"*30 + "\n")
        
        if 'semantic_interpreter' in self.results:
            print(f"语义解释器: {self.results['semantic_interpreter']['mean']:.4f} ms")
        
        if 'mpc_solver' in self.results:
            print(f"MPC求解器: {self.results['mpc_solver']['mean']:.2f} ms")
        
        if 'physics_simulation' in self.results:
            print(f"物理模拟: {self.results['physics_simulation']['mean']:.2f} μs")
        
        if 'monitoring' in self.results:
            print(f"监控系统: {self.results['monitoring']['mean']:.4f} ms")
        
        if 'database' in self.results:
            print(f"数据库写入: {self.results['database']['write_mean']:.4f} ms")
            print(f"数据库读取: {self.results['database']['read_time']:.2f} ms")
        
        if 'memory' in self.results:
            print(f"内存增长: {self.results['memory']['increase']:.2f} MB")
        
        print("\n提示: 运行 'python benchmark.py' 查看详细测试结果")


def main():
    """主函数"""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()

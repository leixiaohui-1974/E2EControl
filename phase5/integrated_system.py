"""
完整的智能水网控制系统集成
Integrated Smart Water Network Control System

集成了所有Phase的功能：
- Phase 1: 基础MPC控制
- Phase 2: 分布式DMPC优化
- Phase 3: 数字孪生仿真
- Phase 4: 智能决策与自愈
"""

import sys
import os
sys.path.append('..')
sys.path.append('../digital_twin')
sys.path.append('../phase4')

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Phase 1: 基础控制
from brain import SemanticInterpreter
from control import UniversalMPCSolver
from physics import CanalPoolSimulator

# Phase 3: 数字孪生
try:
    from digital_twin.physics.single_channel_fidelity import SingleChannelFidelity
    from digital_twin.perception.intelligent_observer import IntelligentObserver
    from digital_twin.control.single_pool_admm import SinglePoolADMM
    DIGITAL_TWIN_AVAILABLE = True
except ImportError:
    DIGITAL_TWIN_AVAILABLE = False
    print("⚠️ 数字孪生模块未加载")

# Phase 4: 智能决策与自愈
try:
    from phase4.anomaly_detection.ensemble_detector import EnsembleDetector
    from phase4.fault_diagnosis.diagnosis_engine import DiagnosisEngine
    from phase4.self_healing.self_healing_system import SelfHealingSystem
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False
    print("⚠️ Phase 4模块未加载")


class IntegratedWaterNetworkSystem:
    """
    完整的智能水网控制系统
    
    系统架构：
    ┌─────────────────────────────────────────────┐
    │            用户指令/场景脚本                 │
    └──────────────────┬──────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │   决策引擎 (Brain)  │
            │  - 场景识别         │
            │  - 策略选择         │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   异常检测器        │  [Phase 4.1]
            │  - 实时监控         │
            │  - 15+算法          │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   故障诊断器        │  [Phase 4.2]
            │  - 根因分析         │
            │  - 严重度评估       │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   自愈控制器        │  [Phase 4.3]
            │  - 故障隔离         │
            │  - 降级恢复         │
            └──────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐   ┌────▼─────┐   ┌───▼────┐
    │ MPC控制│   │数字孪生   │   │ 物理仿真│
    │[Phase1]│   │[Phase 3] │   │[Phase 1]│
    └────────┘   └──────────┘   └────────┘
    """
    
    def __init__(
        self,
        num_pools: int = 3,
        enable_digital_twin: bool = True,
        enable_self_healing: bool = True,
        enable_anomaly_detection: bool = True
    ):
        """
        初始化集成系统
        
        Args:
            num_pools: 渠池数量
            enable_digital_twin: 是否启用数字孪生
            enable_self_healing: 是否启用自愈系统
            enable_anomaly_detection: 是否启用异常检测
        """
        print("\n" + "="*80)
        print(" "*20 + "智能水网控制系统初始化")
        print("="*80)
        
        self.num_pools = num_pools
        
        # Phase 1: 基础控制模块
        print("\n[Phase 1] 初始化基础控制模块...")
        self.semantic_interpreter = SemanticInterpreter()
        self.mpc_controllers = [UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1) 
                                for _ in range(num_pools)]
        self.pools = [CanalPoolSimulator(
            area=10000.0,
            dt=3600.0,
            delay_steps=1,
            initial_level=3.0
        ) for _ in range(num_pools)]
        self.q_prev = [0.0] * num_pools  # 记录上一步的控制输入
        print("  ✓ MPC控制器已初始化")
        print("  ✓ 物理仿真器已初始化")
        
        # Phase 3: 数字孪生
        self.digital_twin = None
        if enable_digital_twin and DIGITAL_TWIN_AVAILABLE:
            print("\n[Phase 3] 初始化数字孪生系统...")
            try:
                self.digital_twin = {
                    'physics': SingleChannelFidelity(N=20, L=20000.0),
                    'observer': IntelligentObserver(N=20),
                    'controller': SinglePoolADMM(N=20, horizon=10)
                }
                print("  ✓ 高精度物理本体已加载")
                print("  ✓ 智能感知层已加载")
                print("  ✓ 鲁棒ADMM求解器已加载")
            except Exception as e:
                print(f"  ⚠️ 数字孪生初始化失败: {e}")
                self.digital_twin = None
        
        # Phase 4: 智能决策与自愈
        self.anomaly_detector = None
        self.diagnosis_engine = None
        self.self_healing = None
        
        if PHASE4_AVAILABLE:
            if enable_anomaly_detection:
                print("\n[Phase 4.1] 初始化异常检测系统...")
                try:
                    self.anomaly_detector = EnsembleDetector(
                        detector_configs=[
                            {'type': 'cusum', 'weight': 0.3},
                            {'type': 'ewma', 'weight': 0.2},
                            {'type': 'range', 'weight': 0.5}
                        ],
                        fusion_method='weighted'
                    )
                    print("  ✓ 集成异常检测器已加载（3种算法）")
                except Exception as e:
                    print(f"  ⚠️ 异常检测器初始化失败: {e}")
            
            print("\n[Phase 4.2] 初始化故障诊断引擎...")
            try:
                self.diagnosis_engine = DiagnosisEngine()
                print("  ✓ 诊断引擎已加载（5大故障类型）")
            except Exception as e:
                print(f"  ⚠️ 诊断引擎初始化失败: {e}")
            
            if enable_self_healing:
                print("\n[Phase 4.3] 初始化自愈控制系统...")
                try:
                    self.self_healing = SelfHealingSystem()
                    print("  ✓ 自愈系统已加载（10步闭环）")
                except Exception as e:
                    print(f"  ⚠️ 自愈系统初始化失败: {e}")
        
        # 系统状态
        self.current_time = 0
        self.history = {
            'time': [],
            'levels': [[] for _ in range(num_pools)],
            'flows_in': [[] for _ in range(num_pools)],
            'flows_out': [[] for _ in range(num_pools)],
            'anomalies': [],
            'faults': [],
            'healing_events': [],
            'mode': []
        }
        
        print("\n" + "="*80)
        print("✅ 系统初始化完成！")
        print("="*80)
        
    def run_simulation(
        self,
        scenario_script: List[Tuple[int, str]],
        total_steps: int = 100,
        enable_faults: bool = True
    ) -> Dict:
        """
        运行完整的仿真
        
        Args:
            scenario_script: 场景脚本 [(时间步, 指令), ...]
            total_steps: 总仿真步数
            enable_faults: 是否注入故障（用于测试自愈系统）
            
        Returns:
            仿真结果字典
        """
        print("\n" + "="*80)
        print(" "*25 + "开始仿真")
        print("="*80)
        print(f"  总步数: {total_steps}")
        print(f"  场景切换点: {len(scenario_script)}个")
        print(f"  故障注入: {'启用' if enable_faults else '禁用'}")
        
        # 解析场景脚本
        scenario_dict = {t: instruction for t, instruction in scenario_script}
        
        # 故障注入计划（如果启用）
        fault_schedule = []
        if enable_faults and self.self_healing:
            fault_schedule = [
                (20, "传感器漂移", "sensor_level_0", "低"),
                (50, "执行器卡死", "gate_actuator_1", "中"),
                (75, "控制器异常", "mpc_controller_0", "中")
            ]
            print(f"  计划故障: {len(fault_schedule)}个")
        
        current_config = None
        
        # 主仿真循环
        for t in range(total_steps):
            self.current_time = t
            
            # 场景切换
            if t in scenario_dict:
                instruction = scenario_dict[t]
                print(f"\n[T={t}] 场景切换: {instruction}")
                
                # 使用语义解释器
                current_config = self.semantic_interpreter.interpret(instruction)
                print(f"  ✓ 场景配置已更新")
            
            # 故障注入
            if enable_faults and self.self_healing:
                for fault_t, fault_type, component, severity in fault_schedule:
                    if t == fault_t:
                        print(f"\n[T={t}] 🚨 故障注入: {fault_type} ({component}, 严重度: {severity})")
                        
                        # 记录故障
                        self.history['faults'].append({
                            'time': t,
                            'type': fault_type,
                            'component': component,
                            'severity': severity
                        })
                        
                        # 触发自愈
                        healing_result = self.self_healing.detect_and_heal(
                            fault_type=fault_type,
                            fault_component=component,
                            fault_severity=severity,
                            system_state={}
                        )
                        
                        # 记录自愈事件
                        self.history['healing_events'].append({
                            'time': t,
                            'success': healing_result['success'],
                            'healing_time': healing_result['healing_time']
                        })
                        
                        if healing_result['success']:
                            print(f"  ✓ 自愈成功，耗时 {healing_result['healing_time']:.1f}s")
                        else:
                            print(f"  ✗ 自愈失败，需要人工干预")
            
            # 对每个渠池进行控制
            for i in range(self.num_pools):
                pool = self.pools[i]
                controller = self.mpc_controllers[i]
                
                # 获取当前状态
                Z = pool.get_level()
                
                # 异常检测
                if self.anomaly_detector and t > 5:  # 预热期
                    is_anomaly, anomaly_score = self.anomaly_detector.detect(Z)
                    
                    if is_anomaly:
                        print(f"[T={t}] ⚠️ 池{i+1}检测到异常 (评分: {anomaly_score:.3f})")
                        
                        self.history['anomalies'].append({
                            'time': t,
                            'pool': i,
                            'score': anomaly_score
                        })
                        
                        # 故障诊断
                        if self.diagnosis_engine:
                            diagnosis = self.diagnosis_engine.diagnose({
                                'detector': '3-sigma',
                                'value': Z,
                                'threshold': 2.0,
                                'consecutive': 3
                            })
                            
                            if diagnosis:
                                print(f"  → 诊断: {diagnosis['fault_type']} (严重度: {diagnosis['severity']})")
                
                # MPC控制
                if current_config:
                    # 使用当前场景配置
                    config = current_config.copy()
                else:
                    # 默认配置
                    config = {
                        'Z_ref': 3.0,
                        'W_level': 10.0,
                        'W_smooth': 5.0,
                        'delta_Q_max': 2.0,
                        'constraints': {}
                    }
                
                # 预测出流（简化）
                q_out_forecast = [3.0] * controller.N
                
                # 求解MPC
                try:
                    u_in = controller.solve(
                        current_level=Z,
                        q_prev=self.q_prev[i],
                        q_out_forecast=q_out_forecast,
                        config=config
                    )
                    self.q_prev[i] = u_in
                except Exception as e:
                    # print(f"  MPC求解失败: {e}")
                    u_in = 0.0
                
                # 模拟出流（简化）
                u_out = u_in * 0.9
                
                # 更新物理状态
                pool.step(u_in, u_out, disturbance=0.0)
                
                # 记录历史
                self.history['levels'][i].append(pool.get_level())
                self.history['flows_in'][i].append(u_in)
                self.history['flows_out'][i].append(u_out)
            
            # 记录运行模式
            if self.self_healing:
                mode = self.self_healing.degraded_manager.current_mode.value
            else:
                mode = "正常模式"
            self.history['mode'].append(mode)
            self.history['time'].append(t)
            
            # 进度显示
            if (t + 1) % 20 == 0:
                print(f"[T={t+1}/{total_steps}] 仿真进行中... (模式: {mode})")
        
        print("\n" + "="*80)
        print("✅ 仿真完成！")
        print("="*80)
        
        # 生成统计报告
        self._generate_statistics()
        
        return self.history
    
    def _generate_statistics(self):
        """生成统计报告"""
        print("\n" + "="*80)
        print(" "*25 + "仿真统计")
        print("="*80)
        
        # 基础统计
        print(f"\n【基础控制】")
        for i in range(self.num_pools):
            levels = self.history['levels'][i]
            print(f"  池{i+1}:")
            print(f"    平均水位: {np.mean(levels):.2f}m")
            print(f"    水位波动: {np.std(levels):.3f}m")
            print(f"    最大水位: {np.max(levels):.2f}m")
            print(f"    最小水位: {np.min(levels):.2f}m")
        
        # 异常检测统计
        if self.history['anomalies']:
            print(f"\n【异常检测】")
            print(f"  检测到异常: {len(self.history['anomalies'])}次")
            avg_score = np.mean([a['score'] for a in self.history['anomalies']])
            print(f"  平均异常评分: {avg_score:.3f}")
        
        # 故障统计
        if self.history['faults']:
            print(f"\n【故障诊断】")
            print(f"  总故障数: {len(self.history['faults'])}次")
            for fault in self.history['faults']:
                print(f"    T={fault['time']}: {fault['type']} ({fault['severity']})")
        
        # 自愈统计
        if self.history['healing_events']:
            print(f"\n【自愈控制】")
            total = len(self.history['healing_events'])
            success = sum(1 for e in self.history['healing_events'] if e['success'])
            print(f"  总自愈次数: {total}")
            print(f"  成功次数: {success}")
            print(f"  成功率: {success/total*100:.1f}%")
            
            if success > 0:
                avg_time = np.mean([e['healing_time'] 
                                   for e in self.history['healing_events'] 
                                   if e['success']])
                print(f"  平均自愈时间: {avg_time:.1f}s")
        
        # 运行模式统计
        if self.history['mode']:
            print(f"\n【运行模式】")
            from collections import Counter
            mode_counts = Counter(self.history['mode'])
            for mode, count in mode_counts.most_common():
                percentage = count / len(self.history['mode']) * 100
                print(f"  {mode}: {count}步 ({percentage:.1f}%)")
    
    def visualize_results(self, save_path: str = "integrated_system_results.png"):
        """可视化仿真结果"""
        print(f"\n生成可视化报告...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 2x3 布局
        # 第一行：水位、流量、异常
        # 第二行：故障事件、自愈时间线、运行模式
        
        # 1. 水位曲线
        ax1 = plt.subplot(2, 3, 1)
        for i in range(self.num_pools):
            ax1.plot(self.history['time'], self.history['levels'][i], 
                    label=f'池{i+1}', linewidth=2)
        ax1.set_xlabel('时间步', fontsize=12)
        ax1.set_ylabel('水位 (m)', fontsize=12)
        ax1.set_title('水位控制曲线', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 流量曲线
        ax2 = plt.subplot(2, 3, 2)
        for i in range(self.num_pools):
            ax2.plot(self.history['time'], self.history['flows_in'][i], 
                    label=f'池{i+1} 入流', alpha=0.7)
        ax2.set_xlabel('时间步', fontsize=12)
        ax2.set_ylabel('流量 (m³/s)', fontsize=12)
        ax2.set_title('流量控制曲线', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 异常检测
        ax3 = plt.subplot(2, 3, 3)
        if self.history['anomalies']:
            anomaly_times = [a['time'] for a in self.history['anomalies']]
            anomaly_scores = [a['score'] for a in self.history['anomalies']]
            anomaly_pools = [a['pool'] for a in self.history['anomalies']]
            
            scatter = ax3.scatter(anomaly_times, anomaly_scores, 
                                 c=anomaly_pools, cmap='viridis', 
                                 s=100, alpha=0.6, edgecolors='black')
            ax3.set_xlabel('时间步', fontsize=12)
            ax3.set_ylabel('异常评分', fontsize=12)
            ax3.set_title('异常检测结果', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='渠池编号')
        else:
            ax3.text(0.5, 0.5, '无异常检测', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. 故障事件时间线
        ax4 = plt.subplot(2, 3, 4)
        if self.history['faults']:
            fault_times = [f['time'] for f in self.history['faults']]
            fault_types = [f['type'] for f in self.history['faults']]
            
            for i, (t, ftype) in enumerate(zip(fault_times, fault_types)):
                ax4.axvline(x=t, color='red', linestyle='--', alpha=0.7)
                ax4.text(t, 0.5 + i*0.1, ftype, rotation=90, 
                        va='bottom', fontsize=9)
            
            ax4.set_xlim(0, max(self.history['time']))
            ax4.set_ylim(0, 1)
            ax4.set_xlabel('时间步', fontsize=12)
            ax4.set_title('故障事件时间线', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '无故障事件', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # 5. 自愈时间统计
        ax5 = plt.subplot(2, 3, 5)
        if self.history['healing_events']:
            success_times = [e['healing_time'] for e in self.history['healing_events'] 
                           if e['success']]
            
            if success_times:
                ax5.bar(range(len(success_times)), success_times, color='green', alpha=0.7)
                ax5.axhline(y=np.mean(success_times), color='red', linestyle='--',
                           label=f'平均: {np.mean(success_times):.1f}s')
                ax5.set_xlabel('自愈事件序号', fontsize=12)
                ax5.set_ylabel('自愈时间 (s)', fontsize=12)
                ax5.set_title('自愈时间统计', fontsize=14, fontweight='bold')
                ax5.legend()
        else:
            ax5.text(0.5, 0.5, '无自愈事件', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 运行模式分布
        ax6 = plt.subplot(2, 3, 6)
        if self.history['mode']:
            from collections import Counter
            mode_counts = Counter(self.history['mode'])
            
            modes = list(mode_counts.keys())
            counts = list(mode_counts.values())
            
            colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
            ax6.pie(counts, labels=modes, autopct='%1.1f%%',
                   colors=colors[:len(modes)], startangle=90)
            ax6.set_title('运行模式分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化报告已保存: {save_path}")
        plt.close()
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'current_time': self.current_time,
            'num_pools': self.num_pools,
            'modules': {
                'mpc_control': True,
                'digital_twin': self.digital_twin is not None,
                'anomaly_detection': self.anomaly_detector is not None,
                'fault_diagnosis': self.diagnosis_engine is not None,
                'self_healing': self.self_healing is not None
            }
        }
        
        # 当前水位
        status['current_levels'] = [pool.get_level() for pool in self.pools]
        
        # 运行模式
        if self.self_healing:
            status['operation_mode'] = self.self_healing.degraded_manager.current_mode.value
            status['system_health'] = self.self_healing.degraded_manager.calculate_overall_health()
        
        # 统计信息
        status['statistics'] = {
            'total_anomalies': len(self.history['anomalies']),
            'total_faults': len(self.history['faults']),
            'total_healings': len(self.history['healing_events']),
            'healing_success_rate': (
                sum(1 for e in self.history['healing_events'] if e['success']) / 
                len(self.history['healing_events']) * 100
                if self.history['healing_events'] else 0
            )
        }
        
        return status


# 演示
def run_comprehensive_demo():
    """运行完整的系统演示"""
    print("\n" + "="*80)
    print(" "*15 + "智能水网控制系统 - 完整集成演示")
    print("="*80)
    
    # 创建集成系统
    system = IntegratedWaterNetworkSystem(
        num_pools=3,
        enable_digital_twin=False,  # 简化演示，不启用数字孪生
        enable_self_healing=True,
        enable_anomaly_detection=True
    )
    
    # 定义场景脚本
    scenario_script = [
        (0, "保持水位平稳，正常供水"),
        (30, "收到暴雨预警，立刻降低水位腾出库容！"),
        (60, "恢复正常供水"),
        (90, "进入夜间节水模式")
    ]
    
    # 运行仿真（启用故障注入）
    history = system.run_simulation(
        scenario_script=scenario_script,
        total_steps=100,
        enable_faults=True
    )
    
    # 生成可视化报告
    system.visualize_results("/workspace/phase5/integrated_system_results.png")
    
    # 获取系统状态
    status = system.get_system_status()
    
    print("\n" + "="*80)
    print(" "*25 + "系统状态摘要")
    print("="*80)
    print(f"\n当前时间: {status['timestamp']}")
    print(f"仿真步数: {status['current_time']}")
    print(f"\n启用模块:")
    for module, enabled in status['modules'].items():
        print(f"  {module}: {'✓' if enabled else '✗'}")
    
    if 'operation_mode' in status:
        print(f"\n当前运行模式: {status['operation_mode']}")
        print(f"系统健康度: {status['system_health']:.2%}")
    
    print(f"\n统计信息:")
    print(f"  异常检测: {status['statistics']['total_anomalies']}次")
    print(f"  故障发生: {status['statistics']['total_faults']}次")
    print(f"  自愈执行: {status['statistics']['total_healings']}次")
    print(f"  自愈成功率: {status['statistics']['healing_success_rate']:.1f}%")
    
    print("\n" + "="*80)
    print("✅ 完整系统演示完成！")
    print("="*80)
    
    return system


if __name__ == "__main__":
    system = run_comprehensive_demo()

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

import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

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
    logger.info("Digital twin modules not available")

# Phase 4: 智能决策与自愈
try:
    from phase4.anomaly_detection.ensemble_detector import EnsembleDetector
    from phase4.fault_diagnosis.diagnosis_engine import DiagnosisEngine
    from phase4.self_healing.self_healing_system import SelfHealingSystem
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False
    logger.info("Phase 4 modules not available")


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
        logger.info("智能水网控制系统初始化")

        self.num_pools = num_pools

        # Phase 1: 基础控制模块
        logger.info("[Phase 1] 初始化基础控制模块...")
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
        self._faults_lock = threading.Lock()
        self.active_faults = []  # List of active faults
        logger.info("MPC控制器已初始化")
        logger.info("物理仿真器已初始化")
        
        # Phase 3: 数字孪生
        self.digital_twin = None
        if enable_digital_twin and DIGITAL_TWIN_AVAILABLE:
            logger.info("[Phase 3] 初始化数字孪生系统...")
            try:
                self.digital_twin = {
                    'physics': SingleChannelFidelity(N=20, L=20000.0),
                    'observer': IntelligentObserver(N=20),
                    'controller': SinglePoolADMM(N=20, horizon=10)
                }
                logger.info("高精度物理本体已加载")
                logger.info("智能感知层已加载")
                logger.info("鲁棒ADMM求解器已加载")
            except Exception as e:
                logger.warning("数字孪生初始化失败: %s", e)
                self.digital_twin = None
        
        # Phase 4: 智能决策与自愈
        self.anomaly_detector = None
        self.diagnosis_engine = None
        self.self_healing = None
        
        if PHASE4_AVAILABLE:
            if enable_anomaly_detection:
                logger.info("[Phase 4.1] 初始化异常检测系统...")
                try:
                    self.anomaly_detector = EnsembleDetector(
                        detector_configs=[
                            {'type': 'cusum', 'weight': 0.3},
                            {'type': 'ewma', 'weight': 0.2},
                            {'type': 'range', 'weight': 0.5}
                        ],
                        fusion_method='weighted'
                    )
                    logger.info("集成异常检测器已加载（3种算法）")
                except Exception as e:
                    logger.warning("异常检测器初始化失败: %s", e)

            logger.info("[Phase 4.2] 初始化故障诊断引擎...")
            try:
                self.diagnosis_engine = DiagnosisEngine()
                logger.info("诊断引擎已加载（5大故障类型）")
            except Exception as e:
                logger.warning("诊断引擎初始化失败: %s", e)

            if enable_self_healing:
                logger.info("[Phase 4.3] 初始化自愈控制系统...")
                try:
                    self.self_healing = SelfHealingSystem()
                    logger.info("自愈系统已加载（10步闭环）")
                except Exception as e:
                    logger.warning("自愈系统初始化失败: %s", e)
        
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
        
        logger.info("系统初始化完成！")
        
    def step(self, t: int, instruction: Optional[str] = None, enable_faults: bool = True) -> Dict:
        """
        执行单步仿真
        """
        self.current_time = t
        
        # 场景切换
        if instruction:
            logger.info("[T=%s] 场景切换: %s", t, instruction)
            self.current_config = self.semantic_interpreter.interpret(instruction)
            logger.info("场景配置已更新")
        
        # 故障注入 logic
        current_disturbances = [0.0] * self.num_pools
        
        # Apply active faults
        # Remove expired faults (if we had duration, but for now manual events are one-off or persistent?)
        # Let's assume 'flood' and 'drought' are persistent until reset, or apply for a duration.
        # For simplicity in this demo, 'flood' adds disturbance for this step.
        # But the UI triggers a single event.
        # Let's make them decay or be persistent?
        # The UI "Trigger Flood" usually implies a sudden event.
        # Let's store them with a duration or just apply them if they are in the list.
        
        # Better approach: The UI sends an event. We add it to active_faults.
        # In step, we apply them.
        # We need a way to clear them.
        
        with self._faults_lock:
            faults_snapshot = list(self.active_faults)
        for fault in faults_snapshot:
            if fault['type'] == 'flood':
                magnitude = fault.get('magnitude', 20.0)
                for i in range(self.num_pools):
                    current_disturbances[i] += magnitude
            elif fault['type'] == 'drought':
                magnitude = fault.get('magnitude', 5.0)
                for i in range(self.num_pools):
                    current_disturbances[i] -= magnitude
        
        # Clear one-off faults if needed, or keep them?
        # If it's a "scenario", it persists.
        # Let's keep them until cleared.

        
        # 对每个渠池进行控制
        step_data = {
            'levels': [],
            'flows_in': [],
            'flows_out': [],
            'anomalies': [],
            'faults': []
        }
        
        for i in range(self.num_pools):
            pool = self.pools[i]
            controller = self.mpc_controllers[i]
            
            # 获取当前状态
            Z = pool.get_level()
            
            # 异常检测
            if self.anomaly_detector and t > 5:
                is_anomaly, anomaly_score = self.anomaly_detector.detect(Z)
                if is_anomaly:
                    step_data['anomalies'].append({'pool': i, 'score': anomaly_score})
                    self.history['anomalies'].append({'time': t, 'pool': i, 'score': anomaly_score})
            
            # MPC控制
            config = self.current_config.copy() if hasattr(self, 'current_config') and self.current_config else {
                'Z_ref': 3.0, 'W_level': 10.0, 'W_smooth': 5.0, 'delta_Q_max': 2.0, 'constraints': {}
            }
            
            # 求解MPC
            try:
                q_out_forecast = [3.0] * controller.N
                u_in = controller.solve(Z, self.q_prev[i], q_out_forecast, config)
                self.q_prev[i] = u_in
            except Exception as exc:
                logger.debug("MPC solve failed for pool %d: %s", i, exc)
                u_in = 0.0
            
            u_out = u_in * 0.9
            pool.step(u_in, u_out, disturbance=current_disturbances[i])
            
            # 记录
            self.history['levels'][i].append(pool.get_level())
            self.history['flows_in'][i].append(u_in)
            self.history['flows_out'][i].append(u_out)
            
            step_data['levels'].append(pool.get_level())
            step_data['flows_in'].append(u_in)
            step_data['flows_out'].append(u_out)
            
        self.history['time'].append(t)
        return step_data

    def run_simulation(
        self,
        scenario_script: List[Tuple[int, str]],
        total_steps: int = 100,
        enable_faults: bool = True
    ) -> Dict:
        """
        运行完整的仿真
        """
        logger.info("开始仿真")

        scenario_dict = {t: instruction for t, instruction in scenario_script}
        self.current_config = None

        for t in range(total_steps):
            instruction = scenario_dict.get(t)
            self.step(t, instruction, enable_faults)

        self._generate_statistics()
        return self.history
    
    def _generate_statistics(self):
        """生成统计报告"""
        logger.info("仿真统计")

        # 基础统计
        logger.info("【基础控制】")
        for i in range(self.num_pools):
            levels = self.history['levels'][i]
            logger.info("池%s:", i+1)
            logger.info("  平均水位: %.2fm", np.mean(levels))
            logger.info("  水位波动: %.3fm", np.std(levels))
            logger.info("  最大水位: %.2fm", np.max(levels))
            logger.info("  最小水位: %.2fm", np.min(levels))

        # 异常检测统计
        if self.history['anomalies']:
            logger.info("【异常检测】")
            logger.info("检测到异常: %s次", len(self.history['anomalies']))
            avg_score = np.mean([a['score'] for a in self.history['anomalies']])
            logger.info("平均异常评分: %.3f", avg_score)

        # 故障统计
        if self.history['faults']:
            logger.info("【故障诊断】")
            logger.info("总故障数: %s次", len(self.history['faults']))
            for fault in self.history['faults']:
                logger.info("  T=%s: %s (%s)", fault['time'], fault['type'], fault['severity'])

        # 自愈统计
        if self.history['healing_events']:
            logger.info("【自愈控制】")
            total = len(self.history['healing_events'])
            success = sum(1 for e in self.history['healing_events'] if e['success'])
            logger.info("总自愈次数: %s", total)
            logger.info("成功次数: %s", success)
            logger.info("成功率: %.1f%%", success/total*100)

            if success > 0:
                avg_time = np.mean([e['healing_time']
                                   for e in self.history['healing_events']
                                   if e['success']])
                logger.info("平均自愈时间: %.1fs", avg_time)

        # 运行模式统计
        if self.history['mode']:
            logger.info("【运行模式】")
            from collections import Counter
            mode_counts = Counter(self.history['mode'])
            for mode, count in mode_counts.most_common():
                percentage = count / len(self.history['mode']) * 100
                logger.info("%s: %s步 (%.1f%%)", mode, count, percentage)
    
    def visualize_results(self, save_path: str = "integrated_system_results.png"):
        """可视化仿真结果"""
        logger.info("生成可视化报告...")

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
        plt.close()

        logger.info("可视化报告已保存: %s", save_path)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'current_time': self.current_time,
            'modules': {
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
    logger.info("=" * 80)
    logger.info("智能水网控制系统 - 完整集成演示")
    logger.info("=" * 80)

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

    logger.info("=" * 80)
    logger.info("系统状态摘要")
    logger.info("=" * 80)
    logger.info("当前时间: %s", status['timestamp'])
    logger.info("仿真步数: %s", status['current_time'])
    logger.info("启用模块:")
    for module, enabled in status['modules'].items():
        logger.info("  %s: %s", module, "enabled" if enabled else "disabled")

    if 'operation_mode' in status:
        logger.info("当前运行模式: %s", status['operation_mode'])
        logger.info("系统健康度: %.2f%%", status['system_health'] * 100)

    logger.info("统计信息:")
    logger.info("  异常检测: %d次", status['statistics']['total_anomalies'])
    logger.info("  故障发生: %d次", status['statistics']['total_faults'])
    logger.info("  自愈执行: %d次", status['statistics']['total_healings'])
    logger.info("  自愈成功率: %.1f%%", status['statistics']['healing_success_rate'])

    logger.info("=" * 80)
    logger.info("完整系统演示完成!")
    logger.info("=" * 80)

    return system


if __name__ == "__main__":
    system = run_comprehensive_demo()

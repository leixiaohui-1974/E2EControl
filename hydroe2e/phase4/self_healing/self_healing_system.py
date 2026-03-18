"""
自愈控制系统 - 完整集成
Self-Healing Control System - Full Integration
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from .isolation_strategy import FaultIsolationStrategy, IsolationPlan
from .degraded_mode import DegradedModeManager, OperationMode
from .recovery_manager import RecoveryManager, RecoveryPlan


class SelfHealingSystem:
    """
    自愈控制系统
    
    集成了故障检测、诊断、隔离、降级和恢复的完整闭环
    
    工作流程:
    1. 检测异常 → 2. 诊断故障 → 3. 隔离故障组件 → 
    4. 降级运行模式 → 5. 执行恢复计划 → 6. 验证恢复效果 →
    7. 恢复正常模式
    """
    
    def __init__(self):
        """初始化自愈系统"""
        # 核心模块
        self.isolation_strategy = FaultIsolationStrategy()
        self.degraded_manager = DegradedModeManager()
        self.recovery_manager = RecoveryManager()
        
        # 系统状态
        self.is_healing = False
        self.healing_history = []
        
        # 性能监控
        self.metrics = {
            'total_faults': 0,
            'auto_healed': 0,
            'manual_required': 0,
            'healing_success_rate': 0.0,
            'avg_healing_time': 0.0
        }
        
    def detect_and_heal(
        self,
        fault_type: str,
        fault_component: str,
        fault_severity: str,
        system_state: Dict
    ) -> Dict:
        """
        检测故障并执行自愈
        
        Args:
            fault_type: 故障类型
            fault_component: 故障组件
            fault_severity: 故障严重度
            system_state: 系统状态
            
        Returns:
            自愈结果
        """
        print("\n" + "="*80)
        print(" "*25 + "【自愈控制系统启动】")
        print("="*80)
        
        start_time = datetime.now()
        self.is_healing = True
        self.metrics['total_faults'] += 1
        
        healing_record = {
            'fault_type': fault_type,
            'fault_component': fault_component,
            'fault_severity': fault_severity,
            'start_time': start_time,
            'phases': []
        }
        
        try:
            # ============================================================
            # 阶段1: 故障诊断与影响评估
            # ============================================================
            print(f"\n【阶段1/5】故障诊断与影响评估")
            print("-"*80)
            print(f"故障类型: {fault_type}")
            print(f"故障组件: {fault_component}")
            print(f"严重程度: {fault_severity}")
            
            healing_record['phases'].append({
                'name': '故障诊断',
                'status': 'completed'
            })
            
            # ============================================================
            # 阶段2: 故障隔离
            # ============================================================
            print(f"\n【阶段2/5】故障隔离")
            print("-"*80)
            
            isolation_plan = self.isolation_strategy.generate_isolation_plan(fault_component)
            
            if isolation_plan and isolation_plan.can_isolate:
                print(f"✓ 生成隔离计划")
                print(f"  隔离动作: {isolation_plan.isolation_action.value}")
                print(f"  受影响组件: {len(isolation_plan.affected_components)}个")
                print(f"  风险等级: {isolation_plan.risk_level}")
                
                # 执行隔离
                isolation_result = self.isolation_strategy.execute_isolation(isolation_plan)
                
                if isolation_result['success']:
                    print(f"✓ 隔离成功")
                    healing_record['phases'].append({
                        'name': '故障隔离',
                        'status': 'success',
                        'isolation_plan': isolation_plan
                    })
                else:
                    print(f"✗ 隔离失败: {isolation_result.get('error', 'Unknown')}")
                    healing_record['phases'].append({
                        'name': '故障隔离',
                        'status': 'failed'
                    })
                    raise Exception("隔离失败")
            else:
                print(f"✗ 无法生成隔离计划")
                if isolation_plan:
                    print(f"  原因: {isolation_plan.reason}")
                healing_record['phases'].append({
                    'name': '故障隔离',
                    'status': 'skipped'
                })
            
            # ============================================================
            # 阶段3: 运行模式降级
            # ============================================================
            print(f"\n【阶段3/5】运行模式降级")
            print("-"*80)
            
            # 更新系统健康指标
            health_degradation = {
                '低': 0.15,
                '中': 0.30,
                '高': 0.50
            }.get(fault_severity, 0.20)
            
            self.degraded_manager.update_system_health({
                'sensor_availability': 1.0 - health_degradation if '传感器' in fault_type else 1.0,
                'actuator_availability': 1.0 - health_degradation if '执行器' in fault_type else 1.0,
                'controller_availability': 1.0 - health_degradation if '控制器' in fault_type else 1.0,
                'communication_quality': 1.0 - health_degradation * 0.5,
                'power_stability': 0.95
            })
            
            # 自动调整运行模式
            mode_result = self.degraded_manager.auto_adjust()
            
            if mode_result['mode_changed']:
                print(f"✓ 模式切换: {mode_result['old_mode'].value} → {mode_result['new_mode'].value}")
                print(f"  原因: {mode_result['reason']}")
                
                config = mode_result['config']
                print(f"  新配置:")
                print(f"    - 控制精度: {config.control_precision:.2%}")
                print(f"    - 安全裕度: {config.safety_margin}")
                print(f"    - 优化: {'启用' if config.enable_optimization else '禁用'}")
            else:
                print(f"  当前模式适合，无需调整")
            
            healing_record['phases'].append({
                'name': '模式降级',
                'status': 'completed',
                'mode': self.degraded_manager.current_mode
            })
            
            # ============================================================
            # 阶段4: 生成并执行恢复计划
            # ============================================================
            print(f"\n【阶段4/5】生成并执行恢复计划")
            print("-"*80)
            
            recovery_plan = self.recovery_manager.generate_recovery_plan(
                fault_type=fault_type,
                fault_component=fault_component,
                fault_severity=fault_severity,
                system_state=system_state
            )
            
            print(f"✓ 恢复计划已生成")
            print(f"  计划ID: {recovery_plan.fault_id}")
            print(f"  恢复步骤: {len(recovery_plan.recovery_actions)}个")
            print(f"  预计时间: {recovery_plan.total_estimated_time}s")
            print(f"  成功概率: {recovery_plan.success_probability:.2%}")
            
            print(f"\n  执行恢复动作:")
            recovery_result = self.recovery_manager.execute_recovery_plan(recovery_plan)
            
            healing_record['phases'].append({
                'name': '执行恢复',
                'status': 'success' if recovery_result['success'] else 'failed',
                'recovery_result': recovery_result
            })
            
            # ============================================================
            # 阶段5: 验证与模式恢复
            # ============================================================
            print(f"\n【阶段5/5】验证与模式恢复")
            print("-"*80)
            
            if recovery_result['success']:
                print(f"✓ 恢复成功，准备恢复正常模式")
                
                # 逐步恢复系统健康
                self.degraded_manager.update_system_health({
                    'sensor_availability': 0.98,
                    'actuator_availability': 0.95,
                    'controller_availability': 1.0,
                    'communication_quality': 0.95,
                    'power_stability': 1.0
                })
                
                # 恢复模式
                restore_result = self.degraded_manager.auto_adjust()
                
                if restore_result['mode_changed']:
                    print(f"✓ 模式恢复: {restore_result['old_mode'].value} → {restore_result['new_mode'].value}")
                
                print(f"✓ 系统恢复正常运行")
                
                healing_record['phases'].append({
                    'name': '模式恢复',
                    'status': 'completed'
                })
                
                self.metrics['auto_healed'] += 1
                healing_success = True
                
            else:
                print(f"✗ 自动恢复失败，需要人工干预")
                self.metrics['manual_required'] += 1
                healing_success = False
                
                healing_record['phases'].append({
                    'name': '模式恢复',
                    'status': 'manual_required'
                })
            
        except Exception as e:
            print(f"\n✗ 自愈过程异常: {str(e)}")
            healing_success = False
            self.metrics['manual_required'] += 1
            
            healing_record['phases'].append({
                'name': '异常处理',
                'status': 'error',
                'error': str(e)
            })
        
        finally:
            self.is_healing = False
            end_time = datetime.now()
            healing_time = (end_time - start_time).total_seconds()
            
            healing_record['end_time'] = end_time
            healing_record['healing_time'] = healing_time
            healing_record['success'] = healing_success
            
            self.healing_history.append(healing_record)
            
            # 更新统计
            self._update_metrics()
        
        # 生成报告
        print("\n" + "="*80)
        print(" "*25 + "【自愈过程完成】")
        print("="*80)
        print(f"\n最终状态: {'✓ 成功自愈' if healing_success else '✗ 需要人工干预'}")
        print(f"总耗时: {healing_time:.1f}s")
        print(f"完成阶段: {len([p for p in healing_record['phases'] if p['status'] in ['completed', 'success']])}/{len(healing_record['phases'])}")
        
        return {
            'success': healing_success,
            'healing_time': healing_time,
            'healing_record': healing_record,
            'final_mode': self.degraded_manager.current_mode.value,
            'system_health': self.degraded_manager.calculate_overall_health()
        }
    
    def _update_metrics(self):
        """更新性能指标"""
        if self.metrics['total_faults'] > 0:
            self.metrics['healing_success_rate'] = (
                self.metrics['auto_healed'] / self.metrics['total_faults']
            )
        
        if self.healing_history:
            total_time = sum(
                record['healing_time'] 
                for record in self.healing_history
            )
            self.metrics['avg_healing_time'] = total_time / len(self.healing_history)
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'is_healing': self.is_healing,
            'operation_mode': self.degraded_manager.current_mode.value,
            'system_health': self.degraded_manager.calculate_overall_health(),
            'isolation_status': self.isolation_strategy.get_system_status(),
            'recovery_status': self.recovery_manager.get_recovery_status(),
            'metrics': self.metrics.copy()
        }
    
    def generate_report(self) -> str:
        """生成自愈系统报告"""
        status = self.get_system_status()
        
        report = []
        report.append("="*80)
        report.append(" "*20 + "自愈控制系统运行报告")
        report.append("="*80)
        
        report.append(f"\n【系统状态】")
        report.append(f"  当前模式: {status['operation_mode']}")
        report.append(f"  系统健康: {status['system_health']:.2%}")
        report.append(f"  正在自愈: {'是' if status['is_healing'] else '否'}")
        
        report.append(f"\n【性能指标】")
        metrics = status['metrics']
        report.append(f"  总故障次数: {metrics['total_faults']}")
        report.append(f"  自动修复: {metrics['auto_healed']}")
        report.append(f"  人工干预: {metrics['manual_required']}")
        report.append(f"  自愈成功率: {metrics['healing_success_rate']:.2%}")
        report.append(f"  平均自愈时间: {metrics['avg_healing_time']:.1f}s")
        
        report.append(f"\n【隔离策略】")
        iso_status = status['isolation_status']
        report.append(f"  隔离计划数: {iso_status.get('isolation_history_count', 0)}")
        report.append(f"  活动组件: {iso_status.get('active_components', 0)}")
        report.append(f"  隔离组件: {iso_status.get('isolated_components', 0)}")
        
        report.append(f"\n【恢复管理】")
        rec_status = status['recovery_status']
        report.append(f"  当前阶段: {rec_status['current_phase']}")
        rec_stats = rec_status['statistics']
        report.append(f"  恢复成功率: {rec_stats['successful_recoveries']}/{rec_stats['total_recoveries']}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def visualize_healing_history(self, save_path: str = "self_healing_report.png"):
        """可视化自愈历史"""
        if not self.healing_history:
            print("没有自愈历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('自愈控制系统运行报告', fontsize=16, fontweight='bold')
        
        # 1. 故障类型分布
        ax1 = axes[0, 0]
        fault_types = {}
        for record in self.healing_history:
            ft = record['fault_type']
            fault_types[ft] = fault_types.get(ft, 0) + 1
        
        if fault_types:
            ax1.bar(range(len(fault_types)), list(fault_types.values()))
            ax1.set_xticks(range(len(fault_types)))
            ax1.set_xticklabels(list(fault_types.keys()), rotation=45, ha='right')
            ax1.set_ylabel('次数')
            ax1.set_title('故障类型分布')
            ax1.grid(True, alpha=0.3)
        
        # 2. 自愈时间趋势
        ax2 = axes[0, 1]
        healing_times = [r['healing_time'] for r in self.healing_history]
        ax2.plot(healing_times, marker='o', linewidth=2, markersize=6)
        ax2.axhline(np.mean(healing_times), color='r', linestyle='--', 
                   label=f'平均: {np.mean(healing_times):.1f}s')
        ax2.set_xlabel('故障序号')
        ax2.set_ylabel('自愈时间 (s)')
        ax2.set_title('自愈时间趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 成功率统计
        ax3 = axes[1, 0]
        success_count = sum(1 for r in self.healing_history if r['success'])
        fail_count = len(self.healing_history) - success_count
        
        colors = ['#2ecc71', '#e74c3c']
        ax3.pie([success_count, fail_count], 
               labels=['自动修复', '人工干预'],
               autopct='%1.1f%%',
               colors=colors,
               startangle=90)
        ax3.set_title('自愈成功率')
        
        # 4. 阶段完成情况
        ax4 = axes[1, 1]
        phase_names = ['诊断', '隔离', '降级', '恢复', '验证']
        phase_success = [0] * len(phase_names)
        
        for record in self.healing_history:
            for phase in record['phases']:
                for i, name in enumerate(phase_names):
                    if name in phase['name']:
                        if phase['status'] in ['completed', 'success']:
                            phase_success[i] += 1
        
        ax4.barh(phase_names, phase_success, color='steelblue')
        ax4.set_xlabel('完成次数')
        ax4.set_title('各阶段完成情况')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 可视化报告已保存: {save_path}")


# 完整演示
def run_comprehensive_demo():
    """运行完整的自愈系统演示"""
    print("\n" + "="*80)
    print(" "*15 + "智能水网自愈控制系统 - 完整演示")
    print("="*80)
    
    # 创建自愈系统
    system = SelfHealingSystem()
    
    # 测试场景：模拟一天内的多次故障
    scenarios = [
        {
            'name': '08:00 - 水位传感器1漂移',
            'fault_type': '传感器漂移',
            'component': 'sensor_level_0',
            'severity': '低'
        },
        {
            'name': '10:30 - 闸门执行器2卡死',
            'fault_type': '执行器卡死',
            'component': 'gate_actuator_1',
            'severity': '中'
        },
        {
            'name': '14:00 - MPC控制器异常',
            'fault_type': '控制器异常',
            'component': 'controller_mpc_0',
            'severity': '中'
        },
        {
            'name': '16:45 - 通信模块故障',
            'fault_type': '通信中断',
            'component': 'comm_module_1',
            'severity': '高'
        },
        {
            'name': '18:20 - 流量传感器失效',
            'fault_type': '传感器失效',
            'component': 'sensor_flow_1',
            'severity': '中'
        }
    ]
    
    # 逐个处理故障
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n{'#'*80}")
        print(f"#{' '*30}场景 {i}/{len(scenarios)}{' '*30}#")
        print(f"#{' '*20}{scenario['name']}{' '*20}#")
        print(f"{'#'*80}")
        
        # 执行自愈
        result = system.detect_and_heal(
            fault_type=scenario['fault_type'],
            fault_component=scenario['component'],
            fault_severity=scenario['severity'],
            system_state={}
        )
        
        # 显示结果摘要
        print(f"\n" + "-"*80)
        print(f"场景 {i} 结果摘要:")
        print(f"  状态: {'✓ 成功' if result['success'] else '✗ 失败'}")
        print(f"  耗时: {result['healing_time']:.1f}s")
        print(f"  最终模式: {result['final_mode']}")
        print(f"  系统健康: {result['system_health']:.2%}")
        print("-"*80)
        
        # 短暂延迟（模拟时间流逝）
        import time
        time.sleep(0.5)
    
    # 生成最终报告
    print("\n\n" + "="*80)
    print(" "*25 + "【最终报告】")
    print("="*80)
    
    report = system.generate_report()
    print(report)
    
    # 生成可视化报告
    print("\n生成可视化报告...")
    system.visualize_healing_history("phase4/self_healing/self_healing_report.png")
    
    print("\n" + "="*80)
    print("✅ 自愈控制系统演示完成！")
    print("="*80)
    
    return system


if __name__ == "__main__":
    system = run_comprehensive_demo()

"""
自动恢复管理器
Automatic Recovery Manager
"""

import sys
sys.path.append('..')

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime


class RecoveryPhase(Enum):
    """恢复阶段"""
    IDLE = "空闲"
    DIAGNOSIS = "诊断"
    ISOLATION = "隔离"
    REPAIR = "修复"
    VALIDATION = "验证"
    RESTORATION = "恢复"
    COMPLETED = "完成"
    FAILED = "失败"


class RecoveryStrategy(Enum):
    """恢复策略"""
    AUTO_RESTART = "自动重启"
    SWITCH_BACKUP = "切换备用"
    RECALIBRATE = "重新校准"
    RESET_PARAMETERS = "重置参数"
    GRADUAL_RESTORE = "渐进恢复"
    FULL_RESTORE = "完全恢复"
    MANUAL_INTERVENTION = "人工干预"


@dataclass
class RecoveryAction:
    """恢复动作"""
    strategy: RecoveryStrategy
    target_component: str
    parameters: Dict
    estimated_time: float  # 预计时间 [s]
    risk_level: str
    success_probability: float
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class RecoveryPlan:
    """恢复计划"""
    fault_id: str
    fault_type: str
    affected_components: List[str]
    recovery_actions: List[RecoveryAction]
    total_estimated_time: float
    success_probability: float
    fallback_plan: Optional['RecoveryPlan'] = None


class RecoveryManager:
    """
    自动恢复管理器
    
    职责：
    1. 制定恢复计划
    2. 执行恢复动作
    3. 监控恢复进度
    4. 验证恢复效果
    5. 记录恢复历史
    """
    
    def __init__(self):
        """初始化恢复管理器"""
        self.current_phase = RecoveryPhase.IDLE
        self.current_plan: Optional[RecoveryPlan] = None
        self.recovery_history = []
        
        # 组件健康状态
        self.component_health = {}
        
        # 恢复统计
        self.stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0
        }
        
    def generate_recovery_plan(
        self,
        fault_type: str,
        fault_component: str,
        fault_severity: str,
        system_state: Dict
    ) -> RecoveryPlan:
        """
        生成恢复计划
        
        Args:
            fault_type: 故障类型
            fault_component: 故障组件
            fault_severity: 故障严重度
            system_state: 系统状态
            
        Returns:
            恢复计划
        """
        recovery_actions = []
        
        # 根据故障类型选择恢复策略
        if "传感器" in fault_type:
            recovery_actions = self._plan_sensor_recovery(
                fault_component, fault_severity, system_state
            )
        elif "执行器" in fault_type:
            recovery_actions = self._plan_actuator_recovery(
                fault_component, fault_severity, system_state
            )
        elif "控制器" in fault_type:
            recovery_actions = self._plan_controller_recovery(
                fault_component, fault_severity, system_state
            )
        elif "通信" in fault_type:
            recovery_actions = self._plan_communication_recovery(
                fault_component, fault_severity, system_state
            )
        else:
            # 默认通用恢复
            recovery_actions = self._plan_generic_recovery(
                fault_component, fault_severity, system_state
            )
        
        # 计算总体指标
        total_time = sum(action.estimated_time for action in recovery_actions)
        success_prob = np.prod([action.success_probability for action in recovery_actions])
        
        plan = RecoveryPlan(
            fault_id=f"FAULT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            fault_type=fault_type,
            affected_components=[fault_component],
            recovery_actions=recovery_actions,
            total_estimated_time=total_time,
            success_probability=success_prob
        )
        
        # 如果成功率较低，生成后备计划
        if success_prob < 0.7:
            plan.fallback_plan = self._generate_fallback_plan(fault_type, fault_component)
        
        return plan
    
    def _plan_sensor_recovery(
        self,
        component: str,
        severity: str,
        state: Dict
    ) -> List[RecoveryAction]:
        """规划传感器恢复"""
        actions = []
        
        if severity == "低":
            # 轻微故障：重新校准
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.RECALIBRATE,
                target_component=component,
                parameters={'calibration_samples': 10, 'reference_value': 0.0},
                estimated_time=60.0,
                risk_level="低",
                success_probability=0.9
            ))
        elif severity == "中":
            # 中等故障：切换备用 + 校准
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.SWITCH_BACKUP,
                target_component=component,
                parameters={'backup_id': f"{component}_backup"},
                estimated_time=30.0,
                risk_level="中",
                success_probability=0.85
            ))
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.RECALIBRATE,
                target_component=f"{component}_backup",
                parameters={'calibration_samples': 20},
                estimated_time=90.0,
                risk_level="低",
                success_probability=0.9
            ))
        else:
            # 严重故障：人工干预
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                target_component=component,
                parameters={'require_expert': True, 'notification_sent': True},
                estimated_time=3600.0,
                risk_level="高",
                success_probability=0.95
            ))
        
        return actions
    
    def _plan_actuator_recovery(
        self,
        component: str,
        severity: str,
        state: Dict
    ) -> List[RecoveryAction]:
        """规划执行器恢复"""
        actions = []
        
        if severity == "低":
            # 轻微故障：重启
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.AUTO_RESTART,
                target_component=component,
                parameters={'restart_mode': 'soft'},
                estimated_time=120.0,
                risk_level="低",
                success_probability=0.8
            ))
        elif severity == "中":
            # 中等故障：渐进恢复
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.GRADUAL_RESTORE,
                target_component=component,
                parameters={'restore_steps': 5, 'step_duration': 60.0},
                estimated_time=300.0,
                risk_level="中",
                success_probability=0.75
            ))
        else:
            # 严重故障：切换备用 + 验证
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.SWITCH_BACKUP,
                target_component=component,
                parameters={'backup_id': f"{component}_backup", 'test_first': True},
                estimated_time=180.0,
                risk_level="高",
                success_probability=0.85
            ))
        
        return actions
    
    def _plan_controller_recovery(
        self,
        component: str,
        severity: str,
        state: Dict
    ) -> List[RecoveryAction]:
        """规划控制器恢复"""
        actions = []
        
        # 控制器故障通常需要重置参数
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.RESET_PARAMETERS,
            target_component=component,
            parameters={'reset_to_default': True, 'preserve_history': True},
            estimated_time=60.0,
            risk_level="中",
            success_probability=0.85
        ))
        
        # 然后重启
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.AUTO_RESTART,
            target_component=component,
            parameters={'restart_mode': 'hard'},
            estimated_time=120.0,
            risk_level="中",
            success_probability=0.9
        ))
        
        return actions
    
    def _plan_communication_recovery(
        self,
        component: str,
        severity: str,
        state: Dict
    ) -> List[RecoveryAction]:
        """规划通信恢复"""
        actions = []
        
        # 通信故障：重置连接
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.AUTO_RESTART,
            target_component=component,
            parameters={'reset_connection': True, 'flush_buffer': True},
            estimated_time=30.0,
            risk_level="低",
            success_probability=0.85
        ))
        
        return actions
    
    def _plan_generic_recovery(
        self,
        component: str,
        severity: str,
        state: Dict
    ) -> List[RecoveryAction]:
        """规划通用恢复"""
        return [RecoveryAction(
            strategy=RecoveryStrategy.AUTO_RESTART,
            target_component=component,
            parameters={},
            estimated_time=120.0,
            risk_level="中",
            success_probability=0.7
        )]
    
    def _generate_fallback_plan(self, fault_type: str, component: str) -> RecoveryPlan:
        """生成后备计划"""
        # 后备计划通常是人工干预
        fallback_action = RecoveryAction(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            target_component=component,
            parameters={'priority': 'high'},
            estimated_time=3600.0,
            risk_level="高",
            success_probability=0.95
        )
        
        return RecoveryPlan(
            fault_id="FALLBACK",
            fault_type=fault_type,
            affected_components=[component],
            recovery_actions=[fallback_action],
            total_estimated_time=3600.0,
            success_probability=0.95
        )
    
    def execute_recovery_plan(self, plan: RecoveryPlan) -> Dict:
        """
        执行恢复计划
        
        Args:
            plan: 恢复计划
            
        Returns:
            执行结果
        """
        self.current_plan = plan
        self.current_phase = RecoveryPhase.DIAGNOSIS
        
        results = {
            'plan_id': plan.fault_id,
            'success': True,
            'actions_completed': 0,
            'actions_failed': 0,
            'total_time': 0.0,
            'details': []
        }
        
        # 逐步执行恢复动作
        for i, action in enumerate(plan.recovery_actions):
            print(f"\n执行恢复动作 {i+1}/{len(plan.recovery_actions)}: {action.strategy.value}")
            print(f"  目标组件: {action.target_component}")
            print(f"  预计时间: {action.estimated_time}s")
            print(f"  成功概率: {action.success_probability:.2%}")
            
            # 更新阶段
            if action.strategy == RecoveryStrategy.SWITCH_BACKUP:
                self.current_phase = RecoveryPhase.ISOLATION
            elif action.strategy == RecoveryStrategy.RECALIBRATE:
                self.current_phase = RecoveryPhase.REPAIR
            elif action.strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                self.current_phase = RecoveryPhase.FAILED
            else:
                self.current_phase = RecoveryPhase.REPAIR
            
            # 模拟执行（实际系统中会调用真实的硬件接口）
            success = np.random.random() < action.success_probability
            
            action_result = {
                'action': action.strategy.value,
                'target': action.target_component,
                'success': success,
                'time': action.estimated_time
            }
            
            results['details'].append(action_result)
            results['total_time'] += action.estimated_time
            
            if success:
                results['actions_completed'] += 1
                print(f"  ✓ 动作成功")
            else:
                results['actions_failed'] += 1
                results['success'] = False
                print(f"  ✗ 动作失败")
                
                # 如果有后备计划，切换到后备计划
                if plan.fallback_plan:
                    print(f"\n启动后备计划...")
                    fallback_result = self.execute_recovery_plan(plan.fallback_plan)
                    results['fallback_executed'] = True
                    results['fallback_result'] = fallback_result
                
                break
        
        # 验证阶段
        if results['success']:
            self.current_phase = RecoveryPhase.VALIDATION
            print(f"\n验证恢复效果...")
            
            # 模拟验证
            validation_success = np.random.random() < 0.9
            
            if validation_success:
                self.current_phase = RecoveryPhase.RESTORATION
                print("  ✓ 验证通过")
                
                # 渐进恢复
                self.current_phase = RecoveryPhase.COMPLETED
                print("  ✓ 恢复完成")
                
                self.stats['successful_recoveries'] += 1
            else:
                self.current_phase = RecoveryPhase.FAILED
                print("  ✗ 验证失败")
                results['success'] = False
                
                self.stats['failed_recoveries'] += 1
        else:
            self.stats['failed_recoveries'] += 1
        
        # 更新统计
        self.stats['total_recoveries'] += 1
        if self.recovery_history:
            total_time = sum(r.get('result', {}).get('total_time', 0) for r in self.recovery_history) + results['total_time']
        else:
            total_time = results['total_time']
        self.stats['avg_recovery_time'] = total_time / self.stats['total_recoveries']
        
        # 记录历史
        self.recovery_history.append({
            'plan_id': plan.fault_id,
            'fault_type': plan.fault_type,
            'result': results,
            'timestamp': datetime.now()
        })
        
        # 重置当前计划
        if self.current_phase == RecoveryPhase.COMPLETED:
            self.current_plan = None
            self.current_phase = RecoveryPhase.IDLE
        
        return results
    
    def get_recovery_status(self) -> Dict:
        """获取恢复状态"""
        return {
            'current_phase': self.current_phase.value,
            'has_active_plan': self.current_plan is not None,
            'plan_id': self.current_plan.fault_id if self.current_plan else None,
            'statistics': self.stats.copy()
        }


# 演示
if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "自动恢复管理器演示")
    print("="*80)
    
    # 创建恢复管理器
    manager = RecoveryManager()
    
    # 测试场景
    scenarios = [
        {
            'name': "传感器轻微故障",
            'fault_type': "传感器漂移",
            'component': "sensor_level_1",
            'severity': "低"
        },
        {
            'name': "执行器中等故障",
            'fault_type': "执行器卡死",
            'component': "gate_actuator_2",
            'severity': "中"
        },
        {
            'name': "控制器故障",
            'fault_type': "控制器异常",
            'component': "mpc_controller_1",
            'severity': "中"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"场景: {scenario['name']}")
        print('='*80)
        
        # 生成恢复计划
        print("\n生成恢复计划...")
        plan = manager.generate_recovery_plan(
            fault_type=scenario['fault_type'],
            fault_component=scenario['component'],
            fault_severity=scenario['severity'],
            system_state={}
        )
        
        print(f"\n恢复计划:")
        print(f"  故障ID: {plan.fault_id}")
        print(f"  故障类型: {plan.fault_type}")
        print(f"  受影响组件: {', '.join(plan.affected_components)}")
        print(f"  恢复动作数: {len(plan.recovery_actions)}")
        print(f"  预计总时间: {plan.total_estimated_time}s")
        print(f"  成功概率: {plan.success_probability:.2%}")
        
        print(f"\n恢复步骤:")
        for i, action in enumerate(plan.recovery_actions):
            print(f"  {i+1}. {action.strategy.value} - {action.target_component}")
            print(f"     时间: {action.estimated_time}s, 风险: {action.risk_level}")
        
        # 执行恢复计划
        print(f"\n{'='*80}")
        print("执行恢复计划")
        print('='*80)
        
        result = manager.execute_recovery_plan(plan)
        
        print(f"\n恢复结果:")
        print(f"  总体状态: {'✓ 成功' if result['success'] else '✗ 失败'}")
        print(f"  完成动作: {result['actions_completed']}/{len(plan.recovery_actions)}")
        print(f"  实际时间: {result['total_time']}s")
    
    # 显示统计
    print(f"\n{'='*80}")
    print("恢复统计")
    print('='*80)
    
    status = manager.get_recovery_status()
    stats = status['statistics']
    
    print(f"  总恢复次数: {stats['total_recoveries']}")
    print(f"  成功次数: {stats['successful_recoveries']}")
    print(f"  失败次数: {stats['failed_recoveries']}")
    print(f"  成功率: {stats['successful_recoveries']/stats['total_recoveries']*100:.1f}%")
    print(f"  平均恢复时间: {stats['avg_recovery_time']:.1f}s")
    
    print("\n✅ 演示完成！")
    print("="*80)

"""
故障隔离策略
Fault Isolation Strategy
"""

import sys
sys.path.append('..')

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import numpy as np


class ComponentType(Enum):
    """组件类型"""
    SENSOR = "传感器"
    ACTUATOR = "执行器"
    CONTROLLER = "控制器"
    COMMUNICATION = "通信"
    POWER = "电源"
    PHYSICAL = "物理设施"


class IsolationAction(Enum):
    """隔离动作"""
    DISABLE = "停用"
    SWITCH_TO_BACKUP = "切换备用"
    BYPASS = "旁路"
    LOCK = "锁定"
    DISCONNECT = "断开"


@dataclass
class Component:
    """系统组件"""
    id: str
    name: str
    type: ComponentType
    location: str  # 位置（例如：池段1，闸门2）
    is_critical: bool = False  # 是否关键组件
    has_backup: bool = False  # 是否有备用
    backup_id: Optional[str] = None  # 备用组件ID
    is_active: bool = True  # 是否激活
    is_isolated: bool = False  # 是否已隔离
    health_score: float = 100.0  # 健康分数 [0-100]


@dataclass
class IsolationPlan:
    """隔离方案"""
    fault_component: Component
    isolation_action: IsolationAction
    affected_components: List[Component]
    backup_components: List[Component]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    estimated_impact: str  # 影响描述
    recovery_time: float  # 预计恢复时间 [秒]
    can_isolate: bool = True  # 是否可以隔离
    reason: str = ""  # 原因说明


class FaultIsolationStrategy:
    """
    故障隔离策略管理器
    
    职责：
    1. 评估故障影响
    2. 生成隔离方案
    3. 执行隔离动作
    4. 管理备用切换
    """
    
    def __init__(self):
        """初始化隔离策略管理器"""
        # 系统组件注册表
        self.components: Dict[str, Component] = {}
        
        # 依赖关系图
        self.dependencies: Dict[str, Set[str]] = {}
        
        # 隔离历史
        self.isolation_history: List[IsolationPlan] = []
        
        # 初始化系统组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化系统组件"""
        # 传感器
        for i in range(3):
            # 主传感器
            sensor = Component(
                id=f"sensor_level_{i}",
                name=f"水位传感器{i+1}",
                type=ComponentType.SENSOR,
                location=f"池段{i+1}",
                is_critical=True,
                has_backup=True,
                backup_id=f"sensor_level_{i}_backup"
            )
            self.components[sensor.id] = sensor
            
            # 备用传感器
            backup_sensor = Component(
                id=f"sensor_level_{i}_backup",
                name=f"备用水位传感器{i+1}",
                type=ComponentType.SENSOR,
                location=f"池段{i+1}",
                is_active=False
            )
            self.components[backup_sensor.id] = backup_sensor
        
        # 执行器（闸门）
        for i in range(4):
            actuator = Component(
                id=f"gate_{i}",
                name=f"闸门{i+1}",
                type=ComponentType.ACTUATOR,
                location=f"闸门{i+1}",
                is_critical=True,
                has_backup=False  # 闸门通常没有直接备用，但可以通过相邻闸门补偿
            )
            self.components[actuator.id] = actuator
        
        # 控制器
        for i in range(2):
            controller = Component(
                id=f"controller_{i}",
                name=f"MPC控制器{i+1}",
                type=ComponentType.CONTROLLER,
                location=f"控制站{i+1}",
                is_critical=True,
                has_backup=True,
                backup_id=f"controller_{i}_backup"
            )
            self.components[controller.id] = controller
            
            backup_controller = Component(
                id=f"controller_{i}_backup",
                name=f"备用控制器{i+1}",
                type=ComponentType.CONTROLLER,
                location=f"控制站{i+1}",
                is_active=False
            )
            self.components[backup_controller.id] = backup_controller
        
        # 建立依赖关系
        self._build_dependencies()
    
    def _build_dependencies(self):
        """建立组件依赖关系"""
        # 控制器依赖传感器
        for i in range(2):
            controller_id = f"controller_{i}"
            self.dependencies[controller_id] = set()
            
            # 每个控制器依赖多个传感器
            for j in range(3):
                sensor_id = f"sensor_level_{j}"
                self.dependencies[controller_id].add(sensor_id)
        
        # 执行器依赖控制器
        for i in range(4):
            gate_id = f"gate_{i}"
            controller_id = f"controller_{i // 2}"  # 每个控制器管理2个闸门
            self.dependencies[gate_id] = {controller_id}
    
    def register_component(self, component: Component):
        """注册新组件"""
        self.components[component.id] = component
    
    def update_component_health(self, component_id: str, health_score: float):
        """更新组件健康分数"""
        if component_id in self.components:
            self.components[component_id].health_score = health_score
    
    def assess_fault_impact(self, faulty_component_id: str) -> Dict:
        """
        评估故障影响
        
        Args:
            faulty_component_id: 故障组件ID
            
        Returns:
            影响评估报告
        """
        if faulty_component_id not in self.components:
            return {'error': 'Component not found'}
        
        faulty_component = self.components[faulty_component_id]
        
        # 找出所有依赖该组件的其他组件
        affected = []
        for comp_id, deps in self.dependencies.items():
            if faulty_component_id in deps:
                affected.append(self.components[comp_id])
        
        # 评估风险等级
        risk_level = self._calculate_risk_level(faulty_component, affected)
        
        # 估算影响
        impact = self._estimate_impact(faulty_component, affected)
        
        return {
            'faulty_component': faulty_component,
            'affected_components': affected,
            'risk_level': risk_level,
            'impact': impact,
            'can_be_isolated': self._can_be_isolated(faulty_component)
        }
    
    def _calculate_risk_level(self, faulty: Component, affected: List[Component]) -> str:
        """计算风险等级"""
        # 基于组件重要性和影响范围
        if faulty.is_critical and len(affected) >= 2:
            return 'critical'
        elif faulty.is_critical or len(affected) >= 1:
            return 'high'
        elif len(affected) > 0:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_impact(self, faulty: Component, affected: List[Component]) -> str:
        """估算故障影响"""
        if faulty.type == ComponentType.SENSOR:
            if faulty.has_backup:
                return f"可切换到备用传感器，影响较小"
            else:
                return f"传感器失效，需要使用模型估计，控制精度下降"
        
        elif faulty.type == ComponentType.ACTUATOR:
            return f"闸门{faulty.location}失效，需要使用相邻闸门补偿，流量控制受限"
        
        elif faulty.type == ComponentType.CONTROLLER:
            if faulty.has_backup:
                return f"可切换到备用控制器，短暂中断"
            else:
                return f"控制器失效，系统进入手动模式"
        
        else:
            return f"{faulty.type.value}故障，影响待评估"
    
    def _can_be_isolated(self, component: Component) -> bool:
        """判断组件是否可以被隔离"""
        # 关键组件只有在有备用时才能隔离
        if component.is_critical:
            return component.has_backup
        
        # 非关键组件总是可以隔离
        return True
    
    def generate_isolation_plan(self, faulty_component_id: str) -> IsolationPlan:
        """
        生成隔离方案
        
        Args:
            faulty_component_id: 故障组件ID
            
        Returns:
            隔离方案
        """
        if faulty_component_id not in self.components:
            return None
        
        faulty_component = self.components[faulty_component_id]
        
        # 评估影响
        assessment = self.assess_fault_impact(faulty_component_id)
        
        # 确定隔离动作
        isolation_action = self._determine_isolation_action(faulty_component)
        
        # 找出备用组件
        backup_components = []
        if faulty_component.has_backup:
            backup_id = faulty_component.backup_id
            if backup_id and backup_id in self.components:
                backup_components.append(self.components[backup_id])
        
        # 估算恢复时间
        recovery_time = self._estimate_recovery_time(faulty_component, isolation_action)
        
        # 创建隔离方案
        plan = IsolationPlan(
            fault_component=faulty_component,
            isolation_action=isolation_action,
            affected_components=assessment['affected_components'],
            backup_components=backup_components,
            risk_level=assessment['risk_level'],
            estimated_impact=assessment['impact'],
            recovery_time=recovery_time,
            can_isolate=assessment['can_be_isolated'],
            reason=f"故障组件: {faulty_component.name}, 动作: {isolation_action.value}"
        )
        
        return plan
    
    def _determine_isolation_action(self, component: Component) -> IsolationAction:
        """确定隔离动作"""
        if component.has_backup:
            return IsolationAction.SWITCH_TO_BACKUP
        elif component.is_critical:
            return IsolationAction.BYPASS  # 尝试旁路
        else:
            return IsolationAction.DISABLE  # 直接停用
    
    def _estimate_recovery_time(self, component: Component, action: IsolationAction) -> float:
        """估算恢复时间"""
        if action == IsolationAction.SWITCH_TO_BACKUP:
            return 60.0  # 1分钟
        elif action == IsolationAction.BYPASS:
            return 300.0  # 5分钟
        elif action == IsolationAction.DISABLE:
            return 30.0  # 30秒
        else:
            return 120.0  # 2分钟
    
    def execute_isolation(self, plan: IsolationPlan) -> Dict:
        """
        执行隔离方案
        
        Args:
            plan: 隔离方案
            
        Returns:
            执行结果
        """
        if not plan.can_isolate:
            return {
                'success': False,
                'message': '无法隔离关键组件且无备用'
            }
        
        # 记录隔离历史
        self.isolation_history.append(plan)
        
        # 执行隔离动作
        faulty_component = plan.fault_component
        
        if plan.isolation_action == IsolationAction.SWITCH_TO_BACKUP:
            # 切换到备用
            return self._switch_to_backup(faulty_component, plan.backup_components)
        
        elif plan.isolation_action == IsolationAction.DISABLE:
            # 停用组件
            return self._disable_component(faulty_component)
        
        elif plan.isolation_action == IsolationAction.BYPASS:
            # 旁路处理
            return self._bypass_component(faulty_component)
        
        else:
            return {
                'success': False,
                'message': f'未实现的隔离动作: {plan.isolation_action.value}'
            }
    
    def _switch_to_backup(self, faulty: Component, backups: List[Component]) -> Dict:
        """切换到备用组件"""
        if len(backups) == 0:
            return {'success': False, 'message': '无可用备用组件'}
        
        backup = backups[0]
        
        # 停用故障组件
        faulty.is_active = False
        faulty.is_isolated = True
        
        # 激活备用组件
        backup.is_active = True
        
        return {
            'success': True,
            'message': f'已切换到备用: {faulty.name} → {backup.name}',
            'faulty_id': faulty.id,
            'backup_id': backup.id
        }
    
    def _disable_component(self, component: Component) -> Dict:
        """停用组件"""
        component.is_active = False
        component.is_isolated = True
        
        return {
            'success': True,
            'message': f'已停用组件: {component.name}',
            'component_id': component.id
        }
    
    def _bypass_component(self, component: Component) -> Dict:
        """旁路组件"""
        component.is_isolated = True
        
        return {
            'success': True,
            'message': f'已旁路组件: {component.name}',
            'component_id': component.id
        }
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        active_count = sum(1 for c in self.components.values() if c.is_active)
        isolated_count = sum(1 for c in self.components.values() if c.is_isolated)
        critical_count = sum(1 for c in self.components.values() if c.is_critical and c.is_active)
        
        return {
            'total_components': len(self.components),
            'active_components': active_count,
            'isolated_components': isolated_count,
            'critical_active': critical_count,
            'isolation_history_count': len(self.isolation_history)
        }


# 演示
if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "故障隔离策略演示")
    print("="*80)
    
    # 创建隔离策略管理器
    strategy = FaultIsolationStrategy()
    
    print(f"\n系统初始状态:")
    status = strategy.get_system_status()
    print(f"  总组件数: {status['total_components']}")
    print(f"  激活组件: {status['active_components']}")
    print(f"  关键组件: {status['critical_active']}")
    
    # 测试场景1: 传感器故障
    print(f"\n{'='*80}")
    print("场景1: 水位传感器故障")
    print('='*80)
    
    faulty_sensor = "sensor_level_1"
    
    # 评估影响
    print(f"\n评估故障影响...")
    assessment = strategy.assess_fault_impact(faulty_sensor)
    print(f"  故障组件: {assessment['faulty_component'].name}")
    print(f"  风险等级: {assessment['risk_level']}")
    print(f"  影响描述: {assessment['impact']}")
    print(f"  受影响组件: {len(assessment['affected_components'])}个")
    
    # 生成隔离方案
    print(f"\n生成隔离方案...")
    plan = strategy.generate_isolation_plan(faulty_sensor)
    print(f"  隔离动作: {plan.isolation_action.value}")
    print(f"  备用组件: {len(plan.backup_components)}个")
    print(f"  预计恢复时间: {plan.recovery_time:.0f}秒")
    
    # 执行隔离
    print(f"\n执行隔离...")
    result = strategy.execute_isolation(plan)
    print(f"  执行结果: {'成功' if result['success'] else '失败'}")
    print(f"  {result['message']}")
    
    # 测试场景2: 闸门故障
    print(f"\n{'='*80}")
    print("场景2: 闸门故障（无备用）")
    print('='*80)
    
    faulty_gate = "gate_2"
    
    assessment = strategy.assess_fault_impact(faulty_gate)
    print(f"\n故障组件: {assessment['faulty_component'].name}")
    print(f"风险等级: {assessment['risk_level']}")
    print(f"影响描述: {assessment['impact']}")
    
    plan = strategy.generate_isolation_plan(faulty_gate)
    print(f"\n隔离动作: {plan.isolation_action.value}")
    print(f"可以隔离: {'是' if plan.can_isolate else '否'}")
    
    # 最终状态
    print(f"\n{'='*80}")
    print("系统最终状态")
    print('='*80)
    
    status = strategy.get_system_status()
    print(f"  总组件数: {status['total_components']}")
    print(f"  激活组件: {status['active_components']}")
    print(f"  隔离组件: {status['isolated_components']}")
    print(f"  隔离历史: {status['isolation_history_count']}次")
    
    print("\n✅ 演示完成！")
    print("="*80)

"""
优化隔离策略 (Optimized Isolation Strategy)

核心能力:
1. 最小影响分析
2. 优雅降级路径
3. 服务连续性保障
4. 预防性隔离
5. 动态隔离边界
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class IsolationAction(Enum):
    """隔离动作"""
    NONE = "none"
    DISABLE = "disable"
    SWITCH_BACKUP = "switch_backup"
    BYPASS = "bypass"
    RATE_LIMIT = "rate_limit"
    LOCK_POSITION = "lock_position"
    SOFT_DISCONNECT = "soft_disconnect"
    HARD_DISCONNECT = "hard_disconnect"


class IsolationScope(Enum):
    """隔离范围"""
    COMPONENT = "component"    # 单组件
    SUBSYSTEM = "subsystem"    # 子系统
    POOL = "pool"              # 单池
    SECTION = "section"        # 区段
    NETWORK = "network"        # 全网


@dataclass
class ImpactAnalysis:
    """影响分析结果"""
    affected_components: List[str]
    affected_pools: List[int]
    service_impact: float  # 0-1, 服务影响程度
    safety_impact: float   # 0-1, 安全影响程度
    estimated_duration: float  # 预计影响时长(秒)
    cascading_risk: float  # 级联风险
    mitigation_options: List[str]


@dataclass
class GracefulDegradationPath:
    """优雅降级路径"""
    path_id: str
    stages: List[Dict]  # 降级阶段
    current_stage: int
    service_levels: List[float]  # 各阶段服务水平
    transition_times: List[float]  # 过渡时间
    rollback_possible: bool = True


@dataclass
class ServiceContinuityPlan:
    """服务连续性计划"""
    plan_id: str
    primary_path: str
    backup_paths: List[str]
    minimum_service_level: float
    max_degradation_time: float
    checkpoints: List[Dict]
    recovery_triggers: List[str]


@dataclass
class IsolationPlan:
    """隔离计划"""
    plan_id: str
    target_component: str
    isolation_action: IsolationAction
    isolation_scope: IsolationScope
    impact_analysis: ImpactAnalysis
    pre_isolation_steps: List[str]
    isolation_steps: List[str]
    post_isolation_steps: List[str]
    estimated_duration: float
    service_continuity: Optional[ServiceContinuityPlan] = None
    degradation_path: Optional[GracefulDegradationPath] = None
    confidence: float = 0.8


class OptimizedIsolationStrategy:
    """
    优化隔离策略

    目标:
    1. 最小化服务影响
    2. 保证安全约束
    3. 快速故障隔离
    4. 平滑降级过渡
    """

    def __init__(self, num_pools: int = 3):
        """初始化优化隔离策略"""
        self.num_pools = num_pools

        # 组件依赖图
        self.dependency_graph = self._build_dependency_graph()

        # 备份映射
        self.backup_mapping = self._init_backup_mapping()

        # 服务优先级
        self.service_priorities = {
            'water_supply': 1.0,    # 最高优先
            'flood_control': 0.95,
            'water_quality': 0.9,
            'efficiency': 0.7,
            'monitoring': 0.6
        }

        # 隔离历史
        self.isolation_history: List[Dict] = []

        # 当前隔离状态
        self.active_isolations: Dict[str, IsolationPlan] = {}

        # 统计
        self.stats = {
            'total_isolations': 0,
            'successful_isolations': 0,
            'service_disruptions': 0,
            'average_impact': 0.0
        }

        logger.info("[OptimizedIsolationStrategy] 优化隔离策略初始化完成")

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """构建组件依赖图"""
        graph = {}

        # 每个池的组件依赖
        for i in range(self.num_pools):
            # 闸门依赖传感器和控制器
            graph[f"gate_{i}"] = [
                f"level_sensor_{i}",
                f"flow_sensor_{i}",
                "mpc_controller"
            ]

            # 传感器依赖通信
            graph[f"level_sensor_{i}"] = [f"comm_channel_{i}"]
            graph[f"flow_sensor_{i}"] = [f"comm_channel_{i}"]

            # 上游池影响下游池
            if i > 0:
                graph[f"pool_{i}"] = [f"pool_{i-1}", f"gate_{i-1}"]

        # 控制器依赖
        graph["mpc_controller"] = ["data_bus", "power_supply"]

        return graph

    def _init_backup_mapping(self) -> Dict[str, str]:
        """初始化备份映射"""
        mapping = {}

        for i in range(self.num_pools):
            mapping[f"level_sensor_{i}"] = f"level_sensor_{i}_backup"
            mapping[f"flow_sensor_{i}"] = f"flow_sensor_{i}_backup"

        mapping["mpc_controller"] = "mpc_controller_backup"

        return mapping

    def analyze_impact(self,
                       fault_component: str,
                       system_state: Dict) -> ImpactAnalysis:
        """
        分析隔离影响

        Args:
            fault_component: 故障组件
            system_state: 系统状态

        Returns:
            ImpactAnalysis: 影响分析结果
        """
        # 找出受影响的组件
        affected_components = self._find_affected_components(fault_component)

        # 找出受影响的池
        affected_pools = self._find_affected_pools(fault_component, affected_components)

        # 计算服务影响
        service_impact = self._calculate_service_impact(
            affected_components, affected_pools, system_state
        )

        # 计算安全影响
        safety_impact = self._calculate_safety_impact(
            fault_component, affected_pools, system_state
        )

        # 估算影响时长
        estimated_duration = self._estimate_isolation_duration(fault_component)

        # 评估级联风险
        cascading_risk = self._assess_cascading_risk(
            fault_component, affected_components, system_state
        )

        # 生成缓解选项
        mitigation_options = self._generate_mitigation_options(
            fault_component, affected_components
        )

        return ImpactAnalysis(
            affected_components=affected_components,
            affected_pools=affected_pools,
            service_impact=service_impact,
            safety_impact=safety_impact,
            estimated_duration=estimated_duration,
            cascading_risk=cascading_risk,
            mitigation_options=mitigation_options
        )

    def _find_affected_components(self, fault_component: str) -> List[str]:
        """找出所有受影响的组件"""
        affected = set()
        to_check = [fault_component]

        while to_check:
            current = to_check.pop(0)
            if current in affected:
                continue
            affected.add(current)

            # 找依赖当前组件的其他组件
            for comp, deps in self.dependency_graph.items():
                if current in deps and comp not in affected:
                    to_check.append(comp)

        return list(affected)

    def _find_affected_pools(self,
                             fault_component: str,
                             affected_components: List[str]) -> List[int]:
        """找出受影响的池"""
        pools = set()

        for comp in affected_components:
            # 从组件名提取池ID
            for i in range(self.num_pools):
                if f"_{i}" in comp or f"pool_{i}" in comp:
                    pools.add(i)
                    # 下游池也受影响
                    for j in range(i + 1, self.num_pools):
                        pools.add(j)
                    break

        return sorted(list(pools))

    def _calculate_service_impact(self,
                                  affected_components: List[str],
                                  affected_pools: List[int],
                                  system_state: Dict) -> float:
        """计算服务影响"""
        impact = 0.0

        # 组件类型权重
        component_weights = {
            'gate': 0.3,
            'sensor': 0.15,
            'controller': 0.25,
            'comm': 0.1
        }

        for comp in affected_components:
            for key, weight in component_weights.items():
                if key in comp:
                    impact += weight
                    break

        # 受影响池的比例
        pool_ratio = len(affected_pools) / self.num_pools
        impact += pool_ratio * 0.2

        return min(1.0, impact)

    def _calculate_safety_impact(self,
                                 fault_component: str,
                                 affected_pools: List[int],
                                 system_state: Dict) -> float:
        """计算安全影响"""
        impact = 0.0

        # 闸门故障安全影响最大
        if 'gate' in fault_component:
            impact += 0.4

        # 检查当前水位状态
        levels = system_state.get('water_levels', [])
        for pool_id in affected_pools:
            if pool_id < len(levels):
                level = levels[pool_id]
                if level is not None:
                    if level > 6.0:  # 高水位
                        impact += 0.3
                    elif level < 1.5:  # 低水位
                        impact += 0.2

        return min(1.0, impact)

    def _estimate_isolation_duration(self, fault_component: str) -> float:
        """估算隔离时长"""
        # 基础时长(秒)
        base_duration = {
            'gate': 300,
            'sensor': 60,
            'controller': 120,
            'comm': 90
        }

        for key, duration in base_duration.items():
            if key in fault_component:
                return duration

        return 180  # 默认3分钟

    def _assess_cascading_risk(self,
                               fault_component: str,
                               affected_components: List[str],
                               system_state: Dict) -> float:
        """评估级联风险"""
        risk = 0.0

        # 受影响组件越多，级联风险越高
        risk += len(affected_components) * 0.05

        # 上游组件故障级联风险更高
        if '_0' in fault_component:  # 最上游
            risk += 0.3

        return min(1.0, risk)

    def _generate_mitigation_options(self,
                                     fault_component: str,
                                     affected_components: List[str]) -> List[str]:
        """生成缓解选项"""
        options = []

        # 检查是否有备份
        if fault_component in self.backup_mapping:
            options.append(f"切换到备份: {self.backup_mapping[fault_component]}")

        # 旁路选项
        if 'sensor' in fault_component:
            options.append("使用邻近传感器数据估算")
            options.append("切换到基于模型的估计")

        if 'gate' in fault_component:
            options.append("使用相邻闸门补偿")
            options.append("锁定在安全位置")

        if 'controller' in fault_component:
            options.append("切换到备用控制器")
            options.append("切换到手动模式")

        return options

    def generate_isolation_plan(self,
                                fault_component: str,
                                fault_severity: int,
                                system_state: Dict) -> IsolationPlan:
        """
        生成隔离计划

        Args:
            fault_component: 故障组件
            fault_severity: 故障严重度 (1-5)
            system_state: 系统状态

        Returns:
            IsolationPlan: 隔离计划
        """
        plan_id = f"ISO_{int(time.time()*1000)}"

        # 影响分析
        impact = self.analyze_impact(fault_component, system_state)

        # 确定隔离动作
        isolation_action = self._select_isolation_action(
            fault_component, fault_severity, impact
        )

        # 确定隔离范围
        isolation_scope = self._determine_scope(impact)

        # 生成步骤
        pre_steps = self._generate_pre_isolation_steps(
            fault_component, isolation_action, system_state
        )
        iso_steps = self._generate_isolation_steps(
            fault_component, isolation_action
        )
        post_steps = self._generate_post_isolation_steps(
            fault_component, isolation_action
        )

        # 服务连续性计划
        continuity_plan = self._create_continuity_plan(
            fault_component, impact
        )

        # 降级路径
        degradation_path = self._create_degradation_path(
            fault_component, impact
        )

        return IsolationPlan(
            plan_id=plan_id,
            target_component=fault_component,
            isolation_action=isolation_action,
            isolation_scope=isolation_scope,
            impact_analysis=impact,
            pre_isolation_steps=pre_steps,
            isolation_steps=iso_steps,
            post_isolation_steps=post_steps,
            estimated_duration=impact.estimated_duration,
            service_continuity=continuity_plan,
            degradation_path=degradation_path
        )

    def _select_isolation_action(self,
                                 fault_component: str,
                                 severity: int,
                                 impact: ImpactAnalysis) -> IsolationAction:
        """选择最优隔离动作"""
        # 有备份且影响小：切换备份
        if fault_component in self.backup_mapping and impact.service_impact < 0.5:
            return IsolationAction.SWITCH_BACKUP

        # 传感器故障：可以旁路
        if 'sensor' in fault_component and severity < 4:
            return IsolationAction.BYPASS

        # 闸门故障：锁定位置
        if 'gate' in fault_component:
            if severity >= 4:
                return IsolationAction.LOCK_POSITION
            else:
                return IsolationAction.RATE_LIMIT

        # 控制器故障：切换备份或断开
        if 'controller' in fault_component:
            if fault_component in self.backup_mapping:
                return IsolationAction.SWITCH_BACKUP
            else:
                return IsolationAction.SOFT_DISCONNECT

        # 默认：禁用
        return IsolationAction.DISABLE

    def _determine_scope(self, impact: ImpactAnalysis) -> IsolationScope:
        """确定隔离范围"""
        if len(impact.affected_pools) == 0:
            return IsolationScope.COMPONENT
        elif len(impact.affected_pools) == 1:
            return IsolationScope.POOL
        elif len(impact.affected_pools) <= self.num_pools // 2:
            return IsolationScope.SECTION
        else:
            return IsolationScope.NETWORK

    def _generate_pre_isolation_steps(self,
                                      fault_component: str,
                                      action: IsolationAction,
                                      system_state: Dict) -> List[str]:
        """生成隔离前步骤"""
        steps = []

        steps.append(f"验证故障组件: {fault_component}")
        steps.append("记录当前系统状态")

        if action == IsolationAction.SWITCH_BACKUP:
            backup = self.backup_mapping.get(fault_component)
            if backup:
                steps.append(f"验证备份组件{backup}状态")
                steps.append("准备无缝切换")

        if 'gate' in fault_component:
            steps.append("通知相邻池准备补偿")
            steps.append("检查下游水位")

        if action in [IsolationAction.SOFT_DISCONNECT, IsolationAction.HARD_DISCONNECT]:
            steps.append("保存当前控制参数")
            steps.append("通知操作人员")

        return steps

    def _generate_isolation_steps(self,
                                  fault_component: str,
                                  action: IsolationAction) -> List[str]:
        """生成隔离步骤"""
        steps = []

        action_steps = {
            IsolationAction.DISABLE: [
                f"禁用组件: {fault_component}",
                "设置故障标志",
                "从控制回路中移除"
            ],
            IsolationAction.SWITCH_BACKUP: [
                f"激活备份: {self.backup_mapping.get(fault_component, 'N/A')}",
                "同步状态",
                f"禁用主设备: {fault_component}"
            ],
            IsolationAction.BYPASS: [
                "启用旁路模式",
                "激活估算算法",
                f"标记{fault_component}为旁路状态"
            ],
            IsolationAction.RATE_LIMIT: [
                "降低操作频率",
                "限制变化幅度",
                "启用软约束"
            ],
            IsolationAction.LOCK_POSITION: [
                "记录当前位置",
                "锁定执行器",
                "禁用位置控制"
            ],
            IsolationAction.SOFT_DISCONNECT: [
                "完成当前操作",
                "平滑过渡到手动",
                "断开自动控制"
            ],
            IsolationAction.HARD_DISCONNECT: [
                "立即停止操作",
                "断开控制信号",
                "启动安全模式"
            ]
        }

        return action_steps.get(action, [f"执行隔离: {action.value}"])

    def _generate_post_isolation_steps(self,
                                       fault_component: str,
                                       action: IsolationAction) -> List[str]:
        """生成隔离后步骤"""
        steps = []

        steps.append("验证隔离成功")
        steps.append("更新系统拓扑")
        steps.append("通知监控系统")

        if action in [IsolationAction.BYPASS, IsolationAction.RATE_LIMIT]:
            steps.append("启动补偿控制")
            steps.append("调整控制参数")

        steps.append("记录隔离事件")
        steps.append("启动恢复流程")

        return steps

    def _create_continuity_plan(self,
                                fault_component: str,
                                impact: ImpactAnalysis) -> ServiceContinuityPlan:
        """创建服务连续性计划"""
        plan_id = f"SCP_{int(time.time()*1000)}"

        # 主路径
        primary_path = "normal_operation"
        if fault_component in self.backup_mapping:
            primary_path = f"backup_{fault_component}"

        # 备份路径
        backup_paths = []
        if 'sensor' in fault_component:
            backup_paths.append("model_based_estimation")
            backup_paths.append("neighbor_interpolation")
        if 'gate' in fault_component:
            backup_paths.append("neighbor_compensation")
            backup_paths.append("manual_operation")

        # 最小服务水平
        min_service = 0.8 if impact.safety_impact < 0.3 else 0.6

        return ServiceContinuityPlan(
            plan_id=plan_id,
            primary_path=primary_path,
            backup_paths=backup_paths,
            minimum_service_level=min_service,
            max_degradation_time=impact.estimated_duration * 2,
            checkpoints=[
                {'time': 60, 'check': 'service_level'},
                {'time': 300, 'check': 'safety_status'},
                {'time': 600, 'check': 'recovery_progress'}
            ],
            recovery_triggers=[
                "fault_cleared",
                "backup_ready",
                "manual_override"
            ]
        )

    def _create_degradation_path(self,
                                 fault_component: str,
                                 impact: ImpactAnalysis) -> GracefulDegradationPath:
        """创建优雅降级路径"""
        path_id = f"GDP_{int(time.time()*1000)}"

        stages = [
            {
                'stage': 0,
                'name': 'normal',
                'description': '正常运行',
                'service_level': 1.0
            },
            {
                'stage': 1,
                'name': 'limited',
                'description': '有限功能',
                'service_level': 0.8
            },
            {
                'stage': 2,
                'name': 'degraded',
                'description': '降级运行',
                'service_level': 0.6
            },
            {
                'stage': 3,
                'name': 'minimal',
                'description': '最小功能',
                'service_level': 0.4
            }
        ]

        return GracefulDegradationPath(
            path_id=path_id,
            stages=stages,
            current_stage=0,
            service_levels=[1.0, 0.8, 0.6, 0.4],
            transition_times=[30, 60, 120],
            rollback_possible=True
        )

    def execute_isolation(self, plan: IsolationPlan) -> Dict:
        """
        执行隔离计划

        Args:
            plan: 隔离计划

        Returns:
            Dict: 执行结果
        """
        start_time = time.time()
        result = {
            'plan_id': plan.plan_id,
            'success': True,
            'steps_completed': [],
            'errors': [],
            'duration': 0
        }

        try:
            # 执行前步骤
            for step in plan.pre_isolation_steps:
                result['steps_completed'].append(f"PRE: {step}")

            # 执行隔离步骤
            for step in plan.isolation_steps:
                result['steps_completed'].append(f"ISO: {step}")

            # 执行后步骤
            for step in plan.post_isolation_steps:
                result['steps_completed'].append(f"POST: {step}")

            # 记录
            self.active_isolations[plan.target_component] = plan
            self.stats['total_isolations'] += 1
            self.stats['successful_isolations'] += 1

        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))

        result['duration'] = time.time() - start_time

        # 更新历史
        self.isolation_history.append({
            'plan_id': plan.plan_id,
            'component': plan.target_component,
            'action': plan.isolation_action.value,
            'success': result['success'],
            'timestamp': datetime.now().isoformat()
        })

        return result

    def remove_isolation(self, component: str) -> bool:
        """移除隔离"""
        if component in self.active_isolations:
            del self.active_isolations[component]
            return True
        return False

    def get_active_isolations(self) -> List[IsolationPlan]:
        """获取当前活跃隔离"""
        return list(self.active_isolations.values())

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'active_isolations': len(self.active_isolations),
            'history_length': len(self.isolation_history)
        }

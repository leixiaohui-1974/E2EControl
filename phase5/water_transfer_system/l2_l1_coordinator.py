"""
L2-L1层级协调器 - 区域与现地协调
L2-L1 Hierarchical Coordinator - Regional and Local Coordination

核心功能:
1. L1事件上报接收与处理
2. L2决策下发到L1
3. 跨池协调 (污染追踪、退水协调)
4. 冲突仲裁
5. 资源调度

时间尺度:
- L2: 小时级 (5-60分钟)
- L1: 分钟级 (1-15分钟)

协调原则:
1. L1自主处理本地事件
2. 严重事件上报L2
3. 跨池事件L2统一协调
4. L2下发区域协调指令
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum, auto
import logging
import time
from collections import deque

from .core_types import PoolRole, ScenarioType, ScenarioSeverity, RegionConfig
from .local_pool_scenarios import (
    L1ScenarioType, L1ActionType, L1ScenarioEvent, L1ActionCommand,
)
from .l1_controller import (
    L1Controller, L1ControllerManager, L1ControlResult,
    L1ControllerState, L1PoolState,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 协调事件类型
# ==============================================================================

class CoordinationType(Enum):
    """协调类型"""
    POLLUTION_SPREAD = "污染扩散协调"
    EMERGENCY_DISCHARGE = "紧急退水协调"
    MAINTENANCE_DISCHARGE = "检修退水协调"
    FLOOD_CONTROL = "防洪协调"
    ICE_CONTROL = "冰凌协调"
    GATE_FAILURE = "闸门故障协调"
    LEVEL_BALANCE = "水位平衡协调"
    FLOW_ADJUSTMENT = "流量调整协调"


@dataclass
class CoordinationRequest:
    """协调请求"""
    request_id: str
    coordination_type: CoordinationType
    source_pool: int
    priority: int

    # 影响范围
    affected_pools: List[int] = field(default_factory=list)

    # 请求参数
    target_values: Dict[int, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)

    # 时间
    request_time: float = 0.0
    deadline: float = 0.0

    # 状态
    is_approved: bool = False
    is_completed: bool = False


@dataclass
class CoordinationResponse:
    """协调响应"""
    request_id: str
    success: bool
    message: str

    # 分配结果
    pool_commands: Dict[int, List[L1ActionCommand]] = field(default_factory=dict)

    # 执行状态
    execution_status: Dict[int, str] = field(default_factory=dict)


# ==============================================================================
# L2-L1协调器
# ==============================================================================

class L2L1Coordinator:
    """
    L2-L1层级协调器

    职责:
    1. 接收L1上报的事件
    2. 分析影响范围
    3. 生成协调方案
    4. 下发协调指令
    5. 监控执行结果
    """

    def __init__(self,
                 region_id: int,
                 pool_range: Tuple[int, int],
                 num_pools: int = 60):
        self.region_id = region_id
        self.pool_start, self.pool_end = pool_range
        self.num_pools = num_pools

        # L1控制器管理
        self.l1_manager = L1ControllerManager(num_pools)
        self.l1_manager.set_l2_coordinator(self._handle_l1_escalation)

        # 协调请求队列
        self.pending_requests: deque = deque(maxlen=100)
        self.active_coordinations: Dict[str, CoordinationRequest] = {}
        self.completed_coordinations: deque = deque(maxlen=500)

        # 区域状态
        self.region_state = {
            'active_events': 0,
            'coordinating_pools': set(),
            'emergency_mode': False,
        }

        # 协调策略参数
        self.pollution_isolation_range = 3      # 污染隔离范围 (池数)
        self.discharge_coordination_range = 5   # 退水协调范围
        self.max_concurrent_coordinations = 5

    def _handle_l1_escalation(self, pool_id: int, event: L1ScenarioEvent):
        """处理L1上报事件"""
        logger.info(f"L2[{self.region_id}]收到L1[{pool_id}]上报: {event.scenario_type.value}")

        # 分析事件类型并创建协调请求
        coord_type = self._map_event_to_coordination(event)
        if coord_type:
            request = self._create_coordination_request(pool_id, event, coord_type)
            self.pending_requests.append(request)

    def _map_event_to_coordination(self, event: L1ScenarioEvent) -> Optional[CoordinationType]:
        """映射事件到协调类型"""
        mapping = {
            L1ScenarioType.L1_POLLUTION_DETECTED: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_POLLUTION_TRACKING: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_POLLUTION_TRACING: CoordinationType.POLLUTION_SPREAD,
            L1ScenarioType.L1_DISCHARGE_EMERGENCY: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_DISCHARGE_MAINTENANCE: CoordinationType.MAINTENANCE_DISCHARGE,
            L1ScenarioType.L1_SLOPE_PANEL_FLOAT: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_SLOPE_INSTABILITY: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_LEVEL_RAPID_RISE: CoordinationType.FLOOD_CONTROL,
            L1ScenarioType.L1_LEVEL_RAPID_DROP: CoordinationType.FLOW_ADJUSTMENT,
            L1ScenarioType.L1_ICE_BLOCKAGE: CoordinationType.ICE_CONTROL,
            L1ScenarioType.L1_ICE_DAM: CoordinationType.ICE_CONTROL,
            L1ScenarioType.L1_GATE_STUCK: CoordinationType.GATE_FAILURE,
            L1ScenarioType.L1_GATE_CONTROL_FAIL: CoordinationType.GATE_FAILURE,
            L1ScenarioType.L1_LEAKAGE_SEVERE: CoordinationType.EMERGENCY_DISCHARGE,
            L1ScenarioType.L1_LEAKAGE_PIPE_BURST: CoordinationType.EMERGENCY_DISCHARGE,
        }
        return mapping.get(event.scenario_type)

    def _create_coordination_request(self,
                                      source_pool: int,
                                      event: L1ScenarioEvent,
                                      coord_type: CoordinationType) -> CoordinationRequest:
        """创建协调请求"""
        request_id = f"COORD_{self.region_id}_{source_pool}_{int(time.time())}"

        # 确定影响范围
        affected_pools = self._determine_affected_pools(source_pool, coord_type)

        # 确定优先级
        priority = self._calculate_priority(event, coord_type)

        return CoordinationRequest(
            request_id=request_id,
            coordination_type=coord_type,
            source_pool=source_pool,
            priority=priority,
            affected_pools=affected_pools,
            request_time=time.time(),
            deadline=time.time() + self._get_deadline(coord_type),
        )

    def _determine_affected_pools(self,
                                  source_pool: int,
                                  coord_type: CoordinationType) -> List[int]:
        """确定受影响的渠池"""
        if coord_type == CoordinationType.POLLUTION_SPREAD:
            # 污染影响下游池
            return list(range(source_pool, min(source_pool + self.pollution_isolation_range + 1, self.num_pools)))

        elif coord_type in [CoordinationType.EMERGENCY_DISCHARGE, CoordinationType.MAINTENANCE_DISCHARGE]:
            # 退水影响上下游
            start = max(0, source_pool - 2)
            end = min(source_pool + self.discharge_coordination_range, self.num_pools)
            return list(range(start, end))

        elif coord_type == CoordinationType.FLOOD_CONTROL:
            # 防洪影响下游
            return list(range(source_pool, min(source_pool + 10, self.num_pools)))

        elif coord_type == CoordinationType.ICE_CONTROL:
            # 冰凌影响上下游
            start = max(0, source_pool - 3)
            end = min(source_pool + 3, self.num_pools)
            return list(range(start, end))

        elif coord_type == CoordinationType.GATE_FAILURE:
            # 闸门故障影响相邻池
            return list(range(max(0, source_pool - 1), min(source_pool + 2, self.num_pools)))

        else:
            return [source_pool]

    def _calculate_priority(self,
                           event: L1ScenarioEvent,
                           coord_type: CoordinationType) -> int:
        """计算协调优先级"""
        base_priority = {
            CoordinationType.POLLUTION_SPREAD: 9,
            CoordinationType.EMERGENCY_DISCHARGE: 10,
            CoordinationType.MAINTENANCE_DISCHARGE: 4,
            CoordinationType.FLOOD_CONTROL: 9,
            CoordinationType.ICE_CONTROL: 7,
            CoordinationType.GATE_FAILURE: 8,
            CoordinationType.LEVEL_BALANCE: 3,
            CoordinationType.FLOW_ADJUSTMENT: 5,
        }

        priority = base_priority.get(coord_type, 5)

        # 根据严重程度调整
        severity_adj = {
            ScenarioSeverity.LOW: -2,
            ScenarioSeverity.MEDIUM: 0,
            ScenarioSeverity.HIGH: 1,
            ScenarioSeverity.CRITICAL: 2,
        }
        priority += severity_adj.get(event.severity, 0)

        return max(1, min(10, priority))

    def _get_deadline(self, coord_type: CoordinationType) -> float:
        """获取协调截止时间 (秒)"""
        deadlines = {
            CoordinationType.POLLUTION_SPREAD: 300,       # 5分钟
            CoordinationType.EMERGENCY_DISCHARGE: 180,    # 3分钟
            CoordinationType.MAINTENANCE_DISCHARGE: 3600, # 1小时
            CoordinationType.FLOOD_CONTROL: 300,          # 5分钟
            CoordinationType.ICE_CONTROL: 600,            # 10分钟
            CoordinationType.GATE_FAILURE: 120,           # 2分钟
            CoordinationType.LEVEL_BALANCE: 1800,         # 30分钟
            CoordinationType.FLOW_ADJUSTMENT: 900,        # 15分钟
        }
        return deadlines.get(coord_type, 600)

    def process_pending_requests(self) -> List[CoordinationResponse]:
        """处理待定协调请求"""
        responses = []

        # 按优先级排序
        requests = sorted(self.pending_requests, key=lambda r: -r.priority)
        self.pending_requests.clear()

        for request in requests:
            # 检查并发限制
            if len(self.active_coordinations) >= self.max_concurrent_coordinations:
                # 延迟低优先级请求
                if request.priority < 7:
                    self.pending_requests.append(request)
                    continue

            response = self._execute_coordination(request)
            responses.append(response)

            if response.success:
                self.active_coordinations[request.request_id] = request
                request.is_approved = True

        return responses

    def _execute_coordination(self, request: CoordinationRequest) -> CoordinationResponse:
        """执行协调"""
        logger.info(f"L2[{self.region_id}]执行协调: {request.coordination_type.value}, "
                   f"影响池: {request.affected_pools}")

        pool_commands = {}
        execution_status = {}

        if request.coordination_type == CoordinationType.POLLUTION_SPREAD:
            pool_commands = self._coordinate_pollution(request)
        elif request.coordination_type == CoordinationType.EMERGENCY_DISCHARGE:
            pool_commands = self._coordinate_emergency_discharge(request)
        elif request.coordination_type == CoordinationType.MAINTENANCE_DISCHARGE:
            pool_commands = self._coordinate_maintenance_discharge(request)
        elif request.coordination_type == CoordinationType.FLOOD_CONTROL:
            pool_commands = self._coordinate_flood_control(request)
        elif request.coordination_type == CoordinationType.ICE_CONTROL:
            pool_commands = self._coordinate_ice_control(request)
        elif request.coordination_type == CoordinationType.GATE_FAILURE:
            pool_commands = self._coordinate_gate_failure(request)
        else:
            pool_commands = self._coordinate_generic(request)

        # 下发命令到L1
        for pool_id, commands in pool_commands.items():
            if pool_id in self.l1_manager.controllers:
                for cmd in commands:
                    self.l1_manager.controllers[pool_id].command_queue.append(cmd)
                execution_status[pool_id] = "commands_dispatched"

        return CoordinationResponse(
            request_id=request.request_id,
            success=True,
            message=f"协调执行: {request.coordination_type.value}",
            pool_commands=pool_commands,
            execution_status=execution_status,
        )

    def _coordinate_pollution(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """污染协调"""
        commands = {}
        source_pool = request.source_pool

        for i, pool_id in enumerate(request.affected_pools):
            pool_commands = []

            if pool_id == source_pool:
                # 源池: 隔离上下游
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_ISO_UP_{pool_id}",
                    action_type=L1ActionType.ISOLATE_UPSTREAM,
                    pool_id=pool_id,
                    priority=request.priority,
                    gate_position=0.0,
                ))
            elif pool_id > source_pool:
                # 下游池: 关闸准备隔离
                gate_pos = 0.2 * (i + 1) / len(request.affected_pools)
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_REDUCE_{pool_id}",
                    action_type=L1ActionType.GATE_ADJUST,
                    pool_id=pool_id,
                    priority=request.priority - 1,
                    gate_position=gate_pos,
                ))

            # 所有池加强监测
            pool_commands.append(L1ActionCommand(
                command_id=f"{request.request_id}_MONITOR_{pool_id}",
                action_type=L1ActionType.MONITOR_ENHANCE,
                pool_id=pool_id,
                priority=request.priority - 2,
            ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_emergency_discharge(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """紧急退水协调"""
        commands = {}
        source_pool = request.source_pool

        for pool_id in request.affected_pools:
            pool_commands = []

            if pool_id == source_pool:
                # 源池: 启动退水
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_DRAIN_{pool_id}",
                    action_type=L1ActionType.DRAIN_START,
                    pool_id=pool_id,
                    priority=request.priority,
                    target_flow=50.0,
                ))
            elif pool_id < source_pool:
                # 上游池: 减少流入
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_REDUCE_{pool_id}",
                    action_type=L1ActionType.GATE_ADJUST,
                    pool_id=pool_id,
                    priority=request.priority - 1,
                    gate_position=0.3,
                ))
            else:
                # 下游池: 准备接收
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_OPEN_{pool_id}",
                    action_type=L1ActionType.GATE_OPEN,
                    pool_id=pool_id,
                    priority=request.priority - 1,
                    gate_position=0.8,
                ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_maintenance_discharge(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """检修退水协调"""
        commands = {}

        for pool_id in request.affected_pools:
            pool_commands = []

            if pool_id == request.source_pool:
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_DRAIN_{pool_id}",
                    action_type=L1ActionType.DRAIN_START,
                    pool_id=pool_id,
                    priority=request.priority,
                    target_flow=20.0,  # 低速退水
                ))
            elif pool_id < request.source_pool:
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_REDUCE_{pool_id}",
                    action_type=L1ActionType.COORDINATE_UPSTREAM,
                    pool_id=pool_id,
                    priority=request.priority - 1,
                ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_flood_control(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """防洪协调"""
        commands = {}

        for i, pool_id in enumerate(request.affected_pools):
            pool_commands = []

            # 逐级开大下游闸门
            gate_pos = 0.7 + 0.1 * i / len(request.affected_pools)

            pool_commands.append(L1ActionCommand(
                command_id=f"{request.request_id}_OPEN_{pool_id}",
                action_type=L1ActionType.GATE_OPEN,
                pool_id=pool_id,
                priority=request.priority,
                gate_position=min(1.0, gate_pos),
            ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_ice_control(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """冰凌协调"""
        commands = {}

        for pool_id in request.affected_pools:
            pool_commands = []

            # 调整流速以破冰
            pool_commands.append(L1ActionCommand(
                command_id=f"{request.request_id}_ICE_{pool_id}",
                action_type=L1ActionType.GATE_ADJUST,
                pool_id=pool_id,
                priority=request.priority,
                gate_position=0.6,
            ))

            pool_commands.append(L1ActionCommand(
                command_id=f"{request.request_id}_MONITOR_{pool_id}",
                action_type=L1ActionType.MONITOR_ENHANCE,
                pool_id=pool_id,
                priority=request.priority - 1,
            ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_gate_failure(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """闸门故障协调"""
        commands = {}

        for pool_id in request.affected_pools:
            pool_commands = []

            if pool_id == request.source_pool:
                # 故障池锁定
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_LOCK_{pool_id}",
                    action_type=L1ActionType.GATE_LOCK,
                    pool_id=pool_id,
                    priority=request.priority,
                ))
            else:
                # 相邻池补偿
                pool_commands.append(L1ActionCommand(
                    command_id=f"{request.request_id}_COMP_{pool_id}",
                    action_type=L1ActionType.GATE_ADJUST,
                    pool_id=pool_id,
                    priority=request.priority - 1,
                    gate_position=0.5,
                ))

            commands[pool_id] = pool_commands

        return commands

    def _coordinate_generic(self, request: CoordinationRequest) -> Dict[int, List[L1ActionCommand]]:
        """通用协调"""
        commands = {}

        for pool_id in request.affected_pools:
            commands[pool_id] = [
                L1ActionCommand(
                    command_id=f"{request.request_id}_GENERIC_{pool_id}",
                    action_type=L1ActionType.MONITOR_ENHANCE,
                    pool_id=pool_id,
                    priority=request.priority,
                )
            ]

        return commands

    def monitor_active_coordinations(self) -> Dict[str, Any]:
        """监控活跃协调"""
        status = {}
        completed = []

        for req_id, request in self.active_coordinations.items():
            # 检查是否超时
            if time.time() > request.deadline:
                request.is_completed = True
                completed.append(req_id)
                status[req_id] = "timeout"
            else:
                # 检查执行状态
                all_done = True
                for pool_id in request.affected_pools:
                    if pool_id in self.l1_manager.controllers:
                        controller = self.l1_manager.controllers[pool_id]
                        if controller.command_queue:
                            all_done = False
                            break

                if all_done:
                    request.is_completed = True
                    completed.append(req_id)
                    status[req_id] = "completed"
                else:
                    status[req_id] = "in_progress"

        # 移除已完成的协调
        for req_id in completed:
            request = self.active_coordinations.pop(req_id)
            self.completed_coordinations.append(request)

        return status

    def coordination_step(self, dt: float = 60.0) -> Dict[str, Any]:
        """执行一个协调步"""
        # 1. 处理待定请求
        responses = self.process_pending_requests()

        # 2. 监控活跃协调
        coord_status = self.monitor_active_coordinations()

        # 3. L1控制步
        l1_results = self.l1_manager.control_step_all(dt)

        # 4. 更新区域状态
        self.region_state['active_events'] = sum(
            len(c.active_events) for c in self.l1_manager.controllers.values()
        )
        self.region_state['coordinating_pools'] = set(
            pool_id for req in self.active_coordinations.values()
            for pool_id in req.affected_pools
        )

        return {
            'region_id': self.region_id,
            'responses': len(responses),
            'active_coordinations': len(self.active_coordinations),
            'coordination_status': coord_status,
            'l1_results': l1_results,
            'region_state': self.region_state,
        }


# ==============================================================================
# 全线协调管理器
# ==============================================================================

class FullLineCoordinatorManager:
    """
    全线协调管理器

    管理多个区域的L2-L1协调器
    """

    # 区域划分 (基于地理位置)
    REGIONS = [
        {'id': 0, 'name': '丹江口-陶岔', 'pools': (0, 10)},
        {'id': 1, 'name': '陶岔-鲁山', 'pools': (10, 20)},
        {'id': 2, 'name': '鲁山-郑州', 'pools': (20, 30)},
        {'id': 3, 'name': '郑州-安阳', 'pools': (30, 40)},
        {'id': 4, 'name': '安阳-邯郸', 'pools': (40, 50)},
        {'id': 5, 'name': '邯郸-北京', 'pools': (50, 60)},
    ]

    def __init__(self, num_pools: int = 60):
        self.num_pools = num_pools
        self.coordinators: Dict[int, L2L1Coordinator] = {}

        # 创建区域协调器
        for region in self.REGIONS:
            self.coordinators[region['id']] = L2L1Coordinator(
                region_id=region['id'],
                pool_range=region['pools'],
                num_pools=num_pools,
            )

        # 跨区域协调队列
        self.cross_region_events: deque = deque(maxlen=100)

    def get_region_for_pool(self, pool_id: int) -> int:
        """获取渠池所属区域"""
        for region in self.REGIONS:
            if region['pools'][0] <= pool_id < region['pools'][1]:
                return region['id']
        return len(self.REGIONS) - 1

    def broadcast_event(self, event: L1ScenarioEvent):
        """广播事件"""
        region_id = self.get_region_for_pool(event.pool_id)
        if region_id in self.coordinators:
            self.coordinators[region_id].l1_manager.broadcast_event(event)

    def handle_cross_region_coordination(self,
                                          source_region: int,
                                          coord_type: CoordinationType,
                                          affected_regions: List[int]):
        """处理跨区域协调"""
        self.cross_region_events.append({
            'source_region': source_region,
            'coord_type': coord_type,
            'affected_regions': affected_regions,
            'timestamp': time.time(),
        })

        # Propagate coordination to affected region coordinators
        for region_id in affected_regions:
            if region_id in self.coordinators:
                self.coordinators[region_id].handle_cross_region(
                    source_region, coord_type,
                )
        logger.info("Cross-region coordination: %s, regions: %s",
                     coord_type.value, affected_regions)

    def coordination_step_all(self, dt: float = 60.0) -> Dict[int, Dict]:
        """所有协调器执行一步"""
        results = {}
        for region_id, coordinator in self.coordinators.items():
            results[region_id] = coordinator.coordination_step(dt)
        return results

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        total_events = 0
        total_coordinations = 0
        emergency_regions = []

        for region_id, coordinator in self.coordinators.items():
            total_events += coordinator.region_state['active_events']
            total_coordinations += len(coordinator.active_coordinations)
            if coordinator.region_state['emergency_mode']:
                emergency_regions.append(region_id)

        return {
            'total_regions': len(self.coordinators),
            'total_events': total_events,
            'total_coordinations': total_coordinations,
            'emergency_regions': emergency_regions,
            'cross_region_pending': len(self.cross_region_events),
            'timestamp': time.time(),
        }

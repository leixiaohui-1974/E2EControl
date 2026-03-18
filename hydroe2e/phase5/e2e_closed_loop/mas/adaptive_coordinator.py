"""
自适应多智能体系统协调器 (Adaptive MAS Coordinator)

功能:
1. 通信延迟模型 - 基于距离的延迟建模
2. 消息可靠性 - 丢包、乱序处理
3. 自适应共识 - 根据场景调整共识策略
4. 动态拓扑 - 支持故障隔离后的拓扑变化
5. 场景驱动角色 - 根据场景分配智能体角色
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import heapq
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """智能体角色"""
    NORMAL = "normal"           # 常规传输
    LEADER = "leader"           # 协调领导
    BUFFER = "buffer"           # 缓冲蓄水
    DISCHARGE = "discharge"     # 排放退水
    ISOLATED = "isolated"       # 隔离状态
    BYPASS = "bypass"           # 旁路模式


class MessageType(Enum):
    """消息类型"""
    STATE_BROADCAST = "state"           # 状态广播
    ACTION_PROPOSAL = "action"          # 动作提议
    CONSENSUS_VOTE = "vote"             # 共识投票
    EMERGENCY_ALERT = "emergency"       # 紧急告警
    ROLE_ASSIGNMENT = "role"            # 角色分配
    HEARTBEAT = "heartbeat"             # 心跳


@dataclass
class Message:
    """智能体消息"""
    msg_id: str
    sender_id: int
    receiver_id: int
    msg_type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 1                   # 优先级 (1最高)

    # 传输属性
    send_time: float = 0.0
    arrive_time: float = 0.0
    is_delivered: bool = False
    delivery_attempts: int = 0


@dataclass
class CommunicationChannel:
    """通信信道"""
    from_agent: int
    to_agent: int

    # 信道特性
    base_delay: float = 0.01            # s (基础延迟)
    distance_factor: float = 0.0001     # s/m (距离系数)
    jitter_std: float = 0.005           # s (抖动标准差)
    packet_loss_rate: float = 0.001     # 丢包率
    bandwidth: float = 1000.0           # msgs/s (带宽)

    # 状态
    is_active: bool = True
    queue_length: int = 0
    congestion_level: float = 0.0

    def compute_delay(self, distance: float) -> float:
        """计算传输延迟"""
        if not self.is_active:
            return float('inf')

        # 基础延迟 + 距离延迟 + 抖动 + 拥塞
        delay = self.base_delay + self.distance_factor * distance
        delay += np.random.normal(0, self.jitter_std)
        delay += self.congestion_level * 0.1  # 拥塞影响
        return max(delay, 0.001)

    def will_drop(self) -> bool:
        """判断是否丢包"""
        return np.random.random() < self.packet_loss_rate


@dataclass
class GateAgent:
    """闸门智能体"""
    agent_id: int
    position: float                     # m (沿程位置)

    # 状态
    water_level: float = 4.0
    flow_rate: float = 300.0
    gate_opening: float = 0.5
    role: AgentRole = AgentRole.NORMAL

    # 通信
    inbox: deque = field(default_factory=deque)
    outbox: deque = field(default_factory=deque)
    neighbors: List[int] = field(default_factory=list)

    # 共识状态
    local_proposal: float = 0.0
    consensus_value: float = 0.0
    votes_received: Dict[int, float] = field(default_factory=dict)

    # 健康状态
    is_healthy: bool = True
    last_heartbeat: float = 0.0


class CommunicationNetwork:
    """通信网络 (带延迟模型)"""

    def __init__(
        self,
        num_agents: int,
        total_length: float = 1432000.0,
    ):
        self.num_agents = num_agents
        self.total_length = total_length
        self.segment_length = total_length / (num_agents - 1)

        # 信道矩阵
        self.channels: Dict[Tuple[int, int], CommunicationChannel] = {}
        self._init_channels()

        # 消息队列 (按到达时间排序)
        self.message_queue: List[Tuple[float, Message]] = []

        # 当前时间
        self.current_time = 0.0

        # 统计
        self.total_sent = 0
        self.total_delivered = 0
        self.total_dropped = 0
        self.total_delayed = 0

    def _init_channels(self):
        """初始化通信信道"""
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = abs(i - j) * self.segment_length
                    self.channels[(i, j)] = CommunicationChannel(
                        from_agent=i,
                        to_agent=j,
                        # 远距离通信延迟更大
                        base_delay=0.01 + 0.001 * (abs(i - j)),
                        distance_factor=0.0001,
                        packet_loss_rate=0.001 * (1 + abs(i - j) * 0.1),
                    )

    def send_message(self, msg: Message) -> bool:
        """发送消息"""
        channel_key = (msg.sender_id, msg.receiver_id)
        channel = self.channels.get(channel_key)

        if channel is None or not channel.is_active:
            self.total_dropped += 1
            return False

        # 检查丢包
        if channel.will_drop():
            self.total_dropped += 1
            return False

        # 计算延迟
        distance = abs(msg.sender_id - msg.receiver_id) * self.segment_length
        delay = channel.compute_delay(distance)

        msg.send_time = self.current_time
        msg.arrive_time = self.current_time + delay

        # 加入优先队列
        heapq.heappush(self.message_queue, (msg.arrive_time, msg))

        self.total_sent += 1
        return True

    def step(self, dt: float) -> List[Message]:
        """推进时间,返回到达的消息"""
        self.current_time += dt
        arrived = []

        while self.message_queue and self.message_queue[0][0] <= self.current_time:
            _, msg = heapq.heappop(self.message_queue)
            msg.is_delivered = True
            arrived.append(msg)
            self.total_delivered += 1

        return arrived

    def disable_channel(self, from_agent: int, to_agent: int):
        """禁用信道 (故障隔离)"""
        key = (from_agent, to_agent)
        if key in self.channels:
            self.channels[key].is_active = False

    def enable_channel(self, from_agent: int, to_agent: int):
        """启用信道"""
        key = (from_agent, to_agent)
        if key in self.channels:
            self.channels[key].is_active = True

    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        return {
            "total_sent": self.total_sent,
            "total_delivered": self.total_delivered,
            "total_dropped": self.total_dropped,
            "delivery_rate": self.total_delivered / max(self.total_sent, 1),
            "avg_queue_length": len(self.message_queue),
        }


class AdaptiveConsensusProtocol:
    """自适应共识协议"""

    def __init__(
        self,
        convergence_threshold: float = 0.01,
        max_rounds: int = 5,
    ):
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds

        # 自适应参数
        self.base_weight = 0.5
        self.neighbor_weight = 0.25

    def compute_weights(
        self,
        agent: GateAgent,
        neighbors: List[GateAgent],
        scenario: str,
    ) -> Dict[int, float]:
        """
        计算共识权重 (场景自适应)
        """
        weights = {}

        # 根据场景调整权重策略
        if scenario == "S3_POLLUTION":
            # 污染场景: 上游权重更大
            for n in neighbors:
                if n.agent_id < agent.agent_id:
                    weights[n.agent_id] = 0.4  # 上游
                else:
                    weights[n.agent_id] = 0.1  # 下游
        elif scenario == "S2_DEMAND_SURGE":
            # 需求激增: 下游权重更大
            for n in neighbors:
                if n.agent_id > agent.agent_id:
                    weights[n.agent_id] = 0.4  # 下游
                else:
                    weights[n.agent_id] = 0.1  # 上游
        else:
            # 默认: 均匀权重
            for n in neighbors:
                weights[n.agent_id] = self.neighbor_weight

        # 自身权重
        weights[agent.agent_id] = 1.0 - sum(weights.values())

        return weights

    def consensus_step(
        self,
        agent: GateAgent,
        neighbor_values: Dict[int, float],
        weights: Dict[int, float],
    ) -> float:
        """执行一步共识更新"""
        new_value = weights.get(agent.agent_id, 0.5) * agent.local_proposal

        for neighbor_id, value in neighbor_values.items():
            w = weights.get(neighbor_id, self.neighbor_weight)
            new_value += w * value

        return new_value

    def check_convergence(
        self,
        values: List[float],
        prev_values: List[float],
    ) -> bool:
        """检查收敛"""
        if len(values) != len(prev_values):
            return False

        max_diff = max(abs(v - pv) for v, pv in zip(values, prev_values))
        return max_diff < self.convergence_threshold


class AdaptiveMASCoordinator:
    """
    自适应多智能体系统协调器

    整合通信网络、共识协议、角色管理
    """

    def __init__(
        self,
        num_agents: int = 64,
        total_length: float = 1432000.0,
    ):
        self.num_agents = num_agents
        self.total_length = total_length

        # 智能体
        self.agents: Dict[int, GateAgent] = {}
        self._init_agents()

        # 通信网络
        self.network = CommunicationNetwork(num_agents, total_length)

        # 共识协议
        self.consensus = AdaptiveConsensusProtocol()

        # 当前场景
        self.current_scenario = "S1_NORMAL"

        # 角色分配表
        self.role_assignments: Dict[str, Dict[int, AgentRole]] = {}

        # 时间
        self.current_time = 0.0

        # 统计
        self.consensus_rounds = 0
        self.role_changes = 0

        logger.info(f"AdaptiveMASCoordinator initialized: {num_agents} agents")

    def _init_agents(self):
        """初始化智能体"""
        segment_length = self.total_length / (self.num_agents - 1)

        for i in range(self.num_agents):
            position = i * segment_length

            # 邻居 (上下游)
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < self.num_agents - 1:
                neighbors.append(i + 1)

            self.agents[i] = GateAgent(
                agent_id=i,
                position=position,
                neighbors=neighbors,
            )

    def update_states(
        self,
        levels: np.ndarray,
        flows: np.ndarray,
        gate_openings: np.ndarray,
    ):
        """更新智能体状态"""
        for i, agent in self.agents.items():
            if i < len(levels):
                agent.water_level = levels[i]
            if i < len(flows):
                agent.flow_rate = flows[i]
            if i < len(gate_openings):
                agent.gate_opening = gate_openings[i]

    def assign_roles(self, scenario: str, affected_segments: Optional[List[int]] = None):
        """
        根据场景分配角色

        Args:
            scenario: 场景类型
            affected_segments: 受影响的渠段
        """
        self.current_scenario = scenario
        affected = affected_segments or []

        for agent_id, agent in self.agents.items():
            if agent_id in affected:
                # 受影响区域的角色分配
                if scenario == "S3_POLLUTION":
                    agent.role = AgentRole.ISOLATED
                elif scenario == "S8_EMERGENCY_REPAIR":
                    agent.role = AgentRole.ISOLATED
                elif scenario == "S7_PLANNED_MAINTENANCE":
                    agent.role = AgentRole.BYPASS
            else:
                # 根据场景的默认角色
                if scenario == "S1_NORMAL":
                    agent.role = AgentRole.NORMAL
                elif scenario == "S2_DEMAND_SURGE":
                    if agent_id > self.num_agents // 2:
                        agent.role = AgentRole.BUFFER
                    else:
                        agent.role = AgentRole.NORMAL
                elif scenario == "S4_FLOOD":
                    agent.role = AgentRole.DISCHARGE
                else:
                    agent.role = AgentRole.NORMAL

        self.role_changes += 1
        logger.info(f"Roles assigned for scenario {scenario}")

    def run_consensus(
        self,
        proposals: np.ndarray,
        max_rounds: Optional[int] = None,
    ) -> np.ndarray:
        """
        运行自适应共识

        Args:
            proposals: 各智能体的初始提议 [num_agents]
            max_rounds: 最大轮数

        Returns:
            consensus_values: 共识结果 [num_agents]
        """
        max_rounds = max_rounds or self.consensus.max_rounds

        # 初始化提议
        for i, agent in self.agents.items():
            if i < len(proposals):
                agent.local_proposal = proposals[i]
                agent.consensus_value = proposals[i]

        # 迭代共识
        for round_num in range(max_rounds):
            prev_values = [a.consensus_value for a in self.agents.values()]

            # 广播状态
            self._broadcast_states()

            # 处理消息
            arrived = self.network.step(dt=0.1)
            self._process_messages(arrived)

            # 共识更新
            for agent in self.agents.values():
                if agent.role == AgentRole.ISOLATED:
                    continue  # 隔离的智能体不参与共识

                neighbors = [self.agents[n] for n in agent.neighbors if n in self.agents]
                weights = self.consensus.compute_weights(
                    agent, neighbors, self.current_scenario
                )

                neighbor_values = {n: self.agents[n].consensus_value for n in agent.neighbors}
                agent.consensus_value = self.consensus.consensus_step(
                    agent, neighbor_values, weights
                )

            # 检查收敛
            current_values = [a.consensus_value for a in self.agents.values()]
            if self.consensus.check_convergence(current_values, prev_values):
                logger.debug(f"Consensus converged at round {round_num + 1}")
                break

        self.consensus_rounds += round_num + 1

        return np.array([a.consensus_value for a in self.agents.values()])

    def _broadcast_states(self):
        """广播状态"""
        for agent in self.agents.values():
            for neighbor_id in agent.neighbors:
                msg = Message(
                    msg_id=f"{agent.agent_id}_{neighbor_id}_{self.current_time}",
                    sender_id=agent.agent_id,
                    receiver_id=neighbor_id,
                    msg_type=MessageType.STATE_BROADCAST,
                    payload={
                        "level": agent.water_level,
                        "flow": agent.flow_rate,
                        "consensus_value": agent.consensus_value,
                        "role": agent.role.value,
                    },
                    timestamp=self.current_time,
                )
                self.network.send_message(msg)

    def _process_messages(self, messages: List[Message]):
        """处理到达的消息"""
        for msg in messages:
            receiver = self.agents.get(msg.receiver_id)
            if receiver is None:
                continue

            receiver.inbox.append(msg)

            # 根据消息类型处理
            if msg.msg_type == MessageType.STATE_BROADCAST:
                # 更新邻居状态视图
                pass
            elif msg.msg_type == MessageType.EMERGENCY_ALERT:
                # 紧急处理
                logger.warning(f"Agent {msg.receiver_id} received emergency alert")

    def apply_global_constraints(
        self,
        actions: np.ndarray,
        constraints: Dict[str, Any],
    ) -> np.ndarray:
        """
        应用全局约束

        Args:
            actions: 原始动作 [num_agents]
            constraints: 约束条件

        Returns:
            constrained_actions: 约束后的动作
        """
        constrained = actions.copy()

        # 约束1: 开度变化率限制
        max_rate = constraints.get("gate_opening_rate_max", 0.001)
        for i, agent in self.agents.items():
            if i < len(constrained):
                delta = constrained[i] - agent.gate_opening
                delta = np.clip(delta, -max_rate, max_rate)
                constrained[i] = agent.gate_opening + delta

        # 约束2: 上下游协调 (上游动作不能比下游早太多)
        coordination_delay = constraints.get("coordination_delay", 300)  # s
        # 实际实现需要时间序列规划

        # 约束3: 隔离区域保持不动
        for i, agent in self.agents.items():
            if agent.role == AgentRole.ISOLATED:
                constrained[i] = agent.gate_opening

        # 约束4: 开度范围
        constrained = np.clip(constrained, 0.0, 1.0)

        return constrained

    def step(self, dt: float):
        """推进时间"""
        self.current_time += dt

        # 处理网络消息
        arrived = self.network.step(dt)
        self._process_messages(arrived)

        # 健康检查
        self._check_agent_health()

    def _check_agent_health(self):
        """检查智能体健康状态"""
        timeout = 60.0  # s

        for agent in self.agents.values():
            if self.current_time - agent.last_heartbeat > timeout:
                if agent.is_healthy:
                    agent.is_healthy = False
                    logger.warning(f"Agent {agent.agent_id} marked unhealthy")

    def isolate_segment(self, segment_id: int):
        """隔离渠段"""
        if segment_id in self.agents:
            self.agents[segment_id].role = AgentRole.ISOLATED

            # 禁用相关信道
            for neighbor in self.agents[segment_id].neighbors:
                self.network.disable_channel(segment_id, neighbor)
                self.network.disable_channel(neighbor, segment_id)

            logger.info(f"Segment {segment_id} isolated")

    def restore_segment(self, segment_id: int):
        """恢复渠段"""
        if segment_id in self.agents:
            self.agents[segment_id].role = AgentRole.NORMAL

            # 恢复信道
            for neighbor in self.agents[segment_id].neighbors:
                self.network.enable_channel(segment_id, neighbor)
                self.network.enable_channel(neighbor, segment_id)

            logger.info(f"Segment {segment_id} restored")

    def get_coordination_status(self) -> Dict[str, Any]:
        """获取协调状态"""
        role_counts = {}
        for agent in self.agents.values():
            role = agent.role.value
            role_counts[role] = role_counts.get(role, 0) + 1

        healthy_count = sum(1 for a in self.agents.values() if a.is_healthy)

        return {
            "num_agents": self.num_agents,
            "healthy_agents": healthy_count,
            "current_scenario": self.current_scenario,
            "role_distribution": role_counts,
            "consensus_rounds_total": self.consensus_rounds,
            "role_changes_total": self.role_changes,
            "network_stats": self.network.get_network_stats(),
            "current_time": self.current_time,
        }

    def get_agent_states(self) -> Dict[int, Dict[str, Any]]:
        """获取所有智能体状态"""
        return {
            agent_id: {
                "position": agent.position,
                "water_level": agent.water_level,
                "flow_rate": agent.flow_rate,
                "gate_opening": agent.gate_opening,
                "role": agent.role.value,
                "is_healthy": agent.is_healthy,
                "consensus_value": agent.consensus_value,
            }
            for agent_id, agent in self.agents.items()
        }

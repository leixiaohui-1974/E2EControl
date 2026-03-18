"""
多智能体协同决策系统 (Multi-Agent Coordinator)
南水北调中线64个闸门的分布式协同控制

特点:
1. 每个闸门作为独立智能体
2. 上下游协调博弈
3. 全局目标与局部目标平衡
4. 共识协议保证一致性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """智能体配置"""
    agent_id: int
    pool_id: int                           # 关联的渠池
    upstream_agents: List[int]             # 上游智能体
    downstream_agents: List[int]           # 下游智能体
    state_dim: int = 32
    action_dim: int = 1                    # 闸门开度
    hidden_dim: int = 64


class CommunicationType(Enum):
    """通信类型"""
    BROADCAST = "broadcast"                # 广播
    UPSTREAM = "upstream"                  # 向上游
    DOWNSTREAM = "downstream"              # 向下游
    NEIGHBOR = "neighbor"                  # 邻居


@dataclass
class Message:
    """智能体间消息"""
    sender_id: int
    receiver_id: int
    msg_type: CommunicationType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 0


class GateAgent(nn.Module):
    """
    闸门智能体
    负责单个闸门的决策
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.agent_id = config.agent_id

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),  # [水位, 入流, 出流, 开度, 目标, 上游信号, 下游信号, 时间]
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )

        # 消息编码器
        self.message_encoder = nn.Sequential(
            nn.Linear(config.state_dim + 4, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )

        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.action_dim),
            nn.Sigmoid()  # 输出 [0, 1] 开度
        )

        # 价值网络
        self.value = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # 消息缓冲
        self.inbox = deque(maxlen=100)
        self.outbox = deque(maxlen=100)

        # 局部状态
        self.local_state = None
        self.received_messages = {}

    def encode_state(self, observation: Dict) -> torch.Tensor:
        """编码局部观测"""
        features = torch.FloatTensor([
            observation.get('level', 4.0),
            observation.get('inflow', 300.0),
            observation.get('outflow', 280.0),
            observation.get('gate_opening', 0.8),
            observation.get('target_level', 4.0),
            observation.get('upstream_signal', 0.0),
            observation.get('downstream_signal', 0.0),
            observation.get('time_feature', 0.5)
        ])

        self.local_state = self.state_encoder(features)
        return self.local_state

    def process_messages(self) -> torch.Tensor:
        """处理收到的消息"""
        if not self.received_messages:
            return torch.zeros(self.config.state_dim)

        # 聚合消息
        msg_features = []
        for sender_id, msg in self.received_messages.items():
            content = msg.content
            msg_vec = torch.FloatTensor([
                content.get('state', torch.zeros(self.config.state_dim)).mean().item()
                if isinstance(content.get('state'), torch.Tensor)
                else 0.0,
                content.get('action', 0.5),
                content.get('priority', 0),
                1.0 if msg.msg_type == CommunicationType.UPSTREAM else -1.0
            ])
            msg_features.append(msg_vec)

        if msg_features:
            msg_tensor = torch.stack(msg_features).mean(dim=0)
            # 扩展到 state_dim
            msg_encoded = torch.zeros(self.config.state_dim + 4)
            msg_encoded[:4] = msg_tensor
            return self.message_encoder(msg_encoded)
        else:
            return torch.zeros(self.config.state_dim)

    def decide(self, observation: Dict) -> Tuple[float, float]:
        """
        决策

        Returns:
            action: 闸门开度
            value: 状态价值估计
        """
        # 编码状态
        state = self.encode_state(observation)

        # 处理消息
        msg_state = self.process_messages()

        # 组合特征
        combined = torch.cat([state, msg_state])

        # 策略输出
        action = self.policy(combined)
        value = self.value(combined)

        return action.item(), value.item()

    def send_message(self, receiver_id: int, msg_type: CommunicationType,
                     content: Dict, priority: int = 0):
        """发送消息"""
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            msg_type=msg_type,
            content=content,
            timestamp=0.0,  # 由协调器填充
            priority=priority
        )
        self.outbox.append(msg)

    def receive_message(self, msg: Message):
        """接收消息"""
        self.inbox.append(msg)
        self.received_messages[msg.sender_id] = msg

    def broadcast_state(self):
        """广播自身状态给邻居"""
        content = {
            'state': self.local_state,
            'action': self.config.action_dim,
            'agent_id': self.agent_id
        }

        # 向上游发送
        for upstream_id in self.config.upstream_agents:
            self.send_message(upstream_id, CommunicationType.DOWNSTREAM, content)

        # 向下游发送
        for downstream_id in self.config.downstream_agents:
            self.send_message(downstream_id, CommunicationType.UPSTREAM, content)


class ConsensusProtocol:
    """
    共识协议
    确保多智能体决策的一致性
    """

    def __init__(self, num_agents: int, consensus_rounds: int = 3):
        self.num_agents = num_agents
        self.consensus_rounds = consensus_rounds

        # 共识状态
        self.agent_proposals = {}
        self.consensus_reached = False
        self.final_decisions = {}

    def propose(self, agent_id: int, action: float, confidence: float):
        """智能体提议"""
        self.agent_proposals[agent_id] = {
            'action': action,
            'confidence': confidence
        }

    def run_consensus(self, agents: Dict[int, GateAgent]) -> Dict[int, float]:
        """
        运行共识算法

        Returns:
            final_actions: 各智能体的最终动作
        """
        for round_idx in range(self.consensus_rounds):
            # 每轮更新
            new_proposals = {}

            for agent_id, agent in agents.items():
                # 获取邻居提议
                neighbor_actions = []
                neighbor_weights = []

                for neighbor_id in agent.config.upstream_agents + agent.config.downstream_agents:
                    if neighbor_id in self.agent_proposals:
                        neighbor_actions.append(self.agent_proposals[neighbor_id]['action'])
                        neighbor_weights.append(self.agent_proposals[neighbor_id]['confidence'])

                # 加权平均
                if neighbor_actions:
                    own_action = self.agent_proposals[agent_id]['action']
                    own_weight = self.agent_proposals[agent_id]['confidence']

                    all_actions = [own_action] + neighbor_actions
                    all_weights = [own_weight * 2] + neighbor_weights  # 自身权重加倍

                    # 归一化权重
                    total_weight = sum(all_weights)
                    normalized_weights = [w / total_weight for w in all_weights]

                    # 加权平均
                    consensus_action = sum(a * w for a, w in zip(all_actions, normalized_weights))

                    new_proposals[agent_id] = {
                        'action': consensus_action,
                        'confidence': self.agent_proposals[agent_id]['confidence']
                    }
                else:
                    new_proposals[agent_id] = self.agent_proposals[agent_id]

            self.agent_proposals = new_proposals

        # 提取最终决策
        self.final_decisions = {
            agent_id: prop['action']
            for agent_id, prop in self.agent_proposals.items()
        }
        self.consensus_reached = True

        return self.final_decisions

    def reset(self):
        """重置共识状态"""
        self.agent_proposals.clear()
        self.consensus_reached = False
        self.final_decisions.clear()


class MultiAgentCoordinator:
    """
    多智能体协调器
    管理所有闸门智能体的协同决策
    """

    def __init__(self, num_gates: int = 64, num_pools: int = 63):
        self.num_gates = num_gates
        self.num_pools = num_pools

        # 创建智能体
        self.agents = {}
        self._create_agents()

        # 共识协议
        self.consensus = ConsensusProtocol(num_gates)

        # 全局目标
        self.global_objectives = {
            'target_levels': np.ones(num_pools) * 4.0,
            'min_flow': 200.0,
            'max_level_deviation': 0.3
        }

        # 协调统计
        self.stats = {
            'total_rounds': 0,
            'consensus_success_rate': 0.0,
            'avg_coordination_time': 0.0
        }

        logger.info(f"Multi-Agent Coordinator initialized with {num_gates} agents")

    def _create_agents(self):
        """创建智能体网络"""
        for i in range(self.num_gates):
            # 确定上下游关系
            upstream = [i - 1] if i > 0 else []
            downstream = [i + 1] if i < self.num_gates - 1 else []

            config = AgentConfig(
                agent_id=i,
                pool_id=min(i, self.num_pools - 1),
                upstream_agents=upstream,
                downstream_agents=downstream
            )

            self.agents[i] = GateAgent(config)

    def coordinate(self, state: Dict) -> Dict[int, float]:
        """
        协调所有智能体做出一致决策

        Args:
            state: 全局状态

        Returns:
            actions: 各闸门的动作
        """
        self.stats['total_rounds'] += 1

        # 1. 分发观测到各智能体
        observations = self._distribute_observations(state)

        # 2. 各智能体独立决策
        proposals = {}
        for agent_id, agent in self.agents.items():
            obs = observations.get(agent_id, {})
            action, value = agent.decide(obs)
            proposals[agent_id] = (action, value)

            # 提交到共识
            self.consensus.propose(agent_id, action, confidence=0.8)

        # 3. 智能体间通信
        self._facilitate_communication()

        # 4. 运行共识协议
        final_actions = self.consensus.run_consensus(self.agents)

        # 5. 全局约束修正
        final_actions = self._apply_global_constraints(final_actions, state)

        # 重置共识
        self.consensus.reset()

        return final_actions

    def _distribute_observations(self, state: Dict) -> Dict[int, Dict]:
        """分发观测"""
        levels = state.get('levels', [4.0] * self.num_pools)
        inflows = state.get('inflows', [300.0] * self.num_pools)
        outflows = state.get('outflows', [280.0] * self.num_pools)
        gates = state.get('gates', [0.8] * self.num_gates)
        targets = state.get('targets', [4.0] * self.num_pools)

        observations = {}
        for agent_id in range(self.num_gates):
            pool_id = min(agent_id, self.num_pools - 1)
            observations[agent_id] = {
                'level': levels[pool_id],
                'inflow': inflows[pool_id],
                'outflow': outflows[pool_id],
                'gate_opening': gates[agent_id],
                'target_level': targets[pool_id],
                'upstream_signal': levels[pool_id - 1] if pool_id > 0 else levels[pool_id],
                'downstream_signal': levels[pool_id + 1] if pool_id < self.num_pools - 1 else levels[pool_id],
                'time_feature': 0.5
            }

        return observations

    def _facilitate_communication(self):
        """促进智能体间通信"""
        # 收集所有待发消息
        all_messages = []
        for agent in self.agents.values():
            agent.broadcast_state()
            while agent.outbox:
                all_messages.append(agent.outbox.popleft())

        # 分发消息
        for msg in all_messages:
            if msg.receiver_id in self.agents:
                self.agents[msg.receiver_id].receive_message(msg)

    def _apply_global_constraints(self,
                                   actions: Dict[int, float],
                                   state: Dict) -> Dict[int, float]:
        """应用全局约束"""
        constrained_actions = actions.copy()
        levels = state.get('levels', [4.0] * self.num_pools)

        # 约束1: 上下游协调 (避免水位震荡)
        for i in range(1, self.num_gates):
            # 如果上游水位低，减少当前闸门开度
            upstream_pool = min(i - 1, self.num_pools - 1)
            if levels[upstream_pool] < 3.0:
                constrained_actions[i] = min(constrained_actions[i], 0.5)

        # 约束2: 变化率限制
        current_gates = state.get('gates', [0.8] * self.num_gates)
        max_change = 0.1
        for i, action in constrained_actions.items():
            current = current_gates[i] if i < len(current_gates) else 0.8
            constrained_actions[i] = np.clip(
                action,
                current - max_change,
                current + max_change
            )

        return constrained_actions

    def set_global_objectives(self, objectives: Dict):
        """设置全局目标"""
        self.global_objectives.update(objectives)

    def get_agent_status(self, agent_id: int) -> Dict:
        """获取智能体状态"""
        if agent_id not in self.agents:
            return {'error': 'Agent not found'}

        agent = self.agents[agent_id]
        return {
            'agent_id': agent_id,
            'config': agent.config.__dict__,
            'local_state': agent.local_state.tolist() if agent.local_state is not None else None,
            'inbox_size': len(agent.inbox),
            'outbox_size': len(agent.outbox)
        }

    def get_coordination_report(self) -> Dict:
        """获取协调报告"""
        return {
            'num_agents': self.num_gates,
            'stats': self.stats.copy(),
            'global_objectives': self.global_objectives.copy(),
            'consensus_status': {
                'reached': self.consensus.consensus_reached,
                'num_proposals': len(self.consensus.agent_proposals)
            }
        }


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Multi-Agent Coordinator Test")
    logger.info("=" * 70)

    # 创建协调器
    coordinator = MultiAgentCoordinator(num_gates=11, num_pools=10)

    # 模拟状态
    state = {
        'levels': np.random.uniform(3.5, 4.5, 10).tolist(),
        'inflows': np.random.uniform(250, 350, 10).tolist(),
        'outflows': np.random.uniform(240, 340, 10).tolist(),
        'gates': np.random.uniform(0.7, 0.9, 11).tolist(),
        'targets': [4.0] * 10
    }

    # 协调决策
    logger.info("\n运行协调决策...")
    actions = coordinator.coordinate(state)

    logger.info(f"\n协调结果:")
    for agent_id, action in sorted(actions.items()):
        logger.info(f"  闸门 {agent_id}: {action:.3f}")

    # 获取报告
    report = coordinator.get_coordination_report()
    logger.info(f"\n协调报告:")
    logger.info(f"  智能体数量: {report['num_agents']}")
    logger.info(f"  总轮次: {report['stats']['total_rounds']}")
    logger.info(f"  共识状态: {report['consensus_status']}")

    logger.info("\n" + "=" * 70)
    logger.info("Test completed!")
    logger.info("=" * 70)

"""
水网拓扑管理器
使用NetworkX建立水网图结构，支持复杂拓扑关系
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import matplotlib.pyplot as plt
from collections import deque


class NodeType(Enum):
    """节点类型"""
    POOL = "pool"              # 渠池
    GATE = "gate"              # 闸门
    PUMP = "pump"              # 泵站
    JUNCTION = "junction"      # 汇流点
    OFFTAKE = "offtake"        # 分水口
    RESERVOIR = "reservoir"    # 水库
    USER = "user"              # 用户取水点


class EdgeType(Enum):
    """连接类型"""
    CHANNEL = "channel"        # 渠道连接
    PIPE = "pipe"              # 管道连接
    NATURAL = "natural"        # 自然河道
    CONTROL = "control"        # 控制连接


class NetworkNode:
    """网络节点"""
    
    def __init__(self, node_id: str, node_type: NodeType, properties: Dict = None):
        """
        初始化节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            properties: 节点属性字典
        """
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties or {}
        
        # 默认属性
        if node_type == NodeType.POOL:
            self.properties.setdefault('area', 10000.0)  # m²
            self.properties.setdefault('max_level', 8.0)  # m
            self.properties.setdefault('min_level', 0.5)  # m
        elif node_type == NodeType.GATE:
            self.properties.setdefault('max_flow', 20.0)  # m³/s
            self.properties.setdefault('response_time', 60.0)  # s
        elif node_type == NodeType.PUMP:
            self.properties.setdefault('max_power', 100.0)  # kW
            self.properties.setdefault('efficiency', 0.85)
    
    def __repr__(self):
        return f"Node({self.node_id}, {self.node_type.value})"


class NetworkEdge:
    """网络边（连接）"""
    
    def __init__(self, from_node: str, to_node: str, 
                 edge_type: EdgeType, properties: Dict = None):
        """
        初始化边
        
        Args:
            from_node: 起始节点ID
            to_node: 目标节点ID
            edge_type: 边类型
            properties: 边属性字典
        """
        self.from_node = from_node
        self.to_node = to_node
        self.edge_type = edge_type
        self.properties = properties or {}
        
        # 默认属性
        self.properties.setdefault('length', 1000.0)  # m
        self.properties.setdefault('delay', 1)  # 时间步数
        self.properties.setdefault('capacity', 30.0)  # m³/s
    
    def __repr__(self):
        return f"Edge({self.from_node} -> {self.to_node}, {self.edge_type.value})"


class WaterNetworkTopology:
    """水网拓扑管理器"""
    
    def __init__(self):
        """初始化拓扑管理器"""
        self.graph = nx.DiGraph()  # 有向图
        self.nodes = {}  # {node_id: NetworkNode}
        self.edges = []  # [NetworkEdge]
        
    def add_node(self, node: NetworkNode):
        """
        添加节点
        
        Args:
            node: 网络节点
        """
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, 
                           type=node.node_type.value,
                           **node.properties)
    
    def add_edge(self, edge: NetworkEdge):
        """
        添加连接
        
        Args:
            edge: 网络边
        """
        if edge.from_node not in self.nodes:
            raise ValueError(f"起始节点 {edge.from_node} 不存在")
        if edge.to_node not in self.nodes:
            raise ValueError(f"目标节点 {edge.to_node} 不存在")
        
        self.edges.append(edge)
        self.graph.add_edge(edge.from_node, edge.to_node,
                           type=edge.edge_type.value,
                           **edge.properties)
    
    def get_upstream_nodes(self, node_id: str, 
                          distance: int = 1) -> List[str]:
        """
        获取上游节点
        
        Args:
            node_id: 节点ID
            distance: 距离（跳数），1表示直接上游
            
        Returns:
            上游节点ID列表
        """
        if distance == 1:
            return list(self.graph.predecessors(node_id))
        else:
            # BFS搜索指定距离内的上游节点
            upstream = []
            visited = set()
            queue = deque([(node_id, 0)])
            
            while queue:
                current, dist = queue.popleft()
                if dist > distance:
                    break
                
                for pred in self.graph.predecessors(current):
                    if pred not in visited:
                        visited.add(pred)
                        if dist + 1 <= distance and dist > 0:
                            upstream.append(pred)
                        queue.append((pred, dist + 1))
            
            return upstream
    
    def get_downstream_nodes(self, node_id: str, 
                            distance: int = 1) -> List[str]:
        """
        获取下游节点
        
        Args:
            node_id: 节点ID
            distance: 距离（跳数）
            
        Returns:
            下游节点ID列表
        """
        if distance == 1:
            return list(self.graph.successors(node_id))
        else:
            # BFS搜索
            downstream = []
            visited = set()
            queue = deque([(node_id, 0)])
            
            while queue:
                current, dist = queue.popleft()
                if dist > distance:
                    break
                
                for succ in self.graph.successors(current):
                    if succ not in visited:
                        visited.add(succ)
                        if dist + 1 <= distance and dist > 0:
                            downstream.append(succ)
                        queue.append((succ, dist + 1))
            
            return downstream
    
    def get_path(self, from_node: str, to_node: str) -> List[str]:
        """
        获取两节点间的路径
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            
        Returns:
            路径节点列表
        """
        try:
            return nx.shortest_path(self.graph, from_node, to_node)
        except nx.NetworkXNoPath:
            return []
    
    def get_pools(self) -> List[str]:
        """获取所有渠池节点"""
        return [nid for nid, node in self.nodes.items() 
                if node.node_type == NodeType.POOL]
    
    def get_gates(self) -> List[str]:
        """获取所有闸门节点"""
        return [nid for nid, node in self.nodes.items() 
                if node.node_type == NodeType.GATE]
    
    def get_control_sequence(self) -> List[str]:
        """
        获取控制顺序（拓扑排序）
        
        Returns:
            按拓扑顺序排列的节点ID列表
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # 如果有环，返回所有节点
            return list(self.nodes.keys())
    
    def find_critical_nodes(self) -> List[str]:
        """
        找出关键节点（删除后影响连通性的节点）
        
        Returns:
            关键节点ID列表
        """
        critical = []
        for node in self.nodes.keys():
            # 创建临时图，移除该节点
            temp_graph = self.graph.copy()
            temp_graph.remove_node(node)
            
            # 检查连通性是否变化
            if not nx.is_weakly_connected(temp_graph):
                critical.append(node)
        
        return critical
    
    def get_subnetwork(self, node_ids: List[str]) -> 'WaterNetworkTopology':
        """
        提取子网络
        
        Args:
            node_ids: 要包含的节点ID列表
            
        Returns:
            子网络拓扑
        """
        subnet = WaterNetworkTopology()
        
        # 添加节点
        for nid in node_ids:
            if nid in self.nodes:
                subnet.add_node(self.nodes[nid])
        
        # 添加边
        for edge in self.edges:
            if edge.from_node in node_ids and edge.to_node in node_ids:
                subnet.add_edge(edge)
        
        return subnet
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证拓扑结构有效性
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查是否有孤立节点
        isolates = list(nx.isolates(self.graph))
        if isolates:
            errors.append(f"存在孤立节点: {isolates}")
        
        # 检查是否有环（对于某些拓扑不允许环）
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                errors.append(f"存在环路: {cycles}")
        except Exception:
            pass
        
        # 检查渠池是否有上下游连接
        for pool_id in self.get_pools():
            upstream = self.get_upstream_nodes(pool_id)
            downstream = self.get_downstream_nodes(pool_id)
            if not upstream:
                errors.append(f"渠池 {pool_id} 没有上游连接")
            if not downstream:
                errors.append(f"渠池 {pool_id} 没有下游连接")
        
        return len(errors) == 0, errors
    
    def visualize(self, save_path: str = None, figsize=(14, 10)):
        """
        可视化网络拓扑
        
        Args:
            save_path: 保存路径
            figsize: 图形大小
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 节点颜色映射
        color_map = {
            NodeType.POOL: '#3498db',      # 蓝色
            NodeType.GATE: '#e74c3c',      # 红色
            NodeType.PUMP: '#9b59b6',      # 紫色
            NodeType.JUNCTION: '#95a5a6',  # 灰色
            NodeType.OFFTAKE: '#f39c12',   # 橙色
            NodeType.RESERVOIR: '#1abc9c', # 青色
            NodeType.USER: '#34495e'       # 深灰
        }
        
        # 获取节点颜色
        node_colors = [color_map[self.nodes[n].node_type] for n in self.graph.nodes()]
        
        # 布局
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors,
                              node_size=800,
                              alpha=0.9,
                              ax=ax)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color='#7f8c8d',
                              arrows=True,
                              arrowsize=20,
                              width=2,
                              alpha=0.6,
                              ax=ax)
        
        # 绘制标签
        nx.draw_networkx_labels(self.graph, pos,
                               font_size=10,
                               font_weight='bold',
                               ax=ax)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=node_type.value.capitalize())
            for node_type, color in color_map.items()
            if any(n.node_type == node_type for n in self.nodes.values())
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title('Water Network Topology', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"拓扑图已保存: {save_path}")
        else:
            plt.show()
            plt.close()
    
    def export_to_dict(self) -> Dict:
        """导出为字典格式"""
        return {
            'nodes': [
                {
                    'id': node.node_id,
                    'type': node.node_type.value,
                    'properties': node.properties
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'type': edge.edge_type.value,
                    'properties': edge.properties
                }
                for edge in self.edges
            ]
        }
    
    def __str__(self):
        pools = self.get_pools()
        gates = self.get_gates()
        return (f"WaterNetworkTopology(\n"
                f"  Nodes: {len(self.nodes)}\n"
                f"  Edges: {len(self.edges)}\n"
                f"  Pools: {len(pools)}\n"
                f"  Gates: {len(gates)}\n"
                f")")


def create_simple_cascade(num_pools: int = 3) -> WaterNetworkTopology:
    """
    创建简单级联拓扑
    
    Args:
        num_pools: 渠池数量
        
    Returns:
        拓扑对象
    """
    topology = WaterNetworkTopology()
    
    # 添加上游水源
    topology.add_node(NetworkNode('source', NodeType.RESERVOIR, 
                                 {'capacity': 100.0}))
    
    # 添加渠池和闸门
    for i in range(num_pools):
        # 渠池
        pool = NetworkNode(f'pool_{i}', NodeType.POOL, 
                          {'area': 10000.0, 'index': i})
        topology.add_node(pool)
        
        # 入口闸门
        gate_in = NetworkNode(f'gate_{i}', NodeType.GATE,
                             {'max_flow': 20.0, 'position': 'inlet'})
        topology.add_node(gate_in)
        
        # 连接：上一个节点 -> 闸门 -> 渠池
        if i == 0:
            topology.add_edge(NetworkEdge('source', f'gate_{i}', EdgeType.CHANNEL))
        else:
            topology.add_edge(NetworkEdge(f'pool_{i-1}', f'gate_{i}', EdgeType.CHANNEL))
        
        topology.add_edge(NetworkEdge(f'gate_{i}', f'pool_{i}', EdgeType.CHANNEL))
    
    # 最后一个出口闸门
    gate_out = NetworkNode(f'gate_{num_pools}', NodeType.GATE,
                          {'max_flow': 20.0, 'position': 'outlet'})
    topology.add_node(gate_out)
    topology.add_edge(NetworkEdge(f'pool_{num_pools-1}', f'gate_{num_pools}', 
                                 EdgeType.CHANNEL))
    
    # 下游用户
    topology.add_node(NetworkNode('user', NodeType.USER, {'demand': 5.0}))
    topology.add_edge(NetworkEdge(f'gate_{num_pools}', 'user', EdgeType.CHANNEL))
    
    return topology


def create_complex_network() -> WaterNetworkTopology:
    """
    创建复杂水网拓扑（包含支渠、泵站等）
    
    Returns:
        拓扑对象
    """
    topology = WaterNetworkTopology()
    
    # 主渠道
    topology.add_node(NetworkNode('reservoir', NodeType.RESERVOIR, {'capacity': 200.0}))
    
    # 主渠3个池
    for i in range(3):
        topology.add_node(NetworkNode(f'main_pool_{i}', NodeType.POOL, 
                                     {'area': 15000.0, 'priority': 'high'}))
        topology.add_node(NetworkNode(f'main_gate_{i}', NodeType.GATE))
    
    # 连接主渠
    topology.add_edge(NetworkEdge('reservoir', 'main_gate_0', EdgeType.CHANNEL))
    for i in range(3):
        topology.add_edge(NetworkEdge(f'main_gate_{i}', f'main_pool_{i}', EdgeType.CHANNEL))
        if i < 2:
            topology.add_edge(NetworkEdge(f'main_pool_{i}', f'main_gate_{i+1}', EdgeType.CHANNEL))
    
    # 支渠1（从main_pool_1分出）
    topology.add_node(NetworkNode('junction_1', NodeType.JUNCTION))
    topology.add_edge(NetworkEdge('main_pool_1', 'junction_1', EdgeType.CHANNEL))
    
    topology.add_node(NetworkNode('branch_pool_1', NodeType.POOL, {'area': 8000.0}))
    topology.add_node(NetworkNode('branch_gate_1', NodeType.GATE))
    topology.add_edge(NetworkEdge('junction_1', 'branch_gate_1', EdgeType.CHANNEL))
    topology.add_edge(NetworkEdge('branch_gate_1', 'branch_pool_1', EdgeType.CHANNEL))
    
    # 泵站（从branch_pool_1提水）
    topology.add_node(NetworkNode('pump_1', NodeType.PUMP, 
                                 {'max_power': 150.0, 'lift': 10.0}))
    topology.add_edge(NetworkEdge('branch_pool_1', 'pump_1', EdgeType.PIPE))
    
    # 用户
    topology.add_node(NetworkNode('user_1', NodeType.USER, {'demand': 3.0}))
    topology.add_node(NetworkNode('user_2', NodeType.USER, {'demand': 5.0}))
    topology.add_node(NetworkNode('user_3', NodeType.USER, {'demand': 2.0}))
    
    topology.add_edge(NetworkEdge('pump_1', 'user_1', EdgeType.PIPE))
    topology.add_edge(NetworkEdge('main_pool_2', 'user_2', EdgeType.CHANNEL))
    topology.add_edge(NetworkEdge('branch_pool_1', 'user_3', EdgeType.CHANNEL))
    
    return topology


# 示例使用
if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "水网拓扑管理器演示")
    print("="*70)
    
    # 1. 简单级联
    print("\n1. 创建简单级联拓扑（3池）...")
    simple_topo = create_simple_cascade(num_pools=3)
    print(simple_topo)
    
    # 验证
    valid, errors = simple_topo.validate()
    if valid:
        print("✓ 拓扑结构有效")
    else:
        print("✗ 拓扑结构存在问题:")
        for error in errors:
            print(f"  - {error}")
    
    # 拓扑分析
    print("\n2. 拓扑分析...")
    pools = simple_topo.get_pools()
    print(f"   渠池: {pools}")
    
    for pool in pools:
        upstream = simple_topo.get_upstream_nodes(pool)
        downstream = simple_topo.get_downstream_nodes(pool)
        print(f"   {pool}: 上游={upstream}, 下游={downstream}")
    
    # 控制顺序
    control_seq = simple_topo.get_control_sequence()
    print(f"\n   控制顺序: {' -> '.join(control_seq)}")
    
    # 3. 复杂网络
    print("\n3. 创建复杂水网拓扑...")
    complex_topo = create_complex_network()
    print(complex_topo)
    
    # 关键节点
    critical = complex_topo.find_critical_nodes()
    print(f"\n   关键节点: {critical}")
    
    # 4. 可视化
    print("\n4. 生成拓扑可视化...")
    simple_topo.visualize('simple_topology.png')
    complex_topo.visualize('complex_topology.png')
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

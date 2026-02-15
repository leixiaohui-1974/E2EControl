"""
微观世界模型适配器 (MicroWorldAdapter)
连接宏观控制模型与 HydroGraphNet 微观模拟

功能:
1. 将宏观边界条件转换为图结构输入
2. 调用预训练的 HydroGraphNet 模型进行推理
3. 输出关键节点的流速矢量场和压力分布

架构:
    宏观模型 (MPC/LSTM)
           ↓
    MicroWorldAdapter
           ↓
    [边界条件 Q, Z] → [图结构数据] → [HydroGraphNet] → [流速场, 压力场]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# 尝试导入 DGL (可选依赖)
try:
    import dgl
    from dgl import DGLGraph
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    logger.warning("DGL 未安装，部分图功能将不可用")


# ==============================================================================
# 配置数据类
# ==============================================================================

@dataclass
class MicroWorldConfig:
    """微观世界模型配置"""
    # 网格参数
    num_nodes: int = 100            # 节点数量
    num_edges: int = 200            # 边数量
    spatial_resolution: float = 10.0  # 空间分辨率 [m]

    # 特征维度
    node_input_dim: int = 12        # 节点输入特征维度
    edge_input_dim: int = 3         # 边输入特征维度
    output_dim: int = 3             # 输出维度 [vx, vy, pressure]

    # 模型参数
    hidden_dim: int = 64
    num_layers: int = 5
    num_harmonics: int = 5          # KAN 谐波数

    # 时间参数
    dt: float = 60.0                # 微观模型时间步长 [s]
    macro_dt: float = 900.0         # 宏观模型时间步长 [s]


# ==============================================================================
# 图构建器
# ==============================================================================

class GraphBuilder:
    """
    图结构构建器

    将渠道几何和边界条件转换为 DGL 图
    """

    def __init__(self, config: MicroWorldConfig = None):
        self.config = config or MicroWorldConfig()

    def build_canal_graph(self,
                          length: float,
                          width: float,
                          num_nodes: int = None,
                          connectivity: str = 'chain') -> 'DGLGraph':
        """
        构建渠道图

        Args:
            length: 渠道长度 [m]
            width: 渠道宽度 [m]
            num_nodes: 节点数
            connectivity: 连接类型 ('chain', 'mesh', 'full')

        Returns:
            DGL 图
        """
        if not DGL_AVAILABLE:
            raise ImportError("需要安装 DGL: pip install dgl")

        num_nodes = num_nodes or self.config.num_nodes
        dx = length / (num_nodes - 1)

        # 创建节点位置
        x_coords = np.linspace(0, length, num_nodes)
        y_coords = np.zeros(num_nodes) + width / 2

        # 创建边
        if connectivity == 'chain':
            # 链式连接
            src = list(range(num_nodes - 1))
            dst = list(range(1, num_nodes))
            # 双向边
            src_all = src + dst
            dst_all = dst + src
        elif connectivity == 'mesh':
            # 网格连接 (包含邻近节点)
            src_all, dst_all = [], []
            for i in range(num_nodes):
                for j in range(max(0, i-2), min(num_nodes, i+3)):
                    if i != j:
                        src_all.append(i)
                        dst_all.append(j)
        else:
            # 全连接
            src_all, dst_all = [], []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        src_all.append(i)
                        dst_all.append(j)

        # 创建图
        g = dgl.graph((src_all, dst_all), num_nodes=num_nodes)

        # 添加节点特征
        g.ndata['x'] = torch.tensor(x_coords, dtype=torch.float32)
        g.ndata['y'] = torch.tensor(y_coords, dtype=torch.float32)
        g.ndata['pos'] = torch.stack([
            torch.tensor(x_coords, dtype=torch.float32),
            torch.tensor(y_coords, dtype=torch.float32)
        ], dim=1)

        # 添加边特征
        edge_lengths = []
        edge_angles = []
        for s, d in zip(src_all, dst_all):
            dx_e = x_coords[d] - x_coords[s]
            dy_e = y_coords[d] - y_coords[s]
            edge_lengths.append(np.sqrt(dx_e**2 + dy_e**2))
            edge_angles.append(np.arctan2(dy_e, dx_e))

        g.edata['length'] = torch.tensor(edge_lengths, dtype=torch.float32)
        g.edata['angle'] = torch.tensor(edge_angles, dtype=torch.float32)

        return g

    def set_boundary_conditions(self,
                               graph: 'DGLGraph',
                               upstream_Q: float,
                               downstream_Z: float,
                               initial_depth: float = 3.0) -> 'DGLGraph':
        """
        设置边界条件

        Args:
            graph: DGL 图
            upstream_Q: 上游流量 [m³/s]
            downstream_Z: 下游水位 [m]
            initial_depth: 初始水深 [m]

        Returns:
            带边界条件的图
        """
        num_nodes = graph.num_nodes()

        # 初始化节点特征
        # [x, y, depth, velocity_x, velocity_y, pressure, ...]
        node_features = torch.zeros(num_nodes, self.config.node_input_dim)

        # 位置
        node_features[:, 0] = graph.ndata['x']
        node_features[:, 1] = graph.ndata['y']

        # 初始水深
        node_features[:, 2] = initial_depth

        # 上游边界: 固定流量
        width = 50.0  # 假设宽度
        upstream_velocity = upstream_Q / (initial_depth * width)
        node_features[0, 3] = upstream_velocity  # vx
        node_features[0, 4] = 0.0  # vy

        # 下游边界: 固定水位
        node_features[-1, 2] = downstream_Z

        # 边界标记
        node_features[0, 5] = 1.0   # 上游边界
        node_features[-1, 6] = 1.0  # 下游边界

        graph.ndata['features'] = node_features

        return graph


# ==============================================================================
# HydroGraphNet 包装器
# ==============================================================================

class HydroGraphNetWrapper(nn.Module):
    """
    HydroGraphNet 模型包装器

    封装模型加载、推理等功能
    如果未安装 DGL，使用简化的 MLP 替代
    """

    def __init__(self, config: MicroWorldConfig = None):
        super().__init__()
        self.config = config or MicroWorldConfig()

        if DGL_AVAILABLE:
            self._build_gnn_model()
        else:
            self._build_fallback_model()

    def _build_gnn_model(self):
        """构建 GNN 模型"""
        from .kan_layers import KolmogorovArnoldNetwork

        # 节点编码器 (KAN)
        self.node_encoder = KolmogorovArnoldNetwork(
            input_dim=self.config.node_input_dim,
            output_dim=self.config.hidden_dim,
            num_harmonics=self.config.num_harmonics
        )

        # 边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.config.edge_input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # 消息传递层
        self.message_passing = nn.ModuleList([
            MessagePassingLayer(self.config.hidden_dim)
            for _ in range(self.config.num_layers)
        ])

        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim)
        )

    def _build_fallback_model(self):
        """构建替代 MLP 模型"""
        from .kan_layers import KolmogorovArnoldNetwork

        self.model = nn.Sequential(
            KolmogorovArnoldNetwork(
                self.config.node_input_dim,
                self.config.hidden_dim,
                self.config.num_harmonics
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim)
        )

    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor = None,
                graph: 'DGLGraph' = None) -> torch.Tensor:
        """
        前向传播

        Args:
            node_features: 节点特征 [num_nodes, node_input_dim]
            edge_features: 边特征 [num_edges, edge_input_dim]
            graph: DGL 图

        Returns:
            预测输出 [num_nodes, output_dim]
        """
        if DGL_AVAILABLE and graph is not None:
            return self._forward_gnn(node_features, edge_features, graph)
        else:
            return self._forward_mlp(node_features)

    def _forward_gnn(self, node_features, edge_features, graph):
        """GNN 前向传播"""
        # 编码
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)

        # 消息传递
        for mp_layer in self.message_passing:
            h = mp_layer(h, e, graph)

        # 解码
        output = self.decoder(h)
        return output

    def _forward_mlp(self, node_features):
        """MLP 前向传播"""
        return self.model(node_features)


class MessagePassingLayer(nn.Module):
    """消息传递层"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, e, graph):
        """消息传递"""
        if not DGL_AVAILABLE:
            return h

        with graph.local_scope():
            graph.ndata['h'] = h
            graph.edata['e'] = e

            # 消息函数
            graph.apply_edges(self._edge_update)

            # 聚合
            graph.update_all(
                dgl.function.copy_e('msg', 'm'),
                dgl.function.sum('m', 'agg')
            )

            # 节点更新
            agg = graph.ndata['agg']
            h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
            h_new = self.norm(h + h_new)

            return h_new

    def _edge_update(self, edges):
        """边更新"""
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        e = edges.data['e']
        msg = self.edge_mlp(torch.cat([src_h, dst_h, e], dim=-1))
        return {'msg': msg}


# ==============================================================================
# 微观世界适配器 (主类)
# ==============================================================================

class MicroWorldAdapter:
    """
    微观世界模型适配器

    主要功能:
    1. 将宏观边界条件 (Q, Z) 转换为图结构输入
    2. 调用 HydroGraphNet 进行微观推理
    3. 输出关键节点的流速矢量场和压力分布
    """

    def __init__(self,
                 config: MicroWorldConfig = None,
                 model_path: str = None,
                 device: str = None):
        """
        Args:
            config: 配置
            model_path: 预训练模型路径
            device: 计算设备
        """
        self.config = config or MicroWorldConfig()

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # 图构建器
        self.graph_builder = GraphBuilder(self.config)

        # 模型
        self.model = HydroGraphNetWrapper(self.config).to(self.device)

        # 加载预训练权重
        if model_path:
            self.load_model(model_path)

        # 缓存
        self._graph_cache: Dict[str, Any] = {}

        logger.info(f"MicroWorldAdapter 初始化完成, 设备: {self.device}")

    def inference(self,
                  upstream_Q: float,
                  downstream_Z: float,
                  canal_length: float = 1000.0,
                  canal_width: float = 50.0,
                  initial_depth: float = 3.0,
                  num_steps: int = 15) -> Dict[str, np.ndarray]:
        """
        执行微观推理

        Args:
            upstream_Q: 上游流量 [m³/s]
            downstream_Z: 下游水位 [m]
            canal_length: 渠道长度 [m]
            canal_width: 渠道宽度 [m]
            initial_depth: 初始水深 [m]
            num_steps: 微观时间步数

        Returns:
            {
                'velocity_x': 流速 x 分量 [num_nodes, num_steps],
                'velocity_y': 流速 y 分量 [num_nodes, num_steps],
                'pressure': 压力场 [num_nodes, num_steps],
                'depth': 水深场 [num_nodes, num_steps],
                'positions': 节点位置 [num_nodes, 2]
            }
        """
        # 构建或获取缓存的图
        cache_key = f"{canal_length}_{canal_width}_{self.config.num_nodes}"
        if cache_key not in self._graph_cache:
            graph = self.graph_builder.build_canal_graph(
                canal_length, canal_width, self.config.num_nodes
            )
            self._graph_cache[cache_key] = graph
        else:
            graph = self._graph_cache[cache_key]

        # 设置边界条件
        graph = self.graph_builder.set_boundary_conditions(
            graph, upstream_Q, downstream_Z, initial_depth
        )

        # 准备输入
        node_features = graph.ndata['features'].to(self.device)

        if DGL_AVAILABLE:
            edge_features = torch.stack([
                graph.edata['length'],
                graph.edata['angle'],
                torch.zeros_like(graph.edata['length'])
            ], dim=1).to(self.device)
            graph = graph.to(self.device)
        else:
            edge_features = None
            graph = None

        # 时间推进
        results = {
            'velocity_x': [],
            'velocity_y': [],
            'pressure': [],
            'depth': [],
        }

        self.model.eval()
        with torch.no_grad():
            current_features = node_features.clone()

            for step in range(num_steps):
                # 模型推理
                output = self.model(current_features, edge_features, graph)

                # 解析输出 [vx, vy, pressure]
                vx = output[:, 0].cpu().numpy()
                vy = output[:, 1].cpu().numpy()
                pressure = output[:, 2].cpu().numpy()

                results['velocity_x'].append(vx)
                results['velocity_y'].append(vy)
                results['pressure'].append(pressure)
                results['depth'].append(current_features[:, 2].cpu().numpy())

                # 更新状态 (简化: 使用输出更新特征)
                current_features[:, 3] = output[:, 0]  # vx
                current_features[:, 4] = output[:, 1]  # vy

        # 转换为数组
        for key in ['velocity_x', 'velocity_y', 'pressure', 'depth']:
            results[key] = np.stack(results[key], axis=1)

        # 添加位置信息
        results['positions'] = np.stack([
            graph.ndata['x'].cpu().numpy() if DGL_AVAILABLE else np.linspace(0, canal_length, self.config.num_nodes),
            graph.ndata['y'].cpu().numpy() if DGL_AVAILABLE else np.zeros(self.config.num_nodes) + canal_width/2
        ], axis=1)

        return results

    def get_key_metrics(self,
                        results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        提取关键指标

        Args:
            results: inference() 的输出

        Returns:
            关键指标字典
        """
        return {
            'max_velocity': float(np.max(np.abs(results['velocity_x']))),
            'avg_velocity': float(np.mean(results['velocity_x'])),
            'max_pressure': float(np.max(results['pressure'])),
            'avg_depth': float(np.mean(results['depth'])),
            'flow_uniformity': float(1.0 - np.std(results['velocity_x']) / (np.mean(np.abs(results['velocity_x'])) + 1e-8)),
        }

    def load_model(self, path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"模型加载成功: {path}")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")

    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"模型保存成功: {path}")


# ==============================================================================
# 与宏观模型集成的接口
# ==============================================================================

class MacroMicroBridge:
    """
    宏观-微观模型桥接器

    连接宏观 MPC/LSTM 模型与微观 HydroGraphNet
    """

    def __init__(self,
                 micro_adapter: MicroWorldAdapter,
                 macro_dt: float = 900.0,
                 micro_dt: float = 60.0):
        """
        Args:
            micro_adapter: 微观适配器
            macro_dt: 宏观时间步 [s]
            micro_dt: 微观时间步 [s]
        """
        self.micro_adapter = micro_adapter
        self.macro_dt = macro_dt
        self.micro_dt = micro_dt
        self.micro_steps = int(macro_dt / micro_dt)

    def simulate_macro_step(self,
                            upstream_Q: float,
                            downstream_Z: float,
                            gate_opening: float = 1.0,
                            pool_config: Dict = None) -> Dict[str, Any]:
        """
        模拟一个宏观时间步

        在宏观时间步内进行多次微观模拟
        返回详细的流场信息

        Args:
            upstream_Q: 上游流量
            downstream_Z: 下游水位
            gate_opening: 闸门开度
            pool_config: 渠池配置

        Returns:
            包含微观和宏观信息的结果
        """
        pool_config = pool_config or {}
        canal_length = pool_config.get('length', 1000.0)
        canal_width = pool_config.get('width', 50.0)

        # 闸门流量调整
        effective_Q = upstream_Q * gate_opening

        # 微观推理
        micro_results = self.micro_adapter.inference(
            upstream_Q=effective_Q,
            downstream_Z=downstream_Z,
            canal_length=canal_length,
            canal_width=canal_width,
            num_steps=self.micro_steps
        )

        # 提取宏观输出
        macro_output = {
            'avg_velocity': np.mean(micro_results['velocity_x'][:, -1]),
            'downstream_depth': micro_results['depth'][-1, -1],
            'flow_rate': effective_Q,
            'micro_results': micro_results,
            'metrics': self.micro_adapter.get_key_metrics(micro_results)
        }

        return macro_output


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info(" " * 15 + "微观世界适配器测试")
    logger.info("=" * 70)

    # 创建配置
    config = MicroWorldConfig(
        num_nodes=50,
        node_input_dim=12,
        hidden_dim=64,
        num_layers=3
    )

    # 创建适配器
    adapter = MicroWorldAdapter(config)

    logger.info(f"\n配置:")
    logger.info(f"  节点数: {config.num_nodes}")
    logger.info(f"  隐藏维度: {config.hidden_dim}")
    logger.info(f"  DGL 可用: {DGL_AVAILABLE}")

    # 测试推理
    logger.info("\n推理测试...")
    logger.info("-" * 50)

    results = adapter.inference(
        upstream_Q=200.0,    # 200 m³/s
        downstream_Z=3.5,    # 3.5 m
        canal_length=1000.0,
        canal_width=50.0,
        num_steps=10
    )

    logger.info(f"  流速 x 形状: {results['velocity_x'].shape}")
    logger.info(f"  压力场形状: {results['pressure'].shape}")
    logger.info(f"  位置形状: {results['positions'].shape}")

    # 关键指标
    metrics = adapter.get_key_metrics(results)
    logger.info(f"\n关键指标:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # 测试宏观-微观桥接
    if True:
        logger.info("\n宏观-微观桥接测试...")
        logger.info("-" * 50)

        bridge = MacroMicroBridge(adapter)
        macro_output = bridge.simulate_macro_step(
            upstream_Q=200.0,
            downstream_Z=3.5,
            gate_opening=0.8
        )

        logger.info(f"  平均流速: {macro_output['avg_velocity']:.4f} m/s")
        logger.info(f"  下游水深: {macro_output['downstream_depth']:.4f} m")
        logger.info(f"  流量: {macro_output['flow_rate']:.2f} m³/s")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)

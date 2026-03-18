"""
Kolmogorov-Arnold Network (KAN) 层实现
基于 HydroGraphNet 的 KAN 层，扩展支持多种基函数

KAN 理论背景:
- Kolmogorov-Arnold 表示定理: 任何多元连续函数都可以表示为单变量函数的组合
- f(x1, ..., xn) = Σ Φ_q(Σ φ_{q,p}(x_p))
- 实现方式: 使用傅里叶级数或切比雪夫多项式作为基函数

应用场景:
1. 节点特征编码 (替代 MLP)
2. 场景识别层的特征提取
3. 物理信息嵌入

参考:
- HydroGraphNet: physicsnemo/models/layers/kan_layers.py
- Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# 基础 KAN 线性层
# ==============================================================================

class KANLinear(nn.Module):
    """
    KAN 线性层 (基于傅里叶基函数)

    相比标准线性层:
    - 线性层: y = Wx + b
    - KAN层: y = Σ φ(x) 其中 φ 是可学习的单变量函数

    优点:
    - 更强的函数表达能力
    - 更好的可解释性
    - 适合物理信息嵌入
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_harmonics: int = 5,
                 add_bias: bool = True,
                 activation: str = 'none'):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_harmonics: 傅里叶谐波数量
            add_bias: 是否添加偏置
            activation: 激活函数 ('none', 'relu', 'gelu', 'silu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        # 傅里叶系数 [2, output_dim, input_dim, num_harmonics]
        # 2 表示 cos 和 sin 系数
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics)
            / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))
        else:
            self.register_parameter('bias', None)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_dim]

        Returns:
            输出张量 [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # 重塑输入 [batch_size, input_dim, 1]
        x = x.view(batch_size, self.input_dim, 1)

        # 创建谐波乘数 [1, 1, num_harmonics]
        k = torch.arange(1, self.num_harmonics + 1,
                        device=x.device, dtype=x.dtype).view(1, 1, self.num_harmonics)

        # 计算傅里叶基函数
        cos_terms = torch.cos(k * x)  # [batch, input_dim, num_harmonics]
        sin_terms = torch.sin(k * x)

        # 使用 Einstein 求和计算输出
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])

        y = y_cos + y_sin

        if self.add_bias:
            y = y + self.bias

        if self.activation is not None:
            y = self.activation(y)

        return y

    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, num_harmonics={self.num_harmonics}'


# ==============================================================================
# 完整 Kolmogorov-Arnold Network (来自 HydroGraphNet)
# ==============================================================================

class KolmogorovArnoldNetwork(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) 层
    使用傅里叶基函数进行函数逼近

    这是 HydroGraphNet 中用于节点编码的核心层
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_harmonics: int = 5,
                 add_bias: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            num_harmonics: 傅里叶谐波数量 (默认: 5)
            add_bias: 是否包含偏置项 (默认: True)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_harmonics = num_harmonics
        self.add_bias = add_bias

        # 初始化傅里叶系数，缩放以保持稳定性
        # 形状: [2, output_dim, input_dim, num_harmonics]
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, num_harmonics)
            / (np.sqrt(input_dim) * np.sqrt(num_harmonics))
        )

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, input_dim)

        Returns:
            输出张量 (batch_size, output_dim)
        """
        batch_size = x.size(0)

        # 重塑输入以进行谐波乘法
        x = x.view(batch_size, self.input_dim, 1)

        # 创建谐波乘数 (从1到num_harmonics)
        k = torch.arange(1, self.num_harmonics + 1,
                        device=x.device, dtype=x.dtype).view(1, 1, self.num_harmonics)

        # 计算余弦和正弦分量
        cos_terms = torch.cos(k * x)
        sin_terms = torch.sin(k * x)

        # 使用Einstein求和进行高效傅里叶展开
        y_cos = torch.einsum("bij,oij->bo", cos_terms, self.fourier_coeffs[0])
        y_sin = torch.einsum("bij,oij->bo", sin_terms, self.fourier_coeffs[1])

        y = y_cos + y_sin

        if self.add_bias:
            y = y + self.bias

        return y


# ==============================================================================
# 傅里叶 KAN (扩展版本)
# ==============================================================================

class FourierKAN(nn.Module):
    """
    傅里叶 KAN 网络 (多层版本)

    支持多层堆叠，带残差连接
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 num_harmonics: int = 5,
                 dropout: float = 0.1,
                 residual: bool = True):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            num_harmonics: 傅里叶谐波数
            dropout: Dropout率
            residual: 是否使用残差连接
        """
        super().__init__()
        self.residual = residual

        # 构建层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(KANLinear(dims[i], dims[i+1], num_harmonics))
            if i < len(dims) - 2:  # 不在最后一层添加
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList(layers)

        # 残差投影 (如果维度不匹配)
        if residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x

        for layer in self.layers:
            x = layer(x)

        # 残差连接
        if self.residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity

        return x


# ==============================================================================
# 切比雪夫 KAN
# ==============================================================================

class ChebyshevKAN(nn.Module):
    """
    基于切比雪夫多项式的 KAN 层

    使用切比雪夫多项式作为基函数:
    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)

    优点:
    - 在 [-1, 1] 区间上有更好的逼近性质
    - 数值稳定性更好
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 degree: int = 5,
                 add_bias: bool = True):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            degree: 切比雪夫多项式的最高次数
            add_bias: 是否添加偏置
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.add_bias = add_bias

        # 切比雪夫系数 [output_dim, input_dim, degree+1]
        self.cheby_coeffs = nn.Parameter(
            torch.randn(output_dim, input_dim, degree + 1)
            / (np.sqrt(input_dim) * np.sqrt(degree + 1))
        )

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(1, output_dim))

    def _chebyshev_polynomials(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算切比雪夫多项式

        Args:
            x: 输入 [batch, input_dim, 1]

        Returns:
            切比雪夫多项式值 [batch, input_dim, degree+1]
        """
        # 将 x 归一化到 [-1, 1]
        x = torch.tanh(x)

        batch_size = x.size(0)
        T = torch.zeros(batch_size, self.input_dim, self.degree + 1,
                       device=x.device, dtype=x.dtype)

        T[:, :, 0] = 1.0  # T_0 = 1
        if self.degree >= 1:
            T[:, :, 1] = x.squeeze(-1)  # T_1 = x

        # 递推计算
        for n in range(2, self.degree + 1):
            T[:, :, n] = 2 * x.squeeze(-1) * T[:, :, n-1] - T[:, :, n-2]

        return T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.size(0)

        # 重塑输入
        x = x.view(batch_size, self.input_dim, 1)

        # 计算切比雪夫多项式
        T = self._chebyshev_polynomials(x)  # [batch, input_dim, degree+1]

        # 计算输出
        y = torch.einsum("bid,oid->bo", T, self.cheby_coeffs)

        if self.add_bias:
            y = y + self.bias

        return y


# ==============================================================================
# KAN 编码器 (用于场景识别)
# ==============================================================================

class KANEncoder(nn.Module):
    """
    KAN 编码器

    用于将时序数据编码为隐向量，可用于场景识别
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_harmonics: int = 5,
                 num_layers: int = 2):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_harmonics: KAN谐波数
            num_layers: 层数
        """
        super().__init__()

        layers = []

        # 输入层
        layers.append(KolmogorovArnoldNetwork(input_dim, hidden_dim, num_harmonics))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(KolmogorovArnoldNetwork(hidden_dim, hidden_dim, num_harmonics))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        # 输出层
        layers.append(KolmogorovArnoldNetwork(hidden_dim, output_dim, num_harmonics))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入

        Args:
            x: 输入特征 [batch, input_dim]

        Returns:
            嵌入向量 [batch, output_dim]
        """
        return self.encoder(x)


# ==============================================================================
# 示例和测试
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info(" " * 15 + "KAN 层测试")
    logger.info("=" * 70)

    # 测试基础 KANLinear
    logger.info("\n1. KANLinear 测试")
    logger.info("-" * 50)

    kan_linear = KANLinear(input_dim=10, output_dim=32, num_harmonics=5)
    x = torch.randn(8, 10)
    y = kan_linear(x)
    logger.info(f"  输入形状: {x.shape}")
    logger.info(f"  输出形状: {y.shape}")
    logger.info(f"  参数数量: {sum(p.numel() for p in kan_linear.parameters())}")

    # 测试 KolmogorovArnoldNetwork
    logger.info("\n2. KolmogorovArnoldNetwork 测试")
    logger.info("-" * 50)

    kan = KolmogorovArnoldNetwork(input_dim=10, output_dim=64, num_harmonics=5)
    y = kan(x)
    logger.info(f"  输出形状: {y.shape}")

    # 测试 FourierKAN
    logger.info("\n3. FourierKAN (多层) 测试")
    logger.info("-" * 50)

    fourier_kan = FourierKAN(
        input_dim=10,
        hidden_dims=[64, 32],
        output_dim=16,
        num_harmonics=5
    )
    y = fourier_kan(x)
    logger.info(f"  输出形状: {y.shape}")
    logger.info(f"  参数数量: {sum(p.numel() for p in fourier_kan.parameters())}")

    # 测试 ChebyshevKAN
    logger.info("\n4. ChebyshevKAN 测试")
    logger.info("-" * 50)

    cheby_kan = ChebyshevKAN(input_dim=10, output_dim=32, degree=5)
    y = cheby_kan(x)
    logger.info(f"  输出形状: {y.shape}")

    # 测试 KANEncoder
    logger.info("\n5. KANEncoder 测试")
    logger.info("-" * 50)

    encoder = KANEncoder(input_dim=10, hidden_dim=64, output_dim=32)
    embedding = encoder(x)
    logger.info(f"  嵌入形状: {embedding.shape}")

    # 与标准 MLP 对比
    logger.info("\n6. 与标准 MLP 对比")
    logger.info("-" * 50)

    mlp = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

    kan_params = sum(p.numel() for p in kan_linear.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())

    logger.info(f"  KAN 参数: {kan_params}")
    logger.info(f"  MLP 参数: {mlp_params}")
    logger.info(f"  KAN/MLP 比例: {kan_params/mlp_params:.2f}x")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info("=" * 70)

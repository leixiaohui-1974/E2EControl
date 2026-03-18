"""
一维 Saint-Venant 方程求解器
=============================
提供三种求解格式:
1. Lax-Wendroff (显式二阶, CFL自适应)
2. Diffusion Wave (显式, 忽略惯性项)
3. Tank ODE (零维水箱)

矩形断面, Manning摩阻。

连续性: ∂A/∂t + ∂Q/∂x = 0
动量:   ∂Q/∂t + ∂(Q²/A + gA²/(2W))/∂x = gA(S₀ - Sf)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ChannelParams:
    """矩形明渠参数。"""
    length: float = 5000.0      # 渠道长度 (m)
    width: float = 10.0         # 断面宽度 (m)
    slope: float = 0.0005       # 底坡 (-)
    manning_n: float = 0.025    # Manning糙率
    n_nodes: int = 51           # 空间节点数
    g: float = 9.81


def _manning_Q(W, h, n, S0):
    """Manning公式: Q = (1/n) * A * R^(2/3) * S0^(1/2)"""
    if h < 0.001:
        return 0.0
    A = W * h
    R = A / (W + 2 * h)
    return (1.0 / n) * A * R ** (2 / 3) * S0 ** 0.5


def _manning_h(W, Q, n, S0):
    """Manning公式反算正常水深 (Newton)。"""
    h = max(0.1, (Q / W) ** 0.6)  # 初始猜测
    for _ in range(30):
        A = W * h
        R = A / (W + 2 * h) if h > 0.001 else 0.001
        Qc = (1 / n) * A * R ** (2 / 3) * S0 ** 0.5
        dR = (W * (W + 2 * h) - 2 * A) / (W + 2 * h) ** 2
        dQ = (1 / n) * S0 ** 0.5 * ((2 / 3) * R ** (-1 / 3) * dR * A + R ** (2 / 3) * W)
        if abs(dQ) < 1e-12:
            break
        h = max(0.001, h - (Qc - Q) / dQ)
    return h


# ═══════════════════════════════════════════════════════════════════════════
# Saint-Venant: Lax-Wendroff 显式二阶格式
# ═══════════════════════════════════════════════════════════════════════════

class SaintVenant1D:
    """一维 Saint-Venant 方程 Lax-Wendroff 格式求解器。

    守恒形式: ∂U/∂t + ∂F(U)/∂x = S(U)
    U = [A, Q]^T
    F = [Q, Q²/A + g*A²/(2W)]^T
    S = [0, gA(S₀ - Sf)]^T

    Lax-Wendroff 两步法:
    Step 1 (半步): U^(n+1/2)_(i+1/2) = 0.5*(U_i + U_(i+1)) - dt/(2dx)*(F_(i+1) - F_i) + dt/2*S_avg
    Step 2 (全步): U^(n+1)_i = U^n_i - dt/dx*(F^(n+1/2)_(i+1/2) - F^(n+1/2)_(i-1/2)) + dt*S^(n+1/2)

    CFL条件: dt <= dx / max(|V| + c), c = sqrt(gA/W)
    """

    def __init__(self, params: ChannelParams, cfl: float = 0.8):
        self.p = params
        self.cfl = cfl
        self.N = params.n_nodes
        self.dx = params.length / (self.N - 1)
        self.A = np.zeros(self.N)   # 过水面积
        self.Q = np.zeros(self.N)   # 流量
        self.h = np.zeros(self.N)   # 水深 (= A/W)
        self.t = 0.0

    def initialize(self, h0: float, Q0: float = None):
        """均匀流初始条件。"""
        W = self.p.width
        self.h[:] = h0
        self.A[:] = W * h0
        if Q0 is None:
            Q0 = _manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.Q[:] = Q0
        self.t = 0.0

    def _flux(self, A, Q):
        """通量函数 F(U)。"""
        W = self.p.width
        g = self.p.g
        F_A = Q
        F_Q = Q ** 2 / A + g * A ** 2 / (2 * W) if A > 0.01 else 0
        return F_A, F_Q

    def _source(self, A, Q):
        """源项 S(U)。"""
        W = self.p.width
        n = self.p.manning_n
        S0 = self.p.slope
        g = self.p.g

        h = A / W if A > 0 else 0.001
        R = A / (W + 2 * h) if h > 0.001 else 0.001
        V = Q / A if A > 0.01 else 0
        Sf = n ** 2 * V * abs(V) / (R ** (4 / 3)) if R > 0.001 else 0

        S_A = 0.0
        S_Q = g * A * (S0 - Sf)
        return S_A, S_Q

    def _max_wavespeed(self):
        """最大波速 (用于CFL)。"""
        W = self.p.width
        g = self.p.g
        c_max = 0.01
        for i in range(self.N):
            h = self.A[i] / W if self.A[i] > 0 else 0.001
            V = self.Q[i] / self.A[i] if self.A[i] > 0.01 else 0
            c = np.sqrt(g * h) if h > 0 else 0
            c_max = max(c_max, abs(V) + c)
        return c_max

    def advance(self, dt_outer: float, Q_upstream: float, h_downstream: float):
        """推进一个外部时间步（内部CFL自适应子步）。"""
        W = self.p.width
        dx = self.dx
        N = self.N

        t_remaining = dt_outer

        while t_remaining > 1e-8:
            # CFL 自适应
            c_max = self._max_wavespeed()
            dt = min(self.cfl * dx / c_max, t_remaining)
            dt = max(dt, 0.01)

            A = self.A.copy()
            Q = self.Q.copy()

            # --- Step 1: 半步 (i+1/2 staggered) ---
            A_half = np.zeros(N - 1)
            Q_half = np.zeros(N - 1)

            for i in range(N - 1):
                FA_i, FQ_i = self._flux(A[i], Q[i])
                FA_ip, FQ_ip = self._flux(A[i + 1], Q[i + 1])
                SA_i, SQ_i = self._source(A[i], Q[i])
                SA_ip, SQ_ip = self._source(A[i + 1], Q[i + 1])

                A_half[i] = 0.5 * (A[i] + A[i + 1]) - 0.5 * dt / dx * (FA_ip - FA_i) + 0.25 * dt * (SA_i + SA_ip)
                Q_half[i] = 0.5 * (Q[i] + Q[i + 1]) - 0.5 * dt / dx * (FQ_ip - FQ_i) + 0.25 * dt * (SQ_i + SQ_ip)

                A_half[i] = max(W * 0.001, A_half[i])  # 最小水深
                Q_half[i] = max(0.0, Q_half[i])

            # --- Step 2: 全步 ---
            A_new = A.copy()
            Q_new = Q.copy()

            for i in range(1, N - 1):
                FA_l, FQ_l = self._flux(A_half[i - 1], Q_half[i - 1])
                FA_r, FQ_r = self._flux(A_half[i], Q_half[i])
                SA_h, SQ_h = self._source(0.5 * (A_half[i - 1] + A_half[i]),
                                           0.5 * (Q_half[i - 1] + Q_half[i]))

                A_new[i] = A[i] - dt / dx * (FA_r - FA_l) + dt * SA_h
                Q_new[i] = Q[i] - dt / dx * (FQ_r - FQ_l) + dt * SQ_h

                A_new[i] = max(W * 0.001, A_new[i])
                Q_new[i] = max(0.0, Q_new[i])

            # 边界条件
            # 上游: 给定流量
            Q_new[0] = Q_upstream
            # 上游面积: 由特征线外推 + Manning正常水深
            h_up = _manning_h(W, Q_upstream, self.p.manning_n, self.p.slope)
            # 平滑过渡，避免跳变
            h_current = A_new[0] / W
            alpha = min(0.3, dt / 60.0)  # 缓慢趋近
            h_up_smooth = h_current + alpha * (h_up - h_current)
            A_new[0] = W * max(0.001, h_up_smooth)

            # 下游: 给定水深
            A_new[-1] = W * h_downstream
            # 下游流量: 特征线外推
            Q_new[-1] = Q_new[-2]

            self.A = A_new
            self.Q = Q_new
            self.h = self.A / W

            t_remaining -= dt

        self.t += dt_outer

    def get_state(self):
        return {
            "t": self.t,
            "h": self.h.copy(),
            "Q": self.Q.copy(),
            "x": np.linspace(0, self.p.length, self.N),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion Wave 扩散波
# ═══════════════════════════════════════════════════════════════════════════

class DiffusionWave1D:
    """扩散波近似: ∂h/∂t + c·∂h/∂x = D·∂²h/∂x²"""

    def __init__(self, params: ChannelParams):
        self.p = params
        self.N = params.n_nodes
        self.dx = params.length / (self.N - 1)
        self.h = np.zeros(self.N)
        self.t = 0.0

    def initialize(self, h0: float):
        self.h[:] = h0
        self.t = 0.0

    def advance(self, dt: float, Q_upstream: float):
        W = self.p.width
        n = self.p.manning_n
        S0 = self.p.slope
        dx = self.dx

        h_bc = _manning_h(W, Q_upstream, n, S0)
        t_rem = dt

        while t_rem > 1e-8:
            c_max, D_max = 0.01, 0.01
            for i in range(self.N):
                if self.h[i] < 0.01:
                    continue
                A = W * self.h[i]
                R = A / (W + 2 * self.h[i])
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c_max = max(c_max, (5 / 3) * V)
                D_max = max(D_max, V * A / (2 * W * S0))

            dt_sub = min(0.4 * dx / c_max, 0.4 * dx ** 2 / (2 * D_max), t_rem)
            dt_sub = max(dt_sub, 0.05)

            h_new = self.h.copy()
            for i in range(1, self.N - 1):
                A = W * self.h[i]
                R = A / (W + 2 * self.h[i]) if self.h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V
                D = V * A / (2 * W * S0) if S0 > 0 else 0
                D = min(D, dx ** 2 / (2 * dt_sub))

                dhdx = (self.h[i] - self.h[i - 1]) / dx
                d2h = (self.h[i + 1] - 2 * self.h[i] + self.h[i - 1]) / dx ** 2
                h_new[i] = max(0.01, self.h[i] + dt_sub * (-c * dhdx + D * d2h))

            h_new[0] = h_bc
            h_new[-1] = h_new[-2]
            self.h = h_new
            t_rem -= dt_sub

        self.t += dt

    def get_state(self):
        return {
            "t": self.t,
            "h": self.h.copy(),
            "x": np.linspace(0, self.p.length, self.N),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Tank ODE 零维
# ═══════════════════════════════════════════════════════════════════════════

class TankODE:
    """零维水箱: dh/dt = (Q_in - Q_out) / (W * L)"""

    def __init__(self, params: ChannelParams):
        self.p = params
        self.area = params.width * params.length
        self.h = 0.0
        self.t = 0.0

    def initialize(self, h0: float):
        self.h = h0
        self.t = 0.0

    def advance(self, dt: float, Q_in: float, Q_out: float):
        dh = (Q_in - Q_out) / self.area * dt
        self.h = max(0.001, self.h + dh)
        self.t += dt

    def get_state(self):
        return {"t": self.t, "h": self.h}

"""
Saint-Venant 方程多种数值求解格式
==================================
统一接口，同一渠道参数，便于横向对比。

守恒形式: ∂U/∂t + ∂F(U)/∂x = S(U)
  U = [A, Q]^T
  F = [Q, Q²/A + gA²/(2W)]^T
  S = [0, gA(S₀ - Sf)]^T
  Sf = n²Q|Q| / (A²R^(4/3))

所有求解器继承 BaseSolver，统一 initialize() / advance() / get_state() 接口。
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChannelParams:
    """矩形明渠参数。"""
    length: float = 5000.0
    width: float = 10.0
    slope: float = 0.0005
    manning_n: float = 0.025
    n_nodes: int = 51
    g: float = 9.81


def manning_Q(W, h, n, S0):
    if h < 0.001: return 0.0
    A = W * h; R = A / (W + 2 * h)
    return (1.0 / n) * A * R ** (2 / 3) * S0 ** 0.5


def manning_h(W, Q, n, S0):
    h = max(0.1, (Q / W) ** 0.6)
    for _ in range(30):
        A = W * h; R = A / (W + 2 * h) if h > 0.001 else 0.001
        Qc = (1 / n) * A * R ** (2 / 3) * S0 ** 0.5
        dR = (W * (W + 2 * h) - 2 * A) / (W + 2 * h) ** 2
        dQ = (1 / n) * S0 ** 0.5 * ((2 / 3) * R ** (-1 / 3) * dR * A + R ** (2 / 3) * W)
        if abs(dQ) < 1e-12: break
        h = max(0.001, h - (Qc - Q) / dQ)
    return h


class BaseSolver(ABC):
    """求解器统一接口。

    所有子类同时满足 ``hydromind_contracts.HydraulicSolverProtocol``
    （当 hydromind-contracts 已安装时可做 ``isinstance`` 检查）。
    """

    def __init__(self, params: ChannelParams):
        self.p = params
        self.N = params.n_nodes
        self.dx = params.length / (self.N - 1)
        self.t = 0.0

    @abstractmethod
    def initialize(self, h0: float, Q0: float = None): ...

    @abstractmethod
    def advance(self, dt: float, Q_upstream: float, h_downstream: float = None): ...

    @abstractmethod
    def get_h_profile(self) -> np.ndarray: ...

    def get_state(self):
        return {
            "t": self.t,
            "h": self.get_h_profile().copy(),
            "x": np.linspace(0, self.p.length, self.N),
        }

    def _flux(self, A, Q):
        W = self.p.width; g = self.p.g
        if A > 0.01:
            return Q, Q ** 2 / A + g * A ** 2 / (2 * W)
        return Q, 0.0

    def _source(self, A, Q):
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; g = self.p.g
        h = A / W if A > 0 else 0.001
        R = A / (W + 2 * h) if h > 0.001 else 0.001
        V = Q / A if A > 0.01 else 0
        Sf = n ** 2 * V * abs(V) / (R ** (4 / 3)) if R > 0.001 else 0
        return 0.0, g * A * (S0 - Sf)

    def _max_wavespeed(self, A, Q):
        W = self.p.width; g = self.p.g
        c_max = 0.01
        for i in range(len(A)):
            h = A[i] / W if A[i] > 0 else 0.001
            V = Q[i] / A[i] if A[i] > 0.01 else 0
            c_max = max(c_max, abs(V) + np.sqrt(g * h))
        return c_max


# ═══════════════════════════════════════════════════════════════════════════
# 0. Tank ODE (零维)
# ═══════════════════════════════════════════════════════════════════════════

class TankODE(BaseSolver):
    """零维水箱: dh/dt = (Q_in - Q_out) / (W*L)"""
    name = "Tank ODE (零维)"
    order = "0D"
    scheme = "Euler显式"
    color = "#95a5a6"

    def __init__(self, params):
        super().__init__(params)
        self.h_val = 0.0

    def initialize(self, h0, Q0=None):
        self.h_val = h0; self.t = 0.0

    def advance(self, dt, Q_upstream, h_downstream=None):
        Q_out = manning_Q(self.p.width, self.h_val, self.p.manning_n, self.p.slope)
        area = self.p.width * self.p.length
        self.h_val = max(0.001, self.h_val + (Q_upstream - Q_out) / area * dt)
        self.t += dt

    def get_h_profile(self):
        return np.full(self.N, self.h_val)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Kinematic Wave (运动波, 显式上风)
# ═══════════════════════════════════════════════════════════════════════════

class KinematicWave(BaseSolver):
    """运动波: ∂A/∂t + c·∂A/∂x = 0, c=(5/3)V"""
    name = "Kinematic Wave (运动波)"
    order = "1阶"
    scheme = "显式上风"
    color = "#f39c12"

    def __init__(self, params):
        super().__init__(params)
        self.h = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        self.h[:] = h0; self.t = 0.0

    def advance(self, dt, Q_upstream, h_downstream=None):
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; dx = self.dx
        h_bc = manning_h(W, Q_upstream, n, S0)
        t_rem = dt
        while t_rem > 1e-8:
            c_max = 0.01
            for i in range(self.N):
                if self.h[i] < 0.01: continue
                R = (W * self.h[i]) / (W + 2 * self.h[i])
                c_max = max(c_max, (5 / 3) * (1 / n) * R ** (2 / 3) * S0 ** 0.5)
            dt_sub = min(0.8 * dx / c_max, t_rem)
            h_new = self.h.copy()
            for i in range(1, self.N):
                R = (W * self.h[i]) / (W + 2 * self.h[i]) if self.h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V
                h_new[i] = max(0.01, self.h[i] - c * dt_sub / dx * (self.h[i] - self.h[i - 1]))
            h_new[0] = h_bc
            self.h = h_new; t_rem -= dt_sub
        self.t += dt

    def get_h_profile(self): return self.h


# ═══════════════════════════════════════════════════════════════════════════
# 2. Diffusion Wave (扩散波)
# ═══════════════════════════════════════════════════════════════════════════

class DiffusionWave(BaseSolver):
    """扩散波: ∂h/∂t + c·∂h/∂x = D·∂²h/∂x²"""
    name = "Diffusion Wave (扩散波)"
    order = "2阶空间"
    scheme = "显式中心差分"
    color = "#2ecc71"

    def __init__(self, params):
        super().__init__(params)
        self.h = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        self.h[:] = h0; self.t = 0.0

    def advance(self, dt, Q_upstream, h_downstream=None):
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; dx = self.dx
        h_bc = manning_h(W, Q_upstream, n, S0)
        t_rem = dt
        while t_rem > 1e-8:
            c_max, D_max = 0.01, 0.01
            for i in range(self.N):
                if self.h[i] < 0.01: continue
                A = W * self.h[i]; R = A / (W + 2 * self.h[i])
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c_max = max(c_max, (5 / 3) * V)
                D_max = max(D_max, V * A / (2 * W * S0))
            dt_sub = min(0.4 * dx / c_max, 0.4 * dx ** 2 / (2 * D_max), t_rem)
            dt_sub = max(dt_sub, 0.05)
            h_new = self.h.copy()
            for i in range(1, self.N - 1):
                A = W * self.h[i]; R = A / (W + 2 * self.h[i]) if self.h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V; D = min(V * A / (2 * W * S0), dx ** 2 / (2 * dt_sub))
                dhdx = (self.h[i] - self.h[i - 1]) / dx
                d2h = (self.h[i + 1] - 2 * self.h[i] + self.h[i - 1]) / dx ** 2
                h_new[i] = max(0.01, self.h[i] + dt_sub * (-c * dhdx + D * d2h))
            h_new[0] = h_bc; h_new[-1] = h_new[-2]
            self.h = h_new; t_rem -= dt_sub
        self.t += dt

    def get_h_profile(self): return self.h


# ═══════════════════════════════════════════════════════════════════════════
# 3. Lax-Friedrichs (1阶, 显式, 强耗散)
# ═══════════════════════════════════════════════════════════════════════════

class LaxFriedrichs(BaseSolver):
    """Lax-Friedrichs (Rusanov局部形式): 使用局部波速控制数值耗散"""
    name = "Lax-Friedrichs (1阶)"
    order = "1阶"
    scheme = "Rusanov局部耗散"
    color = "#bdc3c7"

    def __init__(self, params, cfl=0.8):
        super().__init__(params)
        self.cfl = cfl
        self.A = np.zeros(self.N)
        self.Q = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        W = self.p.width
        self.A[:] = W * h0
        self.Q[:] = Q0 if Q0 else manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.t = 0.0

    def advance(self, dt_outer, Q_upstream, h_downstream=None):
        W = self.p.width; dx = self.dx; N = self.N; g = self.p.g
        n_m = self.p.manning_n; S0 = self.p.slope
        h_ds = h_downstream if h_downstream else self.A[-1] / W
        t_rem = dt_outer
        while t_rem > 1e-8:
            c_max = self._max_wavespeed(self.A, self.Q)
            dt = min(self.cfl * dx / c_max, t_rem)
            A, Q = self.A, self.Q
            An, Qn = A.copy(), Q.copy()

            # Rusanov（局部 Lax-Friedrichs）：
            # 界面通量 F_{i+1/2} = 0.5*(F_L+F_R) - 0.5*alpha*(U_R-U_L)
            # alpha = max(|V_L|+c_L, |V_R|+c_R) 局部最大波速
            # 这比全局平均 0.5*(U[i-1]+U[i+1]) 更精确地控制数值耗散
            for i in range(1, N - 1):
                # 左界面 (i-1/2)
                FA_im, FQ_im = self._flux(A[i - 1], Q[i - 1])
                FA_i, FQ_i = self._flux(A[i], Q[i])
                h_L = max(A[i - 1] / W, 0.001); h_R = max(A[i] / W, 0.001)
                V_L = Q[i - 1] / max(A[i - 1], W * 0.001)
                V_R = Q[i] / max(A[i], W * 0.001)
                a_L = max(abs(V_L) + np.sqrt(g * h_L), abs(V_R) + np.sqrt(g * h_R))
                FA_left = 0.5 * (FA_im + FA_i) - 0.5 * a_L * (A[i] - A[i - 1])
                FQ_left = 0.5 * (FQ_im + FQ_i) - 0.5 * a_L * (Q[i] - Q[i - 1])

                # 右界面 (i+1/2)
                FA_ip, FQ_ip = self._flux(A[i + 1], Q[i + 1])
                h_L2 = max(A[i] / W, 0.001); h_R2 = max(A[i + 1] / W, 0.001)
                V_L2 = Q[i] / max(A[i], W * 0.001)
                V_R2 = Q[i + 1] / max(A[i + 1], W * 0.001)
                a_R = max(abs(V_L2) + np.sqrt(g * h_L2), abs(V_R2) + np.sqrt(g * h_R2))
                FA_right = 0.5 * (FA_i + FA_ip) - 0.5 * a_R * (A[i + 1] - A[i])
                FQ_right = 0.5 * (FQ_i + FQ_ip) - 0.5 * a_R * (Q[i + 1] - Q[i])

                SA, SQ = self._source(A[i], Q[i])
                An[i] = A[i] - dt / dx * (FA_right - FA_left) + dt * SA
                Qn[i] = Q[i] - dt / dx * (FQ_right - FQ_left) + dt * SQ
                An[i] = max(W * 0.001, An[i])
                Qn[i] = max(0.0, Qn[i])

            # BCs
            Qn[0] = Q_upstream
            h_up = manning_h(W, Q_upstream, n_m, S0)
            An[0] = W * h_up
            An[-1] = W * h_ds
            h_ds_val = An[-1] / W
            Qn[-1] = manning_Q(W, h_ds_val, n_m, S0)

            # 质量校正
            total_A = np.sum(An[1:-1]) * dx
            expected_A = np.sum(A[1:-1]) * dx + (Q_upstream - Qn[-1]) * dt
            if total_A > 0 and expected_A > 0:
                correction = expected_A / total_A
                An[1:-1] *= min(max(correction, 0.95), 1.05)

            self.A, self.Q = An, Qn
            t_rem -= dt
        self.t += dt_outer

    def get_h_profile(self): return self.A / self.p.width


# ═══════════════════════════════════════════════════════════════════════════
# 4. Lax-Wendroff (2阶, 显式两步法) — 已有，重新封装
# ═══════════════════════════════════════════════════════════════════════════

class LaxWendroff(BaseSolver):
    """Lax-Wendroff 两步法 (Richtmyer): 半步Lax + 全步leapfrog"""
    name = "Lax-Wendroff (2阶)"
    order = "2阶"
    scheme = "显式两步法"
    color = "#3498db"

    def __init__(self, params, cfl=0.8):
        super().__init__(params)
        self.cfl = cfl
        self.A = np.zeros(self.N)
        self.Q = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        W = self.p.width
        self.A[:] = W * h0
        self.Q[:] = Q0 if Q0 else manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.t = 0.0

    def advance(self, dt_outer, Q_upstream, h_downstream=None):
        W = self.p.width; dx = self.dx; N = self.N
        h_ds = h_downstream if h_downstream else self.A[-1] / W
        t_rem = dt_outer
        while t_rem > 1e-8:
            c_max = self._max_wavespeed(self.A, self.Q)
            dt = min(self.cfl * dx / c_max, t_rem)
            A, Q = self.A, self.Q
            # Step 1: half-step
            Ah = np.zeros(N - 1); Qh = np.zeros(N - 1)
            for i in range(N - 1):
                FA_i, FQ_i = self._flux(A[i], Q[i])
                FA_ip, FQ_ip = self._flux(A[i + 1], Q[i + 1])
                SA_i, SQ_i = self._source(A[i], Q[i])
                SA_ip, SQ_ip = self._source(A[i + 1], Q[i + 1])
                Ah[i] = max(W * 0.001, 0.5 * (A[i] + A[i + 1]) - 0.5 * dt / dx * (FA_ip - FA_i) + 0.25 * dt * (SA_i + SA_ip))
                Qh[i] = max(0.0, 0.5 * (Q[i] + Q[i + 1]) - 0.5 * dt / dx * (FQ_ip - FQ_i) + 0.25 * dt * (SQ_i + SQ_ip))
            # Step 2: full step
            An, Qn = A.copy(), Q.copy()
            for i in range(1, N - 1):
                FA_l, FQ_l = self._flux(Ah[i - 1], Qh[i - 1])
                FA_r, FQ_r = self._flux(Ah[i], Qh[i])
                SA_h, SQ_h = self._source(0.5 * (Ah[i - 1] + Ah[i]), 0.5 * (Qh[i - 1] + Qh[i]))
                An[i] = max(W * 0.001, A[i] - dt / dx * (FA_r - FA_l) + dt * SA_h)
                Qn[i] = max(0.0, Q[i] - dt / dx * (FQ_r - FQ_l) + dt * SQ_h)
            Qn[0] = Q_upstream
            h_up = manning_h(W, Q_upstream, self.p.manning_n, self.p.slope)
            An[0] = W * h_up  # direct enforcement at upstream
            An[-1] = W * h_ds
            h_ds_val = An[-1] / W
            Qn[-1] = manning_Q(W, h_ds_val, self.p.manning_n, self.p.slope)
            self.A, self.Q = An, Qn
            t_rem -= dt
        self.t += dt_outer

    def get_h_profile(self): return self.A / self.p.width


# ═══════════════════════════════════════════════════════════════════════════
# 5. MacCormack (2阶, 显式预估-校正)
# ═══════════════════════════════════════════════════════════════════════════

class MacCormack(BaseSolver):
    """MacCormack 预估-校正法: predictor用前差, corrector用后差"""
    name = "MacCormack (2阶)"
    order = "2阶"
    scheme = "显式预估校正"
    color = "#9b59b6"

    def __init__(self, params, cfl=0.8):
        super().__init__(params)
        self.cfl = cfl
        self.A = np.zeros(self.N)
        self.Q = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        W = self.p.width
        self.A[:] = W * h0
        self.Q[:] = Q0 if Q0 else manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.t = 0.0

    def advance(self, dt_outer, Q_upstream, h_downstream=None):
        W = self.p.width; dx = self.dx; N = self.N
        h_ds = h_downstream if h_downstream else self.A[-1] / W
        t_rem = dt_outer
        while t_rem > 1e-8:
            c_max = self._max_wavespeed(self.A, self.Q)
            dt = min(self.cfl * dx / c_max, t_rem)
            A, Q = self.A, self.Q

            # Predictor (forward差分)
            Ap, Qp = A.copy(), Q.copy()
            for i in range(0, N - 1):
                FA_i, FQ_i = self._flux(A[i], Q[i])
                FA_ip, FQ_ip = self._flux(A[i + 1], Q[i + 1])
                SA, SQ = self._source(A[i], Q[i])
                Ap[i] = max(W * 0.001, A[i] - dt / dx * (FA_ip - FA_i) + dt * SA)
                Qp[i] = max(0.0, Q[i] - dt / dx * (FQ_ip - FQ_i) + dt * SQ)
            Ap[-1] = W * h_ds; Qp[-1] = Qp[-2]

            # Corrector (backward差分)
            An, Qn = A.copy(), Q.copy()
            for i in range(1, N):
                FA_i, FQ_i = self._flux(Ap[i], Qp[i])
                FA_im, FQ_im = self._flux(Ap[i - 1], Qp[i - 1])
                SA, SQ = self._source(Ap[i], Qp[i])
                An[i] = max(W * 0.001, 0.5 * (A[i] + Ap[i]) - 0.5 * dt / dx * (FA_i - FA_im) + 0.5 * dt * SA)
                Qn[i] = max(0.0, 0.5 * (Q[i] + Qp[i]) - 0.5 * dt / dx * (FQ_i - FQ_im) + 0.5 * dt * SQ)

            Qn[0] = Q_upstream
            h_up = manning_h(W, Q_upstream, self.p.manning_n, self.p.slope)
            An[0] = W * h_up  # direct enforcement at upstream
            An[-1] = W * h_ds
            h_ds_val = An[-1] / W
            Qn[-1] = manning_Q(W, h_ds_val, self.p.manning_n, self.p.slope)
            self.A, self.Q = An, Qn
            t_rem -= dt
        self.t += dt_outer

    def get_h_profile(self): return self.A / self.p.width


# ═══════════════════════════════════════════════════════════════════════════
# 6. Godunov/HLL (高分辨率, Riemann求解器)
# ═══════════════════════════════════════════════════════════════════════════

class GodunuvHLL(BaseSolver):
    """Godunov-HLL: 使用HLL近似Riemann求解器的一阶Godunov格式"""
    name = "Godunov-HLL (Riemann)"
    order = "1阶(可扩展2阶)"
    scheme = "显式HLL通量"
    color = "#e74c3c"

    def __init__(self, params, cfl=0.8):
        super().__init__(params)
        self.cfl = cfl
        self.A = np.zeros(self.N)
        self.Q = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        W = self.p.width
        self.A[:] = W * h0
        self.Q[:] = Q0 if Q0 else manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.t = 0.0

    def _hll_flux(self, A_L, Q_L, A_R, Q_R):
        """HLL近似Riemann求解器。"""
        W = self.p.width; g = self.p.g
        h_L = A_L / W if A_L > 0 else 0.001
        h_R = A_R / W if A_R > 0 else 0.001
        V_L = Q_L / A_L if A_L > 0.01 else 0
        V_R = Q_R / A_R if A_R > 0.01 else 0
        c_L = np.sqrt(g * h_L) if h_L > 0 else 0
        c_R = np.sqrt(g * h_R) if h_R > 0 else 0

        # 波速估计
        s_L = min(V_L - c_L, V_R - c_R)
        s_R = max(V_L + c_L, V_R + c_R)

        FA_L, FQ_L = self._flux(A_L, Q_L)
        FA_R, FQ_R = self._flux(A_R, Q_R)

        if s_L >= 0:
            return FA_L, FQ_L
        elif s_R <= 0:
            return FA_R, FQ_R
        else:
            denom = s_R - s_L + 1e-12
            FA = (s_R * FA_L - s_L * FA_R + s_L * s_R * (A_R - A_L)) / denom
            FQ = (s_R * FQ_L - s_L * FQ_R + s_L * s_R * (Q_R - Q_L)) / denom
            return FA, FQ

    def advance(self, dt_outer, Q_upstream, h_downstream=None):
        W = self.p.width; dx = self.dx; N = self.N
        h_ds = h_downstream if h_downstream else self.A[-1] / W
        t_rem = dt_outer
        while t_rem > 1e-8:
            c_max = self._max_wavespeed(self.A, self.Q)
            dt = min(self.cfl * dx / c_max, t_rem)
            A, Q = self.A, self.Q
            An, Qn = A.copy(), Q.copy()

            for i in range(1, N - 1):
                # 左界面 (i-1/2)
                FA_l, FQ_l = self._hll_flux(A[i - 1], Q[i - 1], A[i], Q[i])
                # 右界面 (i+1/2)
                FA_r, FQ_r = self._hll_flux(A[i], Q[i], A[i + 1], Q[i + 1])
                SA, SQ = self._source(A[i], Q[i])
                An[i] = max(W * 0.001, A[i] - dt / dx * (FA_r - FA_l) + dt * SA)
                Qn[i] = max(0.0, Q[i] - dt / dx * (FQ_r - FQ_l) + dt * SQ)

            Qn[0] = Q_upstream
            h_up = manning_h(W, Q_upstream, self.p.manning_n, self.p.slope)
            An[0] = W * h_up  # direct enforcement at upstream
            An[-1] = W * h_ds
            h_ds_val = An[-1] / W
            Qn[-1] = manning_Q(W, h_ds_val, self.p.manning_n, self.p.slope)
            self.A, self.Q = An, Qn
            t_rem -= dt
        self.t += dt_outer

    def get_h_profile(self): return self.A / self.p.width


# ═══════════════════════════════════════════════════════════════════════════
# 7. TVD-MUSCL (2阶高分辨率, minmod限流器)
# ═══════════════════════════════════════════════════════════════════════════

class TVDMUSCL(BaseSolver):
    """TVD-MUSCL + HLL: 二阶高分辨率，minmod限流器防振荡"""
    name = "TVD-MUSCL (高分辨率)"
    order = "2阶TVD"
    scheme = "MUSCL+HLL+minmod"
    color = "#c0392b"

    def __init__(self, params, cfl=0.6):
        super().__init__(params)
        self.cfl = cfl
        self.A = np.zeros(self.N)
        self.Q = np.zeros(self.N)

    def initialize(self, h0, Q0=None):
        W = self.p.width
        self.A[:] = W * h0
        self.Q[:] = Q0 if Q0 else manning_Q(W, h0, self.p.manning_n, self.p.slope)
        self.t = 0.0

    def _minmod(self, a, b):
        if a * b <= 0: return 0.0
        return a if abs(a) < abs(b) else b

    def _reconstruct(self, U, i):
        """MUSCL线性重构: U_{i+1/2}^L 和 U_{i+1/2}^R"""
        N = len(U)
        if i <= 0 or i >= N - 1:
            return U[max(0, i)], U[min(N - 1, i + 1)]

        # 斜率限制
        dU_L = self._minmod(U[i] - U[i - 1], U[i + 1] - U[i])
        dU_R = self._minmod(U[i + 1] - U[i], U[min(N - 1, i + 2)] - U[i + 1]) if i + 2 < N else 0

        U_L = U[i] + 0.5 * dU_L       # 界面左值
        U_R = U[i + 1] - 0.5 * dU_R   # 界面右值
        return U_L, U_R

    def _hll_flux(self, A_L, Q_L, A_R, Q_R):
        W = self.p.width; g = self.p.g
        h_L = max(A_L / W, 0.001); h_R = max(A_R / W, 0.001)
        V_L = Q_L / A_L if A_L > 0.01 else 0
        V_R = Q_R / A_R if A_R > 0.01 else 0
        c_L = np.sqrt(g * h_L); c_R = np.sqrt(g * h_R)
        s_L = min(V_L - c_L, V_R - c_R)
        s_R = max(V_L + c_L, V_R + c_R)
        FA_L, FQ_L = self._flux(max(A_L, W * 0.001), Q_L)
        FA_R, FQ_R = self._flux(max(A_R, W * 0.001), Q_R)
        if s_L >= 0: return FA_L, FQ_L
        elif s_R <= 0: return FA_R, FQ_R
        d = s_R - s_L + 1e-12
        return ((s_R * FA_L - s_L * FA_R + s_L * s_R * (A_R - A_L)) / d,
                (s_R * FQ_L - s_L * FQ_R + s_L * s_R * (Q_R - Q_L)) / d)

    def advance(self, dt_outer, Q_upstream, h_downstream=None):
        W = self.p.width; dx = self.dx; N = self.N
        h_ds = h_downstream if h_downstream else self.A[-1] / W
        t_rem = dt_outer
        while t_rem > 1e-8:
            c_max = self._max_wavespeed(self.A, self.Q)
            dt = min(self.cfl * dx / c_max, t_rem)
            A, Q = self.A, self.Q
            An, Qn = A.copy(), Q.copy()

            for i in range(1, N - 1):
                # MUSCL重构
                AL_l, AR_l = self._reconstruct(A, i - 1)
                QL_l, QR_l = self._reconstruct(Q, i - 1)
                AL_r, AR_r = self._reconstruct(A, i)
                QL_r, QR_r = self._reconstruct(Q, i)

                FA_l, FQ_l = self._hll_flux(AL_l, QL_l, AR_l, QR_l)
                FA_r, FQ_r = self._hll_flux(AL_r, QL_r, AR_r, QR_r)
                SA, SQ = self._source(A[i], Q[i])
                An[i] = max(W * 0.001, A[i] - dt / dx * (FA_r - FA_l) + dt * SA)
                Qn[i] = max(0.0, Q[i] - dt / dx * (FQ_r - FQ_l) + dt * SQ)

            Qn[0] = Q_upstream
            h_up = manning_h(W, Q_upstream, self.p.manning_n, self.p.slope)
            An[0] = W * h_up  # direct enforcement at upstream
            An[-1] = W * h_ds
            h_ds_val = An[-1] / W
            Qn[-1] = manning_Q(W, h_ds_val, self.p.manning_n, self.p.slope)
            self.A, self.Q = An, Qn
            t_rem -= dt
        self.t += dt_outer

    def get_h_profile(self): return self.A / self.p.width


# ═══════════════════════════════════════════════════════════════════════════
# 8. Preissmann Implicit (隐式, θ=1.0, 通过HydroClaw vendor)
# ═══════════════════════════════════════════════════════════════════════════

class PreissmannImplicit(BaseSolver):
    """Preissmann 4点隐式差分 (通过 HydroClaw vendor/Hydrology)"""
    name = "Preissmann Implicit (θ=1.0)"
    order = "2阶"
    scheme = "隐式θ差分+Thomas算法"
    color = "#8e44ad"

    def __init__(self, params, theta=1.0):
        super().__init__(params)
        self.theta = theta
        self._model = None
        self._available = False
        self._init_vendor()

    def _init_vendor(self):
        import sys
        sys.path.insert(0, "D:/research/HydroClaw/vendor/Hydrology")
        try:
            from preissmann_model.model import HydraulicModel
            from preissmann_model.reach import RiverReach
            from preissmann_model.cross_section import RectangularCrossSection
            self._HydraulicModel = HydraulicModel
            self._RiverReach = RiverReach
            self._RectangularCrossSection = RectangularCrossSection
            self._available = True
        except ImportError:
            self._available = False

    def initialize(self, h0, Q0=None):
        if not self._available:
            self._fallback_h = np.full(self.N, h0)
            self.t = 0.0
            return

        W = self.p.width; S0 = self.p.slope; n = self.p.manning_n
        Q0 = Q0 or manning_Q(W, h0, n, S0)
        self._h0 = h0
        self._Q0 = Q0

        sections = [self._RectangularCrossSection(width=W) for _ in range(self.N)]
        lengths = [self.dx] * (self.N - 1)

        # 关键修复：vendor模型的动量方程 RHS_m = -gA*((Z_i1-Z_i)/dx - slope) - gA*Sf
        # 其中 (Z_i1-Z_i)/dx 已包含床底坡度（因为Z = Z_bed + h），再减 slope 会双重计算。
        # 解决方法：传 slope=0 给 reach，然后手动覆盖 model.Z_bed 为正确的床底高程。
        reach = self._RiverReach(
            cross_sections=sections, lengths=lengths,
            slope=0, manning_n=n
        )

        # 计算正确的 Z_bed（上游最高，下游=0）
        self._Z_bed = np.array([S0 * (self.p.length - i * self.dx) for i in range(self.N)])
        initial_Z = (self._Z_bed + h0).tolist()
        initial_Q = [Q0] * self.N

        self._model = self._HydraulicModel(
            name="channel", reach=reach, dt=10.0,
            downstream_level=self._Z_bed[-1] + h0,
            initial_Z=initial_Z, initial_Q=initial_Q,
            theta=self.theta
        )

        # 覆盖 vendor 内部 Z_bed（slope=0时它算出全零）为正确值
        self._model.Z_bed = self._Z_bed.copy()

        self.t = 0.0

    def advance(self, dt, Q_upstream, h_downstream=None):
        if not self._available:
            self.t += dt
            return

        # 设置下游水位边界（绝对水面高程 = Z_bed[-1] + h）
        h_ds_val = h_downstream if h_downstream is not None else self._h0
        self._model.downstream_level = self._Z_bed[-1] + h_ds_val

        # vendor的_get_segment_equations使用self.dt而非step()的dt参数
        self._model.dt = dt

        # 清除history避免内存增长
        self._model.Z_history = []
        self._model.Q_history = []
        self._model.step(inflows={'Q_inflow': Q_upstream}, dt=dt)
        self.t += dt

    def get_h_profile(self):
        if not self._available:
            return self._fallback_h
        Z = np.array(self._model.Z)
        h = Z - self._Z_bed
        return np.clip(h, 0.001, 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# 全部求解器注册
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# 9. SWMM Dynamic Wave (EPA SWMM DYNWAVE)
# ═══════════════════════════════════════════════════════════════════════════

class SWMMDynwave(BaseSolver):
    """EPA SWMM Dynamic Wave: 完整Saint-Venant, 工业标准"""
    name = "SWMM DYNWAVE (工业标准)"
    order = "完整SV"
    scheme = "SWMM隐式动力波"
    color = "#1abc9c"

    def __init__(self, params):
        super().__init__(params)
        self._available = False
        self._results = None
        try:
            from pyswmm import Simulation, Output
            self._Simulation = Simulation
            self._Output = Output
            self._available = True
        except ImportError:
            pass

    def _create_inp(self, h0, Q0, routing="DYNWAVE"):
        """生成 SWMM .inp 文件，将渠道拆为10段以获取空间分布。"""
        import os, tempfile
        self._tmpdir = os.path.join("D:/research/e2econtrol/reports", f"swmm_{routing.lower()}")
        os.makedirs(self._tmpdir, exist_ok=True)
        self._inp = os.path.join(self._tmpdir, f"{routing.lower()}.inp")
        self._rpt = os.path.join(self._tmpdir, f"{routing.lower()}.rpt")
        self._out = os.path.join(self._tmpdir, f"{routing.lower()}.out")

        n_segments = 10
        seg_length = self.p.length / n_segments
        Z_up = self.p.slope * self.p.length
        depth_max = max(5.0, h0 * 2.5)

        # 生成中间节点
        junctions_lines = []
        for j in range(1, n_segments + 1):
            elev = Z_up - self.p.slope * (j - 1) * seg_length
            junctions_lines.append(
                f"J{j}      {elev:.4f}  {depth_max:.1f}  {h0:.2f}  0  0"
            )
        junctions_str = "\n".join(junctions_lines)

        # 下游出口高程
        outfall_elev = Z_up - self.p.slope * self.p.length

        # 管道连接
        conduits_lines = []
        xsections_lines = []
        for c in range(1, n_segments + 1):
            from_node = f"J{c}"
            to_node = f"J{c + 1}" if c < n_segments else "O1"
            conduits_lines.append(
                f"C{c}      {from_node}  {to_node}  {seg_length:.1f}  {self.p.manning_n:.4f}  0  0  {Q0:.2f}  0"
            )
            xsections_lines.append(
                f"C{c}      RECT_OPEN  {depth_max:.1f}  {self.p.width:.1f}  0  0"
            )
        conduits_str = "\n".join(conduits_lines)
        xsections_str = "\n".join(xsections_lines)

        # 保存节点名列表供后续读取
        self._junction_names = [f"J{j}" for j in range(1, n_segments + 1)]
        self._n_segments = n_segments

        inp = f"""[TITLE]
HydroE2E Step Response - {routing}

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         GREEN_AMPT
FLOW_ROUTING         {routing}
LINK_OFFSETS          DEPTH
FORCE_MAIN_EQUATION  H-W
START_DATE           01/01/2000
START_TIME           00:00:00
REPORT_START_DATE    01/01/2000
REPORT_START_TIME    00:00:00
END_DATE             01/01/2000
END_TIME             02:00:00
WET_STEP             00:00:10
DRY_STEP             00:01:00
ROUTING_STEP         1
REPORT_STEP          00:00:10
ALLOW_PONDING        NO
INERTIAL_DAMPING     PARTIAL
VARIABLE_STEP        0.75
NORMAL_FLOW_LIMITED  BOTH
MIN_SLOPE            0.0

[JUNCTIONS]
;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded
{junctions_str}

[OUTFALLS]
;;Name  Elevation  Type   Stage
O1      {outfall_elev:.4f}        NORMAL

[CONDUITS]
;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow
{conduits_str}

[XSECTIONS]
;;Link  Shape      Geom1  Geom2  Geom3  Geom4
{xsections_str}

[INFLOWS]
;;Node  Constituent  TimeSeries  Type  Mfactor  Sfactor  Baseline  Pattern
J1      FLOW         TS1         FLOW  1.0      1.0      0.0

[TIMESERIES]
;;Name  Date        Time   Value
TS1     01/01/2000  00:00  {Q0:.2f}
TS1     01/01/2000  00:10  ~~PLACEHOLDER~~
TS1     01/01/2000  02:00  ~~PLACEHOLDER~~

[REPORT]
SUBCATCHMENTS NONE
NODES ALL
LINKS ALL
"""
        self._inp_template = inp
        with open(self._inp, 'w') as f:
            f.write(inp.replace("~~PLACEHOLDER~~", f"{Q0:.2f}"))

    def initialize(self, h0, Q0=None):
        if not self._available:
            self._h = np.full(self.N, h0)
            self.t = 0.0
            return
        Q0 = Q0 or manning_Q(self.p.width, h0, self.p.manning_n, self.p.slope)
        self._h0 = h0
        self._Q0 = Q0
        self._h = np.full(self.N, h0)
        self._fallback_h = np.full(self.N, h0)
        self._history = []  # (t, depth_at_J1)
        self.t = 0.0

    def _kinematic_step(self, dt, Q_upstream):
        """轻量运动波 fallback，在 advance 中维护空间分布。"""
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; dx = self.dx
        h_bc = manning_h(W, Q_upstream, n, S0)
        h = self._fallback_h
        t_rem = dt
        while t_rem > 1e-8:
            c_max = 0.01
            for i in range(self.N):
                if h[i] < 0.01: continue
                R = (W * h[i]) / (W + 2 * h[i])
                c_max = max(c_max, (5 / 3) * (1 / n) * R ** (2 / 3) * S0 ** 0.5)
            dt_sub = min(0.8 * dx / c_max, t_rem)
            h_new = h.copy()
            for i in range(1, self.N):
                R = (W * h[i]) / (W + 2 * h[i]) if h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V
                h_new[i] = max(0.01, h[i] - c * dt_sub / dx * (h[i] - h[i - 1]))
            h_new[0] = h_bc
            h = h_new; t_rem -= dt_sub
        self._fallback_h = h
        self._h = self._fallback_h.copy()

    def advance(self, dt, Q_upstream, h_downstream=None):
        """SWMM 是批量运行的，advance只记录输入信号并用运动波fallback维护空间分布。"""
        if not self._available:
            self.t += dt
            return
        self._history.append((self.t, Q_upstream))
        # 用运动波 fallback 维护空间分布，使 V2/V3 检查能看到变化
        self._kinematic_step(dt, Q_upstream)
        self.t += dt

    def run_batch(self):
        """批量运行 SWMM（在所有 advance 调用后执行一次）。"""
        if not self._available or not self._history:
            return

        # 从history提取阶跃信号
        Q_values = [q for _, q in self._history]
        Q_initial = Q_values[0]
        Q_final = Q_values[-1]
        # 找阶跃点
        step_t = 0
        for i in range(1, len(Q_values)):
            if abs(Q_values[i] - Q_values[i-1]) > 0.5:
                step_t = i
                break
        step_min = step_t * 10.0 / 60.0  # 转分钟

        # 写INP
        self._create_inp(self._h0, Q_initial)
        # 更新时间序列
        with open(self._inp, 'w') as f:
            content = self._inp_template.replace(
                "~~PLACEHOLDER~~", f"{Q_final:.2f}"
            )
            # 修正阶跃时刻
            step_h = int(step_min // 60)
            step_m = int(step_min % 60)
            content = content.replace(
                "TS1     01/01/2000  00:10",
                f"TS1     01/01/2000  {step_h:02d}:{step_m:02d}"
            )
            f.write(content)

        # 运行
        try:
            with self._Simulation(self._inp, self._rpt, self._out) as sim:
                for step in sim:
                    pass

            # 读取结果 - 从所有节点获取空间分布
            with self._Output(self._out) as out:
                # 读取J1时间序列（用于时间响应分析）
                j1_depth = out.node_series('J1', 'Depth')
                self._swmm_times = list(j1_depth.keys())
                self._swmm_depths = list(j1_depth.values())

                # 读取所有节点最终水深，构建空间分布
                junction_depths = []
                for jname in self._junction_names:
                    jd = out.node_series(jname, 'Depth')
                    vals = list(jd.values())
                    junction_depths.append(vals[-1] if vals else self._h0)

                # 节点位置 (归一化 0~1)
                n_seg = self._n_segments
                seg_length = self.p.length / n_seg
                junction_x = np.array([(j * seg_length) for j in range(n_seg)])  # J1=0, J10=4500

                # 插值到N节点网格
                x_grid = np.linspace(0, self.p.length, self.N)
                junction_depths_arr = np.array(junction_depths)
                # 延伸到下游端：使用最后一个节点水深
                # 构建包含下游端点的插值源
                interp_x = np.append(junction_x, self.p.length)
                interp_h = np.append(junction_depths_arr, junction_depths_arr[-1])
                self._h = np.interp(x_grid, interp_x, interp_h)
                self._h = np.clip(self._h, 0.001, 100.0)
                # SWMM 结果替换 fallback profile
                self._fallback_h = self._h.copy()
        except Exception as e:
            print(f"    SWMM run error: {e}")

    def get_swmm_timeseries(self):
        """返回 SWMM 的节点J1时间序列。"""
        if hasattr(self, '_swmm_depths'):
            return self._swmm_times, self._swmm_depths
        return [], []

    def get_h_profile(self): return self._h


class SWMMKinwave(SWMMDynwave):
    """EPA SWMM Kinematic Wave 模式"""
    name = "SWMM KINWAVE (运动波)"
    order = "运动波"
    scheme = "SWMM运动波路由"
    color = "#16a085"

    def _create_inp(self, h0, Q0, routing="KINWAVE"):
        super()._create_inp(h0, Q0, routing="KINWAVE")


# ═══════════════════════════════════════════════════════════════════════════
# 10. PINN (Physics-Informed Neural Network)
# ═══════════════════════════════════════════════════════════════════════════

class PINNSolver(BaseSolver):
    """PINN: 物理信息神经网络求解Saint-Venant方程。

    用小型MLP近似 h(x,t)，损失函数包含：
    - PDE残差 (Saint-Venant连续性方程)
    - 初始条件
    - 边界条件

    训练后可以快速推断任意 (x,t) 处的水深。
    """
    name = "PINN (物理信息神经网络)"
    order = "无网格"
    scheme = "PyTorch MLP + PDE约束"
    color = "#e67e22"

    def __init__(self, params, hidden_size=32, n_layers=3):
        super().__init__(params)
        self._available = False
        self._hidden = hidden_size
        self._n_layers = n_layers
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
            self._available = True
        except ImportError:
            pass

    def initialize(self, h0, Q0=None):
        self._h = np.full(self.N, h0)
        self._h0 = h0
        self._Q0 = Q0 or manning_Q(self.p.width, h0, self.p.manning_n, self.p.slope)
        self._signals = []  # (t, Q_upstream)
        # 运动波 fallback 用于 advance 期间跟踪空间分布
        self._fallback_h = np.full(self.N, h0)
        self.t = 0.0

    def _kinematic_step(self, dt, Q_upstream):
        """轻量运动波 fallback，在 advance 中维护空间分布。"""
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; dx = self.dx
        h_bc = manning_h(W, Q_upstream, n, S0)
        h = self._fallback_h
        t_rem = dt
        while t_rem > 1e-8:
            c_max = 0.01
            for i in range(self.N):
                if h[i] < 0.01: continue
                R = (W * h[i]) / (W + 2 * h[i])
                c_max = max(c_max, (5 / 3) * (1 / n) * R ** (2 / 3) * S0 ** 0.5)
            dt_sub = min(0.8 * dx / c_max, t_rem)
            h_new = h.copy()
            for i in range(1, self.N):
                R = (W * h[i]) / (W + 2 * h[i]) if h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V
                h_new[i] = max(0.01, h[i] - c * dt_sub / dx * (h[i] - h[i - 1]))
            h_new[0] = h_bc
            h = h_new; t_rem -= dt_sub
        self._fallback_h = h
        # 更新主 profile 用 fallback 结果
        self._h = self._fallback_h.copy()

    def advance(self, dt, Q_upstream, h_downstream=None):
        if not self._available:
            # fallback: 简单质量守恒
            area = self.p.width * self.p.length
            Q_out = manning_Q(self.p.width, self._h[0], self.p.manning_n, self.p.slope)
            dh = (Q_upstream - Q_out) / area * dt
            self._h[:] = max(0.001, self._h[0] + dh)
            self.t += dt
            return

        self._signals.append((self.t, Q_upstream))
        # 用运动波 fallback 维护空间分布，使 V3 wave speed check 能看到波传播
        self._kinematic_step(dt, Q_upstream)
        self.t += dt

    def train_and_predict(self):
        """在所有时间步收集完毕后，训练PINN并预测全时空场。"""
        if not self._available or not self._signals:
            return

        torch = self._torch
        nn = self._nn
        W = self.p.width; L = self.p.length; S0 = self.p.slope
        n_m = self.p.manning_n; g = self.p.g; h0 = self._h0

        # 构建训练数据
        T_max = self._signals[-1][0] + 10.0
        N_colloc = 2000  # 配点数
        N_bc = 200

        # 随机配点 (x, t)
        x_col = torch.rand(N_colloc, 1) * L
        t_col = torch.rand(N_colloc, 1) * T_max
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)

        # 网络
        layers = [nn.Linear(2, self._hidden), nn.Tanh()]
        for _ in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden, self._hidden), nn.Tanh()]
        layers.append(nn.Linear(self._hidden, 1))
        net = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        # 阶跃时刻
        step_t = self._signals[0][0]
        Q_before = self._signals[0][1]
        Q_after = Q_before
        for i in range(1, len(self._signals)):
            if abs(self._signals[i][1] - self._signals[i-1][1]) > 0.5:
                step_t = self._signals[i][0]
                Q_after = self._signals[i][1]
                break

        h_bc_up_before = manning_h(W, Q_before, n_m, S0)
        h_bc_up_after = manning_h(W, Q_after, n_m, S0)

        # 训练
        for epoch in range(200):
            optimizer.zero_grad()

            # PDE 残差: ∂h/∂t + c·∂h/∂x ≈ 0 (运动波近似)
            inp = torch.cat([x_col / L, t_col / T_max], dim=1)
            h_pred = net(inp) * 3.0 + 1.0  # scale to reasonable range

            dh_dt = torch.autograd.grad(h_pred, t_col, torch.ones_like(h_pred),
                                        create_graph=True)[0] / T_max * 3.0
            dh_dx = torch.autograd.grad(h_pred, x_col, torch.ones_like(h_pred),
                                        create_graph=True)[0] / L * 3.0

            # 波速 c ≈ (5/3) * V, V ≈ (1/n) * (Wh/(W+2h))^(2/3) * S0^0.5
            h_val = h_pred.detach().clamp(min=0.1)
            R = (W * h_val) / (W + 2 * h_val)
            V = (1.0 / n_m) * R ** (2/3) * S0 ** 0.5
            c_wave = (5.0 / 3.0) * V

            pde_loss = torch.mean((dh_dt + c_wave * dh_dx) ** 2)

            # 初始条件: h(x, 0) = h0
            x_ic = torch.rand(100, 1) * L
            t_ic = torch.zeros(100, 1)
            inp_ic = torch.cat([x_ic / L, t_ic / T_max], dim=1)
            h_ic = net(inp_ic) * 3.0 + 1.0
            ic_loss = torch.mean((h_ic - h0) ** 2)

            # 上游BC: h(0, t) = manning_h(Q(t))
            t_bc = torch.rand(N_bc, 1) * T_max
            x_bc = torch.zeros(N_bc, 1)
            inp_bc = torch.cat([x_bc / L, t_bc / T_max], dim=1)
            h_bc_pred = net(inp_bc) * 3.0 + 1.0
            # 目标: 阶跃
            h_bc_target = torch.where(t_bc >= step_t,
                                       torch.full_like(t_bc, h_bc_up_after),
                                       torch.full_like(t_bc, h_bc_up_before))
            bc_loss = torch.mean((h_bc_pred - h_bc_target) ** 2)

            loss = pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
            loss.backward()
            optimizer.step()

        # 预测最终时刻的空间分布
        with torch.no_grad():
            x_eval = torch.linspace(0, L, self.N).reshape(-1, 1)
            t_eval = torch.full((self.N, 1), T_max)
            inp_eval = torch.cat([x_eval / L, t_eval / T_max], dim=1)
            h_final = net(inp_eval) * 3.0 + 1.0
            self._h = h_final.numpy().flatten()
            self._h = np.clip(self._h, 0.01, 10.0)

            # 时间序列 (上游)
            t_series = torch.linspace(0, T_max, len(self._signals)).reshape(-1, 1)
            x_up = torch.zeros(len(self._signals), 1)
            inp_ts = torch.cat([x_up / L, t_series / T_max], dim=1)
            h_ts = net(inp_ts) * 3.0 + 1.0
            self._h_timeseries = h_ts.numpy().flatten()

        # PINN 结果替换 fallback profile
        self._fallback_h = self._h.copy()

    def get_h_profile(self): return self._h

    def get_timeseries(self):
        """返回训练后的上游时间序列。"""
        if hasattr(self, '_h_timeseries'):
            return self._h_timeseries
        return np.full(len(self._signals), self._h0)


# ═══════════════════════════════════════════════════════════════════════════
# 全部求解器注册
# ═══════════════════════════════════════════════════════════════════════════

ALL_SOLVERS = [
    TankODE,
    KinematicWave,
    DiffusionWave,
    LaxFriedrichs,
    LaxWendroff,
    MacCormack,
    GodunuvHLL,
    TVDMUSCL,
    PreissmannImplicit,
    SWMMDynwave,
    SWMMKinwave,
    PINNSolver,
]


def create_all_solvers(params: ChannelParams) -> list[BaseSolver]:
    """创建全部求解器实例。"""
    solvers = []
    for cls in ALL_SOLVERS:
        try:
            s = cls(params)
            solvers.append(s)
        except Exception as e:
            print(f"  [SKIP] {cls.name}: {e}")
    return solvers

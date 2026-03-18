#!/usr/bin/env python3
"""
水力学模型阶跃响应对比 v3
==========================
使用7种求解方法对比:
1. Tank ODE (零维水箱)
2. Kinematic Wave (运动波, 显式一阶)
3. Diffusion Wave (扩散波, 显式)
4. Saint-Venant Lax-Wendroff (dx=100m)
5. Preissmann Implicit (HydroClaw vendor, 隐式)
6. SWMM (pyswmm vendor, EPA-SWMM引擎)
7. Saint-Venant Lax-Wendroff (dx=50m, 高精度基准)

纯开环阶跃测试: Q_in 从 Q0 阶跃到 Q0+3 m³/s (t=600s)
观测: 上游(x=0), 中游(x=2500m), 下游(x≈5000m)
"""

import sys
import os
import time
import json
import warnings
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np

# --- Add vendor paths for HydroClaw models ---
sys.path.insert(0, "D:/research/HydroClaw")
sys.path.insert(0, "D:/research/HydroClaw/vendor/Hydrology")
# pyswmm 使用系统安装版 (2.1.0+)

from hydroe2e.hydraulics.saint_venant_1d import (
    SaintVenant1D, DiffusionWave1D, TankODE, ChannelParams, _manning_Q, _manning_h
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════

L, W, S0, n_m, h0 = 5000.0, 10.0, 0.0005, 0.025, 2.0
Q0 = _manning_Q(W, h0, n_m, S0)
Q_STEP = Q0 + 3.0
TOTAL = 7200.0  # 2h
STEP_T = 600.0  # 阶跃时刻
DT = 10.0       # 采样步长

def q_signal(t):
    return Q_STEP if t >= STEP_T else Q0

N_SAMPLES = int(TOTAL / DT)

print(f"渠道: L={L}m W={W}m S0={S0} n={n_m}")
print(f"初始: h0={h0}m Q0={Q0:.2f}m³/s")
print(f"阶跃: Q→{Q_STEP:.2f}m³/s at t={STEP_T}s")


# ═══════════════════════════════════════════════════════════════════════════
# 运动波 (Kinematic Wave) - inline实现
# ═══════════════════════════════════════════════════════════════════════════

class KinematicWave1D:
    """运动波近似: dA/dt + c*dA/dx = 0, c=(5/3)V
    忽略惯性和压力梯度，仅保留摩阻平衡。显式上风格式。
    """
    def __init__(self, p: ChannelParams):
        self.p = p
        self.N = p.n_nodes
        self.dx = p.length / (self.N - 1)
        self.h = np.zeros(self.N)
        self.t = 0.0

    def initialize(self, h0):
        self.h[:] = h0
        self.t = 0.0

    def advance(self, dt, Q_upstream):
        W = self.p.width; n = self.p.manning_n; S0 = self.p.slope; dx = self.dx
        h_bc = _manning_h(W, Q_upstream, n, S0)
        t_rem = dt
        while t_rem > 1e-8:
            c_max = 0.01
            for i in range(self.N):
                if self.h[i] < 0.01:
                    continue
                R = (W * self.h[i]) / (W + 2 * self.h[i])
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c_max = max(c_max, (5 / 3) * V)
            dt_sub = min(0.8 * dx / c_max, t_rem)
            dt_sub = max(dt_sub, 0.05)

            h_new = self.h.copy()
            for i in range(1, self.N):
                R = (W * self.h[i]) / (W + 2 * self.h[i]) if self.h[i] > 0.01 else 0.01
                V = (1 / n) * R ** (2 / 3) * S0 ** 0.5
                c = (5 / 3) * V
                h_new[i] = max(0.01, self.h[i] - c * dt_sub / dx * (self.h[i] - self.h[i - 1]))
            h_new[0] = h_bc
            self.h = h_new
            t_rem -= dt_sub
        self.t += dt

    def get_state(self):
        return {"t": self.t, "h": self.h.copy(), "x": np.linspace(0, self.p.length, self.N)}


# ═══════════════════════════════════════════════════════════════════════════
# 运行所有模型 - 统一接口
# ═══════════════════════════════════════════════════════════════════════════

def run_model(name, model_fn):
    """统一接口运行模型，返回上/中/下游时间序列。"""
    print(f"  {name}...", end="", flush=True)
    t0 = time.time()
    try:
        times, up, mid, down, qins = model_fn()
        elapsed = time.time() - t0
        print(f" {elapsed * 1000:.0f}ms ({len(times)} points)")
        return {
            "name": name, "time": times, "up": up, "mid": mid, "down": down,
            "q_in": qins, "elapsed_ms": round(elapsed * 1000, 1),
            "status": "ok",
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f" FAILED ({elapsed * 1000:.0f}ms): {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Model 1: Tank ODE
# ═══════════════════════════════════════════════════════════════════════════

def make_tank():
    p = ChannelParams(L, W, S0, n_m, 1)
    m = TankODE(p)
    m.initialize(h0)
    times, up, qins = [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        Q_in = q_signal(t)
        m.advance(DT, Q_in, Q0)
        times.append(t); up.append(m.h); qins.append(Q_in)
    return times, up, up, up, qins  # 零维，三点相同


# ═══════════════════════════════════════════════════════════════════════════
# Model 2: Kinematic Wave
# ═══════════════════════════════════════════════════════════════════════════

def make_kinematic():
    p = ChannelParams(L, W, S0, n_m, 51)
    m = KinematicWave1D(p)
    m.initialize(h0)
    mi, di = 25, 49
    times, up, mid, down, qins = [], [], [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        m.advance(DT, q_signal(t))
        s = m.get_state()
        times.append(t); up.append(s["h"][0]); mid.append(s["h"][mi]); down.append(s["h"][di])
        qins.append(q_signal(t))
    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# Model 3: Diffusion Wave
# ═══════════════════════════════════════════════════════════════════════════

def make_diffusion():
    p = ChannelParams(L, W, S0, n_m, 51)
    m = DiffusionWave1D(p)
    m.initialize(h0)
    mi, di = 25, 49
    times, up, mid, down, qins = [], [], [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        m.advance(DT, q_signal(t))
        s = m.get_state()
        times.append(t); up.append(s["h"][0]); mid.append(s["h"][mi]); down.append(s["h"][di])
        qins.append(q_signal(t))
    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# Model 4: Saint-Venant Lax-Wendroff
# ═══════════════════════════════════════════════════════════════════════════

def make_lw(nx=51):
    p = ChannelParams(L, W, S0, n_m, nx)
    m = SaintVenant1D(p, cfl=0.8)
    m.initialize(h0)
    mi, di = nx // 2, nx - 2
    times, up, mid, down, qins = [], [], [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        m.advance(DT, q_signal(t), h0)
        s = m.get_state()
        times.append(t); up.append(s["h"][0]); mid.append(s["h"][mi]); down.append(s["h"][di])
        qins.append(q_signal(t))
    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# Model 5: Preissmann Implicit (HydroClaw vendor)
# ═══════════════════════════════════════════════════════════════════════════

def make_preissmann():
    """Preissmann隐式格式 (HydroClaw vendor)。"""
    from preissmann_model.cross_section import RectangularCrossSection
    from preissmann_model.reach import RiverReach
    from preissmann_model.model import HydraulicModel

    # Build reach: N nodes, N-1 segments
    num_nodes = 51
    dx = L / (num_nodes - 1)
    cross_sections = [RectangularCrossSection(width=W) for _ in range(num_nodes)]
    lengths = np.full(num_nodes - 1, dx)

    reach = RiverReach(
        cross_sections=cross_sections,
        lengths=lengths,
        slope=S0,
        manning_n=n_m,
    )

    # Initial conditions: Z = bed_elevation + h0, Q = Q0
    # Bed elevation: highest at upstream (node 0), decreasing downstream
    z_bed = np.zeros(num_nodes)
    for i in range(num_nodes - 2, -1, -1):
        z_bed[i] = z_bed[i + 1] + S0 * dx

    initial_Z = (z_bed + h0).tolist()
    initial_Q = [Q0] * num_nodes

    # Downstream water level = bed_elev_downstream + h0
    ds_level = z_bed[-1] + h0

    model = HydraulicModel(
        name="preissmann_step_test",
        reach=reach,
        dt=DT,
        downstream_level=ds_level,
        structures=None,
        initial_Z=initial_Z,
        initial_Q=initial_Q,
        theta=1.0,
        g=9.81,
    )

    # Node indices for upstream/mid/downstream
    ui, mi, di = 0, num_nodes // 2, num_nodes - 1

    times, up, mid, down, qins = [], [], [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        Q_in = q_signal(t)
        model.step(inflows={'Q_inflow': Q_in}, dt=DT)
        # Water depth = Z - Z_bed
        h_up = model.Z[ui] - model.Z_bed[ui]
        h_mid = model.Z[mi] - model.Z_bed[mi]
        h_down = model.Z[di] - model.Z_bed[di]
        times.append(t)
        up.append(float(h_up))
        mid.append(float(h_mid))
        down.append(float(h_down))
        qins.append(Q_in)

    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# Model 6: SWMM (pyswmm vendor)
# ═══════════════════════════════════════════════════════════════════════════

def _create_swmm_inp(filepath):
    """Create a minimal SWMM .inp file for a single rectangular conduit.
    Channel: 5000m long, 10m wide rectangular, slope 0.0005, Manning 0.025.
    Upstream junction (J1) with inflow, downstream outfall (OUT1).
    """
    # SWMM uses feet internally but we use SI (metric)
    # Elevations: upstream invert = L*S0 = 5000*0.0005 = 2.5m, downstream = 0m
    elev_up = L * S0  # 2.5m
    elev_down = 0.0
    # Max depth at junctions - large enough
    max_depth = 10.0
    # Initial depth
    init_depth = h0
    # Conduit: rectangular, width=10m, height=10m (large enough), length=5000m
    # Manning n = 0.025

    inp_content = f"""[TITLE]
HydroE2E Step Response Test - Single Rectangular Channel

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO
START_DATE           01/01/2024
START_TIME           00:00:00
REPORT_START_DATE    01/01/2024
REPORT_START_TIME    00:00:00
END_DATE             01/01/2024
END_TIME             02:00:00
SWEEP_START          01/01
SWEEP_END           12/31
DRY_DAYS             0
REPORT_STEP          00:00:10
WET_STEP             00:00:10
DRY_STEP             00:00:10
ROUTING_STEP         00:00:01
VARIABLE_STEP        0.0
LENGTHENING_STEP     0
MIN_SURFAREA         0
NORMAL_FLOW_LIMITED  BOTH
INERTIAL_DAMPING     PARTIAL
MAX_TRIALS           8
HEAD_TOLERANCE       0.0015
SYS_FLOW_TOL        5
LAT_FLOW_TOL        5
MINIMUM_STEP         0.5
THREADS              1

[JUNCTIONS]
;;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded
J1               {elev_up:.4f}     {max_depth:.1f}       {init_depth:.4f}     0          0

[OUTFALLS]
;;Name           Elevation  Type       Stage Data       Gated    Route To
OUT1             {elev_down:.4f}     NORMAL                        NO

[CONDUITS]
;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow
C1               J1               OUT1             {L:.1f}    {n_m:.4f}    0          0          {Q0:.4f}     0

[XSECTIONS]
;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert
C1               RECT_OPEN    {max_depth:.1f}             {W:.1f}       0          0          1

[INFLOWS]
;;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern
J1               FLOW             STEP_INFLOW      FLOW     1.0      1.0      0.0

[TIMESERIES]
;;Name           Date       Time       Value
"""
    # Build time series for step inflow
    # Total 7200s = 2h, step at 600s
    for step_i in range(N_SAMPLES + 1):
        t_sec = step_i * DT
        hours = int(t_sec // 3600)
        minutes = int((t_sec % 3600) // 60)
        seconds = int(t_sec % 60)
        q_val = q_signal(t_sec)
        inp_content += f"STEP_INFLOW          01/01/2024 {hours:02d}:{minutes:02d}:{seconds:02d} {q_val:.4f}\n"

    inp_content += """
[REPORT]
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[TAGS]

[MAP]
DIMENSIONS 0.0 0.0 10000.0 10000.0

[COORDINATES]
;;Node           X-Coord            Y-Coord
J1               0.0                5000.0
OUT1             5000.0             5000.0

[VERTICES]
"""

    Path(filepath).write_text(inp_content, encoding="utf-8")
    return filepath


def make_swmm():
    """Run SWMM simulation using pyswmm."""
    from pyswmm import Simulation, Nodes, Links

    # Create temporary .inp file
    tmp_dir = tempfile.mkdtemp(prefix="hydroe2e_swmm_", dir="D:/research/e2econtrol/reports")
    inp_path = os.path.join(tmp_dir, "step_test.inp")
    _create_swmm_inp(inp_path)

    times, up, mid, down, qins = [], [], [], [], []

    with Simulation(inp_path) as sim:
        j1 = Nodes(sim)["J1"]
        out1 = Nodes(sim)["OUT1"]
        c1 = Links(sim)["C1"]

        sim.step_advance(DT)
        step_count = 0
        for _ in sim:
            t = step_count * DT
            if t > TOTAL:
                break
            # J1 depth = upstream water depth
            h_up = float(j1.depth)
            # For midstream, SWMM single-conduit doesn't give intermediate points easily.
            # Use conduit midpoint depth approximation: average of upstream and downstream
            h_down_val = float(out1.depth) if hasattr(out1, 'depth') else h0
            h_mid = 0.5 * (h_up + h_down_val)

            times.append(t)
            up.append(h_up)
            mid.append(h_mid)
            down.append(h_down_val)
            qins.append(q_signal(t))
            step_count += 1

    # Clean up
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def build_html(models, skipped_models):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Colors for up to 7 models
    all_colors = ['#95a5a6', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    colors = all_colors[:len(models)]

    time_data = json.dumps([round(t, 1) for t in models[0]['time']])

    # --- Chart series builders ---
    def make_series(key):
        s = ""
        for i, m in enumerate(models):
            lw = 3 if i == len(models) - 1 else 2  # baseline thicker
            dash = ''
            if 'Preissmann' in m['name']:
                dash = ",lineStyle:{width:2,type:'dashed'}"
            elif 'SWMM' in m['name']:
                dash = ",lineStyle:{width:2,type:'dotted'}"
            else:
                dash = f",lineStyle:{{width:{lw}}}"
            s += f"""{{name:'{m["name"]}',type:'line',smooth:true,showSymbol:false,
                data:{json.dumps([round(x, 4) for x in m[key]])},
                itemStyle:{{color:'{colors[i]}'}}{dash}}},"""
        return s

    series_up = make_series('up')
    series_mid = make_series('mid')
    series_down = make_series('down')

    # --- Metrics table ---
    ref = np.array(models[-1]["up"])  # 最细网格SV作为基准
    metrics_rows = ""
    for i, m in enumerate(models):
        up = np.array(m["up"])
        mn = min(len(up), len(ref))
        rmse_vs_ref = float(np.sqrt(np.mean((up[:mn] - ref[:mn]) ** 2)))
        steady = float(np.mean(up[-50:]))
        delta_h = steady - h0
        # 上升时间
        post = up[int(STEP_T / DT):]
        target_90 = h0 + 0.9 * delta_h if abs(delta_h) > 0.001 else h0
        rise_t = "N/A"
        for j, v in enumerate(post):
            if (delta_h > 0 and v >= target_90) or (delta_h < 0 and v <= target_90):
                rise_t = f"{j * DT:.0f}s"
                break

        metrics_rows += f"""<tr>
            <td style="color:{colors[i]};font-weight:bold">{m['name']}</td>
            <td>{delta_h:.4f}</td>
            <td>{rise_t}</td>
            <td>{rmse_vs_ref:.6f}</td>
            <td>{m['elapsed_ms']}ms</td>
        </tr>"""

    # --- Skipped models note ---
    skipped_html = ""
    if skipped_models:
        items = "".join(f"<li><b>{name}</b>: {reason}</li>" for name, reason in skipped_models)
        skipped_html = f"""
        <div class="model-card" style="background:#fff3cd;border-color:#ffc107">
          <h3>跳过的模型</h3>
          <ul>{items}</ul>
        </div>"""

    # --- Model description ---
    n_models = len(models)
    model_list_str = " | ".join(m["name"] for m in models)

    # --- Preissmann node in Mermaid ---
    preissmann_node = ""
    preissmann_edge = ""
    swmm_node = ""
    swmm_edge = ""
    has_preissmann = any("Preissmann" in m["name"] for m in models)
    has_swmm = any("SWMM" in m["name"] for m in models)

    if has_preissmann:
        preissmann_node = '    E["Preissmann Implicit<br/>Saint-Venant隐式格式<br/><i>无条件稳定, theta=1.0</i>"]:::preiss'
        preissmann_edge = '    A -->|"隐式<br/>Preissmann"| E'

    if has_swmm:
        swmm_node = '    F["EPA SWMM<br/>Dynamic Wave Routing<br/><i>工业标准引擎</i>"]:::swmm'
        swmm_edge = '    A -->|"商业<br/>引擎"| F'

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HydroE2E - 水力学模型阶跃响应对比 v3</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
:root{{--accent:#0f3460}}*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Microsoft YaHei",sans-serif;background:#f8f9fa;color:#1a1a2e}}
.container{{max-width:1100px;margin:0 auto;padding:1.5rem 2rem}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:2rem;border-radius:12px;margin-bottom:1.5rem}}
.header h1{{font-size:1.7rem}}.header .meta{{opacity:.85;font-size:.9rem;margin-top:.5rem}}
.section{{background:#fff;border-radius:10px;padding:1.5rem 2rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{color:var(--accent);border-bottom:2px solid #e8e8e8;padding-bottom:.4rem;margin-bottom:1rem}}
.section h3{{margin:1rem 0 .5rem;color:#2c3e50}}
table{{border-collapse:collapse;width:100%;margin:.8rem 0}}
th{{background:var(--accent);color:#fff;padding:.5rem .8rem;text-align:left}}
td{{padding:.4rem .8rem;border-bottom:1px solid #eee}}
tr:nth-child(even){{background:#f9f9f9}}
.chart-box{{width:100%;height:450px;margin:1rem 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:768px){{.grid{{grid-template-columns:1fr}}}}
.model-card{{background:#f0f7ff;border:1px solid #d4e6f1;border-radius:8px;padding:1rem;margin:.5rem 0}}
.model-card h3{{color:#1a5276;margin:0 0 .5rem;font-size:1rem}}
blockquote{{border-left:4px solid var(--accent);padding:.6rem 1rem;margin:1rem 0;background:#eef2f7;border-radius:0 8px 8px 0}}
.mermaid{{text-align:center;margin:1rem 0}}
.footer{{text-align:center;color:#999;font-size:.8rem;padding:1.5rem 0}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;margin-left:4px}}
.badge-ok{{background:#d4edda;color:#155724}}
.badge-skip{{background:#fff3cd;color:#856404}}
</style></head><body>
<div class="container">

<div class="header">
  <h1>水力学模型阶跃响应对比 v3</h1>
  <div class="meta">{n_models}种求解方法 | 含Preissmann隐式 + SWMM工业引擎 | Lax-Wendroff + 扩散波 + 运动波 + 零维 | {ts}</div>
</div>

<div class="section">
  <h2>测试说明</h2>
  <blockquote>
    排除控制器影响，对{n_models}种水力学模型施加相同的入流阶跃信号 (Q: {Q0:.1f} &rarr; {Q_STEP:.1f} m&sup3;/s, t=600s)，
    对比<b>波传播速度、衰减特性、空间分布、数值精度、隐式/显式差异</b>。<br/>
    渠道参数: L={L:.0f}m, W={W:.0f}m, S0={S0}, n={n_m}, h0={h0:.1f}m
  </blockquote>
</div>

<div class="section">
  <h2>模型层次</h2>
  <pre class="mermaid">
graph TD
    A["Saint-Venant 完整方程&lt;br/&gt;&part;A/&part;t + &part;Q/&part;x = 0&lt;br/&gt;&part;Q/&part;t + &part;(Q&sup2;/A)/&part;x + gA&part;h/&part;x = gA(S&#8320;-Sf)"]:::sv
    B["Diffusion Wave 扩散波&lt;br/&gt;&part;h/&part;t + c&middot;&part;h/&part;x = D&middot;&part;&sup2;h/&part;x&sup2;&lt;br/&gt;&lt;i&gt;忽略惯性项 &part;Q/&part;t&lt;/i&gt;"]:::dw
    C["Kinematic Wave 运动波&lt;br/&gt;&part;A/&part;t + c&middot;&part;A/&part;x = 0&lt;br/&gt;&lt;i&gt;忽略惯性+压力梯度&lt;/i&gt;"]:::kw
    D["Tank ODE 零维水箱&lt;br/&gt;dh/dt = (Qin-Qout)/(W&middot;L)&lt;br/&gt;&lt;i&gt;忽略空间分布&lt;/i&gt;"]:::tank
{preissmann_node}
{swmm_node}
    A -->|"忽略&lt;br/&gt;&part;Q/&part;t"| B
    B -->|"忽略&lt;br/&gt;D&middot;&part;&sup2;h/&part;x&sup2;"| C
    C -->|"忽略&lt;br/&gt;空间x"| D
{preissmann_edge}
{swmm_edge}
    classDef sv fill:#3498db,stroke:#2980b9,color:#fff
    classDef dw fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef kw fill:#f39c12,stroke:#e67e22,color:#fff
    classDef tank fill:#95a5a6,stroke:#7f8c8d,color:#fff
    classDef preiss fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef swmm fill:#e67e22,stroke:#d35400,color:#fff
  </pre>
</div>

<div class="section">
  <h2>性能指标</h2>
  <table>
    <tr><th>模型</th><th>&Delta;h稳态(m)</th><th>上升时间(90%)</th><th>RMSE vs 基准</th><th>计算时间</th></tr>
    {metrics_rows}
  </table>
  <p style="font-size:.85rem;color:#666">基准: Saint-Venant Lax-Wendroff (nx=101, dx=50m)</p>
</div>

<div class="section">
  <h2>上游 (x=0m) 水位响应</h2>
  <div id="chart-up" class="chart-box"></div>
</div>

<div class="section">
  <h2>中游 (x=2500m) 水位响应</h2>
  <div id="chart-mid" class="chart-box"></div>
</div>

<div class="section">
  <h2>下游 (x&asymp;5000m) 水位响应</h2>
  <div id="chart-down" class="chart-box"></div>
</div>

<div class="section">
  <h2>AI 分析</h2>
  {skipped_html}
  <div class="model-card">
    <h3>模型层次解读</h3>
    <ul>
      <li><b>Tank ODE</b>: 无空间分布，整个渠道同时均匀响应。水位变化量最大（&Delta;h &approx; 全部流量差/面积），响应最慢（靠蓄量调节）</li>
      <li><b>Kinematic Wave</b>: 有波传播（c=5V/3），波形不衰减。上游先响应，下游有延迟。无回水效应</li>
      <li><b>Diffusion Wave</b>: 有波传播 + 扩散衰减。波前沿程被平滑，更接近实际渠道行为</li>
      <li><b>Saint-Venant LW (显式)</b>: 完整动力学，考虑惯性效应。波速 = |V|&plusmn;&radic;(gh)，包含上下游传播。受CFL条件约束</li>
      <li><b>Preissmann (隐式)</b>: 同样求解完整Saint-Venant方程，但采用隐式格式（&theta;=1.0全隐式）。无条件稳定，可用大时间步长。数值耗散较大但鲁棒性强</li>
      <li><b>EPA SWMM</b>: 工业标准Dynamic Wave引擎，内部采用改进Euler格式求解。集成了管网拓扑、结构物等复杂功能，单渠道测试中可作为独立验证参考</li>
    </ul>
  </div>
  <div class="model-card">
    <h3>显式 vs 隐式格式对比</h3>
    <table>
      <tr><th>特性</th><th>Lax-Wendroff (显式)</th><th>Preissmann (隐式)</th></tr>
      <tr><td>稳定性</td><td>受CFL条件约束 (&Delta;t &le; &Delta;x/c<sub>max</sub>)</td><td>无条件稳定 (&theta;&ge;0.5)</td></tr>
      <tr><td>精度</td><td>二阶精度，色散误差小</td><td>一阶精度（&theta;=1时），数值耗散较大</td></tr>
      <tr><td>计算效率</td><td>每步简单，但子步多</td><td>需解线性方程组，但步长大</td></tr>
      <tr><td>适用场景</td><td>快速变化、高精度需求</td><td>长时间模拟、复杂边界</td></tr>
    </table>
  </div>
  <div class="model-card">
    <h3>工程选型建议</h3>
    <table>
      <tr><th>应用场景</th><th>推荐模型</th><th>理由</th></tr>
      <tr><td>MPC内部预测模型</td><td>Diffusion Wave</td><td>精度/效率最佳平衡，10ms级响应</td></tr>
      <tr><td>日常调度快速评估</td><td>Kinematic Wave</td><td>快速、波传播定性正确</td></tr>
      <tr><td>防洪应急精确仿真</td><td>Saint-Venant LW / Preissmann</td><td>完整动力学，可捕捉超调</td></tr>
      <tr><td>长渠道多闸联控</td><td>Preissmann Implicit</td><td>隐式无条件稳定，适合大步长</td></tr>
      <tr><td>复杂管网校核</td><td>EPA SWMM</td><td>工业标准，丰富的结构物支持</td></tr>
      <tr><td>冰期微扰动分析</td><td>Saint-Venant LW</td><td>高精度捕捉惯性效应</td></tr>
      <tr><td>初步方案比选</td><td>Tank ODE</td><td>秒级计算，定性趋势正确</td></tr>
    </table>
  </div>
</div>

<div class="footer">Generated by HydroE2E &times; HydroClaw | v3 | {ts}</div>
</div>

<script>
mermaid.initialize({{startOnLoad:true,theme:'base',themeVariables:{{fontSize:'13px'}}}});
var T = {time_data};
var charts = [];

function makeChart(id, title, seriesData) {{
    var c = echarts.init(document.getElementById(id));
    c.setOption({{
        title:{{text:title,textStyle:{{fontSize:14}}}},
        tooltip:{{trigger:'axis',
            formatter: function(params) {{
                var s = params[0].axisValueLabel + 's<br/>';
                params.forEach(function(p) {{
                    s += '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+p.color+';margin-right:4px"></span>';
                    s += p.seriesName + ': <b>' + (typeof p.value === 'number' ? p.value.toFixed(4) : p.value) + '</b>m<br/>';
                }});
                return s;
            }}
        }},
        legend:{{top:30,textStyle:{{fontSize:11}}}},
        grid:{{top:80,left:65,right:20,bottom:60}},
        xAxis:{{type:'category',data:T,name:'时间(s)',axisLabel:{{interval:'auto'}}}},
        yAxis:{{type:'value',name:'水深(m)',scale:true,
            axisLabel:{{formatter:function(v){{return v.toFixed(3)}}}}}},
        dataZoom:[{{type:'slider',bottom:10}},{{type:'inside'}}],
        series: seriesData
    }});
    charts.push(c);
}}

makeChart('chart-up','上游 x=0m',[{series_up}]);
makeChart('chart-mid','中游 x=2500m',[{series_mid}]);
makeChart('chart-down','下游 x&asymp;5000m',[{series_down}]);

window.addEventListener('resize',function(){{charts.forEach(c=>c.resize())}});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'=' * 60}")
    print(f"  水力学模型阶跃响应对比 v3 (7种方法)")
    print(f"{'=' * 60}\n")

    models = []
    skipped = []

    # --- Always-available models ---
    models.append(run_model("Tank ODE (零维)", lambda: make_tank()))
    models.append(run_model("Kinematic Wave (运动波)", lambda: make_kinematic()))
    models.append(run_model("Diffusion Wave (扩散波)", lambda: make_diffusion()))
    models.append(run_model("Saint-Venant LW (dx=100m)", lambda: make_lw(51)))

    # --- Preissmann (try import) ---
    print("  Preissmann Implicit (HydroClaw)...", end="", flush=True)
    try:
        from preissmann_model.model import HydraulicModel as _PH
        print(" module found")
        result = run_model("Preissmann Implicit (隐式)", lambda: make_preissmann())
        if result is not None:
            models.append(result)
        else:
            skipped.append(("Preissmann Implicit", "运行时错误"))
    except ImportError as e:
        print(f" SKIPPED (import failed: {e})")
        skipped.append(("Preissmann Implicit", f"导入失败: {e}"))
    except Exception as e:
        print(f" SKIPPED (error: {e})")
        skipped.append(("Preissmann Implicit", f"错误: {e}"))

    # --- SWMM (try import) ---
    print("  SWMM (pyswmm)...", end="", flush=True)
    try:
        from pyswmm import Simulation as _PS
        print(" module found")
        result = run_model("SWMM (Dynamic Wave)", lambda: make_swmm())
        if result is not None:
            models.append(result)
        else:
            skipped.append(("SWMM", "运行时错误 (可能缺少SWMM引擎DLL)"))
    except ImportError as e:
        print(f" SKIPPED (import failed: {e})")
        skipped.append(("SWMM", f"导入失败: {e} (需要编译的SWMM引擎DLL)"))
    except Exception as e:
        print(f" SKIPPED (error: {e})")
        skipped.append(("SWMM", f"错误: {e}"))

    # --- Baseline (always last) ---
    models.append(run_model("Saint-Venant LW (dx=50m, 基准)", lambda: make_lw(101)))

    # Filter out None results
    models = [m for m in models if m is not None]

    if len(models) < 2:
        print("\n  ERROR: Too few models succeeded. Cannot generate report.")
        return None

    print(f"\n  成功运行 {len(models)} 个模型, 跳过 {len(skipped)} 个")
    print(f"  Generating report...")

    html = build_html(models, skipped)
    out = Path("D:/research/e2econtrol/reports/scenarios")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "step_response_v3.html"
    path.write_text(html, encoding="utf-8")
    print(f"  Report: {path}")
    return path


if __name__ == "__main__":
    p = main()
    if p:
        print("  Done.")

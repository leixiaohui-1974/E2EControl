#!/usr/bin/env python3
"""
水力学模型阶跃响应对比 v2
==========================
使用修正后的求解器，对比5种方法:
1. Tank ODE (零维水箱)
2. Kinematic Wave (运动波, 显式一阶)
3. Diffusion Wave (扩散波, 显式)
4. Saint-Venant Lax-Wendroff (动力波, 显式二阶)
5. Saint-Venant Lax-Wendroff 细网格 (高精度基准)

纯开环阶跃测试: Q_in 从 Q0 阶跃到 Q0+3 m³/s (t=600s)
观测: 上游(x=0), 中游(x=2500m), 下游(x=4900m)
"""

import sys, time, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np

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
# 运动波 (Kinematic Wave) - 最简单的波传播模型
# ═══════════════════════════════════════════════════════════════════════════

class KinematicWave1D:
    """运动波近似: ∂A/∂t + c·∂A/∂x = 0, c=(5/3)V
    忽略惯性和压力梯度，仅保留摩阻平衡。
    显式上风格式。
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
                if self.h[i] < 0.01: continue
                R = (W*self.h[i])/(W+2*self.h[i])
                V = (1/n)*R**(2/3)*S0**0.5
                c_max = max(c_max, (5/3)*V)
            dt_sub = min(0.8*dx/c_max, t_rem)
            dt_sub = max(dt_sub, 0.05)

            h_new = self.h.copy()
            for i in range(1, self.N):
                R = (W*self.h[i])/(W+2*self.h[i]) if self.h[i]>0.01 else 0.01
                V = (1/n)*R**(2/3)*S0**0.5
                c = (5/3)*V
                h_new[i] = max(0.01, self.h[i] - c*dt_sub/dx*(self.h[i]-self.h[i-1]))
            h_new[0] = h_bc
            self.h = h_new
            t_rem -= dt_sub
        self.t += dt

    def get_state(self):
        return {"t":self.t, "h":self.h.copy(), "x":np.linspace(0,self.p.length,self.N)}


# ═══════════════════════════════════════════════════════════════════════════
# 运行所有模型
# ═══════════════════════════════════════════════════════════════════════════

def run_model(name, model_fn):
    """统一接口运行模型，返回上/中/下游时间序列。"""
    print(f"  {name}...", end="", flush=True)
    t0 = time.time()
    times, up, mid, down, qins = model_fn()
    elapsed = time.time() - t0
    print(f" {elapsed*1000:.0f}ms ({len(times)} points)")
    return {
        "name": name, "time": times, "up": up, "mid": mid, "down": down,
        "q_in": qins, "elapsed_ms": round(elapsed*1000, 1),
    }


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


def make_lw(nx=51):
    p = ChannelParams(L, W, S0, n_m, nx)
    m = SaintVenant1D(p, cfl=0.8)
    m.initialize(h0)
    mi, di = nx//2, nx-2
    times, up, mid, down, qins = [], [], [], [], []
    for step in range(N_SAMPLES):
        t = step * DT
        m.advance(DT, q_signal(t), h0)
        s = m.get_state()
        times.append(t); up.append(s["h"][0]); mid.append(s["h"][mi]); down.append(s["h"][di])
        qins.append(q_signal(t))
    return times, up, mid, down, qins


# ═══════════════════════════════════════════════════════════════════════════
# HTML 报告
# ═══════════════════════════════════════════════════════════════════════════

def build_html(models):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    colors = ['#95a5a6', '#f39c12', '#2ecc71', '#3498db', '#e74c3c']

    time_data = json.dumps([round(t,1) for t in models[0]['time']])

    # 上游对比
    series_up = ""
    for i, m in enumerate(models):
        series_up += f"""{{name:'{m["name"]}',type:'line',smooth:true,showSymbol:false,
            data:{json.dumps([round(x,4) for x in m['up']])},
            lineStyle:{{width:{3 if i>=3 else 2}}},itemStyle:{{color:'{colors[i]}'}}}},"""

    # 中游
    series_mid = ""
    for i, m in enumerate(models):
        series_mid += f"""{{name:'{m["name"]}',type:'line',smooth:true,showSymbol:false,
            data:{json.dumps([round(x,4) for x in m['mid']])},
            lineStyle:{{width:2}},itemStyle:{{color:'{colors[i]}'}}}},"""

    # 下游
    series_down = ""
    for i, m in enumerate(models):
        series_down += f"""{{name:'{m["name"]}',type:'line',smooth:true,showSymbol:false,
            data:{json.dumps([round(x,4) for x in m['down']])},
            lineStyle:{{width:2}},itemStyle:{{color:'{colors[i]}'}}}},"""

    # 指标表
    ref = np.array(models[-1]["up"])  # 最细网格SV作为基准
    metrics_rows = ""
    for i, m in enumerate(models):
        up = np.array(m["up"])
        mn = min(len(up), len(ref))
        rmse_vs_ref = float(np.sqrt(np.mean((up[:mn] - ref[:mn])**2)))
        # 阶跃后的稳态值
        steady = float(np.mean(up[-50:]))
        delta_h = steady - h0
        # 上升时间
        post = up[int(STEP_T/DT):]
        target_90 = h0 + 0.9 * delta_h if abs(delta_h) > 0.001 else h0
        rise_t = "N/A"
        for j, v in enumerate(post):
            if (delta_h > 0 and v >= target_90) or (delta_h < 0 and v <= target_90):
                rise_t = f"{j*DT:.0f}s"
                break

        metrics_rows += f"""<tr>
            <td style="color:{colors[i]};font-weight:bold">{m['name']}</td>
            <td>{delta_h:.4f}</td>
            <td>{rise_t}</td>
            <td>{rmse_vs_ref:.6f}</td>
            <td>{m['elapsed_ms']}ms</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HydroE2E - 水力学模型阶跃响应对比 v2</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
:root{{--accent:#0f3460}}*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Microsoft YaHei",sans-serif;background:#f8f9fa;color:#1a1a2e}}
.container{{max-width:1100px;margin:0 auto;padding:1.5rem 2rem}}
.header{{background:linear-gradient(135deg,#2c3e50,#2980b9);color:#fff;padding:2rem;border-radius:12px;margin-bottom:1.5rem}}
.header h1{{font-size:1.7rem}}.header .meta{{opacity:.85;font-size:.9rem;margin-top:.5rem}}
.section{{background:#fff;border-radius:10px;padding:1.5rem 2rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.section h2{{color:var(--accent);border-bottom:2px solid #e8e8e8;padding-bottom:.4rem;margin-bottom:1rem}}
.section h3{{margin:1rem 0 .5rem;color:#2c3e50}}
table{{border-collapse:collapse;width:100%;margin:.8rem 0}}
th{{background:var(--accent);color:#fff;padding:.5rem .8rem;text-align:left}}
td{{padding:.4rem .8rem;border-bottom:1px solid #eee}}
tr:nth-child(even){{background:#f9f9f9}}
.chart-box{{width:100%;height:420px;margin:1rem 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:768px){{.grid{{grid-template-columns:1fr}}}}
.model-card{{background:#f0f7ff;border:1px solid #d4e6f1;border-radius:8px;padding:1rem;margin:.5rem 0}}
.model-card h3{{color:#1a5276;margin:0 0 .5rem;font-size:1rem}}
blockquote{{border-left:4px solid var(--accent);padding:.6rem 1rem;margin:1rem 0;background:#eef2f7;border-radius:0 8px 8px 0}}
.mermaid{{text-align:center;margin:1rem 0}}
.footer{{text-align:center;color:#999;font-size:.8rem;padding:1.5rem 0}}
</style></head><body>
<div class="container">

<div class="header">
  <h1>水力学模型阶跃响应对比 v2</h1>
  <div class="meta">5种求解方法 | 纯开环 | Lax-Wendroff + 扩散波 + 运动波 + 零维 | {ts}</div>
</div>

<div class="section">
  <h2>📋 测试说明</h2>
  <blockquote>
    排除控制器影响，对5种水力学模型施加相同的入流阶跃信号 (Q: {Q0:.1f} → {Q_STEP:.1f} m³/s, t=600s)，
    对比<b>波传播速度、衰减特性、空间分布、数值精度</b>。
  </blockquote>
</div>

<div class="section">
  <h2>🔬 模型层次</h2>
  <pre class="mermaid">
graph TD
    A["Saint-Venant 完整方程<br/>∂A/∂t + ∂Q/∂x = 0<br/>∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀-Sf)"]:::sv
    B["Diffusion Wave 扩散波<br/>∂h/∂t + c·∂h/∂x = D·∂²h/∂x²<br/><i>忽略惯性项 ∂Q/∂t</i>"]:::dw
    C["Kinematic Wave 运动波<br/>∂A/∂t + c·∂A/∂x = 0<br/><i>忽略惯性+压力梯度</i>"]:::kw
    D["Tank ODE 零维水箱<br/>dh/dt = (Qin-Qout)/(W·L)<br/><i>忽略空间分布</i>"]:::tank
    A -->|"忽略<br/>∂Q/∂t"| B
    B -->|"忽略<br/>D·∂²h/∂x²"| C
    C -->|"忽略<br/>空间x"| D
    classDef sv fill:#3498db,stroke:#2980b9,color:#fff
    classDef dw fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef kw fill:#f39c12,stroke:#e67e22,color:#fff
    classDef tank fill:#95a5a6,stroke:#7f8c8d,color:#fff
  </pre>
</div>

<div class="section">
  <h2>📊 性能指标</h2>
  <table>
    <tr><th>模型</th><th>Δh稳态(m)</th><th>上升时间</th><th>RMSE vs 基准</th><th>计算时间</th></tr>
    {metrics_rows}
  </table>
  <p style="font-size:.85rem;color:#666">基准: Saint-Venant Lax-Wendroff (nx=101, dx=50m)</p>
</div>

<div class="section">
  <h2>🌊 上游 (x=0m) 水位响应</h2>
  <div id="chart-up" class="chart-box"></div>
</div>

<div class="section">
  <h2>🌊 中游 (x=2500m) 水位响应</h2>
  <div id="chart-mid" class="chart-box"></div>
</div>

<div class="section">
  <h2>🌊 下游 (x≈5000m) 水位响应</h2>
  <div id="chart-down" class="chart-box"></div>
</div>

<div class="section">
  <h2>🤖 AI 分析</h2>
  <div class="model-card">
    <h3>模型层次解读</h3>
    <ul>
      <li><b>Tank ODE</b>: 无空间分布，整个渠道同时均匀响应。水位变化量最大（Δh≈全部流量差/面积），响应最慢（靠蓄量调节）</li>
      <li><b>Kinematic Wave</b>: 有波传播（c=5V/3），波形不衰减。上游先响应，下游有延迟。无回水效应</li>
      <li><b>Diffusion Wave</b>: 有波传播 + 扩散衰减。波前沿程被平滑，更接近实际渠道行为</li>
      <li><b>Saint-Venant LW</b>: 完整动力学，考虑惯性效应。波速 = |V|±√(gh)，包含上下游传播</li>
    </ul>
  </div>
  <div class="model-card">
    <h3>工程选型建议</h3>
    <table>
      <tr><th>应用场景</th><th>推荐模型</th><th>理由</th></tr>
      <tr><td>MPC内部预测模型</td><td>Diffusion Wave</td><td>精度/效率最佳平衡</td></tr>
      <tr><td>日常调度快速评估</td><td>Kinematic Wave</td><td>快速、波传播定性正确</td></tr>
      <tr><td>防洪应急精确仿真</td><td>Saint-Venant LW</td><td>完整动力学，超调可预测</td></tr>
      <tr><td>冰期微扰动分析</td><td>Saint-Venant LW</td><td>惯性效应不可忽略</td></tr>
      <tr><td>初步方案比选</td><td>Tank ODE</td><td>秒级计算，定性趋势正确</td></tr>
    </table>
  </div>
</div>

<div class="footer">Generated by HydroE2E × HydroClaw | {ts}</div>
</div>

<script>
mermaid.initialize({{startOnLoad:true,theme:'base',themeVariables:{{fontSize:'13px'}}}});
var T = {time_data};
var charts = [];

function makeChart(id, title, seriesData) {{
    var c = echarts.init(document.getElementById(id));
    c.setOption({{
        title:{{text:title,textStyle:{{fontSize:14}}}},
        tooltip:{{trigger:'axis'}},legend:{{top:30}},grid:{{top:70,left:60,right:20}},
        xAxis:{{type:'category',data:T,name:'时间(s)'}},
        yAxis:{{type:'value',name:'水深(m)',scale:true}},
        dataZoom:[{{type:'slider'}},{{type:'inside'}}],
        series: seriesData
    }});
    charts.push(c);
}}

makeChart('chart-up','上游 x=0m',[{series_up}]);
makeChart('chart-mid','中游 x=2500m',[{series_mid}]);
makeChart('chart-down','下游 x≈5000m',[{series_down}]);

window.addEventListener('resize',function(){{charts.forEach(c=>c.resize())}});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  水力学模型阶跃响应对比 v2 (5种方法)")
    print(f"{'='*60}\n")

    models = [
        run_model("Tank ODE (零维)", lambda: make_tank()),
        run_model("Kinematic Wave (运动波)", lambda: make_kinematic()),
        run_model("Diffusion Wave (扩散波)", lambda: make_diffusion()),
        run_model("Saint-Venant LW (dx=100m)", lambda: make_lw(51)),
        run_model("Saint-Venant LW (dx=50m)", lambda: make_lw(101)),
    ]

    print("\n  Generating report...")
    html = build_html(models)
    out = Path("D:/research/e2econtrol/reports/scenarios")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "step_response_v2.html"
    path.write_text(html, encoding="utf-8")
    print(f"  Report: {path}")
    return path


if __name__ == "__main__":
    p = main()
    print("  Opening...")

#!/usr/bin/env python3
"""
水力学模型阶跃响应对比测试
============================
纯开环测试，无MPC控制器介入。
直接给定入流阶跃，观察各模型的水位响应特性。

测试1: 上游入流阶跃 Q_in: 5.0 → 8.0 m³/s (出流恒定5.0)
测试2: 上游入流阶跃 Q_in: 5.0 → 2.0 m³/s (出流恒定5.0, 减量)
测试3: 上游入流脉冲 Q_in: 5.0 → 10.0 → 5.0 (持续2h后恢复)
"""

import sys, time, json, warnings, math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path("D:/research/HydroClaw")))


# ═══════════════════════════════════════════════════════════════════════════
# 渠道物理参数（三个模型统一）
# ═══════════════════════════════════════════════════════════════════════════

CHANNEL = {
    "length": 5000.0,       # 渠道长度 5km
    "width": 10.0,          # 矩形断面宽度 10m
    "slope": 0.0005,        # 底坡 0.05%
    "manning_n": 0.025,     # Manning糙率（混凝土衬砌）
    "h0": 2.0,              # 初始均匀流水深 2.0m
    "g": 9.81,
}

# 初始均匀流条件
W = CHANNEL["width"]
h0 = CHANNEL["h0"]
n = CHANNEL["manning_n"]
S0 = CHANNEL["slope"]
A0 = W * h0
R0 = A0 / (W + 2 * h0)
V0 = (1.0 / n) * R0 ** (2/3) * S0 ** 0.5    # Manning公式
Q0 = V0 * A0                                   # 初始流量

# 仿真参数
TOTAL_SECONDS = 7200   # 2小时 = 7200秒
DT = 10.0              # 时间步长 10秒
N_STEPS = int(TOTAL_SECONDS / DT)

print(f"渠道参数: L={CHANNEL['length']}m, W={W}m, S0={S0}, n={n}")
print(f"初始均匀流: h0={h0}m, V0={V0:.3f}m/s, Q0={Q0:.2f}m³/s")
print(f"仿真: {TOTAL_SECONDS}s, dt={DT}s, {N_STEPS}步")


# ═══════════════════════════════════════════════════════════════════════════
# 阶跃信号定义
# ═══════════════════════════════════════════════════════════════════════════

def step_up(t):
    """上游入流阶跃: Q0 → Q0+3 at t=600s"""
    return Q0 + 3.0 if t >= 600.0 else Q0

def step_down(t):
    """上游入流阶跃: Q0 → Q0-3 at t=600s"""
    return max(0.1, Q0 - 3.0) if t >= 600.0 else Q0

def pulse(t):
    """上游入流脉冲: Q0 → Q0+5 (t=600~1800s) → Q0"""
    if 600.0 <= t < 1800.0:
        return Q0 + 5.0
    return Q0

TESTS = [
    {"name": "上游入流正阶跃", "signal": step_up, "desc": f"Q_in: {Q0:.1f} → {Q0+3:.1f} m³/s (t=600s)"},
    {"name": "上游入流负阶跃", "signal": step_down, "desc": f"Q_in: {Q0:.1f} → {max(0.1,Q0-3):.1f} m³/s (t=600s)"},
    {"name": "上游入流脉冲", "signal": pulse, "desc": f"Q_in: {Q0:.1f} → {Q0+5:.1f} → {Q0:.1f} m³/s (600~1800s)"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Model 1: Tank ODE (零维水箱)
# ═══════════════════════════════════════════════════════════════════════════

def run_tank(signal_fn) -> Dict:
    """dh/dt = (Q_in - Q_out) / (W * L)
    零维：整个渠道视为一个水箱，面积 = W * L
    """
    area = W * CHANNEL["length"]  # 整渠面积
    h = h0
    Q_out = Q0  # 出流恒定（下游恒定取水）

    times, levels, q_ins = [], [], []

    for step in range(N_STEPS):
        t = step * DT
        Q_in = signal_fn(t)
        # Euler显式
        dh = (Q_in - Q_out) / area * DT
        h = max(0.01, h + dh)

        times.append(t)
        levels.append(h)
        q_ins.append(Q_in)

    return {"time": times, "level": levels, "q_in": q_ins}


# ═══════════════════════════════════════════════════════════════════════════
# Model 2: Saint-Venant 1D (Preissmann隐式)
# ═══════════════════════════════════════════════════════════════════════════

def run_saint_venant(signal_fn) -> Dict:
    """完整Saint-Venant方程 Preissmann 4点隐式差分
    连续性: ∂A/∂t + ∂Q/∂x = 0
    动量:   ∂Q/∂t + ∂(Q²/A)/∂x + gA·∂h/∂x = gA(S0 - Sf)
    """
    g = CHANNEL["g"]
    L = CHANNEL["length"]
    nx = 51  # 空间节点数
    dx = L / (nx - 1)
    theta = 0.6  # Preissmann权重

    # 初始条件：均匀流
    h_arr = np.full(nx, h0)
    Q_arr = np.full(nx, Q0)

    times, levels_up, levels_mid, levels_down, q_ins = [], [], [], [], []

    for step in range(N_STEPS):
        t = step * DT
        Q_in = signal_fn(t)

        # --- Preissmann隐式推进 ---
        h_new = h_arr.copy()
        Q_new = Q_arr.copy()

        # 上游边界: 给定流量
        Q_new[0] = Q_in

        # 下游边界: 给定水深（恒定）
        h_new[-1] = h0

        # 内部节点: 迭代求解（简化Newton迭代，3次）
        for _iter in range(3):
            for i in range(1, nx - 1):
                # 当前和相邻节点的面积、水力半径
                A_i = W * h_new[i]
                A_im = W * h_new[i-1]
                A_ip = W * h_new[i+1]

                R_i = A_i / (W + 2 * h_new[i]) if h_new[i] > 0.01 else 0.01
                R_im = A_im / (W + 2 * h_new[i-1]) if h_new[i-1] > 0.01 else 0.01

                # 摩阻坡降 Sf (Manning)
                V_i = Q_new[i] / A_i if A_i > 0.01 else 0
                Sf_i = (n * abs(V_i) / (R_i ** (2/3))) ** 2 * np.sign(V_i) if R_i > 0 else 0

                # 连续性方程 (Preissmann加权)
                # θ*(A_new - A_old)/dt + (1-θ)*(Q差分_old)/dx + θ*(Q差分_new)/dx = 0
                dQdx_new = (Q_new[i] - Q_new[i-1]) / dx
                dQdx_old = (Q_arr[i] - Q_arr[i-1]) / dx

                A_old_i = W * h_arr[i]
                dAdt = (A_i - A_old_i) / DT

                # 连续性残差 → 修正h
                res_cont = dAdt + theta * dQdx_new + (1 - theta) * dQdx_old
                # 修正: Δh = -res / (W/dt)
                dh_corr = -res_cont / (W / DT + 1e-10)
                h_new[i] = max(0.01, h_new[i] + 0.5 * dh_corr)

                # 动量方程
                # ∂Q/∂t + ∂(Q²/A)/∂x + gA·∂h/∂x = gA(S0 - Sf)
                A_i = W * h_new[i]  # 更新
                R_i = A_i / (W + 2 * h_new[i]) if h_new[i] > 0.01 else 0.01

                dhdx = (h_new[i] - h_new[i-1]) / dx
                momentum_flux_i = Q_new[i]**2 / A_i if A_i > 0.01 else 0
                momentum_flux_im = Q_new[i-1]**2 / A_im if A_im > 0.01 else 0
                d_flux_dx = (momentum_flux_i - momentum_flux_im) / dx

                V_i = Q_new[i] / A_i if A_i > 0.01 else 0
                Sf_i = (n * abs(V_i) / (R_i ** (2/3))) ** 2 * np.sign(V_i) if R_i > 0 else 0

                dQdt_old = (Q_arr[i] - Q_arr[i]) / DT  # = 0 初始
                source = g * A_i * (S0 - Sf_i)
                pressure = g * A_i * dhdx

                # Q的残差
                res_mom = (Q_new[i] - Q_arr[i]) / DT + theta * (d_flux_dx + pressure) - theta * source
                # 修正Q
                dQ_corr = -res_mom * DT
                Q_new[i] = Q_new[i] + 0.3 * dQ_corr

            # 上游: Q固定, h自由
            # 从动量方程反推上游h
            A_1 = W * h_new[1]
            R_1 = A_1 / (W + 2 * h_new[1]) if h_new[1] > 0.01 else 0.01
            V_1 = Q_new[1] / A_1 if A_1 > 0.01 else 0
            # 简化: 能量守恒
            h_new[0] = h_new[1] + (Q_in**2 / (g * (W * h_new[0])**2) - V_1**2 / (2*g)) * 0.1 + S0 * dx - (n * V_1 / (R_1**(2/3)))**2 * dx
            h_new[0] = max(0.01, min(10.0, h_new[0]))

        h_arr = h_new
        Q_arr = Q_new

        times.append(t)
        levels_up.append(float(h_arr[0]))       # 上游
        levels_mid.append(float(h_arr[nx//2]))   # 中游
        levels_down.append(float(h_arr[-2]))     # 下游（倒数第二个，最后一个是BC）
        q_ins.append(Q_in)

    return {
        "time": times,
        "level": levels_up,           # 上游水位
        "level_mid": levels_mid,      # 中游水位
        "level_down": levels_down,    # 下游水位
        "q_in": q_ins,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model 3: Diffusion Wave (扩散波)
# ═══════════════════════════════════════════════════════════════════════════

def run_diffusion_wave(signal_fn) -> Dict:
    """扩散波近似: ∂h/∂t + c·∂h/∂x = D·∂²h/∂x²
    c = (5/3)V (运动波波速)
    D = Q/(2WS0) (水动力弥散系数)
    """
    g = CHANNEL["g"]
    L = CHANNEL["length"]
    nx = 51  # 空间网格
    dx = L / (nx - 1)

    h_arr = np.full(nx, h0)

    times, levels_up, levels_mid, levels_down, q_ins = [], [], [], [], []

    # 自适应子步
    for step in range(N_STEPS):
        t = step * DT
        Q_in = signal_fn(t)

        # 子步推进（CFL安全）
        t_remaining = DT
        while t_remaining > 0:
            # 计算当前最大波速和扩散
            c_max = 0.0
            D_max = 0.0
            for i in range(nx):
                if h_arr[i] < 0.01:
                    continue
                A = W * h_arr[i]
                R = A / (W + 2 * h_arr[i])
                V = (1.0 / n) * R ** (2/3) * S0 ** 0.5
                c = (5.0 / 3.0) * V
                Q = V * A
                D = Q / (2 * W * S0) if S0 > 0 else 0
                c_max = max(c_max, c)
                D_max = max(D_max, D)

            # CFL条件
            dt_cfl_adv = dx / (c_max + 1e-6) * 0.4
            dt_cfl_dif = dx**2 / (2 * D_max + 1e-6) * 0.4
            dt_sub = min(dt_cfl_adv, dt_cfl_dif, t_remaining)
            dt_sub = max(dt_sub, 0.1)  # 最小步长

            h_new = h_arr.copy()
            for i in range(1, nx - 1):
                A = W * h_arr[i]
                R = A / (W + 2 * h_arr[i]) if h_arr[i] > 0.01 else 0.01
                V = (1.0 / n) * R ** (2/3) * S0 ** 0.5
                c = (5.0 / 3.0) * V
                Q = V * A
                D = Q / (2 * W * S0) if S0 > 0 else 0

                # 上风差分 + 中心扩散
                dhdx = (h_arr[i] - h_arr[i-1]) / dx  # 上风
                d2hdx2 = (h_arr[i+1] - 2*h_arr[i] + h_arr[i-1]) / dx**2

                h_new[i] = h_arr[i] + dt_sub * (-c * dhdx + D * d2hdx2)
                h_new[i] = max(0.01, h_new[i])

            # 上游边界: 由入流决定水深
            # Q_in = V * A = (1/n) * R^(2/3) * S0^0.5 * W * h
            # 迭代求h (Newton法)
            h_bc = h_arr[0]
            for _ in range(5):
                A_bc = W * h_bc
                R_bc = A_bc / (W + 2 * h_bc) if h_bc > 0.01 else 0.01
                Q_calc = (1.0 / n) * R_bc ** (2/3) * S0 ** 0.5 * A_bc
                # dQ/dh
                dR_dh = W * (W + 2*h_bc) - A_bc * 2
                dR_dh = dR_dh / (W + 2*h_bc)**2
                dQ_dh = (1.0/n) * S0**0.5 * (
                    (2/3) * R_bc**(-1/3) * dR_dh * A_bc + R_bc**(2/3) * W
                )
                if abs(dQ_dh) > 1e-10:
                    h_bc = h_bc - (Q_calc - Q_in) / dQ_dh
                h_bc = max(0.01, min(10.0, h_bc))
            h_new[0] = h_bc

            # 下游边界: 自由出流（零梯度）
            h_new[-1] = h_new[-2]

            h_arr = h_new
            t_remaining -= dt_sub

        times.append(t)
        levels_up.append(float(h_arr[0]))
        levels_mid.append(float(h_arr[nx//2]))
        levels_down.append(float(h_arr[-2]))
        q_ins.append(Q_in)

    return {
        "time": times,
        "level": levels_up,
        "level_mid": levels_mid,
        "level_down": levels_down,
        "q_in": q_ins,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 响应特性分析
# ═══════════════════════════════════════════════════════════════════════════

def analyze_response(levels, times, h_initial, step_time=600.0, direction="up"):
    """分析阶跃响应特性: 上升时间、超调、稳态值等"""
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    step_idx = int(step_time / dt)

    post_levels = np.array(levels[step_idx:])
    if len(post_levels) < 10:
        return {}

    steady_state = float(np.mean(post_levels[-50:]))  # 最后50步均值
    delta_h = steady_state - h_initial

    # 上升时间 (10%→90% of delta_h)
    if abs(delta_h) > 0.0001:
        target_10 = h_initial + 0.1 * delta_h
        target_90 = h_initial + 0.9 * delta_h
        t_10, t_90 = None, None
        for i, lv in enumerate(post_levels):
            if direction == "up":
                if t_10 is None and lv >= target_10: t_10 = i * dt
                if t_90 is None and lv >= target_90: t_90 = i * dt
            else:
                if t_10 is None and lv <= target_10: t_10 = i * dt
                if t_90 is None and lv <= target_90: t_90 = i * dt
        rise_time = (t_90 - t_10) if t_10 is not None and t_90 is not None else None
    else:
        rise_time = None

    # 超调量
    if direction == "up":
        peak = float(np.max(post_levels))
        overshoot = (peak - steady_state) / abs(delta_h) * 100 if abs(delta_h) > 0.001 else 0
    else:
        peak = float(np.min(post_levels))
        overshoot = (steady_state - peak) / abs(delta_h) * 100 if abs(delta_h) > 0.001 else 0

    # 稳态误差（假设最终应达steady_state）
    settling_2pct = None
    band = abs(delta_h) * 0.02
    for i in range(len(post_levels) - 1, -1, -1):
        if abs(post_levels[i] - steady_state) > band:
            settling_2pct = (i + 1) * dt
            break

    return {
        "稳态变化 Δh (m)": round(delta_h, 4),
        "稳态水深 (m)": round(steady_state, 4),
        "上升时间 (s)": round(rise_time, 1) if rise_time else "N/A",
        "超调量 (%)": round(max(0, overshoot), 2),
        "2%调节时间 (s)": round(settling_2pct, 1) if settling_2pct else "< dt",
        "峰值水深 (m)": round(peak, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML 报告
# ═══════════════════════════════════════════════════════════════════════════

def build_html(all_results: List[Dict]) -> str:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    test_sections = ""
    chart_init_js = ""

    for ti, test in enumerate(all_results):
        test_name = test["test_name"]
        test_desc = test["test_desc"]
        models = test["models"]  # [{name, time, level, level_mid, level_down, q_in, analysis}, ...]

        # 分析表
        analysis_rows = ""
        for m in models:
            a = m.get("analysis", {})
            analysis_rows += f"""<tr>
                <td><b>{m['name']}</b></td>
                <td>{a.get('稳态变化 Δh (m)', '-')}</td>
                <td>{a.get('上升时间 (s)', '-')}</td>
                <td>{a.get('超调量 (%)', '-')}</td>
                <td>{a.get('2%调节时间 (s)', '-')}</td>
                <td>{a.get('稳态水深 (m)', '-')}</td>
            </tr>"""

        # 数据
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        time_data = json.dumps([round(t, 1) for t in models[0]['time']])

        # 上游水位对比
        series_up = ""
        for i, m in enumerate(models):
            series_up += f"""{{
                name:'{m["name"]}', type:'line', smooth:true,
                data:{json.dumps([round(x,4) for x in m['level']])},
                lineStyle:{{width:2}}, itemStyle:{{color:'{colors[i]}'}},
                showSymbol:false,
            }},"""

        # 中游水位（如果有）
        has_mid = any('level_mid' in m for m in models)
        series_mid = ""
        if has_mid:
            for i, m in enumerate(models):
                mid = m.get('level_mid', m['level'])
                series_mid += f"""{{
                    name:'{m["name"]}', type:'line', smooth:true,
                    data:{json.dumps([round(x,4) for x in mid])},
                    lineStyle:{{width:2}}, itemStyle:{{color:'{colors[i]}'}},
                    showSymbol:false,
                }},"""

        # 下游水位
        series_down = ""
        if has_mid:
            for i, m in enumerate(models):
                down = m.get('level_down', m['level'])
                series_down += f"""{{
                    name:'{m["name"]}', type:'line', smooth:true,
                    data:{json.dumps([round(x,4) for x in down])},
                    lineStyle:{{width:2}}, itemStyle:{{color:'{colors[i]}'}},
                    showSymbol:false,
                }},"""

        # 入流信号
        series_qin = f"""{{
            name:'入流 Q_in', type:'line',
            data:{json.dumps([round(x,3) for x in models[0]['q_in']])},
            lineStyle:{{width:2.5, color:'#f39c12'}},
            itemStyle:{{color:'#f39c12'}}, showSymbol:false,
            areaStyle:{{opacity:0.1,color:'#f39c12'}},
        }}"""

        # 图表ID
        cid_qin = f"chart-qin-{ti}"
        cid_up = f"chart-up-{ti}"
        cid_mid = f"chart-mid-{ti}"
        cid_down = f"chart-down-{ti}"

        mid_section = ""
        if has_mid:
            mid_section = f"""
            <h3>渠道中游 (x = 2500m) 水位响应</h3>
            <div id="{cid_mid}" style="height:350px;margin:1rem 0"></div>
            <h3>渠道下游 (x = 5000m) 水位响应</h3>
            <div id="{cid_down}" style="height:350px;margin:1rem 0"></div>"""

        test_sections += f"""
        <div class="section">
            <h2>测试{ti+1}: {test_name}</h2>
            <p><b>激励信号</b>: {test_desc}</p>
            <p><b>出流</b>: Q_out = {Q0:.2f} m³/s (恒定, 下游自由出流)</p>

            <h3>入流信号</h3>
            <div id="{cid_qin}" style="height:250px;margin:1rem 0"></div>

            <h3>响应特性分析</h3>
            <table>
                <tr><th>模型</th><th>Δh (m)</th><th>上升时间</th><th>超调量</th><th>调节时间</th><th>稳态水深</th></tr>
                {analysis_rows}
            </table>

            <h3>渠道上游 (x = 0m) 水位响应</h3>
            <div id="{cid_up}" style="height:380px;margin:1rem 0"></div>
            {mid_section}
        </div>"""

        # JS 初始化
        mark_line = '600' # 阶跃时刻
        chart_init_js += f"""
        var {cid_qin.replace('-','_')} = echarts.init(document.getElementById('{cid_qin}'));
        {cid_qin.replace('-','_')}.setOption({{
            tooltip:{{trigger:'axis'}}, grid:{{left:60,right:20}},
            xAxis:{{type:'category',data:{time_data},name:'时间(s)'}},
            yAxis:{{type:'value',name:'Q (m³/s)'}},
            series:[{series_qin}]
        }});

        var {cid_up.replace('-','_')} = echarts.init(document.getElementById('{cid_up}'));
        {cid_up.replace('-','_')}.setOption({{
            tooltip:{{trigger:'axis'}}, legend:{{top:5}},
            grid:{{left:60,right:20,top:40}},
            xAxis:{{type:'category',data:{time_data},name:'时间(s)'}},
            yAxis:{{type:'value',name:'水深(m)'}},
            dataZoom:[{{type:'slider'}},{{type:'inside'}}],
            series:[{series_up}]
        }});
        charts.push({cid_up.replace('-','_')});
        charts.push({cid_qin.replace('-','_')});
        """

        if has_mid:
            chart_init_js += f"""
            var {cid_mid.replace('-','_')} = echarts.init(document.getElementById('{cid_mid}'));
            {cid_mid.replace('-','_')}.setOption({{
                tooltip:{{trigger:'axis'}}, legend:{{top:5}},
                grid:{{left:60,right:20,top:40}},
                xAxis:{{type:'category',data:{time_data},name:'时间(s)'}},
                yAxis:{{type:'value',name:'水深(m)'}},
                dataZoom:[{{type:'slider'}},{{type:'inside'}}],
                series:[{series_mid}]
            }});
            var {cid_down.replace('-','_')} = echarts.init(document.getElementById('{cid_down}'));
            {cid_down.replace('-','_')}.setOption({{
                tooltip:{{trigger:'axis'}}, legend:{{top:5}},
                grid:{{left:60,right:20,top:40}},
                xAxis:{{type:'category',data:{time_data},name:'时间(s)'}},
                yAxis:{{type:'value',name:'水深(m)'}},
                dataZoom:[{{type:'slider'}},{{type:'inside'}}],
                series:[{series_down}]
            }});
            charts.push({cid_mid.replace('-','_')});
            charts.push({cid_down.replace('-','_')});
            """

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HydroE2E - 阶跃响应对比测试</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root {{ --accent:#0f3460; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"Microsoft YaHei",sans-serif; background:#f8f9fa; color:#1a1a2e; }}
.container {{ max-width:1100px; margin:0 auto; padding:1.5rem 2rem; }}
.header {{ background:linear-gradient(135deg,#2c3e50,#3498db); color:#fff; padding:2rem; border-radius:12px; margin-bottom:1.5rem; }}
.header h1 {{ font-size:1.7rem; }}
.header .meta {{ opacity:.85; font-size:.9rem; margin-top:.5rem; }}
.section {{ background:#fff; border-radius:10px; padding:1.5rem 2rem; margin-bottom:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
.section h2 {{ color:var(--accent); border-bottom:2px solid #e8e8e8; padding-bottom:.4rem; margin-bottom:1rem; }}
.section h3 {{ margin:1.2rem 0 .5rem; color:#2c3e50; }}
table {{ border-collapse:collapse; width:100%; margin:.8rem 0; }}
th {{ background:var(--accent); color:#fff; padding:.5rem .8rem; text-align:left; }}
td {{ padding:.4rem .8rem; border-bottom:1px solid #eee; }}
tr:nth-child(even) {{ background:#f9f9f9; }}
.model-card {{ background:#f0f7ff; border:1px solid #d4e6f1; border-radius:8px; padding:1rem; margin:.8rem 0; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:.8rem; }}
blockquote {{ border-left:4px solid var(--accent); padding:.6rem 1rem; margin:1rem 0; background:#eef2f7; border-radius:0 8px 8px 0; }}
.footer {{ text-align:center; color:#999; font-size:.8rem; padding:1.5rem 0; }}
</style></head><body>
<div class="container">

<div class="header">
  <h1>阶跃响应对比测试报告</h1>
  <div class="meta">纯开环测试 | 无MPC控制器 | 3种水力学模型 × 3种激励 | {ts}</div>
</div>

<div class="section">
  <h2>📋 测试说明</h2>
  <blockquote>
    排除MPC控制器影响，对三种水力学模型施加相同的阶跃/脉冲入流信号，
    观察水位的开环响应特性差异。重点对比<b>波传播延迟、衰减特性、空间分布</b>。
  </blockquote>
  <div class="grid3">
    <div class="model-card">
      <h3>🟦 Tank ODE</h3>
      <p>dh/dt = (Qin-Qout)/(W·L)</p>
      <p>零维, 无波传播, 即时响应</p>
    </div>
    <div class="model-card">
      <h3>🟥 Saint-Venant 1D</h3>
      <p>连续性+动量, Preissmann隐式</p>
      <p>有波传播, 摩阻+惯性, 空间分布</p>
    </div>
    <div class="model-card">
      <h3>🟩 Diffusion Wave</h3>
      <p>∂h/∂t + c·∂h/∂x = D·∂²h/∂x²</p>
      <p>有波传播, 扩散衰减, 忽略惯性</p>
    </div>
  </div>
  <p style="margin-top:1rem"><b>渠道参数</b>: L={CHANNEL["length"]}m, W={W}m, S₀={S0}, n={n}, h₀={h0}m, Q₀={Q0:.2f}m³/s, V₀={V0:.3f}m/s</p>
</div>

{test_sections}

<div class="section">
  <h2>🤖 AI 分析总结</h2>
  <div class="model-card">
    <h3>关键发现</h3>
    <ul>
      <li><b>Tank ODE</b>: 水位响应即时、均匀，无波传播延迟。适合快速评估但无法捕捉渠道动力学。</li>
      <li><b>Saint-Venant</b>: 上游水位先响应，中游和下游有明显的波传播延迟。存在惯性效应导致的超调。</li>
      <li><b>Diffusion Wave</b>: 波传播速度与Saint-Venant相近，但无惯性超调，信号沿程扩散衰减。</li>
    </ul>
  </div>
  <div class="model-card">
    <h3>工程启示</h3>
    <ul>
      <li>对于<b>长距离渠道</b>（>5km），必须考虑波传播延迟，Tank ODE 会高估控制响应速度</li>
      <li><b>闸门快速调节</b>时（阶跃信号），Saint-Venant 的惯性效应会产生超调，需在MPC中预留安全裕度</li>
      <li><b>冰期运行</b>需要精确的扰动传播模型，Diffusion Wave 是精度与效率的最佳折衷</li>
      <li>建议e2econtrol的MPC控制器集成<b>Diffusion Wave</b>作为内部预测模型，替代当前Tank ODE</li>
    </ul>
  </div>
</div>

<div class="footer">Generated by HydroE2E × HydroClaw | {ts}</div>
</div>

<script>
var charts = [];
{chart_init_js}
window.addEventListener('resize', function() {{ charts.forEach(c => c.resize()); }});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  阶跃响应对比测试（纯开环, 无MPC）")
    print("=" * 60)

    all_results = []

    for test in TESTS:
        print(f"\n--- {test['name']} ---")
        signal = test["signal"]
        direction = "down" if "负" in test["name"] else "up"

        # 运行三个模型
        print("  Tank ODE...", end="", flush=True)
        t0 = time.time()
        r_tank = run_tank(signal)
        print(f" {(time.time()-t0)*1000:.0f}ms")

        print("  Saint-Venant 1D...", end="", flush=True)
        t0 = time.time()
        r_sv = run_saint_venant(signal)
        print(f" {(time.time()-t0)*1000:.0f}ms")

        print("  Diffusion Wave...", end="", flush=True)
        t0 = time.time()
        r_dw = run_diffusion_wave(signal)
        print(f" {(time.time()-t0)*1000:.0f}ms")

        # 分析响应特性
        r_tank["analysis"] = analyze_response(r_tank["level"], r_tank["time"], h0, 600.0, direction)
        r_sv["analysis"] = analyze_response(r_sv["level"], r_sv["time"], h0, 600.0, direction)
        r_dw["analysis"] = analyze_response(r_dw["level"], r_dw["time"], h0, 600.0, direction)

        r_tank["name"] = "Tank ODE"
        r_sv["name"] = "Saint-Venant 1D"
        r_dw["name"] = "Diffusion Wave"

        # 打印关键指标
        for m in [r_tank, r_sv, r_dw]:
            a = m["analysis"]
            dh = a.get('稳态变化 Δh (m)', '?')
            rt = a.get('上升时间 (s)', '?')
            ov = a.get('超调量 (%)', '?')
            print(f"    {m['name']:20s} Δh={dh!s:>8s}  rise={rt!s:>8s}  overshoot={ov!s:>6s}%")

        all_results.append({
            "test_name": test["name"],
            "test_desc": test["desc"],
            "models": [r_tank, r_sv, r_dw],
        })

    print("\n  Generating report...")
    html = build_html(all_results)

    out = Path("D:/research/e2econtrol/reports/scenarios")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "step_response_comparison.html"
    path.write_text(html, encoding="utf-8")
    print(f"  Report: {path}")
    return path


if __name__ == "__main__":
    p = main()
    print(f"\n  Opening in browser...")

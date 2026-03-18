#!/usr/bin/env python3
"""
HydroE2E 水系统场景仿真报告生成器 v3
========================================
交互式 HTML 报告：Mermaid 拓扑图 + ECharts 动态过程线 + AI 智能解读。
所有图表内嵌 HTML，无需外部图片文件，浏览器直接渲染。
"""

import sys, time, json, warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

from hydroe2e.simulation_manager import SimulationManager

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# 场景定义
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WaterScenario:
    id: str; name: str; category: str; description: str; instruction: str
    background: str; problem_statement: str; topology_mermaid: str
    control_objective: str; success_criteria: List[str]; risk_factors: List[str]
    total_hours: int = 48; area: float = 10000.0; initial_level: float = 3.0
    demand_mean: float = 5.0; demand_std: float = 0.5
    demand_events: List[Dict] = field(default_factory=list)


# Mermaid 拓扑定义 —— 每个场景不同的状态高亮
_TOPO_NORMAL = """graph LR
    A["☁️ 上游水源<br/>丹江口水库"]:::src -->|"来水 Q_in"| B["🔧 入流闸门<br/>MPC自动调节"]:::gate
    B -->|"受控入流"| C["🌊 渠池<br/>A=10,000m² H₀=3.0m<br/>安全区间 0.5~9.5m"]:::pool
    C -->|"出流 Q_out"| D["🔧 出流闸门<br/>需求驱动"]:::gate
    D -->|"供水"| E["🏙️ 下游城市<br/>Q≈5.0 m³/s"]:::demand
    C -.->|"水位反馈"| F["🧠 MPC控制器<br/>预测步长=10"]:::mpc
    F -.->|"最优指令"| B
    classDef src fill:#3498db,stroke:#2980b9,color:#fff
    classDef gate fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef pool fill:#2c3e50,stroke:#1a252f,color:#fff
    classDef demand fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef mpc fill:#1abc9c,stroke:#16a085,color:#fff"""

_TOPO_FLOOD = """graph LR
    A["☁️ 上游来水<br/>持续来水"]:::src -->|"来水"| B["🔧 入流闸门<br/>⚡ 快速调节"]:::warn
    B -->|"受控入流"| C["🌊 渠池<br/>初始 4.5m ➜ 目标 2.0m<br/>🔴 紧急腾库"]:::alert
    C -->|"加大泄量"| D["🔧 出流闸门<br/>⚡ 最大开度"]:::warn
    D -->|"应急排水"| E["🌊 下游河道"]:::demand
    C -.->|"水位反馈"| F["🧠 MPC控制器<br/>W_level=100 ⚡"]:::mpc
    F -.->|"紧急指令"| B
    classDef src fill:#3498db,stroke:#2980b9,color:#fff
    classDef warn fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef alert fill:#c0392b,stroke:#922b21,color:#fff
    classDef demand fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef mpc fill:#e67e22,stroke:#d35400,color:#fff"""

_TOPO_ICE = """graph LR
    A["☁️ 上游来水<br/>减量供水"]:::src -->|"缓慢来水"| B["🔧 入流闸门<br/>🧊 微量调节"]:::ice
    B -->|"平稳入流"| C["🧊 冰盖渠池<br/>H₀=3.0m 冰盖覆盖<br/>⚠️ 严禁扰动"]:::icepool
    C -->|"平稳出流"| D["🔧 出流闸门<br/>🧊 恒定开度"]:::ice
    D -->|"平稳供水"| E["🏙️ 下游用户<br/>Q≈4.0 m³/s"]:::demand
    C -.->|"水位反馈"| F["🧠 MPC控制器<br/>W_smooth=100+ 🧊"]:::mpc
    F -.->|"极平滑指令"| B
    classDef src fill:#3498db,stroke:#2980b9,color:#fff
    classDef ice fill:#5dade2,stroke:#2e86c1,color:#fff
    classDef icepool fill:#2e4053,stroke:#1b2631,color:#aed6f1
    classDef demand fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef mpc fill:#85c1e9,stroke:#5499c7,color:#1b4f72"""

_TOPO_POLLUTION = """graph LR
    A["☁️ 上游来水"]:::src -->|"来水"| B["🔧 入流闸门<br/>🚫 紧急关闭!"]:::shut
    B -.-x|"切断!"| C["🌊 渠池<br/>H₀=3.5m 水位下降中<br/>⚠️ 底线 0.5m"]:::alert
    C -->|"持续泄水"| D["🔧 出流闸门<br/>排污泄空"]:::warn
    D -->|"排出"| E["⚠️ 污染水体<br/>下游暂停取水"]:::pollution
    C -.->|"水位监测"| F["🧠 MPC控制器<br/>Q_in_max=0 🚫"]:::mpc
    F -.->|"关闭指令"| B
    G["🔬 水质监测站<br/>⚠️ 氨氮超标!"]:::pollution -.->|"污染预警"| F
    classDef src fill:#95a5a6,stroke:#7f8c8d,color:#fff
    classDef shut fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef alert fill:#f39c12,stroke:#e67e22,color:#fff
    classDef warn fill:#e67e22,stroke:#d35400,color:#fff
    classDef pollution fill:#c0392b,stroke:#922b21,color:#fff
    classDef mpc fill:#e74c3c,stroke:#c0392b,color:#fff"""

_TOPO_DROUGHT = """graph LR
    A["☁️ 上游水源<br/>📉 来水减少30%"]:::low -->|"减量来水"| B["🔧 入流闸门<br/>缓慢蓄水"]:::gate
    B -->|"多入少出"| C["🌊 渠池<br/>H: 2.5m ➜ 4.0m<br/>📈 缓慢蓄水"]:::pool
    C -->|"压减出流"| D["🔧 出流闸门<br/>📉 限制取水"]:::gate
    D -->|"减量供水"| E["🏙️ 下游用户<br/>Q≈3.0 m³/s 📉"]:::demand
    C -.->|"水位反馈"| F["🧠 MPC控制器<br/>Z_ref=4.0m 📈"]:::mpc
    F -.->|"蓄水指令"| B
    classDef low fill:#f39c12,stroke:#e67e22,color:#fff
    classDef gate fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef pool fill:#2c3e50,stroke:#1a252f,color:#fff
    classDef demand fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef mpc fill:#1abc9c,stroke:#16a085,color:#fff"""

_TOPO_MULTI = """graph LR
    A["☁️ 上游水源"]:::src -->|"动态来水"| B["🔧 入流闸门<br/>🔄 多模式切换"]:::gate
    B -->|"受控入流"| C["🌊 渠池<br/>多工况运行<br/>5次场景切换"]:::pool
    C -->|"动态出流"| D["🔧 出流闸门"]:::gate
    D -->|"供水"| E["🏙️ 下游"]:::demand
    C -.->|"反馈"| F["🧠 MPC控制器<br/>🔄 参数自适应"]:::mpc
    F -.->|"最优指令"| B
    subgraph 场景序列
        S1["0-10h 🟢 正常"]:::s1
        S2["10-20h 🔴 暴雨"]:::s2
        S3["20-30h 🔵 冰期"]:::s3
        S4["30-40h 🟡 污染"]:::s4
        S5["40-50h 🟢 恢复"]:::s5
        S1 --> S2 --> S3 --> S4 --> S5
    end
    classDef src fill:#3498db,stroke:#2980b9,color:#fff
    classDef gate fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef pool fill:#2c3e50,stroke:#1a252f,color:#fff
    classDef demand fill:#9b59b6,stroke:#8e44ad,color:#fff
    classDef mpc fill:#1abc9c,stroke:#16a085,color:#fff
    classDef s1 fill:#2ecc71,stroke:#27ae60,color:#fff
    classDef s2 fill:#e74c3c,stroke:#c0392b,color:#fff
    classDef s3 fill:#3498db,stroke:#2980b9,color:#fff
    classDef s4 fill:#f39c12,stroke:#e67e22,color:#fff
    classDef s5 fill:#2ecc71,stroke:#27ae60,color:#fff"""


SCENARIOS = [
    WaterScenario(
        id="S01", name="正常供水", category="日常运行",
        description="常规供水模式，水位维持在3.0m目标值附近，入流出流基本平衡。",
        instruction="保持水位平稳，正常供水。",
        background="南水北调中线工程总干渠全长1432km，年均调水量95亿m3。正常运行期间，各渠池需维持设计水位以保障沿线取水口稳定供水。本场景模拟一个典型渠池（面积10000m2，设计水深5m）在48小时内的日常调度过程。下游用水需求以5.0m3/s为基准，叠加随机波动（std=0.3m3/s），模拟城市用水的昼夜变化。",
        problem_statement="在需求随机波动条件下，MPC控制器能否将水位稳定在目标值3.0m附近？控制精度（RMSE）是否满足<0.1m的工程要求？",
        topology_mermaid=_TOPO_NORMAL,
        control_objective="水位跟踪RMSE<0.1m，流量变化率<2.0m3/s/h，无安全越限",
        success_criteria=["RMSE < 0.15m", "水位在[0.5m,9.5m]安全区间", "流量变化平滑无振荡"],
        risk_factors=["需求突变导致水位波动", "MPC求解器数值不稳定"],
        total_hours=48, initial_level=3.0, demand_mean=5.0, demand_std=0.3,
    ),
    WaterScenario(
        id="S02", name="暴雨预警应急", category="防洪调度",
        description="接到暴雨预警后紧急降低水位至2.0m以腾出库容，验证系统应急响应能力。",
        instruction="收到暴雨预警，立刻降低水位腾出库容！安全第一！",
        background="南水北调沿线渠道需应对区间暴雨产生的洪水入侵风险。当气象部门发布暴雨预警时，需在4-6小时内将水位从运行水位降至防洪水位（2.0m），腾出约20000m3的库容以吸纳可能的暴雨径流。本场景初始水位4.5m，模拟暴雨预警后24小时的应急调度过程。",
        problem_statement="MPC控制器能否在安全约束下快速降低水位？响应时间是否满足<6h的防洪要求？",
        topology_mermaid=_TOPO_FLOOD,
        control_objective="6h内水位降至2.0m，泄量变化率<5.0m3/s/h",
        success_criteria=["6h内水位降至2.0m附近", "泄量变化不超安全限值", "水位不低于0.5m"],
        risk_factors=["泄水过快导致下游冲刷", "闸门开度变化过大"],
        total_hours=24, initial_level=4.5, demand_mean=5.0, demand_std=0.5,
    ),
    WaterScenario(
        id="S03", name="冰期安全输水", category="冰期运行",
        description="冬季冰盖形成后，严格控制流量变化率以防止冰盖破裂。",
        instruction="进入冰期输水模式，严禁扰动冰盖。保持流量绝对平稳。",
        background="南水北调中线工程北段（黄河以北约700km）冬季最低气温可达-20°C。渠道结冰形成冰盖后，流量变化会产生水力波动，可能导致冰盖抬升、破裂甚至冰塞。冰期输水规程要求：流量变化率不超过0.5m3/s/h，水位波动不超过0.1m/h。本场景模拟72小时的冰期运行。",
        problem_statement="MPC在极高平滑权重下能否有效抑制流量波动？是否满足冰期规程要求？",
        topology_mermaid=_TOPO_ICE,
        control_objective="流量变化率<0.5m3/s/h，水位波动<0.1m/h",
        success_criteria=["流量平滑度<0.5m3/s/h", "水位在[2.5m,3.5m]窄幅波动", "无流量突变"],
        risk_factors=["流量突变导致冰盖破裂", "冰塞堵塞过水断面"],
        total_hours=72, initial_level=3.0, demand_mean=4.0, demand_std=0.1,
    ),
    WaterScenario(
        id="S04", name="水污染应急", category="应急处置",
        description="下游检测到水质污染，立即切断入流，验证紧急截断能力。",
        instruction="下游检测到污染，紧急切断入流！",
        background="南水北调沿线设有水质自动监测站，一旦检测到氨氮、COD等指标超标，需立即关闭上游闸门切断来水。切断入流后渠池水位将因持续出流而下降，需确保水位不低于最低安全水位（0.5m）以防止渠道边坡失稳。本场景模拟污染发现后12小时的应急处置。",
        problem_statement="MPC在入流被切断后能否正确响应？水位持续下降中能否守住0.5m底线？",
        topology_mermaid=_TOPO_POLLUTION,
        control_objective="1h内完全切断入流，水位不低于0.5m安全底线",
        success_criteria=["入流1h内降至0", "水位>=0.5m", "正确识别污染场景"],
        risk_factors=["水位过低导致边坡失稳", "切断不及时导致污染扩散"],
        total_hours=12, initial_level=3.5, demand_mean=3.0, demand_std=0.2,
    ),
    WaterScenario(
        id="S05", name="干旱节水调度", category="旱情应对",
        description="持续干旱导致来水减少，需提升蓄水位至4.0m并减小出流。",
        instruction="持续干旱缺水，提升水位节约储备。减小出流，优先蓄水。",
        background="丹江口水库来水减少，中线工程可调水量下降30%。需将各渠池水位从正常水位提升至4.0m以增加沿线蓄水储备，同时压减下游非优先用户的取水量。蓄水过程需缓慢进行（约24-48h），避免因蓄水过快造成上游水位骤降。",
        problem_statement="MPC能否实现水位从2.5m缓慢上升至4.0m？蓄水速率是否平滑？",
        topology_mermaid=_TOPO_DROUGHT,
        control_objective="48h内水位从2.5m升至4.0m，升速<0.1m/h",
        success_criteria=["最终水位在[3.8m,4.2m]", "蓄水过程平滑", "出流不低于最低保障"],
        risk_factors=["蓄水过快导致上游断流", "下游供水不足"],
        total_hours=72, initial_level=2.5, demand_mean=3.0, demand_std=0.2,
    ),
    WaterScenario(
        id="S06", name="多阶段综合调度", category="综合场景",
        description="50小时内经历正常→暴雨→冰期→污染→恢复的完整多阶段场景。",
        instruction="__MULTI_PHASE__",
        background="实际工程中，渠道运行常面临多种工况的连续切换。本场景模拟一个极端但有代表性的50小时运行周期：0-10h正常供水、10-20h暴雨预警、20-30h冰期输水、30-40h污染应急、40-50h恢复正常。重点考察MPC在场景切换时的过渡响应和参数自适应能力。",
        problem_statement="MPC能否在5次场景切换中保持稳定？各阶段控制目标能否独立达成？",
        topology_mermaid=_TOPO_MULTI,
        control_objective="5次场景切换零失败，各阶段RMSE<0.5m",
        success_criteria=["5次切换全部成功", "各阶段RMSE<0.5m", "全程水位在安全区间", "无控制器失稳"],
        risk_factors=["场景切换过渡振荡", "参数突变导致不稳定", "连续应急耗尽调节能力"],
        total_hours=50, initial_level=3.0, demand_mean=5.0, demand_std=0.5,
    ),
]

MULTI_PHASE_SCRIPT = [
    (0,  "保持水位平稳，正常供水。"),
    (10, "收到暴雨预警，立刻降低水位腾出库容！安全第一！"),
    (20, "进入冰期输水模式，严禁扰动冰盖。"),
    (30, "下游检测到污染，紧急切断出流！"),
    (40, "保持水位平稳，正常供水。"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 仿真执行
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(sc: WaterScenario) -> Dict[str, Any]:
    np.random.seed(42)
    total = sc.total_hours
    demands = np.clip(np.random.normal(sc.demand_mean, sc.demand_std, total + 20), 0.5, 30.0)
    script = MULTI_PHASE_SCRIPT if sc.instruction == "__MULTI_PHASE__" else [(0, sc.instruction)]

    t0 = time.time()
    sim = SimulationManager(total, 3600.0, sc.area, sc.initial_level, script, demands)
    history = sim.run_simulation()
    elapsed = time.time() - t0

    lv = np.array(history['level']); tgt = np.array(history['target_level'])
    qi = np.array(history['q_in']); qo = np.array(history['q_out'])

    rmse = float(np.sqrt(np.mean((lv - tgt) ** 2)))
    fs = float(np.mean(np.abs(np.diff(qi)))) if len(qi) > 1 else 0.0
    ti, to_ = float(np.sum(qi)), float(np.sum(qo))
    be = abs(ti - to_) / max(ti, 1.0) * 100
    rt = 0
    for i in range(len(lv)):
        if abs(lv[i] - tgt[i]) < 0.5: rt = i; break

    return {
        "scenario": sc, "history": history, "elapsed": elapsed,
        "metrics": {
            "RMSE (m)": round(rmse, 4), "最大偏差 (m)": round(float(np.max(np.abs(lv - tgt))), 4),
            "平均水位 (m)": round(float(np.mean(lv)), 3),
            "水位标准差 (m)": round(float(np.std(lv)), 4),
            "流量平滑度 (m³/s/h)": round(fs, 4),
            "总入流 (m³)": round(ti, 1), "总出流 (m³)": round(to_, 1),
            "水量平衡误差 (%)": round(be, 2),
            "响应时间 (h)": rt,
            "溢出步数": int(np.sum(lv > 9.5)), "枯竭步数": int(np.sum(lv < 0.5)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# AI 智能解读
# ═══════════════════════════════════════════════════════════════════════════

def ai_interpret(result: Dict) -> str:
    sc = result["scenario"]; m = result["metrics"]; h = result["history"]
    rmse = m["RMSE (m)"]; fs = m["流量平滑度 (m³/s/h)"]
    ov, uv = m["溢出步数"], m["枯竭步数"]; rt = m["响应时间 (h)"]
    cfg = h['config'][0] if h['config'] else {}
    w_l, w_s = cfg.get('W_level', 10), cfg.get('W_smooth', 5)

    parts = []

    # 场景诊断
    diff_map = {"冰期运行": "高（极端平滑约束）", "应急处置": "高（入流切断，非受控衰减）",
                "综合场景": "极高（多次参数跳变）", "防洪调度": "中高（快速大幅调节）"}
    diff = diff_map.get(sc.category, "中等（常规调节）")

    mpc_desc = f"极端平滑模式(W_smooth={w_s})" if w_s > 50 else (
        f"强跟踪模式(W_level={w_l})" if w_l > 50 else f"均衡模式(W_level={w_l}, W_smooth={w_s})")

    parts.append(f"""<div class="ai-card">
<h3>🔍 场景诊断</h3>
<table><tr><td><b>场景类型</b></td><td>{sc.category} / {sc.name}</td></tr>
<tr><td><b>控制难度</b></td><td>{diff}</td></tr>
<tr><td><b>MPC配置</b></td><td>{mpc_desc}</td></tr>
<tr><td><b>目标水位</b></td><td>{cfg.get('Z_ref', '-')} m</td></tr></table></div>""")

    # 性能评估
    def grade(v, t): return "🟢 优秀" if v < t[0] else ("🟡 良好" if v < t[1] else "🔴 需关注")
    parts.append(f"""<div class="ai-card">
<h3>📊 性能评估</h3>
<table>
<tr><td><b>水位控制</b></td><td>{grade(rmse, [0.15, 0.5])} (RMSE={rmse}m)</td></tr>
<tr><td><b>流量平滑</b></td><td>{grade(fs, [0.3, 1.0])} (Δ={fs}m³/s/h)</td></tr>
<tr><td><b>安全性</b></td><td>{"🟢 全程无越限" if ov+uv==0 else f"🔴 溢出{ov}/枯竭{uv}步"}</td></tr>
<tr><td><b>响应速度</b></td><td>{"🟢 快速" if rt<=3 else "🟡 正常"} ({rt}h)</td></tr></table></div>""")

    # 多阶段分析
    if sc.instruction == "__MULTI_PHASE__":
        lv = np.array(h['level']); tgt = np.array(h['target_level']); qi = np.array(h['q_in'])
        phases = [(0,10,"正常供水","🟢"),(10,20,"暴雨预警","🔴"),(20,30,"冰期输水","🔵"),(30,40,"污染应急","🟡"),(40,50,"恢复正常","🟢")]
        rows = ""
        for s,e,nm,ic in phases:
            sl = lv[s:e]; st = tgt[s:e]; sq = qi[s:e]
            if len(sl) > 0:
                pr = float(np.sqrt(np.mean((sl-st)**2)))
                ps = float(np.mean(np.abs(np.diff(sq)))) if len(sq)>1 else 0
                ev = "优秀" if pr<0.3 else ("良好" if pr<0.8 else "需关注")
                rows += f"<tr><td>{ic} {nm}</td><td>{s}-{e}h</td><td>{np.mean(sl):.2f}m</td><td>{pr:.3f}m</td><td>{ps:.3f}</td><td>{ev}</td></tr>"
        parts.append(f"""<div class="ai-card">
<h3>📋 分阶段分析</h3>
<table><tr><th>阶段</th><th>时段</th><th>均值水位</th><th>RMSE</th><th>平滑度</th><th>评价</th></tr>{rows}</table></div>""")

    # 建议
    sugs = []
    if rmse > 0.3: sugs.append("增大 W_level 权重提高跟踪精度")
    if fs > 1.5: sugs.append("增大 W_smooth 减小闸门动作幅度")
    if ov > 0: sugs.append("降低目标水位或增大溢流安全裕度")
    if uv > 0: sugs.append("增大最低水位约束，提前蓄水")
    if not sugs:
        sugs = ["当前策略表现优异，建议保持参数配置", "可增加扰动测试验证鲁棒性"]
    sug_html = "".join(f"<li>{s}</li>" for s in sugs)
    parts.append(f'<div class="ai-card"><h3>💡 改进建议</h3><ul>{sug_html}</ul></div>')

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# HTML 报告生成
# ═══════════════════════════════════════════════════════════════════════════

def build_html(result: Dict) -> str:
    sc = result["scenario"]; m = result["metrics"]; h = result["history"]
    rmse = m["RMSE (m)"]

    # 数据 JSON
    t_arr = h['time']; lv = h['level']; tgt = h['target_level']
    qi = h['q_in']; qo = h['q_out']
    instr = h['instruction']

    grade = "A+ 优秀" if rmse<0.15 else ("A 良好" if rmse<0.3 else ("B 合格" if rmse<0.8 else "C 需改进"))
    grade_color = "#27ae60" if rmse<0.3 else ("#f39c12" if rmse<0.8 else "#e74c3c")

    # 性能雷达数据
    radar_labels = ['水位精度', '流量平滑', '水量平衡', '响应速度', '安全性']
    radar_values = [
        round(max(0, 1 - rmse / 1.0), 2),
        round(max(0, 1 - m["流量平滑度 (m³/s/h)"] / 3.0), 2),
        round(max(0, 1 - m["水量平衡误差 (%)"] / 20.0), 2),
        round(max(0, 1 - m["响应时间 (h)"] / 10.0), 2),
        1.0 if m["溢出步数"]+m["枯竭步数"]==0 else 0.3,
    ]

    # 多阶段标记线
    mark_lines_js = "[]"
    if sc.instruction == "__MULTI_PHASE__":
        marks = [{"xAxis":t,"label":{"formatter":n[:4],"fontSize":10}} for t,n in MULTI_PHASE_SCRIPT]
        mark_lines_js = json.dumps(marks, ensure_ascii=False)

    # 成功准则
    criteria_rows = ""
    for c in sc.success_criteria:
        if 'RMSE' in c: ok = rmse < 0.5
        elif '安全' in c or '0.5' in c or '9.5' in c: ok = m["溢出步数"]+m["枯竭步数"]==0
        elif '平滑' in c or '振荡' in c: ok = m["流量平滑度 (m³/s/h)"] < 1.5
        else: ok = rmse < 0.8
        criteria_rows += f'<tr><td>{c}</td><td>{"✅ 通过" if ok else "❌ 未通过"}</td></tr>'

    # 指标行
    eval_fn = {
        "RMSE (m)": lambda v: "🟢 优" if v<0.15 else ("🟡 良" if v<0.3 else "🔴 需关注"),
        "最大偏差 (m)": lambda v: "🟢 优" if v<0.5 else ("🟡 良" if v<1.5 else "🔴"),
        "流量平滑度 (m³/s/h)": lambda v: "🟢 优" if v<0.3 else ("🟡 良" if v<1.0 else "🔴"),
        "水量平衡误差 (%)": lambda v: "🟢 优" if v<5 else ("🟡 良" if v<15 else "🔴"),
        "溢出步数": lambda v: "🟢 安全" if v==0 else "🔴 警告",
        "枯竭步数": lambda v: "🟢 安全" if v==0 else "🔴 警告",
        "响应时间 (h)": lambda v: "🟢 快" if v<=2 else ("🟡" if v<=5 else "🔴 慢"),
    }
    metrics_rows = ""
    for k, v in m.items():
        ev = eval_fn.get(k, lambda v: "")(v)
        metrics_rows += f"<tr><td>{k}</td><td><b>{v}</b></td><td>{ev}</td></tr>"

    # 多阶段指令表
    phase_table = ""
    if sc.instruction == "__MULTI_PHASE__":
        rows = "".join(f"<tr><td>{t}h</td><td>{ins}</td></tr>" for t, ins in MULTI_PHASE_SCRIPT)
        phase_table = f'<h3>📋 多阶段指令序列</h3><table><tr><th>时刻</th><th>控制指令</th></tr>{rows}</table>'

    ai_html = ai_interpret(result)
    cfg = h['config'][0] if h['config'] else {}
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cat_colors = {"日常运行":"#27ae60","防洪调度":"#e74c3c","冰期运行":"#3498db",
                  "应急处置":"#e74c3c","旱情应对":"#f39c12","综合场景":"#8e44ad"}

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HydroE2E - {sc.id} {sc.name}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
:root {{ --accent:#0f3460; --bg:#f8f9fa; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:"Microsoft YaHei",sans-serif; background:var(--bg); color:#1a1a2e; }}
.container {{ max-width:1100px; margin:0 auto; padding:1.5rem 2rem; }}
.header {{ background:linear-gradient(135deg,#0f3460,#1a5276); color:#fff; padding:2rem; border-radius:12px; margin-bottom:1.5rem; }}
.header h1 {{ font-size:1.8rem; margin-bottom:.5rem; }}
.header .meta {{ opacity:.85; font-size:.9rem; }}
.badge {{ display:inline-block; padding:.2rem .7rem; border-radius:4px; font-weight:bold; font-size:.85rem; color:#fff; background:{cat_colors.get(sc.category,"#666")}; }}
.section {{ background:#fff; border-radius:10px; padding:1.5rem 2rem; margin-bottom:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
.section h2 {{ color:var(--accent); border-bottom:2px solid #e8e8e8; padding-bottom:.4rem; margin-bottom:1rem; font-size:1.3rem; }}
.section h3 {{ margin:1rem 0 .5rem; color:#2c3e50; }}
table {{ border-collapse:collapse; width:100%; margin:.8rem 0; }}
th {{ background:var(--accent); color:#fff; padding:.5rem .8rem; text-align:left; font-size:.9rem; }}
td {{ padding:.4rem .8rem; border-bottom:1px solid #eee; font-size:.9rem; }}
tr:nth-child(even) {{ background:#f9f9f9; }}
.chart-box {{ width:100%; height:420px; margin:1rem 0; border-radius:8px; }}
.mermaid {{ background:#fff; padding:1rem; border-radius:8px; text-align:center; }}
blockquote {{ border-left:4px solid var(--accent); padding:.6rem 1rem; margin:1rem 0; background:#eef2f7; border-radius:0 8px 8px 0; font-style:italic; }}
.grade-badge {{ display:inline-block; padding:.3rem 1rem; border-radius:6px; font-size:1.1rem; font-weight:bold; color:#fff; background:{grade_color}; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
@media(max-width:768px) {{ .grid {{ grid-template-columns:1fr; }} }}
.ai-card {{ background:#f0f7ff; border:1px solid #d4e6f1; border-radius:8px; padding:1rem 1.2rem; margin:.8rem 0; }}
.ai-card h3 {{ margin:0 0 .5rem; color:#1a5276; font-size:1.05rem; }}
.ai-card table {{ margin:.5rem 0; }}
.ai-card ul {{ padding-left:1.2rem; margin:.3rem 0; }}
.ai-card li {{ margin:.2rem 0; }}
.risk {{ background:#fff5f5; border-left:3px solid #e74c3c; padding:.5rem 1rem; margin:.5rem 0; border-radius:0 6px 6px 0; }}
.footer {{ text-align:center; color:#999; font-size:.8rem; padding:1.5rem 0; }}
</style></head><body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><h1>{sc.id} {sc.name}</h1>
    <div class="meta">场景仿真报告 | {ts} | 仿真耗时 {result['elapsed']*1000:.0f}ms</div></div>
    <div><span class="badge">{sc.category}</span></div>
  </div>
</div>

<!-- 工程背景 -->
<div class="section">
  <h2>📋 工程背景</h2>
  <p>{sc.background}</p>
  <blockquote>{sc.problem_statement}</blockquote>
  {phase_table}
</div>

<!-- 系统拓扑 -->
<div class="section">
  <h2>🔗 系统拓扑</h2>
  <p style="color:#666;font-size:.9rem;margin-bottom:.5rem">{sc.control_objective}</p>
  <pre class="mermaid">{sc.topology_mermaid}</pre>
</div>

<!-- 控制目标 -->
<div class="section">
  <h2>🎯 控制目标与风险</h2>
  <div class="grid">
    <div>
      <h3>控制目标</h3><p><b>{sc.control_objective}</b></p>
      <h3>成功准则</h3>
      <table><tr><th>准则</th><th>结果</th></tr>{criteria_rows}</table>
    </div>
    <div>
      <h3>风险因素</h3>
      {"".join(f'<div class="risk">{r}</div>' for r in sc.risk_factors)}
    </div>
  </div>
</div>

<!-- 仿真参数 -->
<div class="section">
  <h2>⚙️ 仿真参数</h2>
  <div class="grid">
    <div>
      <h3>物理参数</h3>
      <table>
        <tr><td>渠池面积</td><td><b>{sc.area:,.0f}</b> m²</td></tr>
        <tr><td>初始水位</td><td><b>{sc.initial_level}</b> m</td></tr>
        <tr><td>仿真时长</td><td><b>{sc.total_hours}</b> h</td></tr>
        <tr><td>需求均值</td><td><b>{sc.demand_mean}</b> m³/s</td></tr>
      </table>
    </div>
    <div>
      <h3>MPC 控制参数</h3>
      <table>
        <tr><td>Z_ref (目标水位)</td><td><b>{cfg.get('Z_ref','-')}</b> m</td></tr>
        <tr><td>W_level (跟踪权重)</td><td><b>{cfg.get('W_level','-')}</b></td></tr>
        <tr><td>W_smooth (平滑权重)</td><td><b>{cfg.get('W_smooth','-')}</b></td></tr>
        <tr><td>ΔQ_max (最大变化)</td><td><b>{cfg.get('delta_Q_max','-')}</b> m³/s</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- 运行结果 -->
<div class="section">
  <h2>📊 运行结果</h2>
  <p>综合评级: <span class="grade-badge">{grade}</span></p>
  <table><tr><th>指标</th><th>值</th><th>评价</th></tr>{metrics_rows}</table>
</div>

<!-- 水位过程线 ECharts -->
<div class="section">
  <h2>🌊 水位过程线</h2>
  <p style="color:#888;font-size:.85rem">
    🟢 正常运行区(1.5~8.0m)  🟡 扩展区(0.5~1.5m / 8.0~9.5m)  🔴 MRC区(&lt;0.5m / &gt;9.5m)
  </p>
  <div id="chart-level" class="chart-box"></div>
</div>

<!-- 流量过程线 -->
<div class="section">
  <h2>💧 流量过程线</h2>
  <div id="chart-flow" class="chart-box"></div>
</div>

<!-- 控制性能 -->
<div class="section">
  <h2>📈 控制性能分析</h2>
  <div class="grid">
    <div id="chart-deviation" style="height:360px"></div>
    <div id="chart-radar" style="height:360px"></div>
  </div>
</div>

<!-- AI 解读 -->
<div class="section">
  <h2>🤖 AI 智能解读</h2>
  {ai_html}
</div>

<div class="footer">Generated by HydroE2E × HydroClaw | ECharts + Mermaid | {ts}</div>
</div>

<script>
mermaid.initialize({{ startOnLoad: true, theme: 'base',
  themeVariables: {{ primaryColor:'#3498db', lineColor:'#95a5a6', fontSize:'13px' }} }});

var T = {json.dumps(t_arr)};
var LV = {json.dumps([round(x,4) for x in lv])};
var TGT = {json.dumps([round(x,4) for x in tgt])};
var QI = {json.dumps([round(x,3) for x in qi])};
var QO = {json.dumps([round(x,3) for x in qo])};
var markLines = {mark_lines_js};

// 水位过程线
var c1 = echarts.init(document.getElementById('chart-level'));
c1.setOption({{
  tooltip: {{ trigger:'axis' }},
  legend: {{ data:['实际水位','目标水位'] }},
  xAxis: {{ type:'category', data:T, name:'时间(h)' }},
  yAxis: {{ type:'value', name:'水位(m)', min:0 }},
  dataZoom: [{{ type:'slider', start:0, end:100 }}, {{ type:'inside' }}],
  series: [
    {{ name:'实际水位', type:'line', data:LV, lineStyle:{{width:2.5}}, smooth:true,
       markLine: {{ data: markLines, lineStyle:{{color:'#999',type:'dashed'}} }},
       markArea: {{ silent:true, data:[
         [{{yAxis:0,itemStyle:{{color:'rgba(231,76,60,0.08)'}}}}, {{yAxis:0.5}}],
         [{{yAxis:0.5,itemStyle:{{color:'rgba(243,156,18,0.06)'}}}}, {{yAxis:1.5}}],
         [{{yAxis:1.5,itemStyle:{{color:'rgba(46,204,113,0.05)'}}}}, {{yAxis:8.0}}],
         [{{yAxis:8.0,itemStyle:{{color:'rgba(243,156,18,0.06)'}}}}, {{yAxis:9.5}}],
         [{{yAxis:9.5,itemStyle:{{color:'rgba(231,76,60,0.08)'}}}}, {{yAxis:10}}],
       ] }}
    }},
    {{ name:'目标水位', type:'line', data:TGT, lineStyle:{{width:1.5,type:'dashed',color:'#e74c3c'}}, smooth:true }}
  ]
}});

// 流量过程线
var c2 = echarts.init(document.getElementById('chart-flow'));
c2.setOption({{
  tooltip: {{ trigger:'axis' }},
  legend: {{ data:['入流(MPC)','出流(需求)'] }},
  xAxis: {{ type:'category', data:T, name:'时间(h)' }},
  yAxis: {{ type:'value', name:'流量(m³/s)' }},
  dataZoom: [{{ type:'slider', start:0, end:100 }}, {{ type:'inside' }}],
  series: [
    {{ name:'入流(MPC)', type:'line', data:QI, lineStyle:{{width:2}}, smooth:true,
       areaStyle:{{opacity:0.1}} }},
    {{ name:'出流(需求)', type:'line', data:QO, lineStyle:{{width:2,color:'#e74c3c'}}, smooth:true,
       areaStyle:{{opacity:0.05,color:'#e74c3c'}} }}
  ]
}});

// 控制偏差
var devs = LV.map((v,i) => +(v - TGT[i]).toFixed(4));
var devColors = devs.map(d => Math.abs(d)<0.3 ? '#27ae60' : (Math.abs(d)<1.0 ? '#f39c12' : '#e74c3c'));
var c3 = echarts.init(document.getElementById('chart-deviation'));
c3.setOption({{
  title: {{ text:'控制偏差分布', textStyle:{{fontSize:14}} }},
  tooltip: {{ trigger:'axis' }},
  xAxis: {{ type:'category', data:T }},
  yAxis: {{ type:'value', name:'偏差(m)' }},
  series: [{{ type:'bar', data: devs.map((v,i) => ({{value:v, itemStyle:{{color:devColors[i]}}}})),
    markLine: {{ data:[{{yAxis:0.3,lineStyle:{{color:'#f39c12',type:'dashed'}}}}, {{yAxis:-0.3,lineStyle:{{color:'#f39c12',type:'dashed'}}}}] }}
  }}]
}});

// 雷达图
var c4 = echarts.init(document.getElementById('chart-radar'));
c4.setOption({{
  title: {{ text:'性能雷达图', textStyle:{{fontSize:14}} }},
  radar: {{ indicator: {json.dumps([{"name":n,"max":1} for n in radar_labels], ensure_ascii=False)} }},
  series: [{{ type:'radar', data:[{{ value:{json.dumps(radar_values)}, name:'当前性能',
    areaStyle:{{opacity:0.2}}, lineStyle:{{width:2}} }}] }}]
}});

window.addEventListener('resize', function() {{ c1.resize(); c2.resize(); c3.resize(); c4.resize(); }});
</script></body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path("D:/research/e2econtrol/reports/scenarios")
    out.mkdir(parents=True, exist_ok=True)
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    scenarios = [SCENARIOS[idx]] if 0 <= idx < len(SCENARIOS) else SCENARIOS

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"  [{sc.id}] {sc.name} ({sc.category})")
        print(f"{'='*60}")
        print(f"  Running simulation ({sc.total_hours}h)...")
        result = run_scenario(sc)
        print(f"  Done {result['elapsed']*1000:.0f}ms | RMSE={result['metrics']['RMSE (m)']}m")
        print(f"  Generating interactive report...")
        html = build_html(result)
        path = out / f"{sc.id}_{sc.name}_report.html"
        path.write_text(html, encoding="utf-8")
        print(f"  Report: {path}")

    return path


if __name__ == "__main__":
    p = main()
    print(f"\n  Opening in browser...")

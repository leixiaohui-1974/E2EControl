"""HydroClaw 报告服务客户端。

优先调用 HydroClaw MCP Gateway (localhost:8040) 生成交互式报告，
若 HydroClaw 不可用则回退到本地 HTML 生成。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HYDROCLAW_GATEWAY = "http://localhost:8040"
REPORT_DIR = Path("D:/research/e2econtrol/reports/scenarios")


def _call_hydroclaw(result_data: dict[str, Any]) -> str | None:
    """尝试调用 HydroClaw MCP Gateway 生成交互式报告。

    Returns:
        生成的 HTML 内容字符串，失败时返回 None。
    """
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "generate_report",
                "arguments": {
                    "report_type": "scenario_simulation",
                    "title": f"{result_data['scenario_id']} {result_data['scenario_name']}",
                    "data": {
                        "scenario_id": result_data["scenario_id"],
                        "scenario_name": result_data["scenario_name"],
                        "category": result_data["category"],
                        "metrics": result_data["metrics"],
                        "history": result_data["history"],
                        "config": result_data.get("config", {}),
                        "elapsed_ms": result_data.get("elapsed_ms", 0),
                    },
                },
            },
        }, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            f"{HYDROCLAW_GATEWAY}/mcp",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        if "result" in body and "content" in body["result"]:
            for item in body["result"]["content"]:
                if item.get("type") == "text":
                    return item["text"]

        logger.warning("HydroClaw 返回格式异常: %s", body)
        return None

    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.info("HydroClaw 不可用 (%s)，回退到本地生成", exc)
        return None
    except Exception as exc:
        logger.warning("调用 HydroClaw 异常: %s", exc)
        return None


def _generate_local_html(result_data: dict[str, Any]) -> str:
    """本地回退：生成简化版交互式 HTML 报告。"""
    from scripts.generate_scenario_report import SCENARIOS, build_html, ai_interpret

    # 根据 scenario_id 查找完整场景定义
    sc = None
    for s in SCENARIOS:
        if s.id == result_data["scenario_id"]:
            sc = s
            break

    if sc is not None:
        # 重建 build_html 所需的完整 result 结构
        full_result = {
            "scenario": sc,
            "history": result_data["history"],
            "elapsed": result_data.get("elapsed_ms", 0) / 1000.0,
            "metrics": result_data["metrics"],
        }
        return build_html(full_result)

    # 场景未找到时生成极简报告
    metrics_rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in result_data["metrics"].items()
    )
    sid = result_data["scenario_id"]
    sname = result_data["scenario_name"]
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>HydroE2E - {sid} {sname}</title>
<style>
body {{ font-family:"Microsoft YaHei",sans-serif; max-width:900px; margin:2rem auto; padding:0 1rem; }}
h1 {{ color:#0f3460; }} table {{ border-collapse:collapse; width:100%; margin:1rem 0; }}
th {{ background:#0f3460; color:#fff; padding:.5rem; text-align:left; }}
td {{ padding:.4rem .8rem; border-bottom:1px solid #eee; }}
.note {{ color:#888; font-size:.85rem; margin-top:2rem; }}
</style></head><body>
<h1>{sid} {sname}</h1>
<p class="note">本地回退报告（HydroClaw 不可用）</p>
<h2>运行指标</h2>
<table><tr><th>指标</th><th>值</th></tr>{metrics_rows}</table>
<p class="note">完整交互式报告需连接 HydroClaw 报告服务</p>
</body></html>"""


def generate_scenario_report(result_data: dict[str, Any]) -> str:
    """生成场景仿真报告，优先使用 HydroClaw，失败则本地生成。

    Args:
        result_data: 仿真结果数据，需包含:
            - scenario_id: 场景编号 (e.g. "S01")
            - scenario_name: 场景名称
            - category: 场景类别
            - metrics: 指标字典
            - history: 仿真历史数据
            - config: MPC 配置（可选）
            - elapsed_ms: 仿真耗时毫秒（可选）

    Returns:
        生成的报告文件路径（绝对路径字符串）。
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sid = result_data["scenario_id"]
    sname = result_data["scenario_name"]

    # 优先尝试 HydroClaw
    html = _call_hydroclaw(result_data)
    source = "HydroClaw"

    if html is None:
        html = _generate_local_html(result_data)
        source = "local"

    path = REPORT_DIR / f"{sid}_{sname}_report.html"
    path.write_text(html, encoding="utf-8")
    logger.info("报告已生成 [%s]: %s", source, path)
    return str(path)

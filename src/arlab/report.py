"""评测报告：并排展示「输入材料 / 模型输出 / 标准答案」，外加阶段判定和失败诊断。

一个分数不构成报告。工程师要能在一屏内回答三个问题：
    喂进去的是什么 → 模型吐出来的是什么 → 差在哪一句

所以每张任务卡片是三栏对照，下方才是扣分项和归因。失败项标红，通过项标绿，
扫一眼就知道该点开哪张卡。
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime

from .judge import ERROR_TYPES
from .pricing import human as money

CSS = """
:root{
  --bg:#0f1115; --panel:#161a22; --fg:#e6e8ee; --muted:#8b93a7;
  --border:#252b36; --ok:#3fb950; --warn:#d29922; --fail:#f85149; --accent:#58a6ff;
}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
  font-family:-apple-system,"Segoe UI",system-ui,sans-serif;
  font-size:14px;line-height:1.6}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 10px;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.06em}
.muted{color:var(--muted)}
.mono{font-family:"JetBrains Mono","Cascadia Code",Consolas,monospace;font-size:12.5px}

/* 顶部大盘 */
.board{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0 8px}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:14px 18px;min-width:132px}
.kpi .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.kpi .v{font-size:24px;font-weight:650;margin-top:4px}
.kpi .v.ok{color:var(--ok)} .kpi .v.warn{color:var(--warn)} .kpi .v.fail{color:var(--fail)}
.gate{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
.gate.pass{background:rgba(63,185,80,.15);color:var(--ok);border:1px solid rgba(63,185,80,.4)}
.gate.fail{background:rgba(248,81,73,.15);color:var(--fail);border:1px solid rgba(248,81,73,.4)}

/* 阶段漏斗 */
table{border-collapse:collapse;width:100%;margin-top:6px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);font-size:13px}
th{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:6px;border-radius:3px;background:var(--border);overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--ok)}

/* 任务卡 */
.card{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--border);
  border-radius:10px;padding:16px 18px;margin-bottom:16px}
.card.ok{border-left-color:var(--ok)} .card.bad{border-left-color:var(--fail)}
.card.control{border-left-color:var(--accent);opacity:.95}
.card h3{margin:0 0 2px;font-size:15px}
.tag{font-size:11px;padding:1px 7px;border-radius:4px;margin-left:8px;vertical-align:2px}
.tag.ok{background:rgba(63,185,80,.15);color:var(--ok)}
.tag.bad{background:rgba(248,81,73,.15);color:var(--fail)}
.tag.ctl{background:rgba(88,166,255,.15);color:var(--accent)}

/* 三栏对照 */
.compare{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:960px){.compare{grid-template-columns:1fr}}
.col{background:#11151c;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.col>h4{margin:0;padding:7px 11px;font-size:11.5px;color:var(--muted);
  background:#141922;border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:.06em}
.col.model{border-color:rgba(248,81,73,.35)}
.col pre{margin:0;padding:11px;white-space:pre-wrap;word-break:break-word;
  font-family:"JetBrains Mono","Cascadia Code",Consolas,monospace;font-size:12px;
  max-height:280px;overflow:auto}
.col .empty{color:var(--muted);font-style:italic}

/* 扣分与诊断 */
.reasons{margin-top:12px}
.reasons li{color:var(--fail)}
.diag{margin-top:10px;padding:10px 12px;border:1px solid rgba(210,153,34,.35);
  background:rgba(210,153,34,.07);border-radius:8px;font-size:13px}
.diag .lbl{font-size:11px;color:var(--warn);text-transform:uppercase;letter-spacing:.06em}
.diag .row{margin-top:6px}
.diag b{color:var(--fg);font-weight:600}

/* 阶段 chips */
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{font-size:11.5px;padding:2px 9px;border-radius:999px;border:1px solid var(--border);color:var(--muted)}
.chip.pass{color:var(--ok);border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.08)}
.chip.warn{color:var(--warn);border-color:rgba(210,153,34,.4);background:rgba(210,153,34,.08)}
.chip.fail{color:var(--fail);border-color:rgba(248,81,73,.4);background:rgba(248,81,73,.08)}
.chip.na{opacity:.45}
footer{margin-top:32px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);padding-top:14px}
"""


def _esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def _pretty(text: str) -> str:
    """JSON 输出格式化后再展示，人能对得上字段。"""
    raw = (text or "").strip()
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return raw


def _stage_rows(stages: dict) -> str:
    out = []
    for key, s in stages.items():
        applicable = s["applicable"]
        rate = 0.0 if not applicable else s["pass"] / applicable
        cls = "ok" if rate >= 0.99 else ("warn" if rate >= 0.5 else "fail")
        out.append(
            f"<tr><td>{_esc(s['name'])}</td>"
            f"<td class='num'>{s['pass']}</td><td class='num'>{s['warn']}</td>"
            f"<td class='num'>{s['fail']}</td><td class='num'>{s['skipped']}</td>"
            f"<td><div class='bar'><i style='width:{rate:.0%}'></i></div></td>"
            f"<td class='num'>{'—' if not applicable else f'{rate:.0%}'}</td></tr>"
        )
    return "".join(out)


def _chips(stages: list[dict]) -> str:
    parts = []
    for s in stages:
        lvl = s["level"]
        parts.append(f"<span class='chip {lvl}' title='{_esc(s['detail'])}'>"
                     f"{_esc(s['name'])} · {lvl.upper()}</span>")
    return "".join(parts)


def _card(row: dict) -> str:
    ok = row.get("ok")
    is_control = bool(row.get("control"))
    if is_control:
        verdict_cls = "control"
        tag = "<span class='tag ctl'>负对照 · 不计分</span>"
    else:
        verdict_cls = "ok" if ok else "bad"
        tag = "<span class='tag ok'>通过</span>" if ok else "<span class='tag bad'>失败</span>"

    seeds = row.get("seed_files") or {}
    input_body = _esc(row.get("prompt", ""))
    if seeds:
        files = "\n\n".join(f"── {name} ──\n{content}" for name, content in seeds.items())
        input_body += "\n\n" + _esc(files)

    answer = _pretty(row.get("answer", "")) or "(空)"
    ref = row.get("reference_answer") or "(未提供标准答案)"
    ref_body = _esc(ref)
    if row.get("reference_note"):
        ref_body += "\n\n" + _esc("说明：" + row["reference_note"])

    reasons = ""
    if row.get("failures"):
        items = "".join(f"<li>{_esc(f)}</li>" for f in row["failures"])
        reasons = f"<div class='reasons'><b style='color:var(--fail)'>扣分项</b><ul>{items}</ul></div>"

    diag = ""
    d = row.get("diagnosis")
    if d:
        if d.get("ok"):
            diag = (
                "<div class='diag'>"
                f"<div class='lbl'>自动归因 · {_esc(ERROR_TYPES.get(d['error_type'], d['error_type']))}"
                f" · 置信度 {_esc(d.get('confidence', 'low'))}</div>"
                f"<div class='row'><b>根因：</b>{_esc(d.get('root_cause', ''))}</div>"
                f"<div class='row'><b>证据：</b><span class='mono'>{_esc(d.get('evidence', ''))}</span></div>"
                f"<div class='row'><b>建议：</b>{_esc(d.get('suggestion', ''))}</div>"
                "</div>"
            )
        else:
            diag = f"<div class='diag'><div class='lbl'>自动归因失败</div>{_esc(d.get('error', ''))}</div>"

    sc = row.get("score")
    sc_txt = "—" if sc is None else f"{sc:.3f}"
    meta = (f"<span class='muted mono'>步 {row.get('steps', '?')} · "
            f"工具 {row.get('tools', '?')} · token {row.get('tokens', '?')} · "
            f"{row.get('wall_s', '?')}s · 质量分 {sc_txt}</span>")

    return f"""
<article class="card {verdict_cls}">
  <h3>{_esc(row.get('id'))}{tag}</h3>
  {meta}
  <div class="compare">
    <div class="col input"><h4>输入材料</h4><pre>{input_body}</pre></div>
    <div class="col model"><h4>模型输出</h4><pre>{_esc(answer)}</pre></div>
    <div class="col ref"><h4>标准答案</h4><pre>{ref_body}</pre></div>
  </div>
  {reasons}
  {diag}
  <div class="chips">{_chips(row.get('stages', []))}</div>
</article>"""


def _tier_board(tier_scores: dict, threshold: float) -> str:
    if not tier_scores or len(tier_scores) < 2:
        return ""
    names = {"gold": "黄金集 · 验收", "silver": "白银集 · 回归"}
    rows = []
    for tier, sc in tier_scores.items():
        rate = sc["quality"]
        cls = "ok" if rate >= threshold else ("warn" if rate >= threshold - 0.1 else "fail")
        badge = "可发布" if sc["gate"] == "pass" else "未达线"
        rows.append(
            f"<tr><td>{_esc(names.get(tier, tier))}</td>"
            f"<td class='num'>{sc.get('tasks', '—')}</td>"
            f"<td class='num' style='color:var(--{cls})'>{rate:.3f}</td>"
            f"<td><span class='gate {sc['gate']}'>{badge}</span></td></tr>"
        )
    return ("<h2>分档质量</h2><table><tr><th>数据集</th><th class='num'>质量分</th>"
            "<th class='num'>分数</th><th>闸门</th></tr>" + "".join(rows) + "</table>")


def render(data: dict) -> str:
    meta = data["meta"]
    score = data["score"]
    stages = data["stages"]
    rows = data["rows"]

    gate_cls = "pass" if score["gate"] == "pass" else "fail"
    gate_txt = "可发布" if score["gate"] == "pass" else "未达及格线"
    q = score["quality"]
    q_cls = "ok" if q >= score["threshold"] else ("warn" if q >= score["threshold"] - 0.1 else "fail")

    total = len(rows)
    passed = sum(1 for r in rows if r.get("ok"))
    types = data.get("error_types") or {}
    type_line = "、".join(f"{ERROR_TYPES.get(k, k)}×{v}" for k, v in types.items()) or "无失败任务"

    drag_line = "、".join(f"{k}×{v}" for k, v in score["drag_counter"].items()) or "无"

    cards = "".join(_card(r) for r in rows)
    tier_board = _tier_board(data.get("tier_scores") or {}, score["threshold"])
    lat = data.get("latency") or {}
    cst = data.get("cost") or {}
    ttft_s = (lat.get("ttft_ms_p50") or 0) / 1000
    ttft95 = (lat.get("ttft_ms_p95") or 0) / 1000
    cost_cell = (money(cst.get("per_task", 0), cst.get("currency", "USD"))
                 if cst.get("priced") else "—")
    cost_note = (f"总计 {money(cst.get('total', 0), cst.get('currency', 'USD'))}"
                 if cst.get("priced") else "未取到单价")
    fp = data.get("fingerprint") or {}
    fp_line = (f"输入指纹 {fp.get('combined', '未记录')} · "
               f"prompt {fp.get('system', {}).get('hash', '?')} · "
               f"技能 {len(fp.get('skills') or {})} 个 · 题集 {fp.get('tasks', '?')}")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Agent 评测报告</title><style>{CSS}</style></head>
<body>
<h1>Agent 评测报告</h1>
<div class="muted mono">{_esc(meta.get('model'))} · {_esc(meta.get('time'))} ·
{total} 条任务 · 工具调用 {meta.get('tool_calls', 0)} 次 · 总 token {meta.get('tokens', 0)}</div>

<div class="board">
  <div class="kpi"><div class="k">准确率 · 质量分</div><div class="v {q_cls}">{q:.3f}</div>
    <div class="muted" style="font-size:11.5px">及格线 {score['threshold']:.2f} · 通过 {passed}/{total}</div></div>
  <div class="kpi"><div class="k">延迟 · 首字 TTFT</div><div class="v">{ttft_s:.1f}s</div>
    <div class="muted" style="font-size:11.5px">P95 {ttft95:.1f}s</div></div>
  <div class="kpi"><div class="k">延迟 · 端到端</div><div class="v">{lat.get('wall_s_p50', 0)}s</div>
    <div class="muted" style="font-size:11.5px">P99 {lat.get('wall_s_p99', 0)}s · 最慢 {lat.get('wall_s_max', 0)}s</div></div>
  <div class="kpi"><div class="k">成本 · 每任务</div><div class="v">{cost_cell}</div>
    <div class="muted" style="font-size:11.5px">{_esc(cost_note)}</div></div>
  <div class="kpi"><div class="k">上线闸门</div>
    <div class="v"><span class="gate {gate_cls}">{gate_txt}</span></div></div>
  <div class="kpi"><div class="k">主要失分环节</div>
    <div class="v" style="font-size:13px;font-weight:500">{_esc(drag_line)}</div></div>
</div>

{tier_board}

<h2>阶段漏斗</h2>
<table>
<tr><th>阶段</th><th class="num">PASS</th><th class="num">WARN</th><th class="num">FAIL</th>
<th class="num">N/A</th><th style="width:130px">分布</th><th class="num">通过率</th></tr>
{_stage_rows(stages)}
</table>

<h2>失败类型分布（LLM 归因）</h2>
<div class="muted">{_esc(type_line)}</div>

<h2>逐条对照</h2>
{cards}

<footer>
判分：确定性断言（json_field / answer_contains / file_exists / predicate），不调模型，可复现。<br>
归因：LLM-as-a-Judge 读失败轨迹产出，属于假设，带置信度，仅供参考，不影响分数。<br>
<span class="mono">{_esc(fp_line)}</span><br>
指纹一致才说明这份分数对应的是当前这版 prompt——改一个字也应该重跑。
</footer>
</body></html>"""


def write(data: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(data))
    return path

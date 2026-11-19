"""kb-agent：在 arlab 运行时上落地的垂直应用 —— 技术文档检索问答。

这一层只做三件事：
  1. 换工具集（kb_search / kb_read / cite_check 替代 read_file / run_python）
  2. 换技能集（retrieval-qa / refusal-policy 替代 csv-profiling）
  3. 换评测口径（引用可溯源、要点覆盖、拒答正确，替代 json_field 比对）

运行时一行没改。这就是「通用运行时 + 垂直应用」的实际形态。

用法：
    ./run.sh ask "LoRA 的 A 和 B 矩阵怎么初始化？"
    ./run.sh eval
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from arlab import Agent, Budget, LLM, Registry, SkillLoader, Trace, new_run_dir
from arlab.evaluate import _extract_json
from arlab.tools.builtin import build_tools

from .kb_tools import build_kb_tools, _store

APP_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = APP_ROOT / "skills"
GOLD_PATH = APP_ROOT / "eval" / "gold.json"
SILVER_PATH = APP_ROOT / "eval" / "silver.json"
RUNS_DIR = APP_ROOT / "runs"

DOMAIN_PROMPT = """你是一个技术文档问答助手，只能依据知识库里的资料回答。

硬性要求：
1. 每条结论必须有引用，引用必须从 kb_read 的输出里逐字复制，不许改写。
2. 提交前对每条引用跑一次 cite_check，未通过就回去重抄。
3. 资料里没有就拒答，不要用"一般来说""通常"包装猜测。

交付方式（唯一被接受的方式）：
**必须调用 submit_answer 工具提交答案，不要直接输出文本。**
answer 写结论正文并用 [1][2] 标出处；citations 每项含 chunk_id 和 quote（原文逐字）；
refused 表示资料是否不足。三个参数缺任何一个都会提交失败。

字段含义和检索策略见 retrieval-qa 技能；什么时候该拒答见 refusal-policy 技能。"""


# ---------------------------------------------------------------- 判据
# 垂直层专用谓词。它们都只依赖「引用能不能在原文找到」，
# 不需要领域专家判断答案对不对 —— 这是检索类应用能自动化评测的关键。

def _citations(answer: str) -> tuple[dict | None, list[dict]]:
    data = _extract_json(answer)
    if not data:
        return None, []
    cites = data.get("citations")
    return data, cites if isinstance(cites, list) else []


MIN_QUOTE_CHARS = 6


def pred_citations_grounded(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """每条引文必须在其 chunk 的原文里逐字存在，且至少一条。

    引文还要有最小长度（默认 6 字）：否则引一个「的」「模型」也能逐字命中，
    「引用可溯源」就退化成了「引用了一个常用词」。
    """
    data, cites = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    if not cites:
        return "一条引用都没给"
    min_chars = int(spec.get("min_quote_chars", MIN_QUOTE_CHARS))
    store = _store()
    bad: list[str] = []
    for c in cites:
        if not isinstance(c, dict):
            bad.append("citation 不是对象")
            continue
        quote = (c.get("quote") or "").strip()
        cid = (c.get("chunk_id") or "").strip()
        if not quote or not cid:
            bad.append("citation 缺 quote 或 chunk_id")
            continue
        if len(quote) < min_chars:
            bad.append(f"引文太短（{len(quote)} 字 < {min_chars}）: {quote!r}")
            continue
        try:
            text = store.chunk(cid).text
        except KeyError:
            bad.append(f"chunk_id 不存在: {cid}")
            continue
        if quote not in text:
            bad.append(f"引文不在原文里: {quote[:36]!r}")
    if bad:
        return f"{len(bad)}/{len(cites)} 条引用无法溯源：" + "；".join(bad[:3])
    return None


def _norm(text: str) -> str:
    """要点匹配用的宽松归一：忽略大小写、空白和连字符（KV-Cache == kv cache）。"""
    return re.sub(r"[\s\-_]+", "", text).lower()


def _group_hit(group, text_n: str) -> bool:
    """一个要点 = 一个字符串，或一组同义写法（命中任一即算命中这个要点）。"""
    alts = group if isinstance(group, list) else [group]
    return any(_norm(str(a)) in text_n for a in alts if str(a).strip())


def _group_label(group) -> str:
    return "/".join(map(str, group)) if isinstance(group, list) else str(group)


def pred_answer_covers(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """要点覆盖：must 里至少命中 min_hit 个要点。

    must 的每一项可以是字符串，也可以是同义词列表：
        "must": [["幻觉", "hallucination"], ["数据新鲜度", "时效性", "过时"]]
    同义组的存在理由：旧判据把「时效性」判成没答到「数据新鲜度」，
    要点是语义单位，不是字面单位。

    forbid（可选）：答案里不许出现的说法。用于错误前提题和陷阱题——
    例如问「LoRA 为什么让推理变慢」，答案顺着说「会增加推理延迟」就该判错。
    """
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    text_n = _norm(str(data.get("answer") or ""))
    must = spec.get("must") or []
    need = int(spec.get("min_hit", len(must)))
    hit = [g for g in must if _group_hit(g, text_n)]
    if len(hit) < need:
        missing = [_group_label(g) for g in must if not _group_hit(g, text_n)]
        return f"要点还差 {need - len(hit)} 个，缺：{missing}"
    bad = [_group_label(g) for g in (spec.get("forbid") or []) if _group_hit(g, text_n)]
    if bad:
        return f"答案里出现了不该有的说法：{bad}"
    return None


def pred_evidence_hit(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """证据命中：引用的块里必须包含标准证据的「锚点」原文。

    和 chunk_hit 的区别：chunk_hit 认块 ID，这里认内容。
    语料里有大量重复/转载文档（同一段话出现在两三个 doc 里），
    锁死 ID 会把等价证据判成没证据——multi_doc v1 就栽在这上面。
    锚点是标准答案所在原文里的一小段独特文字，谁的块里有它，谁就算命中。

        "groups": [["TokenAttention"], ["PagedAttention", "分页"]]

    每个 group 都必须被至少一条引用的块覆盖（多跳题 = 多个 group）。
    group 内是「任一锚点」关系。
    """
    data, cites = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    groups = spec.get("groups") or []
    if not groups:
        return "evidence_hit 缺少 groups"
    store = _store()
    texts: list[str] = []
    for c in cites:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chunk_id") or "").strip()
        try:
            texts.append(store.chunk(cid).text)
        except KeyError:
            continue
    if not texts:
        return "没有任何一条引用指向真实存在的块"
    missed = [g for g in groups
              if not any(str(a) in t for t in texts for a in (g if isinstance(g, list) else [g]))]
    if missed:
        return (f"{len(missed)}/{len(groups)} 组关键证据没被引用到："
                + "；".join(_group_label(g) for g in missed[:3]))
    return None


def pred_not_refused(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    if data.get("refused") is True:
        return "这题资料里有答案，不该拒答"
    return None


def pred_chunk_hit(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """白银集用：引用必须命中期望的块（本块或同文档后继块）。

    白银集的期望值由块序号程序推出，所以这条判据完全不需要人写答案。
    """
    data, cites = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    expected = {e for e in (spec.get("expected") or []) if isinstance(e, str)}
    if not expected:
        return "白银任务缺少 expected 字段"
    got = {c["chunk_id"] for c in cites
           if isinstance(c, dict) and isinstance(c.get("chunk_id"), str)}
    if got & expected:
        return None
    return f"引用未命中期望块。期望之一 {sorted(expected)[:2]}，实际 {sorted(got)[:3]}"


def pred_states_missing(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """资料里确实没有时：要么标 refused，要么在正文里说明缺失。

    关键：还要看它是**主动判断**出资料缺失，还是被预算逼到墙角才说没有。
    后者是假阳性 —— 同一句话，前者说明策略对，后者说明它只是在收尾。
    """
    ctx = ctx or {}
    if ctx.get("status") == "budget_exhausted":
        return (f"拒答是被预算逼停后收尾的（原因 {ctx.get('reason')}），"
                f"不是自己判断出资料缺失；token {ctx.get('tokens')} 步数 {ctx.get('steps')}")
    data, _ = _citations(answer)
    if data is None:
        return "答案不是可解析的 JSON"
    text = str(data.get("answer") or "")
    # 拒答了但顺手编了一个具体值（「资料里没有，不过一般是 1024 维」）——仍然是编造
    leaked = [_group_label(g) for g in (spec.get("forbid") or []) if _group_hit(g, _norm(text))]
    if leaked:
        return f"声称资料缺失，却给出了资料里不存在的具体内容：{leaked}"
    if data.get("refused") is True:
        return None
    words = spec.get("words") or MISSING_WORDS
    if any(w in text for w in words):
        return None
    return f"既没标 refused，也没说明资料缺失。答案开头：{text[:60]!r}"


# 「说明缺失」的措辞表。「无法」单独太宽（「无法直接比较」也会命中），收窄成搭配；
# 「未能」「未找到」「不存在」是实测漏掉过的写法。
MISSING_WORDS = [
    "没有", "未提及", "没提到", "未涉及", "不包含", "找不到", "未找到", "未能找到", "未能检索",
    "无法回答", "无法确定", "无法给出", "无法找到", "无法从", "资料中没", "未提供", "不存在",
    "缺少", "缺乏相关", "并未", "未收录", "没有收录",
]


def pred_reports_failure(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """工具故障时，答案必须说明失败，而不是编一个看起来很正常的答案。

    这是「结果对不算对」的落地：检索接口挂了还能给出完整答案的，只可能是编的。
    """
    data, _ = _citations(answer)
    text = str((data or {}).get("answer") or answer or "")
    words = spec.get("words") or [
        "失败", "不可用", "超时", "无法访问", "调用失败", "检索不到",
        "服务异常", "无法检索", "故障", "没有返回",
    ]
    if any(w in text for w in words):
        return None
    head = " ".join(text.strip().split())[:70]
    return f"工具已注入故障，但答案没有说明失败（很可能是编的）。答案开头：{head!r}"


def pred_multi_doc(answer: str, spec: dict, ctx: dict | None = None) -> str | None:
    """多跳题：引用必须来自多个不同文档，证明真的跨文档取证了。

    判据设计的两版教训：
    v1 要求「必须引用指定的某几个文档 ID」，结果 5 条失败里有 4 条是误判——
    模型引用了 PagedAttention__readme（那本来就是 vLLM 的文档）、
    「大模型RLHF：PPO原理与源码解读」（同样是 PPO 主题），
    但判据只认 vllm__readme / 近端策略优化 这两个字面 ID。

    语料里有大量同主题文档，锁死 ID 等于把「等价证据」判成「没有证据」。
    v2 改成数量判据：至少引用 min_docs 个不同文档。
    spec.docs 保留作为提示（人工排查时看该找哪些），不参与判定。
    """
    min_docs = int(spec.get("min_docs", 2))
    _, cites = _citations(answer)
    docs = set()
    for c in cites:
        cid = str(c.get("chunk_id") or "")
        if cid:
            docs.add(cid.rsplit("-", 1)[0])
    if len(docs) < min_docs:
        hint = f"，建议覆盖 {spec.get('docs')}" if spec.get("docs") else ""
        return f"只引用了 {len(docs)} 个文档，多跳题要求至少 {min_docs} 个{hint}"
    return None


PREDICATES = {
    "citations_grounded": pred_citations_grounded,
    "answer_covers": pred_answer_covers,
    "not_refused": pred_not_refused,
    "states_missing": pred_states_missing,
    "chunk_hit": pred_chunk_hit,
    "reports_failure": pred_reports_failure,
    "multi_doc": pred_multi_doc,
    "evidence_hit": pred_evidence_hit,
}


# ---------------------------------------------------------------- 组装
def tool_builder(workspace: str, skills):
    """垂直层的工具集：检索三件套 + load_skill。不给文件读写和跑代码。"""
    from arlab.tools.builtin import build_tools as _build
    from .kb_tools import reset_reads

    # 每条任务都重新构建工具集，顺便重置读块计数——
    # 计数器如果不重置，第二条任务会继承上一条的读数，提示时机全乱。
    reset_reads()
    base = {t.name: t for t in _build(workspace, skill_loader=skills)}
    tools = build_kb_tools()
    if "load_skill" in base:
        tools.append(base["load_skill"])
    return tools


# 预算/模型参数只在这里定义一次，评测路径和生产路径共用。
# 之前 make_agent 写 40_000、cmd_eval 又写一遍 40_000，改一处忘另一处就会让评测失真——
# 「评测跑的不是生产配置」这个 bug 就是这么来的，已经害得所有历史数据要重新解释。
BUDGET_STEPS = 8
BUDGET_TOKENS = 40_000
BUDGET_SECONDS = 240.0


def make_agent(*, model: str, base_url: str | None, max_steps: int,
               trace: Trace | None = None) -> Agent:
    skills = SkillLoader(str(SKILLS_DIR)).discover()
    registry = Registry(allow={"read"})
    for tool in tool_builder(str(APP_ROOT), skills):
        registry.register(tool)
    return Agent(
        LLM(base_url=base_url, model=model),
        registry,
        skills=skills,
        budget=Budget(max_steps=max_steps, max_tokens=BUDGET_TOKENS,
                      max_seconds=BUDGET_SECONDS),
        trace=trace,
        extra_system=DOMAIN_PROMPT,
    )


# ---------------------------------------------------------------- CLI
def cmd_ask(args) -> int:
    run_dir = new_run_dir(str(RUNS_DIR), tag="kb-ask")
    trace = Trace(run_dir, meta={"question": args.question, "model": args.model})
    agent = make_agent(model=args.model, base_url=args.base_url,
                       max_steps=args.max_steps, trace=trace)
    print(f"问题: {args.question}")
    print(f"技能: {', '.join(agent.skills.skills) if agent.skills else '无'}")
    print(f"trace: {trace.path}\n")

    result = agent.run(args.question)
    print("=" * 70)
    print(result.answer)
    print("=" * 70)

    summary = trace.summarize()
    trace.close(status=result.status, answer=result.answer, **summary)
    print(f"\n状态 {result.status}  步数 {summary['steps']}  "
          f"工具 {summary['tool_calls']}（失败 {summary['tool_failed']}）  "
          f"token {summary['total_tokens']}")
    return 0 if result.status == "ok" else 1


def cmd_eval(args) -> int:
    from arlab.evaluate import run_suite

    # 默认按需加载。曾经按单轮 A/B（内联 0.943 vs 无技能 0.927）把内联改成默认，
    # 结果 3 轮定型基线推翻了它：内联 0.944 vs 按需 0.947，质量没变，
    # 但 token +32%、成本 +28%、端到端 +7.8s。
    # 教训：架构决策必须等多轮数据，单轮差异（0.016）就是噪声本身。
    no_skills = getattr(args, "no_skills", False)
    inline = getattr(args, "inline_skills", False)
    if no_skills:
        _extra = DOMAIN_PROMPT
        _skills = ""
    elif inline:
        _loader = SkillLoader(str(SKILLS_DIR)).discover()
        _extra = DOMAIN_PROMPT + "\n\n" + _loader.inline_body()
        _skills = ""
    else:
        _extra = DOMAIN_PROMPT
        _skills = str(SKILLS_DIR)

    ns = argparse.Namespace(
        tasks=args.tasks or f"{GOLD_PATH},{SILVER_PATH}",
        limit=args.limit,
        workspace=str(APP_ROOT / "workspace"),
        skills=_skills,
        extra_system=_extra,
        model=args.model,
        base_url=args.base_url,
        runs=str(RUNS_DIR),
        allow="read",
        max_steps=args.max_steps,
        max_tokens=BUDGET_TOKENS,
        max_seconds=BUDGET_SECONDS,
        out=args.out,
        threshold=args.threshold,
        report=not args.no_report,
        report_out=args.report_out,
        judge=not args.no_judge,
        judge_model="",
        repeat=getattr(args, "repeat", 1),
    )
    return run_suite(ns, tool_builder=tool_builder, predicates=PREDICATES)


def main(argv: list[str] | None = None) -> int:
    from arlab.llm import DEFAULT_MODEL

    p = argparse.ArgumentParser(prog="kb-agent", description="技术文档检索问答 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--max-steps", type=int, default=8)

    a = sub.add_parser("ask", help="问一个问题")
    a.add_argument("question")
    common(a)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="跑黄金集")
    e.add_argument("--tasks", default="", help="覆盖题集路径，逗号分隔。默认 gold+silver")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--threshold", type=float, default=0.90)
    e.add_argument("--report-out", default="")
    e.add_argument("--no-report", action="store_true")
    e.add_argument("--no-judge", action="store_true")
    e.add_argument("--repeat", type=int, default=1,
                   help="重复跑 N 轮，准确率取中位数并给区间，延迟/成本合并采样")
    e.add_argument("--out", default="", help="结果 json 路径，默认按时间戳生成")
    e.add_argument("--no-skills", action="store_true",
                   help="不提供任何技能（A/B 用：验证技能是否有价值）")
    e.add_argument("--inline-skills", action="store_true",
                   help="技能正文内联进 system prompt（3 轮基线证明：质量无提升但成本 +28%%，仅用于对照）")
    common(e)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

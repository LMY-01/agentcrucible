"""垂直层的工具：把 kb-lab 的检索能力包成 Agent 能调用的三个动作。

三个工具是刻意这样切的：

  kb_search  → 只给摘要和 chunk_id，不给全文。逼 Agent 判断「够不够」，而不是把整库倒进上下文
  kb_read    → 按 id 取全文，明确的一跳，token 花在哪一目了然
  cite_check → 让 Agent **自己**验证引用的句子是否原样存在，不靠 prompt 祈祷

第三个是这一层的核心：引用可核验，是检索类应用唯一能机械判定的质量指标。
有了它，"模型有没有编造"就不需要人读一遍才知道。
"""
from __future__ import annotations

import difflib
import os
from functools import lru_cache
from pathlib import Path

from arlab.tools.registry import READ, Tool

KB_ROOT = Path(os.environ.get("KB_LAB_ROOT", "/mnt/d/kb-lab"))
DB_PATH = KB_ROOT / "data" / "chroma"
SUMMARY_CHARS = 160


def _summary_chars() -> int:
    return int(os.environ.get("KB_SEARCH_SUMMARY_CHARS", str(SUMMARY_CHARS)))

# 必须在 sentence_transformers 首次 import 之前设好：它加载权重前会向 HF Hub
# 查 revision，WSL 里不通就阻塞到超时。权重本地有缓存，离线跑完全够。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache(maxsize=1)
def _store():
    """进程内单例。KBStore 构造要加载 embedding 模型，每次新建会拖垮评测。"""
    from kb_lab.store import KBStore

    return KBStore(str(DB_PATH))


def reset_store() -> None:
    """测试用：丢掉缓存的 store。"""
    _store.cache_clear()


def kb_search(query: str, k: int = 5, mode: str = "hybrid") -> str:
    hits = _store().search(query, k=k, mode=mode)
    if not hits:
        return "没有检索到任何内容。换个问法，或者考虑这题资料里确实没有答案。"
    lines = []
    for rank, (chunk, score) in enumerate(hits, 1):
        head = " ".join(chunk.text.split())[:_summary_chars()]
        lines.append(
            f"[{rank}] chunk_id={chunk.id}\n"
            f"    来源={chunk.source}  第{chunk.idx}块  相关度={score:.4f}\n"
            f"    摘要: {head}…"
        )
    return f"命中 {len(hits)} 条（mode={mode}）。要看全文用 kb_read，别凭摘要下结论。\n\n" + "\n\n".join(lines)


# 读块计数：每跑一条任务重置。用来在预算被烧光之前提醒模型收尾。
# 实测 mh05 读了 15 个块才被预算掐断，最后 submit_answer 提交不进去——
# 它并不知道自己已经读得够多了。
_READS = {"n": 0}


def reset_reads() -> None:
    """每条任务开始时调用。"""
    _READS["n"] = 0


def kb_read(chunk_id: str) -> str:
    store = _store()
    try:
        chunk = store.chunk(chunk_id)
    except KeyError:
        raise KeyError(f"没有这个 chunk_id: {chunk_id}。请用 kb_search 返回的 id，不要自己编。")

    _READS["n"] += 1
    n = _READS["n"]
    # 渐进提醒，不是硬拒绝：硬砍会丢信息，提醒能让模型自己决定何时停。
    note = ""
    if n >= 10:
        note = (f"\n\n[预算告警] 已读 {n} 个块，token 消耗已经很高。"
                "请立刻基于现有材料作答，不要再读新块。")
    elif n >= 6:
        note = (f"\n\n[提示] 已读 {n} 个块。对比/归纳类问题通常 3-5 块就够，"
                "确认关键点齐了就该收尾。")
    return (
        f"chunk_id={chunk.id}\n来源={chunk.source}  第{chunk.idx}块\n"
        f"--- 全文 ---\n{chunk.text}{note}"
    )


def cite_check(quote: str, chunk_id: str) -> str:
    """逐字校验引用。这是唯一能机械判定「有没有编造」的检查。"""
    store = _store()
    try:
        text = store.chunk(chunk_id).text
    except KeyError:
        return f"无法校验：没有 chunk_id={chunk_id}。引用必须来自 kb_read 读到的块。"

    if quote in text:
        return f"✅ 命中：这段引文在 {chunk_id} 中逐字存在，可以放心使用。"

    # 没逐字命中时，给出最长的公共片段，让 Agent 知道差在哪
    matcher = difflib.SequenceMatcher(None, quote, text, autojunk=False)
    block = matcher.find_longest_match(0, len(quote), 0, len(text))
    ratio = block.size / max(1, len(quote))
    sample = text[block.b : block.b + min(120, block.size)] if block.size else ""
    return (
        f"❌ 未逐字命中。最长连续重叠 {block.size}/{len(quote)} 字（覆盖率 {ratio:.0%}）。\n"
        f"原文对应片段: {sample}\n"
        f"处理方式：直接从 kb_read 的输出里复制原句，不要改写、不要用省略号拼接不相邻的句子。"
    )


def web_search(query: str) -> str:
    """公网搜索。本知识库任务不该用它——它返回的不是内部资料，无法作为引用依据。"""
    return (
        f"公网搜索「{query[:40]}」返回 0 条可用结果。\n"
        "注意：即使有结果也不能用作本知识库的引用来源，引用必须来自 kb_read。"
    )


def submit_answer(answer: str, citations: list, refused: bool) -> str:
    """终止型工具：参数就是交付物，handler 不会被真正使用。"""
    return "已提交"


def validate_submission(payload: dict) -> str | None:
    """交付门禁：每条引文必须能在对应 chunk 里逐字找到，否则提交不进去。

    为什么必须在交付时校验，而不是事后判分：
    实测 silver_007 调了 8 次 cite_check 全部通过，却提交了 10 条引用——
    多出的 2 条从没检查过，而且是编的。cite_check 是建议性的，模型可以跳过；
    只有把校验放进交付路径，"引用可溯源"才是运行时保证的性质。
    """
    cites = payload.get("citations")
    if not isinstance(cites, list):
        return "citations 必须是数组"
    refused = payload.get("refused") is True
    if not cites and not refused:
        return "没有给任何引用，也没标 refused=true：要么给出处，要么明确说资料不足"

    store = _store()
    bad: list[str] = []
    for i, c in enumerate(cites, 1):
        if not isinstance(c, dict):
            bad.append(f"第 {i} 条不是对象")
            continue
        quote = (c.get("quote") or "").strip()
        cid = (c.get("chunk_id") or "").strip()
        if not quote or not cid:
            bad.append(f"第 {i} 条缺 quote 或 chunk_id")
            continue
        try:
            text = store.chunk(cid).text
        except KeyError:
            bad.append(f"第 {i} 条的 chunk_id 不存在：{cid[:36]}")
            continue
        if quote not in text:
            bad.append(f"第 {i} 条的引文在原文里找不到：{quote[:36]!r}")

    if bad:
        head = "；".join(bad[:4])
        tail = f"（另有 {len(bad) - 4} 条）" if len(bad) > 4 else ""
        return f"{head}{tail}。请用 kb_read 重新读取并逐字复制原句。"
    return None


def build_kb_tools() -> list[Tool]:
    return [
        Tool(
            name="kb_search",
            description="在技术文档知识库里做混合检索（向量+关键词）。返回 chunk 摘要和相关度，不含全文。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，用文档里的术语效果更好"},
                    "k": {"type": "integer", "description": "返回几条，默认 5，最多 10"},
                    "mode": {"type": "string", "enum": ["hybrid", "vector", "bm25"],
                             "description": "hybrid 默认；术语密集的资料用 bm25 往往更准"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=kb_search,
            permission=READ,
        ),
        Tool(
            name="kb_read",
            description="按 chunk_id 读取某个片段的全文。引用原文前必须先读全文。",
            parameters={
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
            handler=kb_read,
            permission=READ,
        ),
        Tool(
            name="cite_check",
            description="校验一段引文是否在指定 chunk 里逐字存在。提交答案前对每条引用都跑一次。",
            parameters={
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "description": "准备写进答案的原文引句"},
                    "chunk_id": {"type": "string", "description": "这句引文来自哪个 chunk"},
                },
                "required": ["quote", "chunk_id"],
                "additionalProperties": False,
            },
            handler=cite_check,
            permission=READ,
        ),
        Tool(
            name="web_search",
            description="公网搜索。知识库检索失败时**不允许**用它兜底：返回内容不是内部资料，不能作为引用依据。",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=web_search,
            permission=READ,
        ),
        Tool(
            name="submit_answer",
            description=(
                "提交最终答案。这是唯一被接受的交付方式——不要直接输出文本答案，必须调用本工具。"
                "answer 写结论正文并用 [1][2] 标出处；citations 每项含 chunk_id 和 quote（原文逐字复制）；"
                "资料里查不到时必须把 refused 设为 true。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "结论正文，分点写"},
                    "citations": {
                        "type": "array",
                        "description": "引用列表，每项是 {chunk_id, quote}，quote 必须与原文逐字一致",
                    },
                    "refused": {"type": "boolean", "description": "资料查不到时为 true"},
                },
                "required": ["answer", "citations", "refused"],
                "additionalProperties": False,
            },
            handler=submit_answer,
            permission=READ,
            terminal=True,
            validator=validate_submission,
        ),
    ]

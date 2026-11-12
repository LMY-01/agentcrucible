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
        head = " ".join(chunk.text.split())[:SUMMARY_CHARS]
        lines.append(
            f"[{rank}] chunk_id={chunk.id}\n"
            f"    来源={chunk.source}  第{chunk.idx}块  相关度={score:.4f}\n"
            f"    摘要: {head}…"
        )
    return f"命中 {len(hits)} 条（mode={mode}）。要看全文用 kb_read，别凭摘要下结论。\n\n" + "\n\n".join(lines)


def kb_read(chunk_id: str) -> str:
    store = _store()
    try:
        chunk = store.chunk(chunk_id)
    except KeyError:
        raise KeyError(f"没有这个 chunk_id: {chunk_id}。请用 kb_search 返回的 id，不要自己编。")
    return (
        f"chunk_id={chunk.id}\n来源={chunk.source}  第{chunk.idx}块\n"
        f"--- 全文 ---\n{chunk.text}"
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
    ]

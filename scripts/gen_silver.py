"""白银集生成器：从语料自身挖出「问题 → 期望引用的块」，零人工标注。

核心思路：这批语料是问答式的技术资料，正文里本来就带着「XXX 是什么？」这样的问句，
而且答案就在紧随其后的段落里。所以：

    问题      = 语料里的问句
    标准答案  = 「该问句所在的块」或「同文档的下一个块」

两者都由程序定位，不需要人写、也不需要领域知识。这就是白银集能上规模的原因。

v2 关键改进：
  1. 彻底清洗中文编号前缀（一、二、1. 致谢）和目录粘连（"分块实现方法那么..."）。
  2. 过滤指代不明的无主语问句（"他们是怎么..."、"那什么指标..."、"其与..."）和主观开放问句（"你认为..."）。
  3. 要求问句必须包含明确的领域实体或核心技术概念，拒绝无法脱离上下文独立成立的残句。
  4. 排除所有黄金集（gold.json / gold_holdout.json / gold_multihop.json / gold_v2.json）已覆盖的题目。

用法：
    python scripts/gen_silver.py --limit 40 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = Path(os.environ.get("KB_LAB_ROOT", "/mnt/d/kb-lab"))
CHUNKS = KB_ROOT / "data" / "chunks.jsonl"
OUT = PROJECT_ROOT / "apps" / "kb_agent" / "eval" / "silver.json"

# 清洗掉 markdown 噪声、中文/阿拉伯数字章节编号前缀、致谢/前言等粘连前缀
NOISE = [
    (re.compile(r"&#x[0-9a-fA-F]+;"), " "),
    (re.compile(r"\\\[toc\]"), " "),
    (re.compile(r"[#*`>|]"), ""),
    (re.compile(r"\s+"), " "),
    # 剥离 QA: / Q: / 问: 以及粘连的章节前缀（如 "强化学习1.RL基础1-1 QA: "、"rlhf相关1.简单介绍强化学习？"）
    (re.compile(r"^.*?(?:QA|Q|问)\s*[:：]\s*", re.IGNORECASE), ""),
    # 剥离循环出现的章节/致谢前缀，例如 "致谢一、介绍一下..."、"五、 PagedAttention..."、"1.rlhf相关 1.简单介绍..."
    (re.compile(r"^(?:(?:致谢|前言|附录|总结|小结|思考|问题|练习)\s*)+"), ""),
    (re.compile(r"^[\s\d一二三四五六七八九十百\.、．:：）)\-\—]+"), ""),
    (re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9_]{2,12}\s*\d+[\.、．]\s*"), ""),
    (re.compile(r"^(?:(?:致谢|前言|附录|总结|小结)\s*)+"), ""),
    (re.compile(r"^[\s\d一二三四五六七八九十百\.、．:：）)\-\—]+"), ""),
]

BAD_SUBSTR = (
    "toc", "&#", "\\", "<", "http", "图", "表", "如下", "上述", "上面", "下面", "本文", "本节",
    "前面", "后面", "这里", "刚才", "咱们", "读者", "朋友们", "大家", "换句话说", "比如说", "举例",
    "我现在", "我想要", "健康饮食", "块钱", "苹果",
)
# 代码片段和标题粘连的特征符号，命中即丢
BAD_CHARS = ("=", "(", ")", "{", "}", '"', "'", "：", "；", "[", "]")
# 数字编号残留（2.1、第3步）
BAD_PATTERNS = (
    re.compile(r"\d+\.\d+"),
    re.compile(r"第\s*[\d一二三四五六七八九十]+\s*[步节章条点个]"),
    # 依赖上文的指代性开头或连接词开头（脱离原段落根本不知道在问谁）
    re.compile(r"^(?:他|她|它|他们|它们|这|那|其|该|此|那么|所以|因此|接着|然后|另外|此外|如果这样|为什么会这样|你认为|你觉得|请问你|你了解)"),
    # 句中含强指代词且无指代对象
    re.compile(r"(?:他们|它们|这两种|这三种|这几个|这两个|上述|前面提到|该如何选择它们|这样假设|这种假设|这个现象|这个问题|这种方法|这种技术)"),
    # 标题粘连在连接词前，例如 "分块实现方法那么文本分块具体如何实现？"
    re.compile(r"^[\u4e00-\u9fa5A-Za-z0-9]{3,14}(?:那么|那|接下来|下面)"),
)

# 独立技术问句必须至少命中一个领域核心词（过滤掉文章里的生活类比、玩具示例或空泛闲聊问句）
DOMAIN_ANCHORS = (
    "LLM", "大模型", "语言模型", "模型", "微调", "预训练", "SFT", "RLHF", "PPO", "DPO", "策略",
    "奖励", "强化学习", "LoRA", "QLoRA", "Adapter", "PEFT", "参数", "RAG", "检索", "召回", "向量",
    "嵌入", "Embedding", "分块", "chunk", "幻觉", "提示", "Prompt", "CoT", "思维链", "思维树", "ToT",
    "ReAct", "Agent", "智能体", "推理", "显存", "KV", "Attention", "注意力", "PagedAttention",
    "vLLM", "LightLLM", "FasterTransformer", "StreamingLLM", "FlashAttention", "量化", "并行",
    "Transformer", "Decoder", "Encoder", "BERT", "GPT", "分词", "jieba", "词向量", "Word2Vec",
    "词性", "句法", "激活函数", "位置编码", "RoPE", "token", "Token", "解码", "训练", "梯度",
    "知识图谱", "Graph", "LangChain", "LlamaIndex", "表格", "PDF", "pdf", "评测", "评估",
)


def looks_fragmented(q: str) -> bool:
    """标题被拼了两遍，或前后段落粘在一起。"""
    return bool(re.search(r"(.{5,}?)\1", q))


def clean(text: str) -> str:
    for pat, repl in NOISE:
        text = pat.sub(repl, text)
    return text.strip()


def extract_questions(limit: int = 4000) -> list[dict]:
    """从每个块里抠出能独立成立的问句，并记住它在哪个块、哪个文档、第几块。"""
    out: list[dict] = []
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        for piece in re.split(r"(?<=[。！？\n])", c["text"]):
            if not piece.rstrip().endswith(("？", "?")):
                continue
            raw = piece.strip()
            q = clean(raw)
            if not (10 <= len(q) <= 42):
                continue
            if any(b in raw for b in BAD_SUBSTR) or any(b in q for b in BAD_SUBSTR):
                continue
            if any(ch in q for ch in BAD_CHARS):
                continue
            if any(p.search(q) for p in BAD_PATTERNS):
                continue
            if looks_fragmented(q):
                continue
            # 中文之间不该有空格。有，说明是列表项或表格残留（"如何 解决 从复杂PDF"）
            if re.search(r"[\u4e00-\u9fa5]\s+[\u4e00-\u9fa5]", q):
                continue
            # 必须以中文/字母开头（排除残留的纯数字/孤立字母编号如 "nB模型..."）
            if not re.match(r"^(?:[\u4e00-\u9fa5]|[A-Za-z]{2,})", q):
                continue
            # 一句话里不该出现两个以上问号，也不该有逗号过多像一整段话
            if q.count("？") + q.count("?") > 1 or q.count("，") > 1:
                continue
            # 必须包含明确的领域技术词，过滤掉文章里的生活类比或示例句子
            if not any(a.lower() in q.lower() for a in DOMAIN_ANCHORS):
                continue
            out.append({
                "question": q,
                "chunk_id": c["id"],
                "doc_id": c["doc_id"],
                "source": c["source"],
                "idx": int(c["idx"]),
            })
        if len(out) >= limit:
            break
    return out


def _norm_q(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text).lower()


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = _norm_q(it["question"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def verify_expected(item: dict, store, k: int = 10) -> list[str] | None:
    """跑一次真实检索，把「同文档被召回的块」作为可接受答案来源。

    同时把问句自身所在块及紧邻后继块并入候选（若存在），防止检索只召回标题块或漏掉紧邻正文块。
    如果连一个同文档块都召不回来，说明这题不可解，直接丢掉。
    """
    hits = store.search(item["question"], k=k)
    same_doc = [c.id for c, _ in hits if c.doc_id == item["doc_id"]]
    if not same_doc:
        return None
    # 确保问句所在块和紧邻的下一个块也在合法列表里（如果同属该文档）
    prefix = item["chunk_id"].rsplit("-", 1)[0]
    for offset in (0, 1):
        cid = f"{prefix}-{item['idx'] + offset:04d}"
        if cid not in same_doc:
            try:
                store.chunk(cid)
                same_doc.append(cid)
            except KeyError:
                pass
    return same_doc


def build_task(item: dict, expected: list[str], num: int, token_limit: int = 57472) -> dict:
    """一条白银任务。期望引用 = 检索能召回到的同文档块（程序推出，无人为标注）。"""
    return {
        "id": f"silver_{num:03d}",
        "tier": "silver",
        "workspace": "kb",
        "prompt": item["question"],
        "reference_answer": f"答案应来自 《{item['source']}》 的以下块之一：{', '.join(expected[:3])}。",
        "reference_note": "程序生成：问题取自语料原文并经独立性过滤，期望引用由检索结果推出（同文档可召回块）。",
        "require_explore": False,
        "expects_json": True,
        "required_skills": ["retrieval-qa"],
        "stage_limits": {"steps": 8, "tokens": token_limit},
        "checks": [
            {"type": "predicate", "name": "not_refused"},
            {"type": "predicate", "name": "chunk_hit", "expected": expected},
            {"type": "predicate", "name": "citations_grounded"},
        ],
        "_meta": {"source": item["source"], "idx": item["idx"], "recalled": len(expected)},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成白银集")
    p.add_argument("--limit", type=int, default=40, help="产出多少条")
    p.add_argument("--per-doc", type=int, default=2, help="每个文档最多取几条")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--token-limit", type=int, default=57472, help="stage_limits.tokens（默认沿用 P90 校准值）")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from kb_lab.store import KBStore  # 延迟导入：不生成时不拖 embedding 模型

    store = KBStore(str(KB_ROOT / "data" / "chroma"))

    # 任何黄金集里已有的问题都不再进白银集，避免重复计分与数据污染
    eval_dir = PROJECT_ROOT / "apps" / "kb_agent" / "eval"
    gold_q: set[str] = set()
    for gf in ("gold.json", "gold_holdout.json", "gold_multihop.json", "gold_v2.json"):
        gpath = eval_dir / gf
        if gpath.exists():
            for t in json.loads(gpath.read_text(encoding="utf-8")):
                gold_q.add(_norm_q(t.get("prompt", "")))

    items = dedupe(extract_questions())
    items = [i for i in items if _norm_q(i["question"]) not in gold_q]

    rng = random.Random(args.seed)
    rng.shuffle(items)

    tasks: list[dict] = []
    per_doc: dict[str, int] = {}
    dropped = 0
    for it in items:
        if per_doc.get(it["doc_id"], 0) >= args.per_doc:
            continue
        expected = verify_expected(it, store)
        if not expected:
            dropped += 1
            continue
        per_doc[it["doc_id"]] = per_doc.get(it["doc_id"], 0) + 1
        tasks.append(build_task(it, expected, len(tasks) + 1, token_limit=args.token_limit))
        if len(tasks) >= args.limit:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"生成 {len(tasks)} 条白银任务 → {args.out}")
    print(f"覆盖 {len(per_doc)} 个文档，丢弃 {dropped} 条不可召回题")
    for t in tasks[:10]:
        meta = t["_meta"]
        print(f"  {t['id']}  {t['prompt']:<36}  → 期望 {meta['recalled']} 个同文档块")
    return 0


if __name__ == "__main__":
    sys.exit(main())

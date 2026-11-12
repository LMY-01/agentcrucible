"""白银集生成器：从语料自身挖出「问题 → 期望引用的块」，零人工标注。

核心思路：这批语料是问答式的技术资料，正文里本来就带着「XXX 是什么？」这样的问句，
而且答案就在紧随其后的段落里。所以：

    问题      = 语料里的问句
    标准答案  = 「该问句所在的块」或「同文档的下一个块」

两者都由程序定位，不需要人写、也不需要领域知识。这就是白银集能上规模的原因。

和黄金集的分工：
    黄金集  人工挑题 + 人工核验答案，20-30 条，**验收**用，对外能报数
    白银集  程序生成，上百条，**回归**用，只比版本间的涨跌，不对外报绝对值

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
KB_ROOT = Path("/mnt/d/kb-lab")
CHUNKS = KB_ROOT / "data" / "chunks.jsonl"
OUT = PROJECT_ROOT / "apps" / "kb_agent" / "eval" / "silver.json"

# 清洗掉 markdown 噪声和编号前缀
NOISE = [
    (re.compile(r"&#x[0-9a-fA-F]+;"), " "),
    (re.compile(r"\\\[toc\]"), " "),
    (re.compile(r"^[\s\d\.、）)]+"), ""),
    (re.compile(r"[#*`>|]"), ""),
    (re.compile(r"\s+"), " "),
]

BAD_SUBSTR = ("toc", "&#", "\\", "<", "http", "图", "表", "如下")
# 代码片段和标题粘连的特征符号，命中即丢
BAD_CHARS = ("=", "(", ")", "{", "}", '"', "'", "：", "；")
# 数字编号残留（2.1、第3步）
BAD_PATTERNS = (
    re.compile(r"\d+\.\d+"),
    re.compile(r"第\s*\d+\s*步"),
)


def looks_fragmented(q: str) -> bool:
    """标题被拼了两遍，或前后段落粘在一起。"""
    return bool(re.search(r"(.{6,}?)\1", q))


def clean(text: str) -> str:
    for pat, repl in NOISE:
        text = pat.sub(repl, text)
    return text.strip()


def extract_questions(limit: int = 4000) -> list[dict]:
    """从每个块里抠出问句，并记住它在哪个块、哪个文档、第几块。

    按句末标点切句再筛，不用「从头找问号」的正则——那会跨标题粘连，
    抓出「大模型显存问题面大模型显存问题面1 大模型大概有多大」这种假问句。
    """
    out: list[dict] = []
    for line in open(CHUNKS, encoding="utf-8"):
        c = json.loads(line)
        for piece in re.split(r"(?<=[。！？\n])", c["text"]):
            if not piece.rstrip().endswith(("？", "?")):
                continue
            raw = piece.strip()
            q = clean(raw)
            if not (12 <= len(q) <= 46):
                continue
            if any(b in raw for b in BAD_SUBSTR):
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
            # 必须以中文/字母/数字开头，排除以符号或编号残渣开头的
            if not re.match(r"^[\u4e00-\u9fa5A-Za-z0-9]", q):
                continue
            # 一句话里不该出现两个以上问号
            if q.count("？") + q.count("?") > 1:
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


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = re.sub(r"\s+", "", it["question"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def verify_expected(item: dict, store, k: int = 10) -> list[str] | None:
    """跑一次真实检索，把「同文档被召回的块」作为可接受答案来源。

    为什么必须自校验：问句所在的块常常只是标题+问题，真正答案在后面的块里。
    初版按「本块或后继块」硬编期望值，结果正确的引用被判错。
    这里改成看检索实际能召回什么——同一文档的任意块都算命中，
    如果连一个同文档块都召不回来，说明这题不可解，直接丢掉。
    """
    hits = store.search(item["question"], k=k)
    same_doc = [c.id for c, _ in hits if c.doc_id == item["doc_id"]]
    return same_doc or None


def build_task(item: dict, expected: list[str], num: int) -> dict:
    """一条白银任务。期望引用 = 检索能召回到的同文档块（程序推出，无人为标注）。"""
    return {
        "id": f"silver_{num:03d}",
        "tier": "silver",
        "workspace": "kb",
        "prompt": item["question"],
        "reference_answer": f"答案应来自 《{item['source']}》 的以下块之一：{', '.join(expected[:3])}。",
        "reference_note": "程序生成：问题取自语料原文，期望引用由检索结果推出（同文档可召回块），未经人工核验。",
        "require_explore": False,
        "expects_json": True,
        "expected_skills": ["retrieval-qa"],
        "stage_limits": {"steps": 8, "tokens": 25000},
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
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    # 逐题跑真实检索做自校验：连自己文档的块都召不回来的题直接丢弃
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from kb_lab.store import KBStore  # 延迟导入：不生成时不拖 embedding 模型

    store = KBStore(str(KB_ROOT / "data" / "chroma"))

    # 黄金集里已有的问题不再进白银集，避免重复计分
    gold_path = PROJECT_ROOT / "apps" / "kb_agent" / "eval" / "gold.json"
    gold_q = set()
    if gold_path.exists():
        for t in json.loads(gold_path.read_text(encoding="utf-8")):
            gold_q.add(re.sub(r"\s+", "", t.get("prompt", "")))

    items = dedupe(extract_questions())
    items = [i for i in items if re.sub(r"\s+", "", i["question"]) not in gold_q]

    rng = random.Random(args.seed)
    rng.shuffle(items)

    picked: list[dict] = []
    per_doc: dict[str, int] = {}
    for it in items:
        if per_doc.get(it["doc_id"], 0) >= args.per_doc:
            continue
        per_doc[it["doc_id"]] = per_doc.get(it["doc_id"], 0) + 1
        picked.append(it)
        if len(picked) >= args.limit:
            break

    tasks: list[dict] = []
    dropped = 0
    for it in picked:
        expected = verify_expected(it, store)
        if not expected:
            dropped += 1
            continue
        tasks.append(build_task(it, expected, len(tasks) + 1))
        if len(tasks) >= args.limit:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成 {len(tasks)} 条白银任务 → {args.out}")
    print(f"覆盖 {len(per_doc)} 个文档，丢弃 {dropped} 条不可召回题")
    for t in tasks[:5]:
        meta = t["_meta"]
        print(f"  {t['id']}  {t['prompt'][:40]}  → 期望 {meta['recalled']} 个同文档块")
    print("  …")
    return 0


if __name__ == "__main__":
    sys.exit(main())

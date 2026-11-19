"""在语料 chunks.jsonl 里按关键词检索，出题 / 核验证据用。

为什么需要它：黄金集的每条证据都必须是「原文里逐字存在的一句话」，
出题时得能快速看到某个词在哪些块里、上下文是什么。

用法：
    python scripts/kb_grep.py 低秩 初始化            # 同时包含所有词的块
    python scripts/kb_grep.py --any HYDE HyDE        # 包含任一词
    python scripts/kb_grep.py --doc lora             # 列出 doc_id 含 lora 的所有块
    python scripts/kb_grep.py --id <chunk_id>        # 打印整块原文
    python scripts/kb_grep.py 低秩 --width 120 --max 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

KB_ROOT = Path(os.environ.get("KB_LAB_ROOT", "/mnt/d/kb-lab"))
CHUNKS = KB_ROOT / "data" / "chunks.jsonl"


def load_chunks(path: Path = CHUNKS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("terms", nargs="*")
    ap.add_argument("--any", action="store_true", help="任一词命中即可（默认要求全部命中）")
    ap.add_argument("--doc", help="按 doc_id 子串过滤")
    ap.add_argument("--id", help="打印指定块全文")
    ap.add_argument("--width", type=int, default=80, help="命中处左右各显示多少字")
    ap.add_argument("--max", type=int, default=12)
    args = ap.parse_args(argv)

    chunks = load_chunks()
    if args.id:
        for c in chunks:
            if c["id"] == args.id:
                print(c["text"])
                return 0
        print(f"找不到块 {args.id}", file=sys.stderr)
        return 1

    shown = 0
    for c in chunks:
        if args.doc and args.doc.lower() not in c["doc_id"].lower():
            continue
        text = c["text"]
        if args.terms:
            hits = [t for t in args.terms if t in text]
            if (args.any and not hits) or (not args.any and len(hits) < len(args.terms)):
                continue
            pos = text.find(hits[0])
            lo, hi = max(0, pos - args.width), pos + len(hits[0]) + args.width
            snippet = text[lo:hi]
        else:
            snippet = text[: args.width * 2]
        print(f"[{c['id']}]\n  {snippet.replace(chr(10), ' ')}\n")
        shown += 1
        if shown >= args.max:
            print(f"...（已达 --max {args.max}）")
            break
    if shown == 0:
        print("(无命中)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

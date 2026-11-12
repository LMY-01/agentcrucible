"""JSON 提取与取值。

单独成一个模块是因为 `stages.py`（判定格式合规）和 `evaluate.py`（判定字段）
都要用它，而 evaluate 已经 import stages，再反向 import 会成环。
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict | None:
    """从模型输出里挖第一段可解析的 JSON 对象。

    模型经常把 JSON 包在 ```json 块里，或者前面加一句"结果如下："，
    所以不能直接 json.loads(全文)。先试代码块，再试括号平衡扫描。
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def dig(data: Any, path: str) -> Any:
    """按 a.b.c 取值，取不到返回 None。"""
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur

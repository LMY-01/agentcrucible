"""把 kb-lab 剩余的 qa 题并入黄金集，生成 gold 增量（不覆盖正在评测的 gold.json）。

黄金集扩容的目的是压窄质量分的区间：12 条时一次抖动就能盖过 0.1 的真实改进，
22 条之后中位数才稳得住。

每条的要点（must / min_hit）是从参考答案里挑的可核验短语，不是自己写的领域知识。
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
GOLD = PROJECT / "apps" / "kb_agent" / "eval" / "gold.json"
OUT = PROJECT / "apps" / "kb_agent" / "eval" / "gold_v2.json"
QA = Path("/mnt/d/kb-lab/data/qa.json")

# (qa 索引, id, 要点, 最少命中数)
SPECS = [
    (1, "g14_vague_query", ["召回"], 1),
    (4, "g15_why_chunk", ["信息丢失", "窗口"], 1),
    (5, "g16_eval_dataset_cols", ["question", "ground_truths", "answer", "contexts"], 3),
    (8, "g17_lora_inference", ["不会增加", "不增加", "无额外", "不会有"], 1),
    (10, "g18_paged_idea", ["虚拟内存", "分页"], 1),
    (12, "g19_two_phases", ["Prefill", "Decoding", "预填充", "解码"], 2),
    (13, "g20_hallucination_detect", ["检索", "熵", "采样"], 1),
    (14, "g21_data_defect", ["压缩", "模仿"], 1),
    (15, "g22_zero_shot_cot", ["逐步思考", "step by step", "Let's"], 1),
    (16, "g23_tot_24game", ["5"], 1),
]


def main() -> int:
    tasks = json.loads(GOLD.read_text(encoding="utf-8"))
    qa = json.loads(QA.read_text(encoding="utf-8"))
    existing = {t["id"] for t in tasks}

    added = 0
    for idx, tid, must, min_hit in SPECS:
        if tid in existing:
            continue
        row = qa[idx]
        tasks.append({
            "id": tid,
            "tier": "gold",
            "workspace": "kb",
            "prompt": row["question"],
            "reference_answer": row["reference"],
            "reference_note": f"源自 kb-lab/data/qa.json 第 {idx} 题；要点：{'、'.join(must)}。",
            "expects_json": True,
            "require_explore": False,
            "expected_skills": ["retrieval-qa"],
            "stage_limits": {"steps": 8, "tokens": 25000},
            "checks": [
                {"type": "predicate", "name": "not_refused"},
                {"type": "predicate", "name": "answer_covers", "must": must, "min_hit": min_hit},
                {"type": "predicate", "name": "citations_grounded"},
            ],
        })
        added += 1

    OUT.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    gold_n = sum(1 for t in tasks if t.get("tier") == "gold")
    print(f"新增 {added} 条 → {OUT.name}，总计 {len(tasks)} 条（gold {gold_n} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""质量分：把八阶段结果压成一个能挂 CI 的数字。

为什么不是直接用「通过率」：
  * 5/5 通过率无法区分「干净通过」和「靠撞错纠错过」
  * 通过率对 na 阶段无解，一个任务不考察探查就该和考察它的一样重吗？

权重是**声明式的**，放在这里而不是散在代码里，因为它是口径：
改权重等于改验收标准，必须能被 review 到。

na 阶段剔除后把剩余权重归一化，避免"任务没考察某阶段"被算成扣分。
"""
from __future__ import annotations

from .stages import FAIL, NA, PASS, WARN, Verdict

STAGE_WEIGHTS: dict[str, float] = {
    "answer": 0.40,        # 最终交付物仍然最重要
    "tool_valid": 0.15,    # 调错工具是硬伤，直接决定这轮能不能用
    "args": 0.10,          # 参数不会写，等于工具不会用
    "explore": 0.10,       # 盲猜路径是可靠性问题，线上会放大
    "recovery": 0.10,      # 失败能不能自己爬出来
    "skill_recall": 0.05,  # 有手册却想不起来，属于能力上限问题
    "efficiency": 0.05,    # 花钱，但不是正确性问题
    "termination": 0.05,   # 会不会被预算逼停
}

LEVEL_SCORE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}

DEFAULT_THRESHOLD = 0.90


def task_score(verdicts: list[Verdict]) -> tuple[float, list[str]]:
    """一条任务的加权分。返回 (分数, 拉低分数的阶段名列表)。"""
    total_w = 0.0
    acc = 0.0
    drag: list[tuple[float, str]] = []
    for v in verdicts:
        if v.level == NA:
            continue
        w = STAGE_WEIGHTS.get(v.stage, 0.05)
        total_w += w
        got = LEVEL_SCORE[v.level]
        acc += w * got
        if got < 1.0:
            drag.append((w * (1.0 - got), v.name))
    if total_w == 0:
        return 0.0, []
    # 按失分多少排序，人看报告时最该先看排最前的
    drag.sort(reverse=True)
    return round(acc / total_w, 3), [name for _, name in drag]


def run_score(per_task: list[list[Verdict]], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """整轮的分。per_task 是每条任务的判定列表。"""
    scores = []
    details = []
    all_drag: dict[str, int] = {}
    for verdicts in per_task:
        s, drag = task_score(verdicts)
        scores.append(s)
        details.append({"score": s, "drag": drag})
        for name in drag:
            all_drag[name] = all_drag.get(name, 0) + 1

    quality = round(sum(scores) / len(scores), 3) if scores else 0.0
    worst = min(scores) if scores else 0.0
    return {
        "quality": quality,
        "worst": worst,
        "threshold": threshold,
        "gate": "pass" if quality >= threshold else "fail",
        "per_task": details,
        "drag_counter": dict(sorted(all_drag.items(), key=lambda kv: -kv[1])),
    }

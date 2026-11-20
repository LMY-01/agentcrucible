# P2.1 技能消融与 required/optional 定型报告

日期：2026-11-19  
模型：`bailian/qwen3.8-flash`  
实验任务：g03_hyde_weak、g07_paged_attention、g11_refuse_aime、g12_refuse_count。  
方法：同一批四题，正常技能目录与 `--no-skills` 各跑 3 轮。判定使用答案通过率和逐轮区间，不直接比较两组的技能阶段分。

结果文件：

- 对照组合并：`runs/ablation_skill_with.json`
- 消融组合并：`runs/ablation_skill_without.json`
- 对照组逐轮：`apps/kb_agent/runs/rep_1795150336_0.json`、`_1.json`、`_2.json`
- 消融组逐轮：`apps/kb_agent/runs/rep_1795150756_0.json`、`_1.json`、`_2.json`

## 逐轮结果

| 任务 | 正常技能目录 | 强制不加载技能 | 通过率区间是否重叠 | 最终标注 |
|---|---|---|---|---|
| g03_hyde_weak | 3/3 | 3/3 | 重叠 | `optional_skills: [retrieval-qa]` |
| g07_paged_attention | 3/3 | 3/3 | 重叠 | `optional_skills: [retrieval-qa]` |
| g11_refuse_aime | 3/3 | 3/3 | 重叠 | `optional_skills: [retrieval-qa]` |
| g12_refuse_count | 3/3 | 2/3 | 重叠 | `optional_skills: [retrieval-qa]` |

质量分区间：

| 配置 | 各轮质量分 | 中位数 | 区间 |
|---|---|---:|---|
| 正常技能目录 | 0.992 / 0.984 / 0.984 | **0.984** | **[0.984, 0.992]** |
| 强制不加载技能 | 0.971 / 0.952 / 0.983 | **0.971** | **[0.952, 0.983]** |

两组质量分区间有边界接近但不重叠，然而该分差混入了技能阶段、输出格式和效率判据，不能替代逐题的正确率消融。required 的认定以逐题答案通过率为主。

## 四条结论

### g03 / g07

正常与消融均为 3/3，未加载技能后仍能完成检索、读块、证据引用和正确作答，符合 P1.1 的结论 A。正式降为 optional，不修改 system prompt。

### g11

正常与消融均为 3/3，均能正确拒答。`retrieval-qa` 不是该题答对的必要条件，正式降为 optional。拒答判据保持不动。

消融组有两轮出现“连续 3 步无新增块或工具类型”的效率浪费 FAIL。这是多轮换词检索拒答时的过程信号，答案仍然正确，不影响 optional 认定；本阶段不顺手修改效率判据。

### g12

正常为 3/3，消融入 2/3。唯一失败轮的直接原因是**输出格式 FAIL**：模型直接输出自然语言拒答，未交付可解析 JSON，答案内容阶段因此为 N/A，而不是已证明答案内容错误。

正常组通过率为 100%；消融组逐轮为 100% / 75% / 100%，中位数仍为 100%，区间为 [75%, 100%]，与正常组重叠。按“区间不重叠才算真变化”的纪律，不能把这一轮格式波动认定为 required 证据，因此仍降为 optional，并把格式稳定性列为后续观察项。

## 成本对照

- 正常技能目录：三轮总成本约 $0.0492，平均每任务 $0.00410。
- 强制不加载技能：三轮总成本约 $0.0508，平均每任务 $0.00423。

消融组没有表现出稳定的成本优势，成本不是本次标注决策依据。

## 落地改动

- `apps/kb_agent/eval/gold.json`：四条任务改为 `required_skills: []`、`optional_skills: [retrieval-qa]`。
- `scripts/build_gold.py`：以 `OPTIONAL_RETRIEVAL` 同步这四条的生成规则，避免重新生成时回退。
- 其他任务仍保留 required 候选；未修改 prompt、技能正文、检索或运行时行为。
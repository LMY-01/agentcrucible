# P1.1 技能召回归因报告

日期：2026-11-19  
范围：只读 trace，不修改系统行为。  
方法：用结果 row 的 `steps / tools / tokens` 对照 `run_end` 精确匹配 trace，避免把后来其他 run 误当成本次失败现场。

## 结论总表

| 题目 | 第一个动作 | 检索 / 读块 | 最终结果 | 唯一结论 | 对应修法 |
|---|---|---|---|---|---|
| g03_hyde_weak | 直接检索 | 3 次搜索 / 读 4 块 | 答案正确，证据来自已读块 | A：检索了且答对，技能标宽 | P2.1 消融确认后，将 `retrieval-qa` 降为 optional |
| g07_paged_attention | 直接检索 | 4 次搜索 / 读 6 块 | 答案正确，证据来自已读块 | A：检索了且答对，技能标宽 | P2.1 消融确认后，将 `retrieval-qa` 降为 optional |
| g11_refuse_aime | 直接检索 | 7 次搜索 / 读 0 块 | 正确拒答 | C：拒答，检查拒答判据 | 保留 `states_missing`；检查拒答触发条件与措辞，不强求加载 retrieval-qa |
| g12_refuse_count | 直接检索 | 4 次搜索 / 读 1 块 | 正确拒答 | C：拒答，检查拒答判据 | 保留 `states_missing`；把全称断言改成检索范围内的严谨表述，不强求读取无关块 |

## g03_hyde_weak

- Trace：`apps/kb_agent/runs/20261119T161616__eval-g03_hyde_weak/trace.jsonl`
- 匹配结果：`runs/gold_a_par.json`，5 步 / 12 工具 / 23,494 tokens。
- 第一个动作：两次并行 `kb_search`，没有先加载技能，也没有直接作答。
- 检索过程：
  - Step 1：`HYDE 效果不好 反作用 局限`，`HYDE hypothetical document embedding 假设文档嵌入`，各 `top_k=8`。
  - Step 3：`HyDE 查询转换 假答案 不准确 嵌入空间 不匹配`，`top_k=6`。
  - 读取 `R__recall_strategy-5106c676-0001/0002/0003` 和 `RAG_Failure-57169d68-0031`。
- 答案依据：
  - `...0002` 直接支持“结合微调过的向量化模型时效果没那么好，非常依赖辅助 LLM 能力”。
  - `...0003` 直接支持“小于 52B 的 LLM 生成假答案可能带来负面影响”。
  - 核心引用均来自已读块并通过 `cite_check`；最终 `ok=true`、`refused=false`。
- 唯一结论：**A。实际完成了检索、读块和引用校验，未加载技能不构成失败。** P2.1 做消融确认后降为 optional，不改 system prompt。

## g07_paged_attention

- Trace：`apps/kb_agent/runs/20261119T231309__eval-g07_paged_attention/trace.jsonl`
- 匹配结果：`runs/gold_calibration_missing_full.json`，6 步 / 15 工具 / 35,960 tokens。
- 第一个动作：两次并行 `kb_search`，全程未调用 `load_skill`。
- 检索过程：
  - Step 1：按 `PagedAttention KV cache memory waste` 和 `PagedAttention 显存 浪费 碎片` 检索，各 `top_k=8`。
  - Step 3：按浪费率、内部碎片等中英文术语补查，各 `top_k=8`。
  - 读取 PagedAttention README、LLMs optimize trick、推理框架对比文档共 6 块。
- 答案依据：
  - `PagedAttention__readme-34d1d63b-0003` 直接支持“浪费仅发生在序列最后一个块，浪费率低于 4%”。
  - `LLMs_optimize_trick_no-3aa95a9d-0001` 直接支持“仅浪费不到 4%”。
  - 优化前 60%–80% 的基线也有已读块证据；最终 `ok=true`、`refused=false`。
- 唯一结论：**A。双文档检索、数字证据和引用完整，主答案不是参数知识。** P2.1 做消融确认后降为 optional。

## g11_refuse_aime

- Trace：`apps/kb_agent/runs/20261119T161840__eval-g11_refuse_aime/trace.jsonl`
- 匹配结果：`runs/gold_a_par.json`，5 步 / 9 工具 / 23,852 tokens。
- 第一个动作：直接检索；Step 3 加载了 `refusal-policy`，始终未加载 `retrieval-qa`。
- 检索过程：
  - 共 7 次 `kb_search`，围绕 `o1`、`AIME`、准确率、数学竞赛和评测结果换词检索。
  - 精确术语 `AIME`、`o1` 的 BM25 检索均返回 0 条；其他命中为 OpenAI 兼容服务、ReAct 示例、评测指标定义等无关内容。
  - 没有读取任何可作为 o1/AIME 准确率依据的块。
- 答案依据：最终 `refused=true`、`citations=[]`、`ok=true`，没有编造准确率。这是多轮检索后的主动正确拒答，不是参数知识作答，也不是预算耗尽。
- 唯一结论：**C。拒答正确，应检查拒答判据和触发条件。** 保留 `states_missing`；检查 `refusal-policy` 中“读过三个以上块”的一般条件，不能让“资料缺少具体性能数字”这一明确缺证情形被机械读块要求卡住，也不能为凑技能加载率改成作答。

## g12_refuse_count

- Trace：`apps/kb_agent/runs/20261119T161807__eval-g12_refuse_count/trace.jsonl`
- 匹配结果：`runs/gold_a_par.json`，4 步 / 7 工具 / 18,967 tokens。
- 第一个动作：直接检索；Step 2 加载 `refusal-policy`，未加载 `retrieval-qa`。
- 检索过程：
  - 共 4 次 `kb_search`，围绕知识库文档总数、语料规模、文档清单和元信息换词检索。
  - 读取 `readme-1a14eeb5-0000`；该块只是某个文档自身目录，不是全知识库 manifest 或总数统计。
- 答案依据：最终 `refused=true`、`citations=[]`、`ok=true`，没有编造 `77` 或 `1450`。拒答来自多轮检索和 README 读块，不是参数知识猜测。
- 唯一结论：**C。拒答正确，但全称表述需要更严谨。** 保留“查了什么、找到什么、缺什么”的判据；答案应表述为“本次多轮检索未找到文档总数，需要 manifest/索引统计才能确定”，而不是把检索未命中扩大成绝对不存在。不要为机械满足读块数量而读取无关块。

## 与 P0.3 最新独立 run 的口径区别

本报告按计划分析交接时指定的四条旧失败现场。P0.3 使用最新独立 run 重算后，`skill_attribution` 已变为：

- `g09_refuse_version`
- `g12_refuse_count`
- `g23_tot_24game`
- `g29_ppo_defaults`

两组 ID 不是同一轮结果，即使重叠的 g12 也不是同一次执行：本报告旧 run 为 4 步 / 7 工具 / 18,967 tokens；P0.3 新 run 为 6 步 / 11 工具 / 35,826 tokens。不能用新 WARN 列表替换本报告的旧 trace 结论。

## 交给 P2.1 的输入

1. g03、g07：按结论 A 做正常加载 vs 强制不加载消融；正确率不掉就正式降为 optional。
2. g11、g12：按结论 C 检查拒答策略和技能要求，不把“必须加载 retrieval-qa”凌驾于正确拒答之上。
3. required 的最终认定只由消融结果决定；本报告不把单次行为直接当成 required 证据。
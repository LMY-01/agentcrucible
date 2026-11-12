# kb-agent — 技术文档检索问答

在 `arlab` 通用运行时上落地的**垂直应用**。运行时一行没改，只换了两样东西：工具集、技能集。

```
任务「LoRA 的核心思想是什么？」
  ↓
┌─────────────────────────────────────────────┐
│ arlab 运行时（循环 / 预算 / 工具权限 / trace）│
└─────────────────────────────────────────────┘
   工具: kb_search · kb_read · cite_check        技能: retrieval-qa · refusal-policy
   索引: /mnt/d/kb-lab（1450 个 chunk，77 篇中文 LLM/RAG 技术资料）
```

## 为什么这么切工具

| 工具 | 只允许做什么 | 设计理由 |
|---|---|---|
| `kb_search` | 返回 `chunk_id` + 摘要（160 字）+ 相关度，**不含全文** | 逼 Agent 判断「摘要够不够」。一次把库倒进上下文，既贵又看不出它到底理解了没有 |
| `kb_read` | 按 id 取全文 | 明确的一跳，token 花在哪一目了然 |
| `cite_check(quote, chunk_id)` | 检查这句引文是否在该 chunk 里**逐字**存在 | **这一层的核心**。让 Agent 自己验证引用，而不是靠 prompt 祈祷它别编 |

第三个工具解决的问题：检索类应用最典型的失败不是答错，是**答得像**——把摘要改写成引文、把两段的句子拼成一句、用「一般来说」包装猜测。这些靠人读一遍才能发现。`cite_check` 把它变成一次工具调用。

## 为什么这套判据不需要领域专家

黄金集的标准答案不是「我认为对的答案」，而是「**文档里那句话**」。所以判据全是机械可核验的：

| 谓词 | 判什么 | 需要懂领域吗 |
|---|---|---|
| `citations_grounded` | 每条 `quote` 是否在其 `chunk_id` 的原文里逐字出现 | ❌ 只需要能判断「这句话在不在原文里」 |
| `answer_covers` | 答案是否覆盖预设要点（`must` 命中 N 个） | 🟡 要点从参考答案里挑，不用自己写 |
| `states_missing` | 资料确实没有时，是否老实说明缺失 | ❌ |
| `not_refused` | 资料里有答案时，不许拒答 | ❌ |

这是检索类应用能自动化评测的关键：**引用可溯源，就不需要人来判断内容对不对**。

## 用法

```bash
./run.sh ask "LoRA 里 A 和 B 矩阵为什么要这样初始化？"
./run.sh eval                      # 跑 12 条黄金集
./run.sh eval --limit 3            # 跑前三条
./run.sh eval --no-judge           # 跳过失败归因，省调用
```

复用 `/mnt/d/kb-lab` 的索引和 venv（`~/venvs/kb-lab`），不重复下载 embedding 模型。

## 踩过的坑

**`sentence_transformers` 加载权重前会向 HuggingFace Hub 查 revision。**
WSL 里这个请求不可达，会阻塞到 TCP 超时——表现是「程序卡死」，实测超过 7 分钟没动静。
`faulthandler.dump_traceback_later()` 打栈才看出来卡在 `hf_api.model_info`。

修法：`run.sh` 和 `kb_tools.py` 里都设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，
权重本地已缓存，首跳从几十秒压到 2 秒。

## 已知局限

- 黄金集 12 条，其中 8 条有答案、4 条拒答；标准答案从 kb-lab 已有的 qa.json 迁移，**未经独立复核**
- 只有单跳检索，还没有多跳（「A 和 B 有什么区别」需要分别检索再对比）
- 语料只有 77 篇，规模太小，分数只能横向比较，不能对外宣称准确率

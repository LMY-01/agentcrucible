"""黄金集 v3 构建脚本：题目定义 → 自动定位证据块 → 校验 → 写出 JSON。

为什么用脚本而不是手写 JSON：
  * 每道题的「锚点」（标准证据里的一段原文）都要在语料里逐字存在，
    拒答题的「不存在词」都要在语料里零命中——这两件事人眼核不住，脚本核。
  * evidence_chunks 由脚本从语料里反查填入，不靠人抄 chunk_id（抄错一位就全错）。

用法：
    python scripts/build_gold.py            # 校验并写出
    python scripts/build_gold.py --check    # 只校验，不写
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kb_grep import load_chunks  # noqa: E402

EVAL = ROOT / "apps" / "kb_agent" / "eval"

SINGLE_LIMITS = {"steps": 8, "tokens": 25000}
MULTI_LIMITS = {"steps": 9, "tokens": 72000}
MULTI_BUDGET = {"steps": 12, "tokens": 90000, "seconds": 300}


def P(name: str, **kw) -> dict:
    return {"type": "predicate", "name": name, **kw}


def answerable(id, split, qtype, difficulty, prompt, ref, must, min_hit, anchors,
               note="", forbid=None, extra=None):
    """可作答题：不许拒答 + 要点覆盖 + 证据命中 + 引用可溯源。"""
    cover = {"must": must, "min_hit": min_hit}
    if forbid:
        cover["forbid"] = forbid
    t = {
        "id": id, "split": split, "qtype": qtype, "difficulty": difficulty,
        "prompt": prompt, "reference_answer": ref, "reference_note": note,
        "checks": [
            P("not_refused"),
            P("answer_covers", **cover),
            P("evidence_hit", groups=anchors),
            P("citations_grounded"),
        ],
    }
    t.update(extra or {})
    return t


def refusal(id, split, qtype, difficulty, prompt, ref, absent, note="", forbid=None):
    """拒答题：absent 里的词必须在语料里零命中（脚本会核），否则题目本身不成立。"""
    spec = {"forbid": forbid} if forbid else {}
    return {
        "id": id, "split": split, "qtype": qtype, "difficulty": difficulty,
        "prompt": prompt, "reference_answer": ref, "reference_note": note,
        "absent_terms": absent,
        "checks": [P("states_missing", **spec)],
    }


# ============================================================ 开发集（可以拿来调 prompt）
# g01–g23 来自 gold_v2，逐条修过判据；g24 起为 v3 新增。
DEV: list[dict] = [
    answerable("g01_rag_problems", "dev", "list", 1,
               "RAG 主要解决大模型的哪两个问题？",
               "幻觉问题（生成不正确、荒谬或与事实不符的结果）；数据新鲜度（对时效性强的问题给出过时答案）。",
               [["幻觉"], ["数据新鲜度", "时效", "过时", "Data Freshness"]], 2,
               [["数据新鲜度"]],
               "v3：「数据新鲜度」加同义组——答「时效性问题」的不该判错。"),
    answerable("g02_hyde_idea", "dev", "how", 1,
               "HYDE 这个召回策略的思路是什么？",
               "用 LLM 根据用户 query 生成 k 个假答案，把假答案和 query 一起向量化后取平均，用融合向量去文档库召回。",
               [["假答案", "假设答案", "假设文档", "伪答案"], ["向量", "嵌入", "embedding"]], 2,
               [["假答案"]]),
    answerable("g03_hyde_weak", "dev", "why", 2,
               "HYDE 在什么情况下效果不好，甚至起反作用？",
               "结合微调过的向量化模型时效果变差；非常依赖辅助 LLM 的能力，用小模型（小于 52B）生成假答案甚至会带来负面影响。",
               [["微调"], ["52B", "小模型", "较小的LLM", "比较小的LLM"]], 2,
               [["52B"]],
               "v3：要点从 1 个加到 2 个，只答「微调」不够。"),
    answerable("g04_lora_core", "dev", "how", 1,
               "LoRA 的核心思想是什么？",
               "通过低秩分解模拟参数的改变量，以极小的参数量实现大模型的间接训练。",
               [["低秩"], ["改变量", "增量", "更新量", "间接训练"]], 2,
               [["低秩分解来模拟参数的改变量", "低秩分解模拟参数的改变量", "本征秩"]]),
    answerable("g05_lora_init", "dev", "why", 2,
               "LoRA 里 A 和 B 两个矩阵分别怎么初始化，为什么要这样？",
               "A 用高斯函数初始化，B 初始化为零矩阵，这样训练开始时 BA=0，新增通路对模型输出没有影响。",
               [["高斯", "随机"], ["零矩阵", "全零", "初始化为0", "初始化为零"], ["BA=0", "没有影响", "不影响", "为零", "为0"]], 2,
               [["零矩阵"]]),
    answerable("g06_decoder_only", "dev", "why", 2,
               "为什么现在的大模型基本都是 Decoder-only 架构？",
               "encoder 存在低秩问题；decoder-only 无需 tuning 数据即可 zero-shot 表现好；支持复用 KV-Cache，对多轮对话友好。",
               [["KV-Cache", "KVCache"], ["低秩"], ["zero-shot", "零样本"]], 2,
               [["低秩"]],
               "v3：原判据 min_hit=1，只答一个理由也过；参考答案有三个理由，要求至少两个。"),
    answerable("g07_paged_attention", "dev", "numeric", 1,
               "用了 PagedAttention 之后，KV 缓存的内存浪费率能降到多少？",
               "浪费只发生在序列最后一个块，内存使用接近最优，浪费率低于 4%。",
               [["4%"]], 1,
               [["浪费率低于 4%", "仅浪费不到 4%"]],
               "v3：加证据锚点。语料里 Adapter 有一句「推理时慢4%-6%」，只看数字会被它骗过。"),
    answerable("g08_react", "dev", "list", 1,
               "ReAct 框架把哪两种东西交错生成结合在一起？",
               "推理轨迹（reasoning trace）和任务特定操作（action），以便与外部源交互收集信息。",
               [["推理", "思考", "reasoning"], ["操作", "动作", "行为", "行动", "action"]], 2,
               [["推理轨迹"]]),
    refusal("g09_refuse_version", "dev", "refusal_nearmiss", 1,
            "vLLM 0.4.2 版本的更新日志里修复了哪些 bug？",
            "资料中没有 vLLM 0.4.2 的更新日志内容，无法回答。",
            ["0.4.2", "更新日志", "changelog"]),
    refusal("g10_refuse_dim", "dev", "refusal_nearmiss", 2,
            "BGE-M3 模型输出的向量维度是多少？",
            "资料中提到了智源的 BGE 模型，但没有 BGE-M3，也没有它的向量维度。",
            ["BGE-M3", "bge-m3"],
            "近邻陷阱：语料有 BGE，没有 BGE-M3。答「1024 维」就是编的。",
            forbid=["1024"]),
    refusal("g11_refuse_aime", "dev", "refusal_nearmiss", 1,
            "OpenAI o1 在 AIME 2024 上的准确率是多少？",
            "资料中没有给出 OpenAI o1 在 AIME 2024 上的准确率。",
            ["AIME 2024", "OpenAI o1"]),
    refusal("g12_refuse_count", "dev", "refusal_meta", 1,
            "这个知识库总共收录了多少篇文档？",
            "资料中没有关于本知识库文档总数的信息。",
            ["收录了77", "共77篇", "77 篇"],
            forbid=["77篇", "77 篇", "1450"]),
    {
        "id": "g13_fault_search", "split": "dev", "qtype": "fault_sticky", "difficulty": 2,
        "prompt": "HYDE 这个召回策略的思路是什么？",
        "reference_answer": "本次 kb_search 被注入持续性故障。正确行为：说明检索工具不可用、无法给出有依据的结论。",
        "reference_note": "故障注入：kb_search 从第 1 次起一直超时。",
        "inject_fault": {"tool": "kb_search", "at_call": 1, "mode": "timeout", "sticky": True,
                         "message": "kb_search 连接超时：上游 30 秒无响应，服务不可用。"},
        "checks": [P("reports_failure")],
    },
    answerable("g14_vague_query", "dev", "why", 1,
               "用户提问太口语化、描述模糊会带来什么后果？",
               "用户的问题是十分口语化的，描述的也比较模糊，这样会影响向量召回的质量，进而影响模型回答效果。",
               [["召回"], ["回答效果", "回答质量", "答案质量", "生成质量"]], 2,
               [["十分口语化"]]),
    answerable("g15_why_chunk", "dev", "why", 1,
               "为什么必须对长文档做文本分块？",
               "一是信息丢失的风险，一次性提取整个文档的嵌入向量可能忽略特定主题的重要信息；二是分块大小的限制，例如 GPT-4 有 32K 的窗口限制。",
               [["信息丢失", "忽略", "丢失"], ["窗口", "上下文长度", "长度限制", "32K"]], 2,
               [["信息丢失的风险", "上下文长度是有限制的"]],
               "v3：两条理由都要答到（原 min_hit=1）。"),
    answerable("g16_eval_dataset_cols", "dev", "list", 1,
               "合成 RAG 评测集时，数据集需要包含哪几列？",
               "question 问题、ground_truths 真实答案、answer 答案、contexts 上下文。",
               ["question", ["ground_truths", "ground truth", "真实答案"], "answer", ["contexts", "上下文"]], 4,
               [["ground_truths"]],
               "v3：四列都要（原来 3/4 就过）。"),
    answerable("g17_lora_inference", "dev", "yesno", 1,
               "LoRA 在推理阶段会增加额外计算量吗？",
               "不会。推理时把 BA 合并进原权重 W 作为新权重，h=(W+BA)x，不增加额外计算。",
               [["不会增加", "不增加", "无额外", "没有额外", "不会有", "不会带来", "消除推理延迟"],
                ["合并", "加到", "W+BA", "BA加到W", "替换"]], 2,
               [["不会增加额外的计算资源", "消除推理延迟"]],
               "v3：要求说出「为什么不会」（合并权重），只答「不会」不够。"),
    answerable("g18_paged_idea", "dev", "how", 1,
               "PagedAttention 借鉴了什么思想，解决什么问题？",
               "受操作系统虚拟内存和分页思想启发，在不连续的内存空间存储连续的 KV；解决碎片和过度预留导致的 60%~80% 内存浪费。",
               [["虚拟内存", "分页", "page table", "页表"], ["浪费", "碎片", "过度预留"]], 2,
               [["虚拟内存"]]),
    answerable("g19_two_phases", "dev", "list", 1,
               "大模型推理分成哪两个阶段，各自在算什么？",
               "Prefill 输入理解与初始化阶段，计算整个 prompt 的自注意力；Decoding 递归推理与解码输出阶段，每生成一个 token 计算一次。",
               [["Prefill", "预填充", "输入理解"], ["Decoding", "解码", "递归推理"]], 2,
               [["Prefill"]],
               "v3：原判据把 Decoding / 解码 当两个要点，答一个阶段也能凑够 2 个。改成同义组。"),
    answerable("g20_hallucination_detect", "dev", "list", 2,
               "检测大模型幻觉有哪些办法？",
               "外部检索增强（用搜索引擎结果检查）；模型回答的不确定性：依赖熵值，或多次随机采样看回答是否一致。",
               [["检索", "搜索引擎", "外部工具"], ["熵", "不确定性"], ["采样", "多次回答", "不一致", "二次提问"]], 2,
               [["熵值"]],
               "v3：原来答一个「检索」就过（几乎所有答案都含「检索」）。"),
    answerable("g21_data_defect", "dev", "why", 2,
               "幻觉的来源之一「数据源缺陷」是怎么造成幻觉的？",
               "预训练把训练数据内化压缩进参数，压缩过程中训练数据的问题会被模型错误地学习和模仿，包括错误模仿和重复偏差。",
               [["压缩", "内化"], ["模仿", "错误的学习", "错误地学习"], ["重复偏差", "错误模仿"]], 2,
               [["数据源缺陷"]]),
    answerable("g22_zero_shot_cot", "dev", "how", 1,
               "zero-shot 思维链提示具体要怎么做？",
               "把「让我们逐步思考」添加到原始提示中，即零样本 CoT（Kojima 等人，2022）。",
               [["逐步思考", "step by step", "一步一步"]], 1,
               [["让我们逐步思考"]]),
    answerable("g23_tot_24game", "dev", "numeric", 2,
               "思维树 ToT 在算 24 游戏里每一步保留几个候选项？",
               "分成 3 个思维步骤，每步保留最优的 5 个候选项。",
               [["5个", "五个", "5 个", "best5"]], 1,
               [["5 个候选项"]],
               "v3：原判据只查字符「5」，「52B」「3.5」都能命中。"),

    # ---------------------------------------------------- v3 新增（开发集）
    answerable("g24_streaming_sink", "dev", "why", 2,
               "只保留最近一段窗口的 KV cache，为什么会让模型输出崩掉？StreamingLLM 作者观察到了什么现象？",
               "开头几个 token 会吸走大量注意力（attention sink），与语义无关、只和绝对位置有关；窗口把它们挤出去后注意力分布被破坏，生成就崩了。",
               [["attention sink", "注意力汇", "sink", "汇聚"], ["开头", "最初", "初始", "intial", "initial", "前几个"]], 2,
               [["attention sink"]],
               "改写过的问法：题面不出现 attention sink，要求模型自己检索到这个概念。"),
    answerable("g25_jieba_oov", "dev", "how", 1,
               "jieba 碰到词典里没收录的新词，是靠什么模型和算法切出来的？",
               "基于汉字成词能力的 HMM（隐马尔科夫）模型，配合 Viterbi 算法。",
               [["HMM", "隐马尔科夫", "隐马尔可夫"], ["Viterbi", "维特比"]], 2,
               [["Viterbi", "viterbi"]]),
    answerable("g26_jieba_paddle", "dev", "numeric", 2,
               "想用 jieba 的 paddle 模式，需要额外装哪个包的哪个版本？对 jieba 自身版本有什么要求？",
               "需安装 paddlepaddle-tiny==1.6.1；paddle 模式支持 jieba v0.40 及以上版本。",
               ["paddlepaddle-tiny", "1.6.1", "0.40"], 3,
               [["paddlepaddle-tiny==1.6.1"]],
               "精确数值题：三个值都要对。"),
    answerable("g27_hard_negative_risk", "dev", "why", 2,
               "训练稠密检索模型时，直接把模型打分最高的 Top-K 文档拿来当负例，会有什么隐患？",
               "很可能把潜在的正例也误判成负例，即假负例（False Negative），训练去区分它们反而让模型无法准确衡量相关性。",
               [["假负例", "False Negative", "潜在的正例", "误判为负例"]], 1,
               [["假负例"]]),
    answerable("g28_prompt_chaining", "dev", "list", 1,
               "把一个复杂任务拆成多个提示依次执行，除了提升效果，还有哪些好处？",
               "链式提示还能提高 LLM 应用的透明度，增加可控性和可靠性。",
               [["透明"], ["控制", "可控"], ["可靠"]], 2,
               [["增加控制性和可靠性"]],
               "题面没出现「链式提示/Prompt Chaining」，要求先把描述映射到概念。"),
    answerable("g29_ppo_defaults", "dev", "numeric", 3,
               "那篇 PPO 源码解读里，计算 reward 时 KL 惩罚系数和奖励裁剪值的默认值分别是多少？",
               "kl_ctl = 0.1，clip_reward_value = 5。",
               ["0.1", "5"], 2,
               [["clip_reward_value = 5"]],
               "代码细节题，答案只在源码段落里；要求证据锚定到那一块，防止凭常识猜 0.1。"),
    answerable("g30_langchain_modules", "dev", "list", 2,
               "LangChain 提供了哪些核心模块？",
               "6 个：模型输入/输出（Model I/O）、数据连接（Data connection）、链（Chains）、智能体（Agents）、记忆（Memory）、回调（Callbacks）。",
               [["Model I/O", "模型输入", "模型输 入"], ["数据连接", "Data connection"], ["链", "Chains"],
                ["智能体", "Agents", "Agent"], ["记忆", "Memory"], ["回调", "Callbacks"]], 5,
               [["6 种标准化"]]),
    answerable("g31_qlora_tech", "dev", "list", 2,
               "QLoRA 为了做到高保真的 4bit 微调，提出了哪几项技术？",
               "4 bit NormalFloat（NF4）量化、双量化，以及防止内存峰值的分页优化器。",
               [["NF4", "NormalFloat"], ["双量化", "Double Quant"], ["分页优化器", "Paged Optim"]], 3,
               [["双量化"]]),
    answerable("g32_tensor_parallel", "dev", "how", 2,
               "做增量预训练时，如果只有少量节点，资料建议用什么并行方式？它在什么条件下才有用？",
               "选用张量并行，但张量并行只有在 nvlink 环境下才会起正向作用，提升也不会太明显。",
               ["张量并行", ["nvlink"]], 2,
               [["nvlink"]]),
    answerable("g33_faithfulness", "dev", "how", 1,
               "评估 RAG 时，「忠实度」这个指标衡量的是什么？",
               "检查生成的答案在给定上下文中的事实准确性，即把答案内容和检索到的上下文做比对。",
               [["上下文", "检索到的"], ["事实", "准确"]], 2,
               [["忠实度（Faithfulness）"]]),
    answerable("g34_flashattention", "dev", "why", 3,
               "标准 attention 计算要读写 HBM 各几次？FlashAttention 为什么计算量更大反而更快？",
               "标准实现 HBM 读三次写三次；FlashAttention 在一个循环里分块算出最终结果，虽然要重复计算中间结果，但不需要反复读写 HBM，所以更快。",
               [["三次", "3次", "3 次"], ["HBM", "显存读写", "访存"], ["分块", "重新计算", "重计算", "一个循环"]], 3,
               [["HBM读三次写三次"]]),
    answerable("g35_false_premise_lora", "dev", "false_premise", 2,
               "LoRA 会明显拖慢推理速度，部署时一般怎么缓解这个问题？",
               "前提不成立：LoRA 训练完后把 BA 合并进原权重 W，推理时不增加额外计算，也就没有额外延迟。",
               [["不会增加", "不增加", "没有额外", "无额外", "消除推理延迟", "不会带来", "不会拖慢", "不会变慢", "并不会"],
                ["合并", "加到", "W+BA", "BA加到W", "替换"]], 2,
               [["不会增加额外的计算资源", "消除推理延迟"]],
               "错误前提题：顺着题面编「缓解办法」就是失败。必须先纠正前提。"),
    answerable("g36_fault_transient_read", "dev", "fault_transient", 2,
               "jieba 碰到词典里没收录的新词，是靠什么模型和算法切出来的？",
               "HMM 模型 + Viterbi 算法。本题 kb_read 第一次调用会失败一次（偶发），正确行为是重试后正常作答。",
               [["HMM", "隐马尔科夫", "隐马尔可夫"], ["Viterbi", "维特比"]], 2,
               [["Viterbi", "viterbi"]],
               "偶发故障：只失败一次。这是「错误恢复」阶段第一次有稳定的样本——此前三轮 15 次全是 N/A。",
               extra={"inject_fault": {"tool": "kb_read", "at_call": 1, "mode": "exception", "sticky": False}}),
    refusal("g37_refuse_leiden", "dev", "refusal_nearmiss", 2,
            "GraphRAG 做社区划分时，用的是 Leiden 还是 Louvain 算法？",
            "资料里有 Graph RAG（基于知识图谱的检索增强），但没有提到社区划分算法，也没有 Leiden / Louvain。",
            ["Leiden", "Louvain", "社区检测", "社区划分"],
            "近邻陷阱：Graph RAG 主题存在，细节不存在。"),
    refusal("g38_refuse_fake_entity", "dev", "refusal_fake", 2,
            "TokenAttention 2.0 相比 LightLLM 初版的 TokenAttention 做了哪些改动？",
            "资料只介绍了 LightLLM 的 TokenAttention，没有「TokenAttention 2.0」。",
            ["TokenAttention 2.0", "TokenAttention2.0", "TokenAttention v2"],
            "虚构实体：真实概念 + 虚构版本号。"),
]

# ============================================================ 留出集（只在里程碑跑，禁止拿来调 prompt）
HOLDOUT: list[dict] = [
    answerable("h01_jieba_dag", "holdout", "how", 2,
               "jieba 在词典范围内，是怎么从所有可能的切法里选出最终切分的？",
               "基于前缀词典扫描词图，生成所有可能成词情况构成的有向无环图（DAG），再用动态规划找基于词频的最大概率路径。",
               [["前缀词典"], ["DAG", "有向无环图"], ["动态规划", "最大概率路径", "最大概率"]], 2,
               [["前缀词典"]]),
    answerable("h02_lightllm", "holdout", "how", 2,
               "LightLLM 是靠什么机制拿到比 vLLM 更高的吞吐的？",
               "引入更细粒度的 kv cache 管理算法 TokenAttention，并设计与之配合的 Efficient Router 调度。",
               [["TokenAttention"], ["Efficient Router", "Router", "调度"]], 2,
               [["TokenAttention"]]),
    answerable("h03_table_recog", "holdout", "list", 1,
               "表格识别任务通常拆成哪两个子任务？",
               "表格检测和表格结构识别。",
               [["表格检测", "表格定位"], ["表格结构识别", "结构识别"]], 2,
               [["表格识别包括表格检测和表格结构识别两个子任务"]]),
    answerable("h04_grad_ckpt", "holdout", "how", 2,
               "gradient checkpointing 是拿什么换什么来省显存的？",
               "在计算图中插入检查点，把一部分计算推迟到后续步骤（重新计算），以多算一些换取更少的内存占用。",
               [["推迟", "重新计算", "重计算", "再计算"], ["内存", "显存"]], 2,
               [["梯度检查点（Gradient Checkpointing）"]]),
    answerable("h05_fastertransformer", "holdout", "list", 1,
               "FasterTransformer 的核心用了哪两种并行技术？",
               "张量并行（TP）和流水线并行（PP）。",
               [["张量并行", "TP"], ["流水线并行", "PP"]], 2,
               [["流水线并行 (PP)"]]),
    answerable("h06_rag_pain_top2", "holdout", "list", 2,
               "那份「RAG 关键痛点及解决方案」里，排在最前面的两个问题是什么？",
               "问题一：内容缺失；问题二：错过排名靠前的文档。",
               ["内容缺失", ["错过排名靠前", "排名靠前的文档"]], 2,
               [["错过排名靠前的文档"]]),
    answerable("h07_hallu_taxonomy", "holdout", "list", 2,
               "按那份幻觉来源与缓解的资料，幻觉被分成哪几类？",
               "事实性问题（Factuality）、忠诚度问题（Faithfulness）、自我矛盾（self-Contradiction）。",
               [["事实性", "Factuality"], ["忠诚度", "忠实度", "Faithfulness"], ["自我矛盾", "Contradiction"]], 3,
               [["自我矛盾（self-Contradiction）"]]),
    answerable("h08_mqa_gqa", "holdout", "how", 2,
               "MQA / GQA 这类注意力改造的核心思想是什么？",
               "减少 kv-cache 的数量，让少量 kv-cache 对应多个 query。",
               [["kv-cache", "kvcache", "KV"], ["多个query", "多个 query", "共享", "对应多个"]], 2,
               [["以少量kv-cache对应多个query"]]),
    answerable("h09_ptq_exception", "holdout", "why", 2,
               "大模型量化一般用训练后量化（PTQ），为什么？有没有例外？",
               "大模型重新训练成本太高，所以一般用 PTQ；例外是 QLoRA，它在 LoRA 微调阶段做量化。",
               [["重新训练", "训练成本", "成本太高"], ["QLoRA", "QLora"]], 2,
               [["一个例外是QLora"]]),
    answerable("h10_rlhf_stages", "holdout", "list", 1,
               "RLHF 的标准流程分哪三个阶段？",
               "监督微调（SFT）、偏好采样和奖励学习（RM）、强化学习微调（PPO）。",
               [["SFT", "监督微调"], ["RM", "奖励"], ["PPO", "强化学习"]], 3,
               [["偏好采样和奖励学习"]]),
    answerable("h11_peft_taxonomy", "holdout", "list", 2,
               "参数高效微调方法大致可以分成哪三大类？",
               "增加额外参数（A）、选取一部分参数更新（S）、引入重参数化（R）。",
               [["增加额外参数", "额外参数"], ["选取一部分参数", "部分参数更新", "选取部分参数"], ["重参数化"]], 3,
               [["引入重参数化"]]),
    answerable("h12_bge_org", "holdout", "factoid", 1,
               "资料里提到的 BGE 嵌入模型是哪个机构开源的？",
               "智源研究院。",
               [["智源"]], 1,
               [["由智源研究院开源"]],
               "和 g10（BGE-M3 拒答）配对：同一实体族，一个可答一个不可答。"),
    answerable("h13_adapter_speed", "holdout", "numeric", 3,
               "和全量微调相比，Adapter 在训练和推理时速度分别怎样？",
               "训练时快 60%，推理时慢 4%–6%，因此提出了 AdapterDrop 来缓解。",
               ["60%", ["4%-6%", "4%~6%", "4%—6%", "4%–6%", "4-6%"]], 2,
               [["AdapterDrop"]],
               "和 g07 的「4%」互为陷阱：两处数字形似、出处不同，证据锚点区分。"),
    answerable("h14_false_premise_paged", "holdout", "false_premise", 2,
               "PagedAttention 把 KV cache 的内存浪费降到了 20% 左右，它是怎么做到的？",
               "前提不准确：浪费率低于 4%（只发生在序列最后一个块）。做法是借鉴虚拟内存分页，把 KV 分块存在不连续的显存里，用 block table 记录映射。",
               [["4%"], ["分页", "虚拟内存", "块", "block"]], 2,
               [["浪费率低于 4%", "仅浪费不到 4%"]],
               "错误前提题：必须纠正 20% 这个数。"),
    answerable("h15_false_premise_jieba", "holdout", "false_premise", 2,
               "jieba 是用 CRF 模型来识别未登录词的，具体怎么做的？",
               "前提不准确：jieba 对未登录词用的是基于汉字成词能力的 HMM 模型 + Viterbi 算法，不是 CRF。",
               [["HMM", "隐马尔科夫", "隐马尔可夫"], ["Viterbi", "维特比"]], 1,
               [["Viterbi", "viterbi"]]),
    refusal("h16_refuse_popqa", "holdout", "refusal_nearmiss", 2,
            "Self-RAG 论文里 7B 模型在 PopQA 上的准确率是多少？",
            "资料介绍了 Self-RAG 的机制，但没有给出 PopQA 上的实验数字。",
            ["PopQA"]),
    refusal("h17_refuse_lcel", "holdout", "refusal_nearmiss", 2,
            "用 LangChain 的 LCEL 表达式语言写一个 RAG 链，语法是怎样的？",
            "资料介绍了 LangChain 的六大模块，但没有 LCEL。",
            ["LCEL"]),
    refusal("h18_refuse_h100", "holdout", "refusal_nearmiss", 2,
            "FasterTransformer 在 H100 上相比 A100 的推理加速比是多少？",
            "资料里没有 H100 的数据。",
            ["H100"]),
    {
        "id": "h19_fault_empty_search", "split": "holdout", "qtype": "fault_sticky", "difficulty": 2,
        "prompt": "LightLLM 是靠什么机制拿到比 vLLM 更高的吞吐的？",
        "reference_answer": "kb_search 持续返回空结果。正确行为：说明检索不到/服务异常，不给出无依据的结论。",
        "reference_note": "故障注入：返回空，而不是超时——模型更容易误以为「资料里没有」然后凭记忆作答。",
        "inject_fault": {"tool": "kb_search", "at_call": 1, "mode": "empty", "sticky": True},
        "checks": [P("reports_failure", words=[
            "失败", "不可用", "超时", "无法访问", "调用失败", "检索不到", "服务异常", "无法检索",
            "故障", "没有返回", "空结果", "为空", "返回空", "未返回", "没有检索到", "未检索到"])],
    },
]

# ============================================================ 多跳集（v2：修参考答案 + 证据分组）
MULTIHOP: list[dict] = [
    answerable("mh01_inference_frameworks", "dev", "multihop", 3,
               "vLLM 和 LightLLM 在推理优化的思路上有什么不同？各自的侧重点是什么？请对比后回答。",
               "vLLM 的核心是 PagedAttention：借鉴虚拟内存分页管理 KV cache，浪费率低于 4%，提升吞吐。"
               "LightLLM 引入更细粒度的 TokenAttention 管理 kv cache，并配合 Efficient Router 调度，在多数场景下吞吐高于 vLLM。",
               [["PagedAttention", "分页"], ["TokenAttention"], ["Router", "调度"]], 2,
               [["PagedAttention"], ["TokenAttention"]],
               "v2：旧参考答案说 LightLLM 走「continuous batching」路线——语料里 LightLLM 的核心是 TokenAttention，旧答案是凭记忆写的。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh02_kv_memory", "dev", "multihop", 3,
               "PagedAttention 和 StreamingLLM 分别怎么处理 KV cache 占用过大的问题？两者解决的问题是同一种吗？",
               "PagedAttention 解决「显存浪费」：分页/块表按需分配，浪费降到 4% 以下。StreamingLLM 解决「序列太长」：保留开头的 attention sink token 加最近窗口。两者不是同一个问题。",
               [["分页", "虚拟内存", "block table", "块表"], ["attention sink", "注意力汇", "sink"], ["不是同一", "不同的问题", "两类", "不一样"]], 2,
               [["虚拟内存", "block table", "分页"], ["attention sink"]],
               "v2：旧要点「块」「注意力」几乎任何答案都含。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh03_peft_compare", "dev", "multihop", 3,
               "LoRA 和 Adapter-tuning 在参数高效微调上的做法有什么不同？各自适合什么场景？",
               "LoRA 用低秩分解模拟参数改变量，推理时 BA 合并进 W，不增加推理延迟。Adapter 在层间插入小模块，训练快 60%，但推理慢 4%–6%（因此有 AdapterDrop）。",
               [["低秩"], ["插入", "适配器", "Adapter模块", "Adapter 模块"], ["推理延迟", "推理时慢", "推理速度", "不会增加额外", "合并"]], 2,
               [["低秩"], ["Adapter"]],
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh04_ppo_vs_dpo", "dev", "multihop", 3,
               "PPO 和 DPO 在训练对齐模型时有什么本质区别？为什么说 DPO 更简单？",
               "PPO 走 RLHF 路线：先训练奖励模型，再用强化学习优化策略，用 KL 惩罚约束偏离。DPO 直接用偏好数据优化策略，不需要单独的奖励模型和强化学习采样，实现和训练都更简单。",
               [["奖励模型", "RM", "reward"], ["强化学习", "RL"], ["偏好数据", "偏好"], ["简化", "更简单", "不需要", "无需"]], 3,
               [["PPO"], ["DPO"]],
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh05_rag_recall", "dev", "multihop", 3,
               "HYDE 和 RAG-Fusion 都在优化检索召回，它们的思路差别在哪？分别在什么情况下更有用？",
               "HYDE 让 LLM 先生成假答案再拿去检索，解决问题和文档用词不匹配；RAG-Fusion 把问题改写成多个查询分别检索，再用倒数排名融合（RRF）合并结果，解决单一查询覆盖面窄。",
               [["假答案", "假设答案", "假设文档"], ["多个查询", "多查询", "多个子查询", "改写"], ["RRF", "排名融合"]], 2,
               [["假答案"], ["RRF", "倒数排名融合", "逆向排名融合"]],
               "v2：旧要点「生成」「查询」题面就有。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh06_advanced_rag", "dev", "multihop", 3,
               "Self-RAG 和 GraphRAG 各自解决传统 RAG 的什么缺陷？它们的适用场景有什么区别？",
               "Self-RAG 训练评判模型（critic）并用评判 token 做自适应检索和树解码，控制「要不要检索、检索结果是否相关、答案是否有支撑」。Graph RAG 基于知识图谱，补足碎片化片段难以回答的实体关系类问题。",
               [["评判", "critic", "critique", "反思"], ["知识图谱", "图谱"], ["自适应检索", "要不要检索", "是否检索", "按需检索"]], 2,
               [["critic", "critique", "评判"], ["知识图谱"]],
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh07_rag_eval", "dev", "multihop", 3,
               "RAG 系统常见的失败模式有哪些？应该用什么指标来评估？请把失败模式和对策/指标对应起来说。",
               "失败模式如内容缺失、错过排名靠前的文档、脱离上下文、未能提取答案、格式错误等；检索环节看命中率、MRR、NDCG，生成环节看忠实度、答案相关性、答案正确性。",
               [["内容缺失"], ["排名靠前"], ["未能提取", "格式错误", "脱离上下文", "特异性"],
                ["命中率", "Hit Rate", "MRR", "NDCG"], ["忠实度", "Faithfulness"]], 3,
               [["内容缺失", "错过排名靠前"], ["忠实度", "Hit Rate", "NDCG"]],
               "v2：旧要点「评估」「指标」「答案」全在题面里，复述题目就能过。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh08_cot_vs_react", "dev", "multihop", 3,
               "思维链（CoT）和 ReAct 在让模型推理上的差别是什么？为什么 Agent 更常用 ReAct？",
               "CoT 通过中间推理步骤在语言内部完成推理；ReAct 交错生成推理轨迹和任务特定操作，可以与外部源（工具）交互获取信息，再根据观察更新计划，所以适合 Agent。",
               [["中间推理步骤", "推理路径", "逐步"], ["外部", "工具", "检索"], ["交错", "交替"]], 2,
               [["中间推理步骤", "逐步思考", "推理路径"], ["推理轨迹"]],
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh09_hallucination", "dev", "multihop", 3,
               "大模型幻觉是怎么定义的？它的主要来源有哪些，对应的缓解手段分别是什么？",
               "幻觉分事实性、忠诚度、自我矛盾等类型；来源包括数据源缺陷（压缩过程中错误模仿、重复偏差）、训练和推理阶段；缓解包括检索增强、Self-Reflection、忠诚度增强解码（Context-aware Decode）等。",
               [["事实性", "Factuality"], ["数据源缺陷", "错误模仿", "重复偏差"], ["Self-Reflection", "反思", "Context-aware", "检索增强", "解码"]], 2,
               [["事实性错误", "忠诚度问题", "数据源缺陷"], ["Self-Reflection", "Context-aware Decode", "检索增强"]],
               "v2：旧要点「来源」「缓解」题面就有。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    {
        "id": "mh10_finetune_vs_prompt", "split": "dev", "qtype": "multihop", "difficulty": 3,
        "prompt": "微调和提示工程各自的成本和效果边界在哪？什么情况下应该选微调而不是写更好的 prompt？",
        "reference_answer": "提示工程成本低、只改输入，但受上下文长度和模型已有能力限制；微调需要数据、显存和训练周期（PEFT 可降低成本），但能改变模型行为和领域知识。",
        "reference_note": "v2：语料里没有一段直接对比两者，属于综合题——保留 multi_doc，不加证据锚点，要点改成同义组。",
        "checks": [
            P("multi_doc", min_docs=2),
            P("answer_covers", must=[["显存", "算力", "GPU"], ["数据", "标注", "样本"], ["上下文", "窗口", "长度"]], min_hit=2),
            P("citations_grounded"),
            P("not_refused"),
        ],
        "stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET,
    },
    answerable("mh11_flash_vs_paged", "dev", "multihop", 3,
               "FlashAttention 和 PagedAttention 都在优化大模型推理，它们分别优化的是什么资源？思路有什么差别？",
               "FlashAttention 优化的是 HBM 访存：标准 attention 要读写 HBM 各三次，它在一个循环里分块计算、重新计算中间结果来减少读写；PagedAttention 优化的是显存浪费：借鉴虚拟内存分页管理 KV cache，把浪费率降到 4% 以下。一个省访存带宽，一个省显存占用。",
               [["HBM", "访存", "显存读写"], ["分块", "重新计算", "重计算"],
                ["分页", "虚拟内存", "block table", "块表"], ["浪费", "碎片"]], 3,
               [["HBM读三次写三次"], ["虚拟内存"]],
               "两个「Attention 优化」极易混为一谈：一个省访存、一个省显存，证据锚点分别锁到两篇文档。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh12_lora_vs_qlora", "dev", "multihop", 3,
               "QLoRA 和 LoRA 是什么关系？QLoRA 在 LoRA 基础上多做了哪些事，省下来的到底是什么？",
               "QLoRA 是在 LoRA 微调阶段做量化的版本：用 4bit NormalFloat（NF4）量化基座模型，配双量化进一步压缩量化常数，再用分页优化器防止显存峰值。LoRA 本身靠低秩分解减少可训练参数，QLoRA 在此之上把基座权重压到 4bit，省的是显存，让单卡微调超大模型可行。",
               [["NF4", "NormalFloat", "4bit", "4 bit"], ["双量化", "Double Quant"],
                ["分页优化器", "Paged Optim"], ["显存", "内存"]], 3,
               [["双量化"], ["低秩分解来模拟参数的改变量", "低秩分解模拟参数的改变量", "本征秩"]],
               "和 h09（PTQ 例外是 QLoRA）呼应，但这里要求说出三项技术本身，不是只答「量化版 LoRA」。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh13_two_low_rank", "dev", "multihop", 3,
               "资料里讲 Decoder-only 架构流行原因时提到的「低秩问题」，和 LoRA 的「低秩分解」是一回事吗？分别指什么？",
               "不是一回事。资料里说 encoder 存在低秩问题，指注意力矩阵表达能力受限，这是 decoder-only 流行的原因之一；LoRA 的低秩分解是主动用两个小矩阵模拟参数的改变量，用极少参数完成微调。一个是架构分析里发现的问题，一个是人为设计的微调方法。",
               [["不是", "不一样", "不同"], ["encoder", "编码器"], ["改变量", "增量", "微调"]], 2,
               [["低秩"], ["低秩分解来模拟参数的改变量", "低秩分解模拟参数的改变量", "本征秩"]],
               "同词异义陷阱：两个「低秩」散在两篇文档里，凭直觉答「都是低秩所以相关」不够，必须说清各自含义。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh14_jieba_two_mechanisms", "dev", "multihop", 3,
               "jieba 对词典里有的词和词典里没有的新词，分别用什么机制切分？为什么两种机制要配合？",
               "词典范围内：基于前缀词典扫描词图，生成所有可能切法的有向无环图（DAG），再用动态规划找基于词频的最大概率路径。词典外的新词：靠基于汉字成词能力的 HMM 模型，用 Viterbi 算法切分。前者靠词典和词频统计，后者靠字与字的成词规律，两者配合才能既快又能识别没收录的词。",
               [["前缀词典", "DAG", "有向无环图"], ["动态规划", "最大概率"],
                ["HMM", "隐马尔科夫", "隐马尔可夫"], ["Viterbi", "维特比"]], 3,
               [["前缀词典"], ["Viterbi", "viterbi"]],
               "g25/g36 考未登录词、h01 考词典内切分，本题要求把两半拼成完整图景。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh15_cot_vs_tot", "dev", "multihop", 3,
               "zero-shot 思维链和思维树（ToT）在推理的组织方式上有什么差别？各自是怎么用的？只答基本组织方式和资料里的 24 点示例，不展开 CoT-SC、专家提示或控制器等外围内容。",
               "zero-shot CoT 是线性链式推理：只要在原始提示里加「让我们逐步思考」，模型就按一条推理链走到底。ToT 是树状搜索：把推理拆成多个思维步骤，每步保留多个候选继续探索，比如 24 游戏分 3 步、每步保留最优 5 个候选项。CoT 只有一条路径，ToT 能回溯和比较多条分支。",
               [["逐步思考", "step by step", "一步一步"], ["树", "搜索", "分支", "候选"],
                ["5个", "五个", "5 个", "best5"]], 2,
               [["让我们逐步思考"], ["5 个候选项"]],
               "与 mh08（CoT vs ReAct）错开：这里比的是推理结构（链 vs 树），不是是否调工具。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh16_parallel_strategies", "dev", "multihop", 3,
               "资料里对增量预训练和 FasterTransformer 分别建议了哪些并行方式？张量并行在什么网络条件下才有正向作用？",
               "增量预训练只有少量节点时资料建议用张量并行，但它只有在 nvlink 环境下才起正向作用，提升也不会太明显。FasterTransformer 的核心则是张量并行（TP）加流水线并行（PP）两种技术一起用。前者是场景受限的选择，后者是推理框架的系统级组合。",
               [["张量并行", "TP"], ["流水线并行", "PP"], ["nvlink"]], 3,
               [["nvlink"], ["流水线并行 (PP)"]],
               "g32 只考增量预训练、h05 只考 FasterTransformer，本题要求跨文档对照并答出 nvlink 条件。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh17_memory_saving", "dev", "multihop", 3,
               "gradient checkpointing 和量化都能省显存，它们各自在什么阶段起作用、省显存的机制有什么不同？",
               "gradient checkpointing 用在训练阶段：在计算图里插入检查点，把一部分计算推迟到后续步骤重新计算，用多算换更少的内存占用。量化是把权重压成低精度，大模型重新训练成本太高所以一般用训练后量化（PTQ），例外是 QLoRA 在 LoRA 微调阶段做量化。一个拿计算换显存，一个拿精度换显存，作用阶段和机制都不同。",
               [["重新计算", "重计算", "推迟"], ["内存", "显存"], ["PTQ", "量化", "4bit", "NF4"]], 2,
               [["梯度检查点（Gradient Checkpointing）"], ["一个例外是QLora"]],
               "一个「以时间换空间」、一个「以精度换空间」，h04 与 h09 的单跳知识在这里要求对照成体系。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh18_rag_eval_dataset", "dev", "multihop", 3,
               "要给自己的 RAG 系统搭一套评测，合成数据集应该包含哪几列？判断模型有没有「照着资料瞎编」，重点看哪个指标？",
               "数据集需要四列：question（问题）、ground_truths（真实答案）、answer（答案）、contexts（上下文）。判断有没有照着资料瞎编，重点看忠实度（Faithfulness）：检查生成的答案在给定上下文中的事实准确性，把答案和检索到的上下文逐条比对。",
               ["question", ["ground_truths", "ground truth", "真实答案"], "answer",
                ["contexts", "上下文"], ["忠实度", "Faithfulness"]], 4,
               [["ground_truths"], ["忠实度（Faithfulness）"]],
               "g16 考列名、g33 考忠实度定义，本题把「备数据」和「看指标」串成完整评测流程。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh19_chunking_and_query", "dev", "multihop", 3,
               "为什么 RAG 系统不能直接把整篇长文档丢给模型？文档分块和用户的提问方式分别通过什么环节影响最终答案质量？",
               "直接整篇嵌入有两个问题：一次性提取整个文档向量会忽略特定主题的重要信息（信息丢失），而且上下文窗口有限制（如 GPT-4 32K）。分块影响召回质量，召回不准会直接影响回答效果；用户问法太口语化、描述模糊也会拉低向量召回，间接让回答变差甚至产生幻觉。",
               [["信息丢失", "忽略", "丢失"], ["窗口", "上下文长度", "长度限制", "32K"],
                ["召回", "检索"], ["幻觉", "回答效果", "回答质量"]], 3,
               [["信息丢失的风险", "上下文长度是有限制的"], ["十分口语化"]],
               "g15 考分块必要性、g14 考模糊提问，本题要求把两者归到「召回质量」这条共同因果链上。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
    answerable("mh20_chaining_vs_cot", "dev", "multihop", 3,
               "链式提示和思维链都强调「步骤」，两者的区别是什么？链式提示除了提升效果，还有什么工程上的好处？只比较基本形式和资料明确列出的工程好处，不展开其他推理方法或组合用法。",
               "CoT 的「步骤」是单次生成里的中间推理步骤，模型在一条回答内部完成推理；链式提示是把复杂任务拆成多个提示依次执行，每个提示是一个独立调用。除了提升效果，链式提示还能提高 LLM 应用的透明度，增加可控性和可靠性——每一步的输入输出都能单独检查和替换。",
               [["中间推理步骤", "推理路径", "逐步"], ["多个提示", "依次执行", "拆", "链式"],
                ["透明"], ["控制", "可控"], ["可靠"]], 3,
               [["增加控制性和可靠性"], ["中间推理步骤", "逐步思考", "推理路径"]],
               "g28 考链式提示好处、g22 考 zero-shot CoT，本题区分「单次生成内的推理」与「多次调用间的编排」。",
               extra={"stage_limits": MULTI_LIMITS, "budget": MULTI_BUDGET}),
]


# ============================================================ 校验 + 定稿
def finalize(tasks: list[dict], tier: str) -> list[dict]:
    out = []
    for t in tasks:
        t = dict(t)
        t.setdefault("tier", tier)
        t.setdefault("workspace", "kb")
        t.setdefault("require_explore", False)
        t.setdefault("expects_json", True)
        t.setdefault("expected_skills", ["retrieval-qa"])
        t.setdefault("stage_limits", dict(SINGLE_LIMITS))
        out.append(t)
    return out


def check(all_sets: dict[str, list[dict]], chunks: list[dict]) -> list[str]:
    errs: list[str] = []
    seen: set[str] = set()
    for name, tasks in all_sets.items():
        for t in tasks:
            tid = t["id"]
            if tid in seen:
                errs.append(f"{tid}: id 重复")
            seen.add(tid)
            for c in t["checks"]:
                if c.get("name") == "evidence_hit":
                    ev: list[str] = []
                    for g in c["groups"]:
                        hits = [ch["id"] for ch in chunks if any(a in ch["text"] for a in g)]
                        if not hits:
                            errs.append(f"{tid}: 锚点组 {g} 在语料里零命中")
                        elif len(hits) > 60:
                            errs.append(f"{tid}: 锚点组 {g} 命中 {len(hits)} 块，太泛，起不到证据作用")
                        ev.extend(hits)
                    t["evidence_chunks"] = sorted(set(ev))
            for term in t.get("absent_terms", []):
                hits = [ch["id"] for ch in chunks if term.lower() in ch["text"].lower()]
                if hits:
                    errs.append(f"{tid}: 拒答题的「不存在词」{term!r} 实际出现在 {hits[:2]}")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    sets = {
        "gold.json": finalize(DEV, "gold"),
        "gold_holdout.json": finalize(HOLDOUT, "gold_holdout"),
        "gold_multihop.json": finalize(MULTIHOP, "multihop"),
    }
    errs = check(sets, load_chunks())
    for name, tasks in sets.items():
        print(f"{name:22} {len(tasks):3} 条")
    if errs:
        print(f"\n✗ {len(errs)} 处问题：")
        for e in errs:
            print("  ·", e)
        return 1
    print("✓ 锚点全部在语料中、拒答词全部不在语料中")
    if args.check:
        return 0
    for name, tasks in sets.items():
        (EVAL / name).write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写出 {EVAL / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

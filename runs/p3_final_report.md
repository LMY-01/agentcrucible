# P3 全量三轮基线报告

日期：2026-11-19  
模型：`bailian/qwen3.8-flash`  
范围：gold 38 + gold_holdout 19 + multihop 20 + silver 40，共 **117 条 × 3 轮**。holdout 仅用于本次里程碑验收，未用于调 prompt 或阈值。  
汇总数据：`runs/p3_final_summary.json`。

## 总体结果

| 指标 | 三轮中位数 | 三轮区间 |
|---|---:|---|
| 质量分 | **0.964** | [0.960, 0.975] |
| 通过率 | **92.31%** | [92.31%, 95.73%] |
| 每轮 token 中位数 | 34,260 | [32,407, 34,614] |

三轮总 token：13,470,971。三轮结果文件：`runs/p3_full_r1.json`、`runs/p3_full_r2.json`、`runs/p3_full_r3.json`。

## 分档结果

| 题集 | 条数 | 通过率中位数（区间） | 质量分中位数（区间） | 每轮 token 中位数（区间） |
|---|---:|---|---|---|
| gold | 38 | **94.74%**（94.74%–97.37%） | **0.968**（0.968–0.981） | 26,899.5（25,948.5–29,842） |
| gold_holdout | 19 | **84.21%**（78.95%–84.21%） | **0.924**（0.901–0.927） | 31,916（27,507–34,432） |
| multihop | 20 | **90.00%**（85%–95%） | **0.954**（0.945–0.971） | 54,881（52,415.5–58,192） |
| silver | 40 | **97.50%**（97.5%–100%） | **0.986**（0.984–0.996） | 31,784.5（31,082.5–34,556） |

## 三轮中出现失败的题目

- gold：g10_refuse_dim、g32_tensor_parallel、g37_refuse_leiden
- gold_holdout：h01_jieba_dag、h05_fastertransformer、h08_mqa_gqa、h10_rlhf_stages、h15_false_premise_jieba
- multihop：mh03_peft_compare、mh06_advanced_rag、mh14_jieba_two_mechanisms、mh16_parallel_strategies、mh19_chunking_and_query
- silver：silver_008、silver_030

这些 ID 是三轮并集，不代表每轮都失败；不能据单轮结果调整 holdout。

## 口径说明

- 效率成本只作为 WARN 和三硬指标报告，不再进入质量分；效率浪费仍保留 FAIL。
- 技能召回区分 required/optional；本次四条 optional 标注来自三组消融，样本量仍小。
- `kb_search` 80 字摘要实验已否决，生产默认保持 160 字；多跳 token 仍有优化空间。
- 并行评测现在保留“质量闸门退出码 1 但结果文件完整”的分片，三轮均无结果丢失。
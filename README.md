# arlab — 轻量 Agent Runtime

一个能讲清楚每个设计决定的 Agent 运行时：**skill 渐进披露、工具权限分级、预算耗尽降级、逐步轨迹落盘、断言式评测**。

零第三方依赖（纯标准库），任何 OpenAI 兼容端点都能跑。

```
任务
 └─ Agent.run()
     ├─ system prompt（角色 + 技能目录 + 运行限制）
     ├─ loop: LLM → 解析 tool_calls → 校验 → 执行 → 结果回填
     │    ├─ 预算检查（步数 / token / 时间 / 连续失败）
     │    └─ 每一步写 trace.jsonl
     └─ 无 tool_calls → 最终答案；预算耗尽 → 关工具 + 强制收尾
```

## 三个和"能跑的 demo"不一样的地方

### 1. 技能用渐进披露，不是全塞进 prompt

`skills/*/SKILL.md` 只把 `name` + `description` 放进系统提示，正文由模型自己判断要不要用 `load_skill` 拉。

理由：技能正文动辄几千 token，十几个技能全塞进去既贵又稀释注意力。这是 Anthropic Agent Skills 的机制，也是"skill 体系"在工程上的真实含义——**难点不在格式，在于让模型在正确的时机想起正确的手册**。

```
【可用技能目录】
- csv-profiling: 给 CSV 做数据画像并输出标准化报告。当任务要求分析表格数据…时加载。
- error-recovery: 工具调用失败后的排查流程。当工具返回错误、找不到文件…时加载。
```

### 2. 工具失败被记成指标，不是异常

`Registry.call()` 里任何失败——未知工具名、参数不是合法 JSON、schema 校验不过、权限不足、handler 抛异常——都返回 `ToolResult(ok=False, error=<类型>)`，附一段给人看的中文错误，交给模型自己纠正。

这样"无效工具调用率"变成一个可测的数字。如果失败直接抛异常，这条流水线只能告诉你"崩了"，说不出崩在哪个环节。

至于参数校验，写完单测立刻抓到一个真 bug：`isinstance(True, int)` 在 Python 里成立，导致 JSON Schema 声明 `integer` 的字段会接受 `true`。修法是在类型检查前把 bool 单独挡掉。

### 3. 预算耗尽要降级，不能崩

`Budget.exhausted()` 返回 `steps` / `tokens` / `time` / `tool_failures` 之一。loop 拿到后不是 raise，而是：**关掉工具 + 注入一条"必须收尾"指令 + 再请求一次**。模型被逼着把已有信息整理成答案，`status` 记成 `budget_exhausted`。

线上 agent 超时崩掉会丢掉全部中间结果；这一条是所有玩具实现和能上线的实现之间的分界线。

## 实测

模型 `bailian/qwen3.8-flash`，5 条任务跑两遍（`temperature=0`）：

| 任务 | 结果 | 步数 | 工具调用 | 失败 | token | 耗时 |
|---|---|---|---|---|---|---|
| t01_sum 单文件聚合 | ✅ | 3 | 3 | 0 | 3933 | 12.1s |
| t02_join 双文件关联 | ✅ | 4 | 4 | 0 | 7327 | 18.5s |
| t03_skill 按技能规范出报告 | ✅ | 4 | 4 | 0 | 8759 | 38.0s |
| t04_honest 资料不存在时拒答 | ✅ | 4 | 3 | 0 | 4839 | 12.2s |
| t05_recover 文件名不对后自救 | ✅ | 4 | 4 | 1 | 4862 | 9.3s |

| run | 成功率 | 平均步数 | 工具调用 | 失败 | 无效率 | 平均 token |
|---|---|---|---|---|---|---|
| 第一次 | 5/5 | 3.8 | 19 | 2 | 10.5% | 6132 |
| 第二次 | 5/5 | 3.8 | 18 | 1 | 5.6% | 5944 |

两遍成功率、平均步数完全一致，token 相差 3%。**波动只在工具调用次数上，来源单一**：t04 拒答任务第一遍多撞了一次不存在的文件才转向 `list_dir`，第二遍直接开始探查。

这一点本身就值得记下来：`temperature=0` 不等于确定——网关侧仍有非确定性，所以单个数字不能当结论，得看两遍以上是否稳定。

那 1–2 次失败都是**预期内**的：t05 先读了 `data.csv`（不存在），`read_file` 报错后 `list_dir` 发现真实文件名是 `data_2024.csv` 并纠正——这正是"错误可恢复"的证据，trace 里完整留着现场：

```
step 1  list_dir                     ok    39ms
step 1  read_file  {data.csv}        FAIL   5ms   error=exception
step 2  read_file  {data_2024.csv}   ok    41ms
step 3  run_python                   ok    40ms
step 4  → 答案
```

离线单测 27 条，不联网：`PYTHONPATH=src python -m unittest discover -s tests`

## 跑起来

```bash
./run.sh eval                   # 跑全部评测
./run.sh eval --limit 2         # 只跑前两条
./run.sh run "把工作区里所有 CSV 合并"   # 单任务
./run.sh inspect                # 只看最终要发给模型的 prompt 和工具 schema，不花钱
```

端点通过环境变量指定（`ARLAB_BASE_URL` / `ARLAB_API_KEY` / `ARLAB_MODEL`），默认自动探测 WSL 里的 Windows 主机网关。

```bash
export ARLAB_BASE_URL=http://172.28.208.1:3425/v1
export ARLAB_API_KEY=...
export ARLAB_MODEL=bailian/qwen3.8-flash
```

## 目录

```
src/arlab/
├── llm.py        OpenAI 兼容客户端，含网关地址自动探测和分级重试
├── loop.py       主循环 + 预算降级 + 工具结果回填
├── budget.py     四类预算
├── skills.py     SKILL.md 发现 + 手写 frontmatter 解析 + 渐进披露
├── trace.py      jsonl 落盘与统计（无效工具调用率、工具直方图）
├── evaluate.py   断言式判分 + 汇总
├── cli.py        run / eval / inspect
└── tools/
    ├── registry.py  schema 校验 + 权限分级 + 失败归类
    └── builtin.py   read/write/list/run_python/load_skill，全部锁在工作区内
skills/           两个真在用的技能手册
eval/tasks.json   5 条任务，含 seed 文件和断言
tests/            27 条离线单测
runs/             每次运行一个目录，trace.jsonl 永不覆盖
```

## 已知局限（v0.2 待办）

- **历史压缩没做**：目前只有单条工具结果截断（4000 字符，保留头尾），没有把中间对话摘要成一条。长任务会撞上下文上限。
- **没有子 agent**：任务是线性的，没有"拆解 → 并行 → 汇总"。
- **评测样本太小**：5 条只够验证机制打通，不能当性能结论。要 30 条以上、并且固定 `temperature=0` 复跑才有横向比较的意义。
- **没有多轮对话**：一次 run 一个任务，memory 不跨 run。

## 参考

架构思路参考了 Claude Code 的公开设计和 `Windy3f3f3f3f/claude-code-from-scratch`（MIT，教学仓库）里的章节划分；技能格式沿用 Anthropic Agent Skills 的 `SKILL.md` 约定。代码为本项目独立实现。

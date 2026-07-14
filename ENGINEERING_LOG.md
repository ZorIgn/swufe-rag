# swufe-rag 工程日志

本文件是项目的权威实施记录。每次修改 B 检索、C 生成与溯源、知识库契约、调试接口、评估口径或交付状态时，都应在此追加可复现的记录；聊天内容不作为工程依据。

## 项目基线

- 主仓：<https://github.com/ZorIgn/swufe-rag>
- 默认分支：`main`
- 本轮开发分支：`feature/rag-refinement`
- 稳定基线标签：`local-bc-v1`（此前 B/C 契约版本）
- 当前数据状态：没有真实教务知识库；生产 `data/chunks.jsonl` 保持空占位。
- Demo 数据：`tests/fixtures/chunks.jsonl`，24 条，所有 ID 以 `fixture_` 开头。
- 公共门面：`from swufe_rag.api import retrieve, answer`

## 冻结接口

### B 检索

```python
retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[dict]
```

返回原知识块全部字段和 `score`。结果顺序由内部融合与重排决定，`score` 始终保留为稠密向量余弦相似度，调用方不得据此重新排序。过滤在 Top-K 之前完成，只允许现行、适用学院和适用年级的知识块进入候选集合。

### C 生成与溯源

```python
answer(query: str, chunks: list[dict]) -> dict
```

返回字段严格限定为 `answer_md`、`citations`、`refused`。最高稠密分数低于 `0.35` 或无结果时不调用 LLM，直接返回固定拒答。LLM 不可用抛出 `GenerationUnavailableError`，不能伪装成政策拒答。

### 调试层边界

`app.runtime.RAGRuntime` 和 `/api/debug` 可以增加 `retrieved`、`latency_ms`、`mode` 等观测字段。这些字段只用于调试，不进入 B/C 冻结返回对象。后续 D 模块可以替换路由和前端，但应继续调用 `swufe_rag.api`。

## 2026-07-14：开源实现调研

### Langchain-Chatchat

- 仓库：<https://github.com/chatchat-space/Langchain-Chatchat>
- 研究提交：`49165d6af4438aa7e8a1f71ce276db55f4405151`
- 关注点：稠密与 BM25 集成、分数阈值、CrossEncoder 重排、MMR/多样性。
- 采用方式：只吸收架构模式，按本项目冻结契约重新实现，未复制代码。

### RAGFlow

- 仓库：<https://github.com/infiniflow/ragflow>
- 研究提交：`22dd1ad401d239a3b8a934ca8098937b4c5b58d8`
- 关注点：扩大候选窗口、词项与向量加权、标题和关键词信号、句级引用验证、引用格式修复、上下文预算。
- 采用方式：只吸收架构模式，按本项目数据结构重新实现，未复制代码。

## 2026-07-14：B 检索增强

已完成：

- 领域查询规范化与扩展：挂科映射不及格/重修，保研映射推免/推荐免试，专选映射专业选修。
- 课程代码、条款编号、数字、课程名和显式实体提取。
- 必需实体门：问题中的明确课程或实体在合格候选中不存在时，后续生成必须拒答。
- 标题、条款和正文的加权词法证据。
- 可选 `BAAI/bge-reranker-base` CrossEncoder 重排。
- 测试环境的确定性启发式 reranker。
- 候选去重和 MMR 多样性选择。
- 过滤仍在候选检索前完成，未改变冻结输出结构。
- 生产入口仍只读正式 `data/chunks.jsonl` 和正式索引，不自动加载 fixture。

## 2026-07-14：C 生成与溯源增强

已完成：

- 按问题相关性选择段落，并限制单块和总上下文字符预算。
- 硬性 `0.35` 稠密分数门槛和必需实体门。
- 支持 `【１】`、`[1,2]` 等常见引用格式归一化。
- 引用必须位于事实句句末，每句最多四个来源。
- 数字、课程代码和内容词支持检查。
- 引文从对应知识块原文精确截取，保证 `quote in chunk.text`。
- 首次校验失败只允许一次“修引用、不增事实”的重试；再次失败转为固定拒答。
- 统一处理拒答末尾标点差异。
- 保持 `answer()` 返回结构不变。

### 阈值偏差及修正

早期 Demo 为适配 512 维哈希假编码器曾使用 `0.18`，精确课程代码还可绕过低分门。这与冻结契约不一致。现已改为 128 维 Demo 哈希编码器，并恢复 C 层硬性 `0.35` 门槛；精确信号只参与检索、排序和实体校验，不再绕过拒答阈值。修正后 Demo 三项指标不下降。

## 2026-07-14：Demo 与评估

- 数据：24 条 fixture，覆盖两个学院、三个年级、校级/院级、现行/历史、培养方案、推免、学籍规定和 Markdown 表格。
- 题集：`demo/queries.json` 共 20 题。
- 客户端：`app.demo_llm.DemoGroundedClient`，确定性、无网络、无费用。
- 运行入口：`python -m eval.demo_eval`。
- 当前结果：
  - Recall@5：`1.0`
  - 范围污染数：`0`
  - 拒答准确率：`1.0`

这些结果只验收程序逻辑和接口，不代表真实政策准确率、真实 BGE 分数分布或最终拒答阈值已经验收。

## 2026-07-14：调试 Web

已实现 FastAPI 调试服务和无构建步骤的静态前端：

- `GET /api/debug/health`
- `GET /api/debug/options`
- `GET /api/debug/examples`
- `POST /api/debug/retrieve`
- `POST /api/debug/ask`
- `GET /api/debug/source/{chunk_id}`

工作台包含问题与学院/年级范围、受约束回答、可点击引用、完整知识块原文、召回顺序、余弦分数、范围和耗时。设计上下文记录在 `.impeccable.md`。

浏览器验收发现并修复：

- 回答生成后空状态仍显示：增加全局 `[hidden]` 规则。
- 新问题仍展示上次来源：每次运行前重置来源面板。
- 桌面和 390 px 移动视口均无关键功能缺失或横向内容溢出。
- 正常回答、引用回查、跨学院拒答均通过。
- 浏览器控制台错误和警告为 0。

## 验证证据

2026-07-14 本地验证：

```text
python -m unittest discover -s . -p "test*.py" -v
Ran 60 tests ... OK (skipped=2)

python -m eval.demo_eval
case_count=20
retrieval_recall_at_5=1.0
scope_pollution_count=0
refusal_accuracy=1.0

Python 3.10 AST parse
47 files OK

node --check app/static/debug.js
OK
```

跳过的两项为显式 opt-in 的真实 BGE 下载和真实 FAISS 后端冒烟测试，需设置 `RUN_BGE_SMOKE=1` / `RUN_FAISS_SMOKE=1` 且准备对应依赖后运行。

## 已知限制

- 尚无真实教务文档，不能报告真实检索准确率或政策回答质量。
- Demo 的 `HashingEncoder` 只用于离线调试，其余弦分布不能替代 BGE。
- DemoGroundedClient 不是生成模型，只用于验证上下文、引用和拒答流程。
- 可选 BGE reranker 和真实 BGE 冒烟测试会下载大型模型，默认测试不执行。
- 调试 Web 不是最终学生端产品；认证、限流、审计和正式 D 模块路由尚未实现。
- Debug API 的 URL 仅用于本地联调，不视为最终外部 HTTP 契约。

## 真实知识库到位后的必做项

1. 对 `data/chunks.jsonl` 运行全量契约校验，任何字段或行错误必须阻断建索引。
2. 人工抽查表格、学院、年级、现行状态、页码和 URL。
3. 用正式 BGE 构建 FAISS 索引，核验清单中的模型、维度、块数和源文件 SHA-256。
4. 建立至少 20 条真实检索开发题，要求 Recall@5 不低于 80%、范围污染为 0。
5. 分析真实 BGE 分数分布后校准 `refuse_th`，初始保持 0.35，并记录阈值选择证据。
6. 调整 BM25 分词、候选窗口、权重、reranker 和 MMR 参数。
7. 用真实条款迭代上下文选段、提示词和引用支持规则。
8. 使用独立 30～40 题评估集完成最终验收。
9. 将失败严格归因到数据、切分、过滤、检索、生成或引用层。
10. 原则上不修改 `retrieve()` 和 `answer()` 的公共签名与返回字段。

## 团队协作与交付

- 其他模块从 `origin/main` 创建功能分支，通过 Pull Request 合并。
- A 模块只向正式 `data/chunks.jsonl` 交付真实知识块，不把 fixture 混入生产数据。
- D 模块通过 `swufe_rag.api` 串联 B/C；不要导入内部 pipeline 类作为长期契约。
- 大型原始文档、模型缓存和索引在确定 Git LFS 或发布附件方案前不提交。
- 禁止强制推送 `main`。
- 功能提交：`dbcc1c0 feat: refine RAG pipeline and add debug workbench`。
- 本地合并：`79053ba merge: deliver refined RAG and debug workbench`。
- 远端交付：2026-07-14 已将 `origin/main` 从 `53405f1` 推送到 `79053ba`；本节随紧接的文档提交同步推送。
## 2026-07-14：新电脑续接交付

为迁移到新电脑继续开发，新增：

- `handoff/CONTINUATION_PROMPT.md`：自包含的续接提示词，明确已完成、未完成、冻结接口、测试和 Git 工作流。
- `handoff/NEW_COMPUTER_SETUP.md`：要求从 GitHub 主仓重新 `git clone`，禁止在无历史源码快照中直接 `git init`。
- `handoff/DELIVERY_MANIFEST.md`：定义外部交付 ZIP 的结构和验收要求。

外部交付 ZIP 另外包含当前已提交 HEAD 的 `git archive` 源码快照、原始《西南财大教务RAG问答系统项目计划书.docx》、SHA-256 校验清单和起步说明。计划书保持原样且不提交 Git；正式继续开发始终以 `origin/main` 最新历史为准。

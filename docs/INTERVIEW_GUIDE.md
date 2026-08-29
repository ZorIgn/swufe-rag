# 简历与面试指南

## 项目定位

最准确的一句话：

> 设计并实现了一个面向培养方案与校务制度的证据约束型 SQL + scoped policy RAG 原型：结构化事实走参数化查询，制度正文走范围先行的混合检索，回答经过 operation coverage、fact/evidence 和 comparator 语义校验，数据以可签名评测证明的不可变 release 交付。

这里的关键词是“证据约束”“范围先行”“可审计”“工程原型”。不要把它表述成已上线的学校生产问答系统。

## 30 秒说明

普通 RAG 容易把课程学分这类精确事实交给向量相似度，也容易在不同届别、学院和制度版本之间串数据。我把问题拆成两条路径：课程和培养要求由 typed SQLite tools 查询，制度解释先做 scope filtering，再用 BM25 + dense + reranker 检索。工具结果进入 EvidencePacket，每个输出都绑定具体 operation，每个事实型 claim 都要匹配 value、unit、scope 和 comparator；缺证据或把“至少”说成“至多”都会 fail closed。数据发布则绑定数据库、索引、模型摘要、Git 和受限 holdout 评测，通过 Ed25519 attestation 后才能原子晋级。

## 两分钟展开

可以按“问题—设计—验证—边界”讲：

1. 问题：培养方案包含精确关系和数值，制度文件包含长文本解释；单一向量检索不能同时保证关系正确与文本召回。
2. 设计：用 SQLite 处理课程、学分、模块和进度，用 scope-aware retrieval 处理制度解释；planner 只生成注册过的 typed read-only operations。
3. 证据：tool results 统一进入 EvidencePacket；CoverageGate 检查每个请求输出的 producer operation；ClaimValidator 把最终文字绑定到 fact、evidence、comparator、unit、scope 和时间。
4. 数据：解析阶段保留页码、表格和字段 lineage；低质量抽取进入 review/quarantine；构建产物按内容寻址，禁止数据库与索引拆开漂移。
5. 发布：candidate 在同一受限 holdout 上跑 agent 和 retrieval gates，签名 attestation 绑定 release、模型、Git 和报告，promotion 才更新 active pointer。
6. 边界：公开仓库是合成 fixture；真实学校效果、延迟和成本尚需授权语料、真实查询分布和固定模型环境验证。

## 简历写法

以下 bullet 与仓库可验证事实一致，可按岗位选 2–3 条使用：

- 设计 SQL + scoped policy RAG 双路径，将课程/学分等精确事实交给参数化 SQLite tools，将制度解释交给届别、学院、专业、有效期先行过滤的 BM25+dense hybrid retrieval，避免跨范围证据污染。
- 构建 bounded typed-agent runtime，以 DAG operation lineage、EvidencePacket 和 output coverage gate 约束复合问答；工具超时、部分失败、缺证据和同权威来源冲突均以 typed result fail closed。
- 实现 ClaimAtom 语义校验，将回答中的 subject、predicate、value、unit、scope、temporal 与 equals / at-least / at-most 等 comparator 绑定到 Fact / DerivedFact 和引用证据，覆盖极性翻转与数值边界反例。
- 建立 page/table quality ledger、field lineage、content-addressed release 和 Ed25519 evaluation attestation，把 SQLite、检索 artifact、模型快照摘要、Git provenance 与冻结评测报告绑定后再原子晋级。
- 为单进程和多 worker 场景实现有界 TTL session store 与可选 Redis backend；session 按 principal 和 dataset version 隔离，严格限制消息数、payload 大小，并对损坏或过期状态 fail closed。

不要写具体测试数量、准确率或召回率，除非数字来自当前 commit 的公开 CI 或一份可复现的正式评测报告。

## 架构图怎么讲

~~~text
Question
  |
  v
Understanding -> Normalization(scope/entity/time)
  |
  v
Typed Plan DAG ---------------------------+
  |                                      |
  +-> SQLite academic tools              +-> scoped policy retrieval
  |                                      |
  +---------------- EvidencePacket <-----+
                         |
                         v
              per-operation CoverageGate
                         |
                         v
              deterministic / LLM draft
                         |
                         v
              ClaimAtom semantic binding
                         |
               pass -----+----- fail
                |                 |
             cited answer   one repair / clarify / refuse
~~~

重点不是组件数量，而是三条不变量：

1. scope 在检索前生效；
2. output 与 producer operation 一一对应；
3. 最终 claim 必须能回到兼容语义的 fact 和 evidence。

## 高频追问

### 这到底算不算 RAG

算，但不是“所有问题都向量检索”的 RAG。制度正文需要 retrieval-augmented generation；课程、学分和培养关系是结构化事实，使用 SQL retrieval 更可靠。最终两条路径都进入统一 EvidencePacket 和生成/校验层，因此是 heterogeneous retrieval 的 evidence-grounded RAG。

### 为什么不用纯向量库

向量相似度不能天然表达“2024 级、某学院、某专业、某有效期”的严格过滤，也不适合精确学分、学期和课程关系。当前规模下，先用 SQLite scope filter 保证正确性，再对 scope 内向量做 exact scan。只有真实 benchmark 证明 scope 内扫描成为瓶颈，才考虑带 metadata filter / partition 的 ANN 或向量数据库。

### 你用了 FAISS，为什么在线又不用 FAISS search

FAISS 文件目前是版本化 hybrid artifact 的一部分，用于构建完整性和未来 ANN 演进；在线 dense rank 对 scope 内 <code>vectors.npy</code> 做 deterministic exact dot product。这样不会出现“全局 top-N 被其他专业占满，再过滤后漏掉唯一相关 scoped row”的问题。代价是 <code>O(|scope| * d)</code>，需要用真实 scope-size 和延迟分布决定何时切换。

### 为什么不做 GraphRAG

当前核心问题是严格 scope、精确结构化关系和引用证据，不是长链多跳实体推理。Fact / DerivedFact 形成的是回答审计图，不是图检索系统。若后续问题集中在跨制度的多跳依赖、先修链或政策影响传播，再用带标注查询集比较 graph retrieval 与现有 SQL/RAG，收益成立后再引入。

### Redis 在哪里，解决什么

Redis 是可选共享 session backend，解决多 worker 之间的对话上下文连续性。它不是检索缓存，也没有替代网关的分布式限流。缓存答案的风险更高：key 至少要绑定 dataset version、scope、cohort、as-of、model/index version 和 principal，否则可能返回旧版本或越权答案。先测真实流量和延迟，再决定是否加 cache。

### 为什么默认不用 LLM 生成答案

仓库默认 deterministic synthesizer，保证公开 fixture 可重复，并让 correctness 主要依赖 typed tools 与 validator。外部 LLM 是 request-scoped 可选表达层；即使启用，它生成的 ClaimAtom 也必须通过相同证据校验。这样可以把“语言质量”和“事实正确性”分开评测。

### 怎么防止“至少 10 学分”被说成“至多 10 学分”

仅检查数值 10 不够。ClaimAtom 带 comparator，validator 同时比较 fact 的 predicate/operator、value、unit、条件、scope 和时间。at_least / at_most 只接受方向兼容且单位一致的数值事实；before / after 也按方向验证。方向翻转、同值换 comparator 和单位不一致都有 adversarial tests。

### 复合问题部分失败怎么办

每个 output contract 绑定具体 producer operation IDs。CoverageGate 不按“结果种类全局凑数”，所以另一个同类型 operation 的成功不能覆盖本 operation 的失败。最终响应逐项标记 fulfilled、missing_data、unsupported 或 failed；只有缺少可补证据时允许一次 targeted retrieval。

### PDF 怎么入库，扫描件怎么办

当前 parser 处理文本型 PDF / DOCX。PDF 保留可提取的页码与表格，DOCX 保留文档顺序和表格但不推测页码；结构化 catalog 的字段 lineage 可记录 page / row / cell / span。低质量页面、关键表格失败或 lineage 不完整时进入 review/quarantine，不能静默成为 verified catalog。仓库没有 OCR 引擎或复杂表格后端；扫描件需要外接 OCR/table parser，再经过同一 ledger 和人工复核。

### MCP 做到什么程度

实现的是 typed MCPAdapter：它复用 ToolRegistry、schema、timeout、read-only policy 和 executor。没有实现 MCP server transport、远程发现、会话或认证，因此只应说“transport-independent adapter contract”，不能说“部署了 MCP server”。

### 怎么证明发布版本真的通过评测

build 只产生 candidate，不会更新 active pointer。promotion-mode runner 从 candidate 和 restricted holdout 派生所有输入，拒绝参数覆盖，并要求 evaluator 与 candidate 是同一 clean Git commit。两份报告按固定 policy 重新计算 gates，然后用 Ed25519 签名。promotion 和 runtime 都会验证签名与 release/holdout/model/Git/report 的完整绑定。

### 评测为什么不能说 100% 准确

公开 fixture 很小且人工构造，100% 只说明这些 contract cases 没回归，不代表真实中文提问分布、错别字、歧义、专业别名、跨版本日期或学校语料效果。正式结论需要脱敏真实 query set、 untouched holdout、错误分类和置信区间。

### 这是 production-ready 吗

核心 correctness 和 release contracts 是 production-oriented，但仓库本身不是完整 production deployment。正式上线还需要授权语料、真实 holdout、固定模型与 benchmark、组织 SSO、网关全局限流、Redis 运维、secret rotation、监控告警、SBOM/scanner 和数据治理审批。

## 当前限制与优化触发条件

| 当前边界 | 何时优化 | 推荐方向 |
| --- | --- | --- |
| scoped dense exact scan | p95 随 scope 规模明显超预算 | partitioned ANN、metadata filter、FAISS IDSelector 或向量库 |
| 无 retrieval/result cache | 重复查询比例和模型/检索延迟可量化 | version/scope/principal-aware Redis cache |
| 进程内 rate limiter | 多实例或公网流量 | gateway / service-mesh 全局限流 |
| 无 OCR/table backend | 扫描件和复杂跨页表格占比高 | OCR + table parser，保留 quarantine 和人工 ledger |
| tiny public fixture | 准备真实质量结论 | 脱敏 query set、restricted holdout、error taxonomy |
| 外部 LLM 可选 | 需要更自然表达或复杂理解 | 固定模型快照，单独评测理解与表达，不放宽 validator |
| 无 GraphRAG | 出现稳定多跳实体关系需求 | 先构造对照集，再做 graph/SQL/RAG ablation |
| 无前端 | 面试需要产品交互展示 | 保持 API contract，另做薄 UI；不改变核心可信链路 |

## 现场演示顺序

1. 运行 <code>uv run python -m scripts.run_demo</code>。
2. 指出第一个问题来自 SQL requirement tool，不依赖相似度猜学分。
3. 指出第二个回答同时展示 course code、学分、学期和 citation。
4. 指出第三个问题请求实时选课数据，系统返回 unsupported 而不是拿培养方案冒充。
5. 打开一个 claim/comparator adversarial test 和一个 coverage-lineage test。
6. 最后展示 release attestation tests，说明 correctness 如何进入交付链路。

不要现场承诺下载模型或使用学校正式数据。稳定、可解释的 synthetic vertical slice 比一个依赖私有资产的失败演示更有说服力。

## 面试前自检

- GitHub main 与本地演示 commit 一致，CI 为绿色。
- README 的一键演示在干净 clone 中可运行。
- 能画出 SQL 与 policy retrieval 的分流。
- 能解释 scope-before-ranking、operation lineage 和 comparator binding。
- 能明确区分 public fixture、restricted holdout 和 official corpus。
- 能解释 Redis、FAISS、GraphRAG、OCR、MCP 的已实现范围与采用条件。
- 不把 deterministic fixture 指标说成真实学校效果。
- 不把 static bearer、进程内 limiter 或可选 Redis session 说成完整生产安全平台。
- 准备一个失败案例：跨 scope、方向翻转、部分 operation 失败或未签名 release。

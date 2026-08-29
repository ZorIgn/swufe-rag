# SWUFE Academic RAG

[![CI](https://github.com/ZorIgn/swufe-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/ZorIgn/swufe-rag/actions/workflows/tests.yml)

一个面向培养方案、课程规则和校务制度的证据约束型问答原型。项目的重点不是“让大模型记住学校资料”，而是把不同性质的问题放进不同的可信执行路径：

- 课程、学分、学期和模块要求由参数化 SQLite 工具查询；
- 制度解释先按届别、学院、专业、有效期和主题收窄范围，再执行 BM25 / dense 检索；
- 每个事实保留字段级 lineage，每个回答声明绑定的 fact、comparator 和 evidence；
- 缺范围、缺证据、来源冲突或语义绑定失败时，系统澄清或拒答，不生成貌似完整的答案。

仓库只包含合成测试夹具，不包含学校正式语料、受限评测集、学生数据或模型权重。因此，本项目可验证的是工程契约和失效保护，不是学校真实场景下的准确率、召回率、延迟或成本。

## 一分钟体验

环境要求：Python 3.10–3.12 和 [uv](https://docs.astral.sh/uv/)。

~~~powershell
uv sync --locked --extra dev
uv run python -m scripts.run_demo
~~~

演示会在临时目录构建一份完全合成的 SQLite 数据库，并通过真实运行时执行三个场景：结构化培养要求、课程明细和对实时选课数据的边界拒答。它不下载模型、不调用外部 LLM，也不会写入仓库数据目录。机器可读输出：

~~~powershell
uv run python -m scripts.run_demo --json
~~~

## 核心链路

~~~text
RawQuestion
  -> UnderstandingDraft
  -> NormalizedQuery + explicit scope
  -> bounded typed ExecutionPlan
  -> read-only SQL / scoped policy retrieval
  -> EvidencePacket
  -> ClaimAtom + comparator validation
  -> cited FinalAnswer
~~~

运行时是一个有界状态机：最多执行一次针对缺失证据的定向修复，不执行模型生成的 SQL 或 Python。复合问题的每项输出都绑定到具体 producer operation；同类型工具的成功结果不能替另一项失败输出“凑齐覆盖”。

### 为什么不是纯向量 RAG

| 问题类型 | 执行路径 | 原因 |
| --- | --- | --- |
| 课程、学分、模块、学期 | 参数化 SQLite 工具 | 需要精确过滤、关系约束和可验证数值 |
| 培养进度、可行性 | 只读 DAG 工具 + DerivedFact | 需要显式依赖、差额和规则计算 |
| 制度解释 | scope-aware BM25 / dense hybrid | 需要从正文中找解释，同时阻止跨届别、学院或版本污染 |
| 实时开课、名额、成绩、毕业审核 | 明确不支持 | 仓库不接教务实时系统，不能用培养方案替代实时事实 |

GraphRAG 不是当前问题的必要组件：仓库中的 Fact / DerivedFact 是回答证据图，不是把语料建成图数据库后执行图检索。若未来问题演化为跨文件、多跳实体关系推理，应先用真实查询集证明图检索的收益。

## 已实现的工程边界

### 数据可信

- PDF 解析保留可获得的页码与表格；DOCX 保留文档顺序和表格但不猜测页码；chunk 绑定来源 SHA-256 与抽取质量，结构化字段另外保存 page / row / cell / span lineage；
- 低质量页面或表格进入 warning / quarantine / review-required，不会被静默写成可信结构化事实；
- catalog materialization 要求字段级来源、行列位置和审核状态；
- reviewer ledger 是可验证输入记录，不代表仓库已经实现身份签名、双人复核或不可篡改审批系统。

当前依赖不包含 OCR 引擎或复杂表格识别后端，因此不能把项目描述为“任意扫描 PDF 全自动入库”。

### 检索可信

- 候选集在排序前应用时间、版本、届别、学院、专业和主题 scope；
- hybrid 合约包含 BM25、dense、RRF、CrossEncoder relevance gate 和 MMR；
- dense 运行时对 scope 内向量执行确定性的 NumPy exact scan，代价为 <code>O(|scope| * d)</code> 点积加 <code>O(|scope| log |scope|)</code> 排序；
- FAISS 文件属于版本化检索 artifact 和后续 ANN 载体，当前请求路径不调用 FAISS ANN search；
- 缺模型快照、维度不一致、索引 hash 漂移或 evidence-state 不一致时 readiness 失败。

是否引入向量数据库或 ANN，应由真实语料规模、scope 大小和 p50/p95 benchmark 决定，而不是为了增加技术名词。

### 回答可信

- ClaimAtom 比较 subject、predicate、value、unit、conditions、scope、temporal 和 comparator；
- <code>equals</code>、<code>at_least</code>、<code>at_most</code>、<code>before</code>、<code>after</code> 等方向性语义按事实类型 fail closed；
- claim 必须绑定可追溯 fact 和 evidence，来源冲突或证据不足时拒绝输出确定性结论；
- 文档文本只作为数据，不能修改工具 schema、调用函数、生成 SQL 或覆盖系统约束。

### 发布可信

一次可运行知识库被定义为一个 content-addressed release，SQLite、检索 artifact、数据 manifest、模型快照摘要和 Git provenance 不能拆开替换。正式晋级流程是：

~~~text
clean Git candidate build
  -> restricted frozen holdout agent evaluation
  -> restricted frozen holdout retrieval evaluation
  -> fixed promotion-policy validation
  -> Ed25519 evaluation attestation
  -> atomic active.json promotion
  -> runtime re-verification
~~~

<code>scripts.build_all --release-tier production</code> 会直接拒绝。只有签名评测证明与候选 release、受限 holdout、模型、评测代码 commit 和两份报告完全绑定后，<code>scripts.promote_release</code> 才能更新 active pointer。

## 本地开发

### 运行测试

~~~powershell
uv sync --locked --extra dev --extra ingest
uv run python -m ruff check agent academic app evidence generation ingest query retrieval storage scripts eval tests/canonical
uv run python -m mypy agent academic app evidence generation ingest query retrieval storage scripts eval
uv run python -m pytest -q tests/canonical
~~~

CI 在 Python 3.10 和 3.12 上执行 lint、mypy、compile、合成数据构建、数据完整性校验、agent fixture evaluation、lexical / deterministic hybrid evaluation、完整 canonical tests 和 Docker 依赖 smoke。公开 hybrid fixture 使用确定性本地 encoder / reranker，只验证融合、scope、hard negative、provenance 和失效合约，不代表真实模型效果。

### 启动 HTTP 服务

先从公开合成 fixture 生成一个可保留的演示数据库，再以 lexical 诊断模式启动：

~~~powershell
uv run python -m scripts.run_demo --database-out data/demo.sqlite3
$env:SWUFE_ACADEMIC_DATABASE="data/demo.sqlite3"
$env:SWUFE_RETRIEVAL_MODE="lexical"
uv run python -m app.server
~~~

API 文档位于 <code>http://127.0.0.1:8000/docs</code>。主要接口：

- <code>GET /options</code>
- <code>POST /ask</code>
- <code>GET /source/{chunk_id}</code>
- <code>GET /academic-audit/options</code>
- <code>POST /academic-audit</code>
- <code>GET /health/live</code>
- <code>GET /health/ready</code>

<code>agent.mcp.MCPAdapter</code> 是与 HTTP 共用 ToolRegistry 的 typed adapter，不包含 MCP server、transport 或远程部署能力。

## 从候选构建到签名晋级

学校数据、受限 holdout 和本地模型路径均由部署方提供。以下命令展示合约，尖括号内容不是仓库内置资源。

~~~powershell
uv sync --locked --extra retrieval
uv run python -m scripts.build_all --catalog data/released/curriculum_catalog.json --sources data/released/sources.csv --chunks data/released/chunks.jsonl --source-review data/released/source_review.csv --evidence-review data/released/evidence_review.csv --source-root data/released/raw --retrieval-mode hybrid --embedding-model <local-embedding-snapshot> --reranker-model <local-reranker-snapshot> --holdout-manifest <restricted-holdout/manifest.json>
~~~

候选输出中的 <code>release_id</code> 决定后续 manifest 路径：

~~~powershell
uv run python -m eval.run_agent_eval --candidate-release-manifest artifacts/releases/<release-id>/release_manifest.json --holdout-manifest <restricted-holdout/manifest.json> --output <restricted-reports/agent.json>
uv run python -m eval.run_retrieval_ablation --candidate-release-manifest artifacts/releases/<release-id>/release_manifest.json --holdout-manifest <restricted-holdout/manifest.json> --output <restricted-reports/retrieval.json>
$env:SWUFE_RELEASE_ATTESTATION_PRIVATE_KEY="<base64-ed25519-private-key>"
uv run python -m scripts.create_eval_attestation --candidate-release-manifest artifacts/releases/<release-id>/release_manifest.json --agent-report <restricted-reports/agent.json> --retrieval-report <restricted-reports/retrieval.json> --issuer <issuer-name> --output <restricted-reports/attestation.json>
$env:SWUFE_RELEASE_ATTESTATION_PUBLIC_KEY="<base64-ed25519-public-key>"
uv run python -m scripts.promote_release --release-id <release-id> --attestation <restricted-reports/attestation.json> --trusted-issuer <issuer-name>
~~~

完整约束见 [release 合约](docs/RELEASES.md) 和 [评测合约](docs/EVALUATION.md)。

## 部署边界

<code>SWUFE_DEPLOYMENT_MODE=production</code> 会强制认证，并禁止加载未签名的 active release。运行时还需要配置可信 attestation 公钥和 issuer。内置 static bearer 适合受控演示；正式公网部署应由网关或可信 principal resolver 接入组织 SSO。

模型身份由 release 中的目录摘要绑定；<code>SWUFE_EMBEDDING_MODEL</code> 和 <code>SWUFE_RERANKER_MODEL</code> 只负责把同一份摘要匹配的快照定位到当前主机或容器路径。路径可变，模型内容不能漂移。

默认 session store 是有界、带 TTL 的进程内实现。设置 <code>SWUFE_SESSION_BACKEND=redis</code> 和 <code>SWUFE_REDIS_URL</code> 后，可使用共享 Redis session；Redis 不负责检索缓存，也没有实现分布式 rate limiter。缓存若加入，key 必须包含 dataset version、scope、as-of、模型 / index 版本和 principal，避免旧数据或越权结果复用。

源码环境启用 Redis 前需执行 <code>uv sync --locked --extra redis</code>；Docker 镜像已经包含该 optional dependency。

当前 rate limit 和 concurrency gate 是单进程边界；多实例部署需要在网关层提供全局限流。debug 响应默认关闭，只有已认证 admin、请求 <code>debug=true</code> 且 <code>SWUFE_ENABLE_DEBUG_RESPONSES=true</code> 时才返回受限执行信息。

配置项见 [.env.example](.env.example)，容器示例见 [docker-compose.yml](docker-compose.yml)，安全说明见 [docs/SECURITY.md](docs/SECURITY.md)。

## 仓库导航

~~~text
academic/    SQLite projection、scope-aware repositories、typed tools
agent/       bounded runtime、planner execution、coverage、session、MCP adapter
app/server/  FastAPI contract、auth/debug/resource boundaries
evidence/    Fact、DerivedFact、EvidencePacket、ClaimAtom、provenance
generation/ deterministic / optional LLM synthesis and claim validation
ingest/      source parsing、quality ledger、catalog review/materialization
query/       understanding、normalization、scope and execution planning
retrieval/   lexical、scoped dense、RRF、reranker、MMR、artifact validation
storage/     strict JSON、Git provenance、content-addressed release、attestation
eval/        diagnostic fixture evaluation and promotion-policy validation
scripts/     build、verify、demo、attest and promote commands
tests/       synthetic canonical contract suite
docs/        architecture、security、evaluation、release and interview guide
~~~

## 如何在简历和面试中描述

推荐定位是“证据约束、可审计的 SQL + scoped policy RAG 工程原型”，不是“已上线的学校生产知识库”。可验证亮点、常见追问和回答边界整理在 [面试指南](docs/INTERVIEW_GUIDE.md)。

代码使用 [MIT License](LICENSE)。学校正式资料、受限 holdout、学生数据和第三方模型不随代码授权，详见 [DATA_NOTICE.md](DATA_NOTICE.md)。

README 中的 build、eval、demo 和 promotion 命令面向源码 checkout；构建出的 wheel 只包含运行时库，不是数据、评测夹具和运维 CLI 的完整分发物。

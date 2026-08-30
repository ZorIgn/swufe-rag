<div align="center">

# 🎓 SWUFE 教务智能问答

**证据可追溯的 SQL + RAG 教务知识系统**

让课程、学分、培养要求与制度问答回到原文件、原页码和可复核证据。

[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-structured%20facts-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Retrieval](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense-6C5CE7)](docs/RETRIEVAL.md)
[![CI](https://github.com/ZorIgn/swufe-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/ZorIgn/swufe-rag/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[项目简介](#-项目简介) · [核心能力](#-核心能力) · [系统架构](#️-系统架构) · [快速开始](#-快速开始) · [HTTP API](#-http-api) · [数据构建](#-数据构建与发布) · [测试](#-测试与验证)

</div>

---

## 📖 项目简介

**SWUFE 教务智能问答**面向西南财经大学培养方案、课程规则与校务制度查询，将结构化课程数据和非结构化制度文本组织到同一条证据链中。

课程、学分、学期和模块要求通过类型化 SQLite 工具精确查询；培养目标、办事规则和制度条款通过作用域感知的混合检索获取上下文；学业规划类问题则组合两条路径，生成可回溯到事实、计算过程和来源页码的回答。

| 信息类型 | 处理方式 | 典型问题 |
| --- | --- | --- |
| 课程、代码、学分、学期、性质 | 参数化 SQLite 工具 | “2024 级人工智能专业的离散数学多少学分？” |
| 毕业要求、模块最低学分 | SQLite + DerivedFact | “专业选修模块还差多少学分？” |
| 培养目标、制度条款、办事规则 | 作用域检索与 RAG | “大学英语达到什么条件可以免修？” |
| 课程事实与制度解释的组合问题 | SQL + RAG | “修完这些课程后还需要满足哪些要求？” |

仓库内置合成示例数据，可直接运行完整问答链路；业务资料、模型快照和评测数据通过版本化数据包接入。

> 💡 一句话概括：**模型负责理解和表达，程序负责查询、计算、引用与校验。**

## ✨ 核心能力

- 🧭 **自然语言查询规划** —— 将问题归一化为届别、学院、专业、学期、课程性质、主题和时间等结构化条件，并生成带依赖关系的类型化执行计划。
- 🗃️ **结构化事实查询** —— 课程、学分、培养模块和毕业要求通过预定义的只读 SQLite 工具查询，计算结果保留输入事实和推导过程。
- 🔎 **作用域混合检索** —— 按届别、学院、专业、主题、生效时间和版本收窄资料，再使用 BM25、稠密向量、RRF、CrossEncoder 和 MMR 检索制度正文与表格脚注。
- 🔗 **证据化回答** —— 结构化事实和检索片段统一进入 <code>EvidencePacket</code>，回答返回来源标题、证据片段、物理页码和引用定位。
- ✅ **回答语义校验** —— 对事实主体、属性、数值、单位、条件、时间和比较关系进行绑定，检查“等于”“至少”“至多”等方向性语义。
- 📋 **学业进度计算** —— 根据已修课程计算模块完成度、剩余学分与规则满足情况，并保留 <code>DerivedFact</code> 的输入事实和计算过程。
- 📚 **数据摄取与审核** —— 解析 PDF / DOCX，保留页面、表格和字段位置，通过质量账本、来源登记和审核记录生成结构化目录。
- 📦 **版本化数据发布** —— 使用内容寻址版本绑定 SQLite、检索产物、数据清单、模型摘要和代码来源，根据评测结果与签名证明完成版本晋级。
- 🌐 **统一服务接口** —— FastAPI 提供问答、学业审计、来源查看和健康检查接口，支持请求级模型 Key、内存或 Redis 会话以及 Docker 部署。

## 🏗️ 系统架构

~~~mermaid
flowchart LR
    Q["用户问题"] --> U["Question Understanding"]
    U --> N["实体归一化与作用域"]
    N --> P["Typed DAG Planner"]

    P --> T["Read-only ToolRegistry"]
    T --> S[("SQLite<br/>课程与培养要求")]
    T --> R["Scoped Policy Retrieval"]

    R --> B["BM25"]
    R --> D["Dense"]
    B --> F["RRF Fusion"]
    D --> F
    F --> X["Reranker + MMR"]

    S --> E["EvidencePacket"]
    X --> E
    E --> G["Answer Synthesizer"]
    G --> V["Claim + Citation Validator"]
    V --> A["答案 + 引用 + 原页定位"]
~~~

一次查询依次经过：

1. **问题理解**：识别课程、培养要求、制度解释或学业规划意图。
2. **实体归一化**：统一届别、专业、课程、学期和主题表达。
3. **执行规划**：生成类型化操作及其依赖关系。
4. **工具执行**：调用参数化 SQLite 工具或作用域检索服务。
5. **证据汇总**：将事实、推导结果、检索片段和引用整理为 <code>EvidencePacket</code>。
6. **回答生成**：基于证据组织自然语言答案。
7. **结果校验**：核对事实、数值、单位、比较关系、操作覆盖与引用。

## 🛠️ 技术栈

| 领域 | 选型 |
| --- | --- |
| 服务端 | Python 3.10–3.12、FastAPI、Pydantic、Uvicorn |
| 结构化知识 | SQLite、参数化查询、类型化只读工具 |
| 检索 | BM25、稠密向量、RRF、CrossEncoder 重排序、MMR、FAISS 检索产物 |
| Agent Runtime | 类型化状态、DAG 计划、ToolRegistry、覆盖检查与会话 |
| 证据与生成 | EvidencePacket、Fact、DerivedFact、ClaimAtom、引用校验 |
| 数据摄取 | PDF / DOCX、页面与表格质量账本、结构化目录物化 |
| 发布 | SHA-256 内容寻址版本、Ed25519 评测签名 |
| 模型接口 | OpenAI-compatible API、请求级 BYOK |
| 会话 | 有界内存会话、可选 Redis |
| 质量保障 | Pytest、pytest-cov、Ruff、mypy、GitHub Actions |
| 部署 | Docker、Docker Compose |

## 🚀 快速开始

环境要求：Python 3.10–3.12 和 [uv](https://docs.astral.sh/uv/)。

以下命令以 Windows PowerShell 为例。

### 1. 安装

~~~powershell
git clone https://github.com/ZorIgn/swufe-rag.git
cd swufe-rag
uv sync --locked --extra dev
~~~

### 2. 运行合成演示

~~~powershell
uv run python -m scripts.run_demo
~~~

演示会创建一份合成培养方案数据库，依次查询模块最低学分、课程学分与学期，并展示带引用的回答。机器可读输出：

~~~powershell
uv run python -m scripts.run_demo --json
~~~

输出示例：

~~~text
问题：2024级测试专业X的测试算法多少学分，在哪个学期开设？

回答：测试算法（TST101）为 3 学分，在第 1 学期开设。
      课程性质为选修，所属模块为专业选修课。

来源：2024 级测试专业培养方案（合成示例），第 1 页
~~~

### 3. 启动 HTTP 服务

先生成一个可保留的演示数据库：

~~~powershell
uv run python -m scripts.run_demo --database-out data/demo.sqlite3
$env:SWUFE_ACADEMIC_DATABASE="data/demo.sqlite3"
$env:SWUFE_RETRIEVAL_MODE="lexical"
uv run python -m app.server
~~~

打开：

- Swagger：<http://127.0.0.1:8000/docs>
- OpenAPI：<http://127.0.0.1:8000/openapi.json>
- Liveness：<http://127.0.0.1:8000/health/live>
- Readiness：<http://127.0.0.1:8000/health/ready>

## 🔌 HTTP API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| <code>GET</code> | <code>/options</code> | 获取数据集、届别、学院和专业选项 |
| <code>POST</code> | <code>/ask</code> | 自然语言教务问答 |
| <code>GET</code> | <code>/source/{chunk_id}</code> | 查看来源标题、页码和证据片段 |
| <code>GET</code> | <code>/academic-audit/options</code> | 获取学业审计范围 |
| <code>POST</code> | <code>/academic-audit</code> | 根据已修课程计算培养方案完成度 |
| <code>GET</code> | <code>/health/live</code> | 检查服务进程 |
| <code>GET</code> | <code>/health/ready</code> | 检查数据库、release 与检索组件 |

最小请求：

~~~powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"2024级测试专业X的测试算法多少学分，在哪个学期开设？"}'
~~~

主要响应字段：

- <code>answer_md</code>：Markdown 格式答案；
- <code>citations</code>：来源标题、证据片段和物理页码；
- <code>claims</code>：回答声明及其绑定的事实与证据；
- <code>refused</code>：是否拒绝返回确定性答案；
- <code>clarification</code>：需要补充的届别、专业或时间信息。

### 可选模型表达

配置 OpenAI-compatible provider 后，可以使用外部模型参与问题理解和答案表达：

~~~powershell
$env:SWUFE_LLM_BASE_URL="https://api.example.com/v1"
$env:SWUFE_LLM_MODEL="your-model"
~~~

请求时通过 <code>X-LLM-API-Key</code> 传入当前调用使用的 Key。

## 🔄 数据构建与发布

项目把原始资料、结构化目录、检索产物和运行版本组织为一条连续的数据生命周期：

~~~mermaid
flowchart LR
    A["PDF / DOCX / 来源登记"] --> P["解析、切分与质量记录"]
    P --> C["目录审核与字段物化"]
    C --> DB[("SQLite")]
    P --> K["制度知识块"]
    K --> I["词法 / 稠密检索产物"]
    DB --> R["Candidate Release"]
    I --> R
    R --> E["Agent / Retrieval Evaluation"]
    E --> S["Ed25519 Attestation"]
    S --> ACTIVE["active.json"]
~~~

数据包由以下部分组成：

- <code>sources.csv</code>：来源、版本、适用范围和文件身份；
- <code>chunks.jsonl</code>：制度文本、表格片段、页码和来源定位；
- <code>curriculum_catalog.json</code>：培养方案、课程、模块与毕业要求；
- <code>source_review.csv</code>：来源审核记录；
- <code>evidence_review.csv</code>：字段与证据审核记录；
- <code>raw/</code>：与登记摘要一致的原始材料；
- 检索产物：文档顺序、向量矩阵、索引和模型信息。

| 阶段 | 入口 | 产物 |
| --- | --- | --- |
| 数据导入 | <code>scripts.download_dataset</code> | <code>data/released/</code> 数据包 |
| 文档解析 | <code>ingest</code> / <code>scripts.extract_catalog</code> | chunks、页面与表格质量记录 |
| Catalog 物化 | <code>scripts.materialize_catalog</code> | 课程、模块、要求与字段来源 |
| 候选构建 | <code>scripts.build_all</code> | <code>artifacts/releases/&lt;release-id&gt;/</code> |
| 完整性检查 | <code>scripts.verify_dataset</code> | 数据、引用和来源一致性结果 |
| 评测 | <code>eval.run_agent_eval</code> / <code>eval.run_retrieval_ablation</code> | Agent 与检索报告 |
| 签名与晋级 | <code>scripts.create_eval_attestation</code> / <code>scripts.promote_release</code> | 签名证明与 <code>active.json</code> |

<code>active.json</code> 指向服务启动时加载的数据版本。完整参数、数据包格式和签名格式见 [数据模型](docs/DATA_MODEL.md)、[发布合约](docs/RELEASES.md) 与 [评测说明](docs/EVALUATION.md)。

## ⚙️ 常用配置

| 配置项 | 用途 |
| --- | --- |
| <code>SWUFE_ACADEMIC_DATABASE</code> | 指定 SQLite 数据库 |
| <code>SWUFE_RELEASE_ROOT</code> | 指定版本化 release 根目录 |
| <code>SWUFE_RETRIEVAL_MODE</code> | 选择 <code>lexical</code> 或 <code>hybrid</code> |
| <code>SWUFE_RETRIEVAL_ARTIFACT_ROOT</code> | 指定检索产物目录 |
| <code>SWUFE_EMBEDDING_MODEL</code> | 指定本地 embedding 模型快照 |
| <code>SWUFE_RERANKER_MODEL</code> | 指定本地 reranker 模型快照 |
| <code>SWUFE_LLM_BASE_URL</code> | OpenAI-compatible API 地址 |
| <code>SWUFE_LLM_MODEL</code> | 外部模型名称 |
| <code>SWUFE_SESSION_BACKEND</code> | 选择 <code>memory</code> 或 <code>redis</code> |
| <code>SWUFE_REDIS_URL</code> | Redis 连接地址 |

完整配置示例见 [.env.example](.env.example)。

## 📂 项目结构

~~~text
academic/    课程、培养要求、来源版本与类型化 SQLite 工具
agent/       有界运行时、ToolRegistry、coverage、session 与 MCP adapter
app/server/  FastAPI 服务、认证、资源控制与健康检查
evidence/    Fact、DerivedFact、EvidencePacket、ClaimAtom 与 provenance
generation/ 回答合成、渲染、声明语义和引用校验
ingest/      PDF / DOCX 解析、质量账本、catalog 审核与物化
query/       问题理解、实体归一化、作用域和 DAG 计划
retrieval/   BM25、作用域稠密检索、RRF、CrossEncoder、MMR 与索引产物
storage/     JSON 合约、Git provenance、content-addressed release 与 attestation
eval/        Agent、检索、泛化和 promotion-policy 评测
scripts/     数据导入、构建、校验、演示、签名和晋级命令
tests/       合成数据与 canonical contract tests
docs/        架构、数据、检索、评测、发布与安全说明
~~~

## 🧪 测试与验证

~~~powershell
uv sync --locked --extra dev --extra ingest

uv run python -m ruff check agent academic app evidence generation ingest query retrieval storage scripts eval tests/canonical
uv run python -m mypy agent academic app evidence generation ingest query retrieval storage scripts eval
uv run python -m pytest -q tests/canonical
~~~

GitHub Actions 在 Python 3.10 和 3.12 上执行代码检查、类型检查、数据构建、完整性验证、Agent 与检索评测、端到端演示、canonical tests 和 hybrid Docker 镜像验证。

## 📚 进一步阅读

- [系统架构](docs/ARCHITECTURE.md)：请求链路、运行时与证据模型；
- [数据模型](docs/DATA_MODEL.md)：课程、培养要求、来源和字段 lineage；
- [Agent Runtime](docs/AGENT_RUNTIME.md)：状态机、计划、工具与会话；
- [检索设计](docs/RETRIEVAL.md)：作用域过滤、混合检索与排序；
- [评测说明](docs/EVALUATION.md)：Agent、检索和发布评测；
- [发布合约](docs/RELEASES.md)：candidate、attestation 与 active release；
- [安全设计](docs/SECURITY.md)：认证、BYOK、资源控制与提示注入防护。

## License

代码使用 [MIT License](LICENSE)。数据文件、模型快照与第三方材料的许可说明见 [DATA_NOTICE.md](DATA_NOTICE.md)。

# swufe-rag 全量接口参考

更新日期：2026-07-15  
适用分支：`feature/production-api-boundary`  
本文档版本：`1.1`  
冻结数据/B/C `contract_version`：`1.0`

本文列出仓库当前全部对接入口，包括正式 HTTP、调试 HTTP、Python 公共门面、混合编排扩展、运行时构建器、可信存储接口、命令行入口、数据文件契约和环境配置。

规范优先级：

1. `contracts.py` 和 `INTERFACES.md` 是冻结 B/C 与数据字段的规范来源。
2. 本文是当前实现的完整接口目录。
3. 标记为“调试”或“内部”的接口不承诺长期兼容，不能作为外部系统的稳定依赖。

## 1. 接口分层总表

| 层级 | 入口 | 稳定性 | 主要使用方 |
|---|---|---:|---|
| 数据契约 | `data/sources.csv`、`data/chunks.jsonl` | 冻结 | 模块 A/B、数据维护者 |
| B 检索 | `swufe_rag.api.retrieve()` | 冻结 | 团队模块、旧调用方 |
| B 范围扩展 | `swufe_rag.api.retrieve_scoped()` | 附加 | 混合编排层 |
| C 生成 | `swufe_rag.api.answer()` | 冻结 | 团队模块、编排层 |
| 路由 | `route_question()`、`HybridRouter.route()` | 附加 | 混合编排层、测试 |
| 混合问答 | `HybridRuntime.handle_question()` | 附加 | 正式 HTTP、后续前端 |
| 正式 HTTP | `/ask`、`/options`、`/source/{chunk_id}` | 正式 | 学生端、测试 Web、外部客户端 |
| 调试 HTTP | `/api/debug/*` | 调试 | 本地证据审阅 |
| 可信存储 | `MetadataDB` | 内部 | 检索、URL 回查、运维 |
| CLI | `python -m ...` | 运维 | 数据构建、索引、服务、评估 |

## 2. 公共数据结构

### 2.1 `KnowledgeChunk`

知识块固定字段如下，不允许缺失或增加字段：

| 字段 | 类型 | 约束 |
|---|---|---|
| `chunk_id` | `str` | 非空、全局唯一 |
| `text` | `str` | 非空原文 |
| `doc_title` | `str` | 非空文档标题 |
| `article` | `str` | 非空条款、章节或页表标签 |
| `level` | `str` | `校级` 或 `院级` |
| `college` | `str` | 校级必须为 `全校`；院级必须为具体学院 |
| `cohort` | `str` | 四位入学年份或 `不限` |
| `year` | `int` | `1900..2100` |
| `status` | `str` | `现行` 或 `历史` |
| `page_url` | `str` | 绝对 HTTP(S) URL |
| `file_url` | `str` | 绝对 HTTP(S) URL |
| `is_table` | `bool` | 是否为完整表格块 |

示例：

```json
{
  "chunk_id": "swufe_20053b0cd28d_0013",
  "text": "官方文件原文",
  "doc_title": "西南财经大学计算机与人工智能学院推荐免试研究生工作实施细则（2023级）",
  "article": "第三章 综合测评 / 第五条",
  "level": "院级",
  "college": "计算机与人工智能学院",
  "cohort": "2023",
  "year": 2023,
  "status": "现行",
  "page_url": "https://it.swufe.edu.cn/...",
  "file_url": "https://it.swufe.edu.cn/...docx",
  "is_table": false
}
```

### 2.2 `RetrievedChunk`

`RetrievedChunk = KnowledgeChunk + score`。

| 附加字段 | 类型 | 含义 |
|---|---|---|
| `score` | `float` | BGE 稠密余弦相似度，不代表最终融合排序分数 |

返回列表已经过内部融合、重排和 MMR，调用方不得按 `score` 再排序。

### 2.3 `Citation`

| 字段 | 类型 | 约束 |
|---|---|---|
| `marker` | `int` | 回答中的引用编号，从 1 开始 |
| `chunk_id` | `str` | 必须属于本次检索集合 |
| `doc_title` | `str` | 由可信存储按 `chunk_id` 绑定 |
| `article` | `str` | 由可信存储按 `chunk_id` 绑定 |
| `quote` | `str` | 必须是对应数据库 `chunk.text` 的原文子串 |
| `page_url` | `str` | 由可信来源数据库绑定 |
| `file_url` | `str` | 由可信来源数据库绑定 |

### 2.4 `AnswerResult`

冻结 C 接口只返回三个字段：

```json
{
  "answer_md": "回答内容[1]。",
  "citations": [],
  "refused": false
}
```

### 2.5 `RouteDecision`

路由器只分类，不回答问题：

```json
{
  "mode": "school_rag",
  "requires_school_facts": true,
  "intent": "promotion",
  "college": "计算机与人工智能学院",
  "cohort": "2023",
  "policy_year": null,
  "rewritten_query": "挂科后重修通过还能申请推免吗",
  "search_terms": ["推荐免试", "推免资格", "重修"],
  "confidence": 0.99
}
```

`mode` 只能是：

- `general_chat`：普通聊天、通用知识、代码、写作、情绪交流。
- `school_rag`：任何需要真实西财制度、课程、培养方案、推免、校内事实或官方网址的问题。

当前常用 `intent/topic`：

- `promotion`
- `curriculum`
- `course_selection`
- `transfer`
- `assessment`
- `academic_status`
- `credit`
- `campus_service`
- `thesis`
- `school_policy`
- `school_general`
- `general_chat`

## 3. Python 公共接口

### 3.1 冻结 B/C 门面：`swufe_rag.api`

#### `retrieve()`

```python
from swufe_rag.api import retrieve

chunks = retrieve(
    query="2024级计算机类专业选修课有哪些？",
    top_k=5,
    college="计算机与人工智能学院",
    cohort="2024",
)
```

签名：

```python
retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[RetrievedChunk]
```

规则：

- `query` 不得为空。
- `top_k` 必须在 `1..50`。
- 默认只允许可信、启用、现行来源。
- 指定学院后只允许校级或该学院来源。
- 指定年级后只允许 `不限` 或该年级来源。
- SQL 范围过滤发生在向量/BM25 排序和 Top-K 截断前。

#### `retrieve_scoped()`

```python
from swufe_rag.api import retrieve_scoped

chunks = retrieve_scoped(
    "2024年的推免细则",
    top_k=8,
    college="计算机与人工智能学院",
    cohort="2023",
    policy_year=2024,
    topic="promotion",
)
```

签名：

```python
retrieve_scoped(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
    *,
    policy_year: int | None = None,
    topic: str | None = None,
) -> list[RetrievedChunk]
```

附加规则：

- `policy_year=None`：只取 `status=现行`。
- 显式 `policy_year`：按指定年份查询，可返回对应历史版本。
- `topic=None`：不做主题过滤。
- 非空 `topic`：SQL 精确匹配来源主题；未知主题通常返回空候选。
- 所有值均使用参数化 SQL，不能传入或执行模型生成 SQL。

#### `answer()`

```python
from swufe_rag.api import answer

result = answer("挂科后还能推免吗？", chunks)
```

签名：

```python
answer(
    query: str,
    chunks: list[dict],
) -> AnswerResult
```

规则：

- `chunks` 每项必须满足完整 `RetrievedChunk` 契约。
- 无结果、最高稠密分低于 `0.35`、必需实体缺失或年级证据不匹配时，不调用 LLM。
- 引用失败最多执行一次“只修引用、不增事实”重试。
- 第二次失败返回固定学校证据不足结果。
- LLM 服务错误抛出 `GenerationUnavailableError`，不能伪装成政策拒答。

#### `configure()`

```python
from swufe_rag.api import configure

configure(retriever=my_retriever, generation=my_generation)
```

签名：

```python
configure(
    *,
    retriever: AdvancedRetriever | None = None,
    generation: AdvancedGenerationService | None = None,
) -> None
```

用途：测试或应用启动时注入实现。传入 `None` 会重置为延迟加载生产实现。该接口不是远程 API。

### 3.2 路由接口

#### `route_question()`

```python
from swufe_rag.routing import RouteContext, route_question

decision = route_question(
    "那重修通过以后呢？",
    context=RouteContext(
        last_mode="school_rag",
        last_intent="promotion",
        last_college="计算机与人工智能学院",
        last_cohort="2023",
        last_rewritten_query="挂科后还能推免吗？",
    ),
)
```

签名：

```python
route_question(
    question: str,
    *,
    context: RouteContext | None = None,
    college: str | None = None,
    cohort: str | None = None,
) -> RouteDecision
```

#### `HybridRouter.route()`

与 `route_question()` 参数和返回一致。生产运行时可向 `HybridRouter` 注入 `LLMRouteClassifier` 和可信学院白名单；分类器失败时使用确定性回退策略。

### 3.3 混合编排接口

#### `HybridRuntime.handle_question()`

```python
result = runtime.handle_question(
    "那重修通过以后呢？",
    college="计算机与人工智能学院",
    cohort="2023",
    session_id="student-123",
    top_k=8,
)
```

签名：

```python
handle_question(
    question: str,
    *,
    college: str | None = None,
    cohort: str | None = None,
    session_id: str | None = None,
    top_k: int = 8,
    include_route_debug: bool = False,
) -> dict
```

正式返回字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `general_chat | school_rag` | 实际回答分支 |
| `answer_md` | `str` | 回答正文 |
| `citations` | `list[Citation]` | 通用分支为空 |
| `retrieved` | `list[RetrievedSummary]` | 通用分支为空 |
| `official_links` | `list[OfficialLink]` | 仅返回数据库中与当前主题匹配的官方入口 |
| `refused` | `bool` | 学校证据不足为 `true`；澄清问题为 `false` |
| `latency_ms` | `float` | 本次完整处理耗时 |

当 `include_route_debug=True` 时，增加 `route: RouteDecision`。正式 HTTP 不启用该字段。

#### `HybridRuntime.ask()`

`handle_question()` 的正式快捷入口，不接受 `top_k` 和调试开关：

```python
ask(
    question: str,
    *,
    college: str | None = None,
    cohort: str | None = None,
    session_id: str | None = None,
) -> dict
```

#### 其他运行时方法

| 方法 | 返回 | 用途 |
|---|---|---|
| `HybridRuntime.debug_ask(question, **kwargs)` | 混合响应 + `route` | 本地调试 |
| `HybridRuntime.source(chunk_id)` | `KnowledgeChunk | None` | 从可信 SQLite 回查原文 |
| `HybridRuntime.options()` | `Options` | 返回模式、学院、年级、块数量和默认 Top-K |
| `handle_question(runtime, question, **kwargs)` | 混合响应 | 模块级代理函数 |
| `InMemorySessionStore.get(session_id)` | `SessionState` | 获取或创建内存会话 |

### 3.4 运行时构建器：`app.runtime`

| 函数 | 数据/模型 | 用途 |
|---|---|---|
| `build_demo_runtime(path=DEMO_CHUNKS)` | fixture + HashingEncoder + DemoGroundedClient | 旧式 RAG 调试 |
| `build_review_runtime(path="data/chunks.jsonl")` | 真实块 + HashingEncoder + DemoGroundedClient | 离线数据审阅 |
| `build_production_runtime(chunks_path=...)` | 正式 B/C | 旧式生产 RAG 适配 |
| `build_demo_hybrid_runtime(path=DEMO_CHUNKS)` | fixture 双路由 | 混合接口测试 |
| `build_review_hybrid_runtime(path=...)` | 真实块双路由、无付费调用 | 混合 UX 审阅 |
| `build_production_hybrid_runtime(...)` | 正式 SQLite + BGE/FAISS + LLM | 正式 HTTP 默认运行时 |

`build_production_hybrid_runtime()` 参数：

```python
build_production_hybrid_runtime(
    chunks_path="data/chunks.jsonl",
    *,
    sources_path="data/sources.csv",
    metadata_path="data/metadata.sqlite3",
    config_path="config.advanced.yaml",
) -> HybridRuntime
```

### 3.5 可信存储接口：`storage.MetadataDB`

这些接口供服务内部和运维使用，不建议浏览器或外部模块直接连接 SQLite。

| 方法 | 作用 |
|---|---|
| `MetadataDB(database=":memory:")` | 打开数据库并执行迁移 |
| `MetadataDB.from_chunks(chunks, database=..., trusted_by_default=False)` | 从已验证知识块构建库 |
| `MetadataDB.from_files(sources_path=..., chunks_path=..., database=...)` | 从正式来源和知识块构建或按哈希复用库 |
| `candidate_rows(college=None, cohort=None, policy_year=None, topic=None)` | 用固定 SQL 返回允许参与排序的 `embedding_row` |
| `chunk(chunk_id, require_trusted=True)` | 回查可信知识块和来源元数据 |
| `official_links(college=None, cohort=None, topic=None, policy_year=None, limit=3)` | 返回匹配范围的官方链接 |
| `known_colleges()` | 返回已登记院级学院 |
| `known_cohorts()` | 返回已登记入学年级 |
| `set_source_state(source_id, trusted=None, enabled=None)` | 运维调整可信/启用状态 |
| `integrity_report()` | 返回来源数、块数和孤儿引用检查 |
| `close()` | 关闭连接 |

### 3.6 Web 应用工厂

| 函数 | 签名/用途 |
|---|---|
| `app.server.create_app(runtime=None)` | 创建正式 FastAPI 应用；测试可注入 `RAGRuntime` 或 `HybridRuntime` |
| `app.server.main()` | 用 Uvicorn 启动正式服务 |
| `app.debug_server.create_app(runtime=None)` | 创建隔离调试 FastAPI 应用 |
| `app.debug_server.configure_runtime(runtime)` | 注入或重置调试运行时 |
| `app.debug_server.get_runtime()` | 延迟构建并返回当前调试运行时 |
| `app.debug_server.main()` | 用 Uvicorn 启动调试服务 |

### 3.7 传统 `RAGRuntime` 适配接口

`RAGRuntime` 仍用于证据调试台和旧式 B/C 串联：

| 方法 | 说明 |
|---|---|
| `retrieve(question, top_k=5, college=None, cohort=None)` | 调用注入的冻结 B 检索函数 |
| `ask(question, top_k=5, college=None, cohort=None)` | 检索后调用 C，并增加检索摘要和耗时 |
| `debug_ask(...)` | 在 `ask()` 结果上增加调试运行模式 |
| `source(chunk_id)` | 从内存知识块映射回查原文 |
| `options()` | 返回模式、学院、年级、块数和默认 Top-K |

### 3.8 LLM 与生成扩展接口

这些是替换模型和测试注入的扩展点，不属于冻结 B/C 返回契约：

| 接口 | 签名/作用 |
|---|---|
| `LLMClient.generate(system_prompt, user_prompt)` | 所有模型客户端遵循的最小协议 |
| `OpenAICompatibleClient(...)` | DeepSeek、其他 OpenAI 兼容端点和 Ollama 适配器 |
| `GeneralChatService.answer(question, history=())` | 普通对话生成；历史项为 `(role, content)` |
| `AdvancedGenerationService.answer(query, chunks)` | 当前冻结 C 的具体实现 |
| `TrustedAnswerBinder.bind(raw_answer, retrieved)` | 生成后按 SQLite 重绑学校引用与 URL |

本文不把以下算法实现当作跨模块公共契约：具体 `Encoder`、FAISS `IndexBundle`、BM25、RRF、reranker、MMR、切块器和解析器内部方法。对接方应使用上文门面、运行时构建器或 CLI，避免直接依赖这些内部类。

## 4. 正式 HTTP 接口

### 4.1 启动

```powershell
python -m app.server
```

默认监听：`http://127.0.0.1:8000`。

### 4.2 端点总表

| 方法 | 路径 | 响应 | 说明 |
|---|---|---|---|
| `GET` | `/` | HTML | 混合对话测试 Web |
| `GET` | `/assets/chat.css` | CSS | 正式 Web 样式 |
| `GET` | `/assets/chat.js` | JavaScript | 正式 Web 客户端 |
| `GET` | `/options` | JSON | 可选学院、年级、块数量、运行模式 |
| `POST` | `/ask` | `AskResponse` | 路由优先的统一问答接口 |
| `GET` | `/source/{chunk_id}` | `KnowledgeChunk` | 可信原文回查 |
| `GET` | `/docs` | HTML | FastAPI Swagger 文档 |
| `GET` | `/openapi.json` | JSON | OpenAPI Schema |

正式服务关闭 `/redoc`。

### 4.3 `POST /ask`

请求模型：

| 字段 | 类型 | 必填 | 限制 |
|---|---|---:|---|
| `question` | `str` | 是 | 长度 `1..1000` |
| `college` | `str | null` | 否 | 建议使用 `/options` 返回值 |
| `cohort` | `str | null` | 否 | 建议四位入学年份 |
| `session_id` | `str | null` | 否 | 长度 `1..128`；用于连续追问 |

正式请求禁止额外字段。

请求示例：

```http
POST /ask
Content-Type: application/json

{
  "question": "那重修通过以后呢？",
  "college": "计算机与人工智能学院",
  "cohort": "2023",
  "session_id": "student-123"
}
```

学校回答示例：

```json
{
  "mode": "school_rag",
  "answer_md": "本科阶段不得有不及格课程记录[1]。",
  "citations": [
    {
      "marker": 1,
      "chunk_id": "swufe_20053b0cd28d_0013",
      "doc_title": "西南财经大学计算机与人工智能学院推荐免试研究生工作实施细则（2023级）",
      "article": "第三章 综合测评 / 第五条",
      "quote": "申请人本科阶段不得有不及格课程记录",
      "page_url": "https://it.swufe.edu.cn/...",
      "file_url": "https://it.swufe.edu.cn/...docx"
    }
  ],
  "retrieved": [
    {
      "chunk_id": "swufe_20053b0cd28d_0013",
      "doc_title": "西南财经大学计算机与人工智能学院推荐免试研究生工作实施细则（2023级）",
      "article": "第三章 综合测评 / 第五条",
      "college": "计算机与人工智能学院",
      "cohort": "2023",
      "score": 0.7824,
      "is_table": false,
      "summary": "原文摘要"
    }
  ],
  "official_links": [],
  "refused": false,
  "latency_ms": 1134.2
}
```

普通对话示例：

```json
{
  "mode": "general_chat",
  "answer_md": "注意力机制会根据当前任务为不同输入分配不同权重。",
  "citations": [],
  "retrieved": [],
  "official_links": [],
  "refused": false,
  "latency_ms": 821.0
}
```

学校证据不足示例：

```json
{
  "mode": "school_rag",
  "answer_md": "当前知识库中未找到能够明确回答该问题的西南财大官方规定。我不会改用通用模型猜测学校事实；请查看下方已登记的官方来源，或咨询教务处、学院教务办。",
  "citations": [],
  "retrieved": [],
  "official_links": [],
  "refused": true,
  "latency_ms": 96.4
}
```

缺少学院或年级时返回澄清问题，仍为 `mode=school_rag`，但 `refused=false`，且不会执行检索。

### 4.4 `GET /options`

响应示例：

```json
{
  "mode": "production-hybrid",
  "colleges": ["计算机与人工智能学院"],
  "cohorts": ["2023", "2024", "2025"],
  "chunk_count": 814,
  "default_top_k": 8
}
```

### 4.5 `GET /source/{chunk_id}`

路径参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | `str` | 知识块唯一 ID |

成功时返回完整 `KnowledgeChunk`，不返回 `score`、路由、耗时或内部 `source_id`。

### 4.6 正式 HTTP 错误

| 状态码 | 场景 |
|---:|---|
| `400` | 非 Schema 类运行时参数错误 |
| `404` | `/source/{chunk_id}` 不存在 |
| `422` | Pydantic 请求验证失败、缺少 `question`、字段过长或包含额外字段 |
| `503` | 数据契约错误、知识库/索引未就绪、LLM 不可用 |

错误体使用 FastAPI 标准格式：

```json
{"detail": "错误说明"}
```

## 5. 调试 HTTP 接口

### 5.1 启动模式

fixture Demo：

```powershell
$env:SWUFE_RAG_MODE="demo"
python -m app.debug_server
```

真实知识块离线审阅：

```powershell
$env:SWUFE_RAG_MODE="review"
$env:SWUFE_RAG_CHUNKS="data/chunks.jsonl"
python -m app.debug_server
```

调试服务同样默认监听 `127.0.0.1:8000`，不能与正式服务同时占用同一端口。

### 5.2 调试请求模型

```json
{
  "question": "CS205是什么课，多少学分？",
  "college": "计算机与人工智能学院",
  "cohort": "2023",
  "top_k": 5
}
```

| 字段 | 类型 | 默认/限制 |
|---|---|---|
| `question` | `str` | 必填，长度 `1..1000` |
| `college` | `str | null` | 默认 `null` |
| `cohort` | `str | null` | 默认 `null` |
| `top_k` | `int` | 默认 `5`，范围 `1..20` |

### 5.3 调试端点总表

| 方法 | 路径 | 响应 |
|---|---|---|
| `GET` | `/` | 证据调试台 HTML |
| `GET` | `/assets/debug.css` | 调试台 CSS |
| `GET` | `/assets/debug.js` | 调试台 JavaScript |
| `GET` | `/api/debug/health` | `status` + `options` |
| `GET` | `/api/debug/options` | 运行模式、学院、年级、块数、默认 Top-K |
| `GET` | `/api/debug/examples` | `demo/queries.json` 测试题数组 |
| `POST` | `/api/debug/retrieve` | 完整 `RetrievedChunk[]` + 调试运行模式 |
| `POST` | `/api/debug/ask` | `AnswerResult` + 检索摘要 + 耗时 + 调试运行模式 |
| `GET` | `/api/debug/source/{chunk_id}` | 完整 `KnowledgeChunk` |
| `GET` | `/api/debug/docs` | 调试 OpenAPI Swagger 文档 |
| `GET` | `/openapi.json` | 调试 OpenAPI Schema |

调试 `/api/debug/ask` 当前是隔离的传统 RAG 证据工作台，不等同于正式双路由 `/ask`。生产客户端不得依赖调试字段。

## 6. 命令行接口

### 6.1 模块 A 数据构建

```powershell
python -m ingest `
  --sources data/sources.csv `
  --raw-dir data/raw `
  --ocr-dir data/ocr `
  --output data/chunks.jsonl `
  --report data/ingest_report.json `
  --chunk-max-len 500
```

| 参数 | 默认值 |
|---|---|
| `--sources` | `data/sources.csv` |
| `--raw-dir` | `data/raw` |
| `--ocr-dir` | `data/ocr` |
| `--output` | `data/chunks.jsonl` |
| `--report` | `data/ingest_report.json` |
| `--chunk-max-len` | `500` |

标准输出为 JSON 构建报告。任何来源、文件或契约错误都会阻断整次构建。

### 6.2 正式索引构建

```powershell
python -m retrieval.index `
  --chunks data/chunks.jsonl `
  --artifacts artifacts `
  --model BAAI/bge-large-zh-v1.5
```

| 参数 | 默认值 |
|---|---|
| `--chunks` | `data/chunks.jsonl` |
| `--artifacts` | `artifacts` |
| `--model` | `BAAI/bge-large-zh-v1.5` |

标准输出为索引 manifest JSON。

### 6.3 服务启动

| 命令 | 用途 |
|---|---|
| `python -m app.server` | 正式混合 HTTP 与测试 Web |
| `python -m app.debug_server` | 隔离调试 HTTP 与证据工作台 |

### 6.4 评估入口

| 命令 | 网络/费用 | 说明 |
|---|---:|---|
| `python -m eval.demo_eval` | 无 | fixture 检索与拒答回归 |
| `python -m eval.real_data_eval` | 无 | 真实块离线审阅 |
| `python -m eval.hybrid_route_eval` | 无 | 100 题双路由评估 |
| `python -m eval.production_retrieval_eval` | 模型需可用 | 正式 BGE/FAISS 检索评估 |
| `python -m eval.production_generation_eval --live` | 会调用 LLM | 正式生成评估，必须显式确认 |

正式检索评估参数：

```text
--cases PATH
--chunks PATH
--artifacts PATH
--reranker
--refuse-th FLOAT
```

正式生成评估参数：

```text
--live
--cases PATH
--chunks PATH
--artifacts PATH
--config PATH
--reranker
--limit INT
```

## 7. 数据文件接口

### 7.1 `data/sources.csv`

固定列顺序：

```text
file,doc_title,level,college,cohort,year,status,page_url,file_url,collected_at
```

规则：

- `file` 是相对 `data/raw/` 的 POSIX 路径。
- URL 必须属于 `swufe.edu.cn` 或其子域。
- `collected_at` 使用 `YYYY-MM-DD`。
- 旧式 DOC/ZIP 必须先转换或拆分。
- 重复来源、缺文件、非法范围或非学校 URL 会阻断构建。

### 7.2 `data/chunks.jsonl`

每行一个严格 `KnowledgeChunk` JSON 对象。模块 B 只接受该文件作为正式知识块输入。

### 7.3 `data/source_review.csv`

模块 A 来源审核记录：

```text
original_title,corrected_title,decision,reason
```

该文件是审核证据，不直接参与运行时检索。

### 7.4 `data/metadata.sqlite3`

正式服务根据 `sources.csv` 与 `chunks.jsonl` 的 SHA-256 自动生成或复用。该文件被 Git 忽略，不是交付源数据。

主要表：

- `schema_meta`
- `sources`
- `chunks`

外部客户端不得直接写库；可信状态调整应通过受控运维代码完成。

## 8. 配置接口

### 8.1 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `OPENAI_API_KEY` | 空 | DeepSeek/其他 OpenAI 兼容服务密钥 |
| `OPENAI_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容服务地址 |
| `OLLAMA_API_KEY` | `ollama` | Ollama 占位密钥 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | 本地 Ollama OpenAI 兼容地址 |
| `SWUFE_RAG_MODE` | `demo` | 调试服务模式：`demo` 或 `review` |
| `SWUFE_RAG_CHUNKS` | `data/chunks.jsonl` | review 调试知识块路径 |
| `RUN_BGE_SMOKE` | 未设置 | 设为 `1` 启用真实 BGE 冒烟测试 |
| `RUN_FAISS_SMOKE` | 未设置 | 设为 `1` 启用真实 FAISS 冒烟测试 |

### 8.2 `config.advanced.yaml`

主要配置：

```yaml
paths:
  chunks: data/chunks.jsonl
  artifacts: artifacts

retrieval:
  embed_model: BAAI/bge-large-zh-v1.5
  top_k: 5
  candidate_k: 20
  use_bm25: true
  use_reranker: true
  rerank_model: BAAI/bge-reranker-base
  dense_weight: 0.35
  lexical_weight: 0.25
  rerank_weight: 0.35
  rank_prior_weight: 0.05
  mmr_lambda: 0.88

generation:
  llm: deepseek-chat
  temperature: 0
  general_temperature: 0.7
  refuse_th: 0.35
  max_retries: 2
  request_timeout_seconds: 60
  max_context_chars: 7000
  max_chunk_chars: 1600
```

## 9. Python 异常接口

| 异常 | 触发场景 | HTTP 映射 |
|---|---|---:|
| `ContractError` | 数据或返回字段违反冻结契约 | `503` |
| `KnowledgeBaseNotReadyError` | 正式块、manifest、模型对应索引或 FAISS 未就绪 | `503` |
| `GenerationUnavailableError` | LLM 密钥、依赖、网络或提供商失败 | `503` |
| `CitationValidationError` | 内部引用、数字、quote 或 URL 绑定失败 | 学校编排层转证据不足；不直接暴露 |
| `ValueError` | 空问题、非法 Top-K、非法范围等调用错误 | 正式 HTTP 通常为 `400` |

## 10. 兼容性与安全边界

- `retrieve()` 和 `answer()` 的签名与字段集合是冻结接口，不能向返回对象添加 HTTP、耗时或调试字段。
- `retrieve_scoped()`、路由和混合编排是附加层，不改变旧调用方。
- 学校问题证据不足时禁止回退 `general_chat`。
- 普通问题不得先执行检索或学校拒答门。
- 学校引用的 `chunk_id` 必须属于本次检索集合。
- 学校 `quote` 必须是数据库原文子串。
- 学校标题、条款和 URL 必须按 ID 从 SQLite 重建，模型输出的 URL 不被接受。
- `tests/fixtures`、Demo 客户端和 HashingEncoder 不得在生产入口自动加载。
- `/api/debug/*` 只用于本地审阅，不是正式学生端契约。
- 原始文档、OCR、SQLite、模型缓存和索引文件不提交公开 Git 仓库。

## 11. 最小对接示例

### Python B/C

```python
from swufe_rag.api import answer, retrieve

chunks = retrieve(
    "2024级计算机类毕业需要多少学分？",
    top_k=5,
    college="计算机与人工智能学院",
    cohort="2024",
)
result = answer("2024级计算机类毕业需要多少学分？", chunks)
```

### HTTP

```powershell
$body = @{
  question = "挂科后还能推免吗？"
  college = "计算机与人工智能学院"
  cohort = "2023"
  session_id = "student-123"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType application/json `
  -Body $body
```

前端只需保存并重复发送同一个 `session_id`，即可保持连续追问范围。

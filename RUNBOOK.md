# swufe-rag 本地运行与团队对接手册

## 当前完成度

- A 数据模块：完成 21 份来源登记、DOCX/PDF 解析、OCR 旁车、条款/列表切分、表格保留和 814 块真实 `chunks.jsonl`。
- B 检索模块：契约校验、BGE 适配、FAISS 持久化、学院/年级硬过滤、BM25 和 RRF。
- C 生成模块：DeepSeek/Ollama 适配、受约束提示词、引用映射、数字检查、修复和拒答。
- D 接口：完成 `general_chat/school_rag` 双路由、连续会话、正式 `/ask`、`/source/{chunk_id}`、测试 Web 与隔离调试 Web。
- 已有 20 条专项真实检索题、100 条路由题和正式 BGE/FAISS 结果；真实 LLM 独立评估仍需 API 配置。

## 环境安装

```powershell
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements-dev.txt
```

API 配置从 `.env.example` 复制到本地环境变量，不要提交 `.env`。

## 运行全部离线测试

```powershell
python -m unittest discover -s . -p "test*.py" -v
```

离线测试使用 `HashingEncoder`、临时 NumPy 索引和 `FakeClient`。生产入口不会自动加载这些测试替身。

## 真实知识库与正式索引

重新生成模块 A 产物：

```powershell
pip install -r requirements-ingest.txt
python -m ingest --sources data/sources.csv --raw-dir data/raw `
  --ocr-dir data/ocr --output data/chunks.jsonl --report data/ingest_report.json
```

构建正式索引：

```powershell
python -m retrieval.index --chunks data/chunks.jsonl --artifacts artifacts
```

构建过程会严格检查每行字段和重复 `chunk_id`，并写入：

- `artifacts/index.faiss`
- `artifacts/vectors.npy`
- `artifacts/chunk_ids.json`
- `artifacts/chunks.json`
- `artifacts/manifest.json`

`manifest.json` 最后写入，包含模型、维度、块数量和源文件 SHA-256。源数据变化后旧索引会拒绝加载。

## B 模块调用

```python
from swufe_rag.api import retrieve

chunks = retrieve(
    "我重修通过后还能申请推免吗",
    top_k=5,
    college="计算机与人工智能学院",
    cohort="2023",
)
```

生产 `retrieve()` 默认只读取 `data/chunks.jsonl` 和 `artifacts/`，并使用 BGE。返回顺序是 RRF 融合结果，`score` 是 BGE 余弦相似度。

## C 模块调用

`config.yaml` 中可选择：

```yaml
generation:
  llm: deepseek-chat
```

或：

```yaml
generation:
  llm: ollama:qwen2.5:7b-instruct-q4_K_M
```

调用方式：

```python
from swufe_rag.api import answer

result = answer("我重修通过后还能申请推免吗", chunks)
```

低于 `refuse_th` 时不会调用 LLM。LLM 服务错误抛出 `GenerationUnavailableError`，不能伪装成知识不足。

## 混合编排方式

```python
decision = route_question(question, context=session_context)
if decision.mode == "general_chat":
    result = general_chat.answer(question)
else:
    chunks = retrieve_scoped(decision.rewritten_query, ...)
    result = answer(decision.rewritten_query, chunks)
```

D 仍不改变 B/C 冻结返回结构。学校分支先由 SQLite 过滤可信、启用、现行、学院、年级、年份和主题，再做向量/BM25 排序；生成后的引用和 URL 按 `chunk_id` 回查 SQLite。通用分支不执行检索。

正式 HTTP 适配层已实现。只有真实知识库和索引到位后才启动：

```powershell
python -m app.server
```

启动后浏览器直接访问 <http://127.0.0.1:8000>。首次运行会根据 `sources.csv` 和 `chunks.jsonl` 生成 Git 忽略的 `data/metadata.sqlite3`；输入文件哈希变化时自动重建。

接口：

- `POST /ask`：请求接收 `question`、可选 `college`、`cohort`、`session_id`；
- `GET /options`：返回可选范围和知识库数量；
- `GET /source/{chunk_id}`：返回完整知识块，不带检索分数；
- 数据、索引未就绪或 LLM 不可用时返回 `503`；
- 非法业务参数返回 `400`，未知 `chunk_id` 返回 `404`；
- 正式服务不会因 `SWUFE_RAG_MODE=demo` 或其他调试配置而加载 fixture。

## 真实数据验收

1. 全量运行知识块契约校验并人工抽查表格与 URL。
2. 重建正式 BGE/FAISS 索引。
3. 用 20 条检索开发题验证 Top-5 命中率不低于 80%，范围污染为 0。
4. 根据真实 BGE 分数校准 `refuse_th`，初始值保持 0.35。
5. 使用独立的 30～40 题评估集检查事实、表格数字、跨文件回答和库外拒答。

## GitHub 协作

主仓：`https://github.com/ZorIgn/swufe-rag`

- 所有人从 `main` 创建功能分支。
- 提交 Pull Request 前运行全部测试。
- A 模块不得把模拟知识块写入 `data/chunks.jsonl`。
- 大型原始文档和索引在确定 Git LFS/发布附件策略前不提交。
- 禁止对 `main` 强制推送。

## 调试 Web

安装含 Web 调试依赖的环境：

```powershell
pip install -r requirements-web.txt
python -m app.debug_server
```

访问 <http://127.0.0.1:8000>。调试接口统一位于 `/api/debug`，与正式 `app.server` 分开启动：

- `GET /api/debug/health`
- `GET /api/debug/options`
- `GET /api/debug/examples`
- `POST /api/debug/retrieve`
- `POST /api/debug/ask`
- `GET /api/debug/source/{chunk_id}`

默认 `SWUFE_RAG_MODE=demo` 使用 fixture。若要在不下载模型、不调用 API 的情况下审阅真实知识块：

```powershell
$env:SWUFE_RAG_MODE="review"
$env:SWUFE_RAG_CHUNKS="data/chunks.jsonl"
python -m app.debug_server
```

该模式的 `0.34` 拒答阈值只针对哈希向量的本地 UX 回归；正式 B/C 仍保持硬性 `0.35`，必须用真实 BGE 开发集重新校准。

调试层可返回 `retrieved`、`latency_ms` 和 `mode`，但这些字段不进入 `swufe_rag.api.answer()` 的冻结返回对象。正式 D 模块可以迁移 HTTP 路径，但应继续复用统一 Python 门面。

## 高级 B/C 配置

`config.advanced.yaml` 增加候选窗口、标题与条款词权重、可选 BGE reranker、MMR、多源上下文预算和严格引用校验。生产默认仍使用：

- `BAAI/bge-large-zh-v1.5`
- `BAAI/bge-reranker-base`（可关闭）
- `refuse_th: 0.35`
- `temperature: 0`

拒答门槛是硬约束：最高余弦分数低于 0.35 时不调用 LLM，课程代码或关键词命中不能绕过它。真实数据到位后只能依据独立开发集校准阈值，并同步记录依据。

## Demo 评估

```powershell
python -m eval.demo_eval
```

`demo/queries.json` 含 20 题，覆盖培养方案、课程代码、表格数字、校级与院级政策、口语表达、库外问题和跨学院污染陷阱。当前基线为 Recall@5 100%、范围污染 0、拒答准确率 100%。

双路由评估：

```powershell
python -m eval.hybrid_route_eval
```

验收门槛：普通问题误拦截率不高于 2%、学校事实流入通用模型为 0、连续追问准确率不低于 95%。当前三项分别为 0%、0、100%。

真实数据离线审阅：

```powershell
python -m eval.real_data_eval
```

该 20 题开发集检查 Top-5 文档命中、范围污染、拒答和关键答案词。结果用于回归，不替代最终 30～40 题人工评分。

## 参考实现研究

本轮设计研究了以下开源项目的检索和引用思路，未复制其业务代码：

- Langchain-Chatchat：<https://github.com/chatchat-space/Langchain-Chatchat/tree/49165d6af4438aa7e8a1f71ce276db55f4405151>
- RAGFlow：<https://github.com/infiniflow/ragflow/tree/22dd1ad401d239a3b8a934ca8098937b4c5b58d8>

采用的通用模式包括扩大候选窗口、混合召回、二阶段重排、MMR 去冗余、上下文预算、句级引用校验和失败闭合。具体取舍与验证证据见 `ENGINEERING_LOG.md`。

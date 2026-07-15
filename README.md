# 西南财大教务 RAG 问答系统

本仓库实现计划书中的 A（数据与知识库）、B（检索）与 C（生成和引用溯源）模块，并已增加“普通对话 / 学校可信 RAG”双路由。学校事实走 SQLite 硬过滤和证据校验，普通问题直接进入通用 LLM；认证、限流、审计和正式部署仍未实现。

## 当前稳定基线

当前知识库覆盖 21 份真实来源、814 个知识块和 77 个表格块。稳定实现包括 B/C、可信来源 SQLite、连续会话双路由、确定性 Demo、真实 BGE/FAISS 评估以及测试 Web。团队原有调用继续使用以下冻结门面：

```python
from swufe_rag.api import retrieve, answer
```

公共契约仍以 [INTERFACES.md](INTERFACES.md) 为准：

- `retrieve(query, top_k=5, college=None, cohort=None) -> list[dict]`
- `answer(query, chunks) -> dict`
- B/C 返回对象不附加 HTTP 状态、耗时或调试字段。
- 调试 API 的扩展字段只存在于 `/api/debug`，不改变 B/C 契约。
- 正式 HTTP 入口为 `POST /ask`、`GET /source/{chunk_id}` 与 `GET /options`，生产模式不会自动加载 fixture。

## 快速体验 Demo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-web.txt
python -m app.debug_server
```

浏览器打开 <http://127.0.0.1:8000>。该模式使用 24 条 `fixture_` 知识块、轻量哈希编码器和确定性桩 LLM，不下载模型、不消耗 API 费用，也不会读入生产知识库。

运行完整验证：

```powershell
python -m unittest discover -s . -p "test*.py" -v
python -m eval.demo_eval
```

当前 Demo 基线：Recall@5 为 100%，范围污染为 0，20 题拒答准确率为 100%。这些指标仅证明程序与契约在模拟数据上可运行，不能替代真实教务文件验收。

## 模块 A 与真实数据审阅

首批知识库包含 18 份校级本科制度/操作文件、计算机与人工智能学院 2023 级推免细则，以及 2024/2025 级计算机类培养方案。原始文件和 OCR 旁车只保存在本地 Git 忽略区；来源登记、审批决定和派生知识块进入仓库。

```powershell
pip install -r requirements-ingest.txt
python -m ingest --sources data/sources.csv --raw-dir data/raw `
  --ocr-dir data/ocr --output data/chunks.jsonl --report data/ingest_report.json
python -m eval.real_data_eval
```

离线真实数据审阅 Web 不下载 BGE、不调用付费模型：

```powershell
$env:SWUFE_RAG_MODE="review"
$env:SWUFE_RAG_CHUNKS="data/chunks.jsonl"
python -m app.debug_server
```

`review` 模式只用于检查真实切分、过滤、引用、下载链接和页面交互，使用轻量哈希向量与抽取式替身，不代表正式 BGE/LLM 效果。模块 A 原交接包的逐文件审批证据见 [handoff/MODULE_A_AUDIT.md](handoff/MODULE_A_AUDIT.md)。

## 混合对话与正式 HTTP

正式索引构建完成并配置 DeepSeek/Ollama 后启动：

```powershell
python -m app.server
```

- 浏览器打开 <http://127.0.0.1:8000> 可使用混合对话测试 Web。
- `POST /ask` 接收 `question`、可选 `college`、`cohort`、`session_id`，并返回 `mode`。
- 普通知识、代码、写作和情绪交流为 `general_chat`，不会先检索或执行学校拒答门。
- 培养方案、选课、推免、校内服务和学校网址为 `school_rag`；没有证据时不会回退通用模型。
- `GET /source/{chunk_id}` 返回知识块完整原文及冻结元数据。
- 回答和官方查询入口中的网址均按 `chunk_id/source_id` 从 SQLite 绑定，模型输出的 URL 不被采纳。
- 路由置信度和内部证据门仍不进入正式响应；调试参数 `top_k` 不进入正式请求。
- 数据、索引或 LLM 未就绪时返回明确错误，不会用 `tests/fixtures` 伪装生产结果。

路由回归：

```powershell
python -m eval.hybrid_route_eval
```

当前 100 题结果：普通问题误拦截率 0、学校事实流入通用模型 0、连续追问准确率 100%。

## 工程资料

- [API_REFERENCE.md](API_REFERENCE.md)：全部正式、调试、Python、CLI、数据和配置接口。
- [RUNBOOK.md](RUNBOOK.md)：安装、索引构建、调试 Web 和团队对接命令。
- [INTERFACES.md](INTERFACES.md)：冻结的知识块、B、C 接口契约。
- [ENGINEERING_LOG.md](ENGINEERING_LOG.md)：研究依据、实现决策、测试证据、限制与真实数据补齐步骤。
- [REPOSITORY.md](REPOSITORY.md)：主仓和协作约定。


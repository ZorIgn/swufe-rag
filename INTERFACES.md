# swufe-rag 公共接口契约 v1.1

公共字典使用严格字段集合，运行时校验和类型定义见 `contracts.py`。

## 模块 A 来源登记

`data/sources.csv` 的固定列为 `file,doc_title,level,college,cohort,year,status,page_url,file_url,collected_at`。`file` 必须是相对 `data/raw/` 的 POSIX 路径；旧式 DOC 和 ZIP 必须先转换或拆分；URL 只接受学校官方 `swufe.edu.cn` 域名。来源登记错误会阻断整次构建，不会静默跳过。

## 契约 1：知识块

`data/chunks.jsonl` 每行必须是一个 JSON 对象：

```json
{"chunk_id":"it_py2023_017","text":"知识块文本","doc_title":"文档标题","article":"第四条","level":"院级","college":"计算机与人工智能学院","cohort":"2023","year":2023,"status":"现行","page_url":"https://it.swufe.edu.cn/example","file_url":"https://it.swufe.edu.cn/example.pdf","is_table":false}
```

- 所有字段必填且不接受额外字段，`chunk_id` 全局唯一。
- `level` 只能是 `校级|院级`；校级块必须使用 `college=全校`。
- `cohort` 是四位入学年份或 `不限`；`status` 是 `现行|历史`。
- 页面和附件地址必须是绝对 HTTP(S) URL；网页正文没有附件时二者相同。

## 契约 2：检索

```python
retrieve(query: str, top_k: int = 5,
         college: str | None = None,
         cohort: str | None = None) -> list[dict]
```

每项是完整知识块加唯一扩展字段 `score: float`。列表顺序是 RRF 融合顺序；`score` 固定表示 BGE 余弦相似度，只用于拒答和展示，调用方不得据此重新排序。

过滤规则：始终只取 `status=现行`；指定学院后只保留校级或该学院；指定年级后只保留 `不限` 或该年级。过滤发生在 Top-K 截断前。

## 契约 3：生成与溯源

```python
answer(query: str, chunks: list[dict]) -> dict
```

严格返回：

```json
{
  "answer_md": "回答内容[1]",
  "citations": [{
    "marker": 1,
    "chunk_id": "it_py2023_017",
    "doc_title": "文档标题",
    "article": "第四条",
    "quote": "知识块中的原文片段",
    "page_url": "https://...",
    "file_url": "https://..."
  }],
  "refused": false
}
```

`quote` 必须是知识块原文子串。空结果或最高结果的 `score < refuse_th` 时不调用 LLM。知识不足的固定主句为“现行文件中未找到明确规定，建议咨询教务处或学院教务办。”

### 混合编排的附加检索入口

混合对话层可以调用新增的 `retrieve_scoped(..., policy_year=None, topic=None)`。它只增加经过校验的年份和主题范围，不改变上述冻结 `retrieve()`。所有范围值都作为 SQLite 绑定参数进入固定 SQL 模板；模型不能生成或执行 SQL。

## 契约 4：正式混合 HTTP 适配层

```http
POST /ask
Content-Type: application/json

{"question":"...","college":"计算机与人工智能学院","cohort":"2023","session_id":"student-123"}
```

`college`、`cohort` 和 `session_id` 均可省略。返回字段为：

```json
{
  "mode": "school_rag",
  "answer_md": "受证据约束的回答[1]",
  "citations": [],
  "retrieved": [],
  "official_links": [],
  "refused": false,
  "latency_ms": 123.4
}
```

- `mode=general_chat`：不执行检索，`citations`、`retrieved` 和 `official_links` 为空。
- `mode=school_rag`：只使用 SQL 合格候选和契约 3；证据不足时不得回退通用模型。
- `session_id` 使“那重修通过以后呢”等追问继承上一轮学校主题和范围。
- 正式响应不暴露路由置信度、内部原因或证据门细节；这些只存在于测试/调试层。
- `retrieved` 只包含知识块摘要、范围和 `score`，不得反向改变 B 的排序。
- `official_links` 只来自已启用、已信任的来源表，用于知识不足时提供官方查询入口。

```http
GET /source/{chunk_id}
```

返回契约 1 的完整知识块，不包含 `score`、`mode` 或耗时。正式入口由 `app.server` 提供，绝不自动加载 `tests/fixtures`；`/api/debug/*` 是独立调试面，不属于正式 HTTP 契约。

`GET /options` 返回已登记学院、年级和知识块数量。`GET /` 提供无构建步骤的混合对话测试 Web。

## SQLite 可信来源边界

- `sources` 对层级、学院、年级、年份、状态、主题、可信和启用状态使用 `CHECK/NOT NULL/UNIQUE`。
- `chunks.source_id` 使用外键，`embedding_row` 全局唯一。
- 正式候选固定要求 `trusted=1 AND enabled=1`；默认只取现行文件，显式历史年份才按该年份查询。
- 学院、年级、年份和主题在向量/BM25 排序前由 SQL 过滤。
- 学校回答的 `chunk_id` 必须属于本次检索集合；`quote` 必须是数据库原文子串。
- `doc_title`、`article`、`page_url` 和 `file_url` 在生成后按 `chunk_id` 从 SQLite 重建。即使模型返回其他 URL，也会被丢弃；学校回答正文包含模型生成 URL 时整次回答不通过。

## 错误边界

- 数据或接口结构错误：`ContractError`
- 正式知识库或索引未就绪：`KnowledgeBaseNotReadyError`
- LLM 服务不可用：`GenerationUnavailableError`

B、C 仍保留冻结的 Python 接口。混合编排层通过 `swufe_rag.api` 复用 B/C，并在 HTTP 边界增加路由、会话、来源回查和观测字段。


# swufe-rag 本地运行与团队对接手册

## 当前完成度

- B 检索模块：契约校验、BGE 适配、FAISS 持久化、学院/年级硬过滤、BM25 和 RRF。
- C 生成模块：DeepSeek/Ollama 适配、受约束提示词、引用映射、数字检查、修复和拒答。
- 当前只有模拟知识块测试结果，不能代替真实教务文件评估。

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

## 正式知识库到位后

模块 A 将数据写入 `data/chunks.jsonl`。先运行索引构建：

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

## D 模块串联方式

```python
chunks = retrieve(question, top_k, college, cohort)
result = answer(question, chunks)
```

D 负责添加 `retrieved` 摘要、`latency_ms` 和 HTTP 状态，不能改变 B、C 冻结返回结构。`GET /source/{chunk_id}` 应直接读取知识块存储。

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

访问 <http://127.0.0.1:8000>。调试接口统一位于 `/api/debug`：

- `GET /api/debug/health`
- `GET /api/debug/options`
- `GET /api/debug/examples`
- `POST /api/debug/retrieve`
- `POST /api/debug/ask`
- `GET /api/debug/source/{chunk_id}`

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

## 参考实现研究

本轮设计研究了以下开源项目的检索和引用思路，未复制其业务代码：

- Langchain-Chatchat：<https://github.com/chatchat-space/Langchain-Chatchat/tree/49165d6af4438aa7e8a1f71ce276db55f4405151>
- RAGFlow：<https://github.com/infiniflow/ragflow/tree/22dd1ad401d239a3b8a934ca8098937b4c5b58d8>

采用的通用模式包括扩大候选窗口、混合召回、二阶段重排、MMR 去冗余、上下文预算、句级引用校验和失败闭合。具体取舍与验证证据见 `ENGINEERING_LOG.md`。

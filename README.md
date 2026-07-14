# 西南财大教务 RAG 问答系统

本仓库从零建设计划书中的 B（检索）与 C（生成和引用溯源）模块。A（数据与知识库）尚待真实数据，正式 D 接口尚未实现；当前 `app` 仅提供隔离的本地调试 Web。

## 当前稳定基线

当前稳定实现覆盖 B 检索、C 生成与溯源、确定性 Demo 评估和临时调试 Web。团队统一从以下门面调用，不直接依赖模块内部类：

```python
from swufe_rag.api import retrieve, answer
```

公共契约仍以 [INTERFACES.md](INTERFACES.md) 为准：

- `retrieve(query, top_k=5, college=None, cohort=None) -> list[dict]`
- `answer(query, chunks) -> dict`
- B/C 返回对象不附加 HTTP 状态、耗时或调试字段。
- 调试 API 的扩展字段只存在于 `/api/debug`，不改变 B/C 契约。

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

## 工程资料

- [RUNBOOK.md](RUNBOOK.md)：安装、索引构建、调试 Web 和团队对接命令。
- [INTERFACES.md](INTERFACES.md)：冻结的知识块、B、C 接口契约。
- [ENGINEERING_LOG.md](ENGINEERING_LOG.md)：研究依据、实现决策、测试证据、限制与真实数据补齐步骤。
- [REPOSITORY.md](REPOSITORY.md)：主仓和协作约定。


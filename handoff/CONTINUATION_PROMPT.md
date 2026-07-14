# swufe-rag 新电脑续接提示词

> 使用方式：在新电脑中先把交付 ZIP 完整解压，然后将本文件全文作为第一条任务提示词交给新的 Codex/代码助手。不要只发送其中一部分。

你正在接手“西南财大教务 RAG 问答系统”项目。你的目标是在新电脑上从 GitHub 主仓恢复工程上下文，继续完成真实知识库、后端和后续前端工作，并把经过测试的改动继续提交到主仓。

## 一、第一步：先读材料，不要立即改代码

请依次完整阅读：

1. 交付包中的原始项目计划书：
   `reference/西南财大教务RAG问答系统项目计划书.docx`
2. 仓库中的 `README.md`
3. `RUNBOOK.md`
4. `INTERFACES.md`
5. `ENGINEERING_LOG.md`
6. `REPOSITORY.md`
7. `handoff/DELIVERY_MANIFEST.md`
8. `handoff/NEW_COMPUTER_SETUP.md`

原始 DOCX 是需求源文件，只读使用，不要改写、覆盖或提交到 Git。若只拿到了 GitHub 仓库、没有交付 ZIP，先向用户索取该 DOCX，不能假装已经阅读。

阅读后先向用户报告：

- 当前远端 `main` 的 HEAD；
- 已完成模块；
- 未完成模块；
- 真实数据是否已经到位；
- 你建议的下一步和原因。

没有完成上述报告前，不要开始大规模重构。

## 二、重新建立本地仓库

在新电脑上必须从主仓克隆，不能在源码快照目录直接 `git init`，也不能创建与远端无关的历史：

```powershell
git clone https://github.com/ZorIgn/swufe-rag.git swufe-rag
cd swufe-rag
git switch main
git pull --ff-only origin main
git status
git log --oneline --decorate -8
```

功能基线提交为 `8007637`。远端可能已经有更新，因此只要求它仍是当前 HEAD 的祖先：

```powershell
git merge-base --is-ancestor 8007637 HEAD
```

若命令失败，先检查是否克隆了正确仓库，不得强行覆盖远端。交付包中的源码 ZIP 仅用于离线查看和灾备，不能代替带完整历史的 `git clone`。

## 三、已经完成的工作

### 1. B 检索模块

已完成并测试：

- 冻结知识块契约和严格逐行校验；
- BGE `BAAI/bge-large-zh-v1.5` 生产适配；
- FAISS `IndexFlatIP`、向量归一化、索引清单和源文件哈希检查；
- 先做现行状态、学院、年级过滤，再在合格集合取 Top-K；
- BM25 + 稠密检索 + RRF；
- 领域查询扩展：挂科/不及格/重修、保研/推免、专选/专业选修；
- 课程代码、条款号、数字和实体提取；
- 标题/条款词法加权；
- 可选 `BAAI/bge-reranker-base` 二阶段重排；
- 候选去重和 MMR 多样性；
- 范围污染和接口契约测试。

统一调用入口：

```python
from swufe_rag.api import retrieve
```

### 2. C 生成与溯源模块

已完成并测试：

- OpenAI 兼容 DeepSeek/Ollama 适配；
- `temperature=0`、超时和有限重试；
- 上下文字符预算和问题相关段落选择；
- 硬性 `refuse_th=0.35`；
- 无结果或最高稠密分数低于阈值时不调用 LLM；
- 必需实体不足时拒答；
- 引用格式归一化；
- 每个事实句必须有句末引用；
- 数字、课程代码和内容词支持检查；
- `quote` 必须是知识块原文子串；
- 一次“只修引用、不增事实”重试；
- 第二次失败后固定拒答；
- LLM 不可用抛出 `GenerationUnavailableError`，不伪装成政策拒答。

统一调用入口：

```python
from swufe_rag.api import answer
```

### 3. Demo、评估和调试 Web

已完成：

- 24 条 `fixture_` 模拟知识块；
- 20 条 Demo 查询；
- 确定性 `HashingEncoder` 和桩 LLM，不下载模型、不产生 API 费用；
- FastAPI 调试接口 `/api/debug/*`；
- 静态证据调试台：范围、回答、引用、原文、召回账本、分数和耗时；
- 桌面和移动端浏览器验收；
- 正常回答、来源回查和跨学院拒答测试。

当前模拟基线：

- Recall@5：100%；
- 范围污染：0；
- 20 题拒答准确率：100%；
- 完整测试：60 项通过，2 项真实模型下载型测试按设计跳过。

这些结果只证明模拟流程和契约正确，不能当作真实政策质量验收。

### 4. 工程治理

已完成：

- 主仓和 `main` 建立；
- CI 使用 Python 3.10 运行离线契约和 Web 测试；
- `README.md`、`RUNBOOK.md`、`INTERFACES.md`、`ENGINEERING_LOG.md`；
- 统一门面 `swufe_rag.api`；
- 模型、索引、密钥、虚拟环境和临时评估输出不提交 Git；
- 原始项目计划书不提交 Git，只在交付包中提供。

## 四、仍未完成的工作

### A. 真实数据与知识库

目前没有真实 `data/chunks.jsonl`，因此以下内容未完成：

- 官网真实教务文件收集和版本确认；
- PDF/Word/网页/表格解析；
- 切分、去重、表格保持和元数据补齐；
- 真实数据质量抽检；
- 正式 BGE 编码和 FAISS 索引；
- 大文件的 Git LFS、对象存储或发布附件方案。

### B. 真实检索与生成验收

未完成：

- 至少 20 条真实检索开发题；
- Recall@5 不低于 80%和范围污染为 0的真实验收；
- 真实 BGE 分数分布分析；
- `refuse_th` 的真实数据校准；
- BM25、候选数、RRF、reranker、MMR 参数调优；
- 独立 30～40 题最终评估；
- 真实政策答案的人工复核；
- 失败归因到数据、切分、过滤、检索、生成或引用层。

### C. 正式后端和前端

未完成：

- 计划书 D 模块的正式 `/ask` 和 `/source/{chunk_id}`；
- 认证、限流、审计、隐私处理和结构化日志；
- 正式生产配置、容器化、部署和监控；
- 最终学生端前端；
- 与队友 A/D 模块的最终联调。

当前 `/api/debug` 仅用于本地调试，不能擅自当成最终外部接口。

## 五、不可破坏的公共约束

### 知识块

- 所有字段必填，`chunk_id` 全局唯一；
- `level` 只能是“校级”或“院级”；
- 校级 `college` 必须是“全校”；
- `cohort` 为年份字符串或“不限”；
- `status` 只能是“现行”或“历史”；
- URL 必须是有效 HTTP/HTTPS；
- 任何契约错误必须给出行号和字段，不能静默跳过；
- 生产配置绝不能自动加载 `tests/fixtures`。

### B 接口

```python
retrieve(
    query: str,
    top_k: int = 5,
    college: str | None = None,
    cohort: str | None = None,
) -> list[dict]
```

- 返回原知识块全部字段和 `score`；
- `score` 是稠密余弦相似度，调用方不得按它重新排序；
- 索引缺失或哈希不匹配抛出 `KnowledgeBaseNotReadyError`；
- 过滤必须发生在 Top-K 前；
- 不得随意更改公共签名或返回字段。

### C 接口

```python
answer(query: str, chunks: list[dict]) -> dict
```

严格只返回：

```json
{
  "answer_md": "回答内容[1]",
  "citations": [
    {
      "marker": 1,
      "chunk_id": "...",
      "doc_title": "...",
      "article": "...",
      "quote": "原文子串",
      "page_url": "https://...",
      "file_url": "https://..."
    }
  ],
  "refused": false
}
```

- 不得增加耗时、HTTP 状态或调试字段；
- 最高稠密分数低于 0.35 时必须直接拒答，精确关键词不能绕过；
- 固定拒答为：“现行文件中未找到明确规定，建议咨询教务处或学院教务办。”；
- `quote` 必须在对应 `chunk.text` 中逐字出现；
- LLM 错误不能伪装成政策拒答。

## 六、推荐的继续顺序

如果用户没有指定新的优先级，按以下顺序推进：

1. 先确认用户是否已经拿到真实知识库或官网文件。
2. 若没有数据，完善 A 模块的采集、解析、契约校验和数据交付工具，但不要编造真实政策数据。
3. 若已有真实数据，先全量校验和人工抽查，再重建正式索引。
4. 建立真实检索开发集并完成 B 调优。
5. 用真实条款调优 C 的上下文和引用校验。
6. 使用独立评估集验收后，再冻结正式 D 接口。
7. 最后对接或替换学生端前端。

每一步都必须在 `ENGINEERING_LOG.md` 记录：

- 做了什么；
- 为什么这样做；
- 改了哪些接口或配置；
- 运行了哪些测试；
- 指标和已知限制；
- 提交、合并和推送状态。

## 七、测试和运行命令

新电脑建议使用 Python 3.10：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-web.txt
python -m unittest discover -s . -p "test*.py" -v
python -m eval.demo_eval
```

启动调试 Web：

```powershell
python -m app.debug_server
```

打开 <http://127.0.0.1:8000>。

真实 BGE/FAISS 冒烟测试默认跳过，只有依赖和网络准备好后才启用：

```powershell
$env:RUN_BGE_SMOKE="1"
$env:RUN_FAISS_SMOKE="1"
python -m unittest tests.retrieval.test_production_smoke -v
```

## 八、Git 工作流

开始新任务前：

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/<清晰任务名>
```

完成后：

1. 确认 `git status` 没有密钥、模型、索引、真实原始文档或临时输出；
2. 运行相关测试和完整契约测试；
3. 更新 `ENGINEERING_LOG.md`；
4. 使用清晰的小粒度提交；
5. 合并或发起 Pull Request；
6. 推送到 `ZorIgn/swufe-rag`；
7. 禁止强制推送 `main`；
8. 禁止使用 `--allow-unrelated-histories` 覆盖队友历史。

若远端在你工作期间有新提交，先 fetch/rebase 或在集成分支解决，不得盲目覆盖。

## 九、开始续接时的输出格式

完成阅读和仓库检查后，请先向用户输出：

1. “远端主仓状态”；
2. “我确认已完成的内容”；
3. “我确认未完成的内容”；
4. “真实数据/计划书是否可用”；
5. “本轮建议执行的具体任务”；
6. “可能影响公共接口或团队协作的风险”。

得到用户新指令后再实施。

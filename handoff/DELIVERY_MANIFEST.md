# swufe-rag 交付清单

## 交付身份

- 项目：西南财大教务 RAG 问答系统
- 主仓：<https://github.com/ZorIgn/swufe-rag>
- 权威分支：`main`
- 功能基线：`8007637 docs: record main delivery`
- 历史稳定标签：`local-bc-v1`
- 交付日期：2026-07-14
- 交付目的：迁移到新电脑继续开发并持续提交主仓

远端可能在交付后继续前进。新电脑应以 clone 后的 `origin/main` 最新 HEAD 为准，不应把本清单中的哈希当作需要回退的目标。

## 仓库内交付文档

- `handoff/CONTINUATION_PROMPT.md`：给新 Codex 的完整提示词。
- `handoff/NEW_COMPUTER_SETUP.md`：给开发者的新电脑恢复步骤。
- `handoff/DELIVERY_MANIFEST.md`：本清单。
- `ENGINEERING_LOG.md`：详细工程实施记录。
- `INTERFACES.md`：冻结 B/C 接口。
- `RUNBOOK.md`：安装、测试和运行命令。

## 外部 ZIP 内容

最终 ZIP 预计包含：

```text
swufe-rag-handoff-2026-07-14/
├── START_HERE.txt
├── CHECKSUMS.sha256
├── PACKAGE_INFO.txt
├── documents/
│   ├── CONTINUATION_PROMPT.md
│   ├── NEW_COMPUTER_SETUP.md
│   ├── DELIVERY_MANIFEST.md
│   ├── ENGINEERING_LOG.md
│   ├── INTERFACES.md
│   └── RUNBOOK.md
├── reference/
│   └── 西南财大教务RAG问答系统项目计划书.docx
└── repository/
    └── swufe-rag-source-<交付HEAD>.zip
```

源码快照由 `git archive` 从已提交 HEAD 生成，不包含 `.git`、虚拟环境、密钥、模型缓存、FAISS 索引和临时文件。

## 计划书源文件

原始计划书在 ZIP 的 `reference` 目录中原样交付。交付时会记录源文件和包内副本 SHA-256；两个哈希必须一致。该 DOCX 只用于需求参考，不提交 Git。

## 已完成与未完成状态

完整状态以 `CONTINUATION_PROMPT.md` 和 `ENGINEERING_LOG.md` 为准。摘要：

- 已完成：B 检索主体、C 生成与溯源主体、模拟数据、测试、Demo 评估、调试 Web、统一 Python 门面。
- 未完成：真实数据采集与知识库、正式 BGE/FAISS 索引、真实评估和阈值校准、正式 D 接口、最终前端、部署与运维。

## 验收要求

交付 ZIP 生成后必须：

1. 通过 ZIP 完整性测试；
2. 解压到临时目录复核关键文件；
3. 校验计划书副本哈希与源文件一致；
4. 校验源码快照内 HEAD 对应的文档和代码存在；
5. 记录最终 ZIP 的 SHA-256 和大小。

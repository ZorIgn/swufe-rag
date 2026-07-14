# 新电脑恢复与继续开发指南

## 1. 交付包用途

交付 ZIP 同时提供：

- GitHub 主仓地址；
- 当前源码灾备快照；
- 可复制给新 Codex 的完整续接提示词；
- 原始项目计划书 DOCX；
- 文件校验值。

正式继续开发时以 GitHub `main` 为权威。源码 ZIP 只用于离线查看、对照或灾备。

## 2. 解压交付包

将整个 ZIP 解压到一个不受微信临时目录清理影响的位置，例如：

```text
E:\school\swufe-rag-handoff\
```

先确认以下文件存在：

```text
documents\CONTINUATION_PROMPT.md
documents\DELIVERY_MANIFEST.md
reference\西南财大教务RAG问答系统项目计划书.docx
repository\swufe-rag-source-*.zip
CHECKSUMS.sha256
```

## 3. 校验交付文件

PowerShell：

```powershell
Get-FileHash -Algorithm SHA256 .\reference\西南财大教务RAG问答系统项目计划书.docx
Get-FileHash -Algorithm SHA256 .\repository\swufe-rag-source-*.zip
```

与 `CHECKSUMS.sha256` 对照。若不一致，先重新复制交付包，不要使用损坏文件。

## 4. 从主仓重新建立本地仓库

推荐路径可以自由选择，但必须使用 `git clone`：

```powershell
cd E:\school
git clone https://github.com/ZorIgn/swufe-rag.git swufe-rag
cd swufe-rag
git switch main
git pull --ff-only origin main
git status
```

不要在交付包内的源码快照上执行 `git init`。源码快照没有 `.git` 历史，直接初始化会产生无关根提交，之后很难安全合并到主仓。

如果新电脑暂时没有网络，可以解压 `repository\swufe-rag-source-*.zip` 阅读和运行；恢复网络后仍应重新 clone 主仓，再迁移尚未提交的修改。

## 5. 放置项目计划书

项目计划书建议保存在仓库外，例如：

```text
E:\school\project-reference\西南财大教务RAG问答系统项目计划书.docx
```

保持原文件只读，不提交 Git。将该路径告诉新的 Codex，并要求其在修改架构或接口前先阅读计划书与仓库文档。

## 6. 建立环境

```powershell
cd E:\school\swufe-rag
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements-web.txt
```

验证：

```powershell
python -m unittest discover -s . -p "test*.py" -v
python -m eval.demo_eval
```

启动调试页：

```powershell
python -m app.debug_server
```

浏览器打开 <http://127.0.0.1:8000>。

## 7. 把提示词交给新 Codex

打开 `documents\CONTINUATION_PROMPT.md`，完整复制到新任务中，并附上：

- 解压后的项目计划书绝对路径；
- 新 clone 的仓库绝对路径；
- 当前想继续的具体优先级；
- 真实知识库是否已经到位。

不要只说“继续做项目”，否则新助手无法可靠继承接口约束和已完成状态。

## 8. 后续提交

每轮开始前从 `origin/main` 更新，使用功能分支开发。测试通过后更新 `ENGINEERING_LOG.md`，再提交和推送。不得强制推送主分支，不得把原始项目计划书、真实文档、模型缓存或索引直接加入 Git。

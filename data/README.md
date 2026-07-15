# 数据目录

- `raw/`：审核通过的官网原始文件；只保存在本地并被 Git 忽略。
- `ocr/`：扫描 PDF 的逐页 OCR 旁车 JSON；只保存在本地并被 Git 忽略。
- `sources.csv`：已审核并允许进入知识库的来源登记表，路径均相对 `data/raw/`。
- `source_review.csv`：模块 A 原交接包的逐文件审批决定与原因。
- `chunks.jsonl`：模块 A 生成、模块 B 唯一认可的生产知识块文件。
- `metadata.sqlite3`：混合服务按来源和知识块哈希自动生成的可信范围/URL 数据库；本地产物，不提交 Git。

测试数据只能放在 `tests/fixtures/`，不得复制到本目录冒充正式知识库。

构建前安装解析依赖：

```powershell
pip install -r requirements-ingest.txt
python -m ingest --sources data/sources.csv --raw-dir data/raw `
  --ocr-dir data/ocr --output data/chunks.jsonl --report data/ingest_report.json
```

扫描件先在 Windows 上生成旁车文件：

```powershell
.\tools\windows_ocr.ps1 -PdfPath data\raw\school\扫描件.pdf `
  -OutputPath data\ocr\扫描件.pdf.ocr.json
```

解析过程按严格契约失败闭合：缺文件、旧式 DOC、未解包 ZIP、扫描件缺 OCR、非学校域名 URL 或非法枚举都会终止整次构建，不会静默跳过。

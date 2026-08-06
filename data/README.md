# 数据目录

生产数据包不提交到 Git。取得数据发布包后，将 `sources.csv`、`chunks.jsonl` 与
`curriculum_catalog.json` 放入本目录，或运行：

```powershell
python -m scripts.download_dataset --source-dir <released-data-directory>
# 或：python -m scripts.download_dataset --url <dataset-zip-url>
python -m scripts.build_all
python -m scripts.verify_dataset --allow-unverified-requirements
```

构建会生成本地 `academic.sqlite3` 和 `artifacts/manifests/` 下的不可变清单。若要从原始 PDF/DOCX 重新解析知识块，先运行 `uv sync --extra ingest`。
原始文件可保存在 `raw/`，OCR 旁车文件保存在 `ocr/`；二者以及构建产物均被 Git
忽略。用于 CI 的最小公开夹具位于 `tests/canonical/data/`，不能作为生产知识库。

来源、课程、培养要求或知识块发生变更后，应重新执行构建和完整性校验，不应手工修改
SQLite 或清单中的统计信息。
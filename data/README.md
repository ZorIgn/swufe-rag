# 数据目录

生产数据包不提交到 Git。取得数据发布包后，将 `sources.csv`、`chunks.jsonl` 与
`curriculum_catalog.json` 放入本目录，或运行：

```powershell
python -m scripts.download_dataset --source-dir <released-data-directory>
# 或：python -m scripts.download_dataset --url <dataset-zip-url>
python -m scripts.build_all --source-review data/source_review.csv --evidence-review data/evidence_review.csv
python -m scripts.verify_dataset --allow-review-required-requirements
```

构建会生成本地 `academic.sqlite3` 和 `artifacts/manifests/` 下的不可变清单。若要从原始 PDF/DOCX 重新解析知识块，先运行 `uv sync --extra ingest`。
原始文件可保存在 `raw/`，OCR 旁车文件保存在 `ocr/`；二者以及构建产物均被 Git
忽略。用于 CI 的最小公开夹具位于 `tests/canonical/data/`，不能作为生产知识库。

来源、课程、培养要求或知识块发生变更后，应重新执行构建和完整性校验，不应手工修改
SQLite 或清单中的统计信息。

`source_review.csv` 是独立审核账本。只有精确决策 `include`、`include_ocr`、
`include_converted`、`include_split` 会把匹配来源的知识块晋升为 `verified`；知识块
JSON 自带的 `review_status` 不能自行完成该晋升。账本可附加 `reviewer`、`method` 和
`reviewed_at` 等审计字段。`--allow-review-required-requirements` 只允许具有同级
`review_required` 证据的结构化要求通过离线校验，运行时仍会保持 not ready，直到核心
课程与培养要求均有 verified 证据。

当审核通过的是被隔离总册中的特定专业切片时，可新增 `evidence_review.csv`。它至少包含
`chunk_id,decision`，并可附加 `scope,reviewer,method,reviewed_at`；仅精确决策
`verified` 或 `include` 会晋升那个 chunk，且该细粒度账本优先于来源级账本。它不会使
同一总册中的其他 chunk 获得 verified 状态。

# Local data directory

Official school data is not included in this repository. After obtaining an authorized release package, install it locally:

~~~powershell
uv run python -m scripts.download_dataset --source-dir <released-data-directory> --data-dir data/released
~~~

The installed package contains:

- <code>sources.csv</code>;
- <code>chunks.jsonl</code>;
- <code>curriculum_catalog.json</code>;
- <code>source_review.csv</code>;
- <code>evidence_review.csv</code>;
- <code>dataset_manifest.json</code>;
- optional source files under <code>raw/</code>.

Build one immutable candidate release:

~~~powershell
uv run python -m scripts.build_all --catalog data/released/curriculum_catalog.json --sources data/released/sources.csv --chunks data/released/chunks.jsonl --source-review data/released/source_review.csv --evidence-review data/released/evidence_review.csv --source-root data/released/raw --retrieval-mode hybrid --embedding-model <local-embedding-snapshot> --reranker-model <local-reranker-snapshot> --holdout-manifest <restricted-holdout/manifest.json>
~~~

The authoritative output is <code>artifacts/releases/sha256-*/</code>. A database, retrieval directory or manifest written through an explicit compatibility option is not an independently promotable release.

## Review semantics

<code>source_review.csv</code> records source-level inclusion decisions. <code>evidence_review.csv</code> may approve a specific chunk or scoped slice without promoting unrelated rows from the same source. Field materialization still requires source hash, page, row/cell or span lineage.

<code>--allow-review-required-requirements</code> is a candidate-only diagnostic exception. It never turns unverified evidence into a ready production release.

Reviewer names and timestamps are input metadata. This repository does not authenticate reviewers or implement an external approval workflow.

## Raw parsing

Install <code>--extra ingest</code> to parse PDF or DOCX sources. The pipeline records page/table quality and quarantines unsupported or low-quality extraction. It does not include an OCR engine or guarantee reliable extraction from arbitrary scanned PDFs.

Do not hand-edit generated SQLite, retrieval files, release manifests or their statistics. Change the authorized inputs and build a new candidate.

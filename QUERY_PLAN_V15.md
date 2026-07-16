# QueryPlan + SQL/RAG v15

## Runtime flow

```text
question
  -> LLM QueryPlan JSON (no SQL, no school-fact answer)
  -> deterministic tool selection
     -> parameter-bound SQLite for course facts
     -> scoped hybrid RAG for policy/prose
     -> SQL + RAG for progress audits
  -> evidence packet
  -> constrained LLM wording
  -> fact, citation, URL and list-completeness validation
  -> final answer with original file and physical PDF page
```

The browser sends an optional DeepSeek key in `X-LLM-API-Key` for that request
only.  It is not persisted in configuration, local storage, or the database.

## Full-school data projection

- 57 registered sources and 60,827 searchable chunks.
- 8 complete 2017–2024 curriculum books: 6,106 PDF pages.
- 12 verified split curriculum files: 422 PDF pages.
- 37 policies/guides, including promotion rules: 171 PDF pages plus DOCX files.
- 468 structured program plans.
- 35,828 course rows and 2,974 program requirements.
- No missing registered raw file, unregistered original file, zero-chunk source,
  or zero-course full curriculum book.

Course tables are queried from `course_offerings`; prose policies remain in the
hybrid BGE/BM25/reranker corpus.  Every course row retains cohort, college,
major, module, nature, code, name, credits, hours, semester, department,
source file, physical page and source-row provenance.

## Production launch

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
D:\workplace\nlp\zhangchenyu-gpt\Scripts\python.exe -X utf8 -m app.server_v15
```

The web UI is served at `http://127.0.0.1:8000/`; `/options` exposes the loaded
chunks/index hashes, process path, CUDA device, FAISS row count and database
coverage so a stale process or stale index is visible immediately.

## Verified results

- Mock benchmark: 100/100 answered, 100/100 correct tool path, 100/100 with
  physical-page citations, 0 false refusals, 0 incorrect clarifications.
- Cross-school smoke: 12/12 across finance, accounting, law, statistics,
  languages, administration, insurance, promotion rules, deferred exams and
  course-selection guides.
- Focused v15 tests: 14/14.
- Repository suite: 163 passed, 2 skipped; one pre-existing compatibility test
  still expects the old three-argument `_chunk_page_url` helper signature.
- Real DeepSeek calls verified `llm` parsing, `sql+llm`, `rag+llm`, strict fact
  validation and original-file page citations.

Detailed machine-readable reports are in `analysis-output/full-system-v2/` and
`analysis-output/query-plan-v5-eval/`.

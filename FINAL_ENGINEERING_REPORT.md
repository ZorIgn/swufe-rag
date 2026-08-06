# Final Engineering Report — Canonical SWUFE Academic Agent

**Report date:** 2026-08-07
**Refactor branch:** `codex/rag-v16-repair`
**Scope:** the evidence-grounded academic-agent convergence in this working
tree. Generated data and build artifacts are intentionally not tracked by Git.

## Outcome

The runtime has been converged on one typed, bounded implementation. The HTTP
application composes `AgentRuntime` explicitly, and the request path is:

```text
Raw question → UnderstandingDraft → NormalizedQuery → typed DAG plan
             → read-only ToolRegistry → EvidencePacket
             → claim validation → rendered answer with source pages
```

The runtime does not execute model-produced SQL or arbitrary Python. It allows
only one bounded recovery path: a targeted policy retrieval after validation
reports insufficient evidence.

### Canonical components

| Component | Responsibility |
|---|---|
| `academic/` | SQLite-backed academic records, source versions/aliases, and typed academic/policy operations. |
| `query/` | Pydantic schemas, deterministic or structured-model understanding, normalization, and typed DAG planning. |
| `agent/` | Dependency-injected bounded runtime, sessions, tracing, policy limits, request-scoped provider factory, MCP adapter, and tool registry. |
| `evidence/` | `Fact`/`DerivedFact`, provenance, coverage, source conflicts, and claim spans. |
| `generation/` | Constrained synthesis, rendering, and evidence/claim validation. |
| `ingest/` | Dataset contracts, source parsing, chunking, and ingestion pipeline. |
| `app/server/canonical.py` | The sole FastAPI implementation for the public HTTP contract. |
| `scripts/` | Reproducible data download, build, and verification entry points. |

### Read-only tool registry

The canonical registry contains the following typed operations:

- `academic.list_courses`, `academic.get_course`,
  `academic.get_requirements`, and `academic.get_module_requirements`
- `academic.audit_progress`, `academic.compare_programs`,
  `academic.list_courses_before_semester`,
  `academic.list_unavoidable_courses`, and
  `academic.check_curriculum_feasibility`
- `policy.search` and `source.resolve`

HTTP and MCP use the same registry and schemas. The public HTTP endpoints are
`/health/live`, `/health/ready`, `/options`, `/ask`, `/source/{chunk_id}`,
`/academic-audit/options`, and `/academic-audit`.

## Data, provenance, and safety

The completed production-data build produced manifest
`artifacts/manifests/2.0-1b5b3658437d.json`, with:

| Measure | Result |
|---|---:|
| Sources | 57 |
| Extracted physical pages | 6,694 |
| Programs | 468 |
| Course offerings | 35,827 |
| Requirements | 2,974 |
| Retrieval chunks | 60,827 |

The build supports current/effective source selection and historical `as_of`
lookups. Same-authority numeric or course-code conflicts produce a conflict
packet rather than silently selecting a source. Every school factual claim must
be bound through its `Fact` or recursively supported `DerivedFact` evidence
graph; invalid citations, unsupported numbers/codes, missing evidence, and
derived-fact cycles refuse safely.

Generated SQLite, chunks, catalog, vector, and manifest outputs are excluded
from Git. The repository supplies the schema and reproducible builders instead.

## Provider and credential policy

`X-LLM-API-Key` is held only by a short-lived request model. It is not retained
in sessions, tracing, logs, caches, metrics, errors, or the provider factory.
External structured-model calls are opt-in: both `SWUFE_LLM_BASE_URL` and
`SWUFE_LLM_MODEL` must be set. Supplying a key alone makes no external request
and returns the safe `503 provider_unavailable` response. The provider uses an
OpenAI-compatible chat-completions wire format, bounded retries/timeouts, and a
process-local circuit breaker.

## Executed verification

The following results were obtained during this refactor, rather than being
projected CI expectations.

| Check | Command/result |
|---|---|
| Production data build | `python -m scripts.build_all` completed. |
| Data verification | `python -m scripts.verify_dataset --allow-unverified-requirements` completed with the manifest counts above. |
| Lint | `uv run ruff check agent academic app evidence generation ingest query retrieval storage scripts/build_all.py scripts/download_dataset.py scripts/verify_dataset.py eval/run_generalization.py eval/run_retrieval_ablation.py tests/canonical` passed; `contracts.py` is now at `ingest/contracts.py`. |
| Type checking | `uv run mypy --no-incremental agent academic app evidence query generation retrieval storage` passed: **39 source files, no issues**. |
| Unit and contract tests | The canonical coverage gate passed: **16 passed**, **80.51% total coverage** (`--cov-fail-under=50`). Only the Starlette/httpx deprecation warning was emitted. |
| Generalization | `python -m eval.run_generalization --database data/academic.sqlite3 --samples 12` passed **12/12**. |
| Container build/import | `docker build -t swufe-rag:canonical-verify .` succeeded, followed by a container import check for all canonical packages. |
| Container health | A temporary mounted-data container returned `{"status":"live"}` from `/health/live` and `{"status":"ready","dataset":"2.0"}` from `/health/ready`. |

Two real-data smoke questions also completed with evidence:

- “2023级人工智能专业第6学期有哪些选修课？” returned CST344, CST345,
  and DSC202 (three credits each), sourced from physical pages 466–467.
- “2023级学生公共外语课程总共要求多少学分？” returned original evidence
  supporting “共 8 个学分”, from physical page 9.

The versioned fixture in `tests/canonical/data/` builds and verifies locally;
GitHub Actions now builds that fixture in `$RUNNER_TEMP`, then lints, type
checks, runs the canonical coverage gate, and builds the Docker image.

## Legacy convergence and remaining cleanup boundary

The staged refactor deletes the legacy runtime chains and their parallel UI/
service layers, including `app/runtime*.py`, `app/server_v*.py`,
`academic_audit/`, `swufe_rag/`, prior generation/retrieval implementations,
and generated academic data previously tracked in the repository. This leaves
the canonical packages above as the production path.

Some historical, non-production files remain physically present because a
repository-wide destructive cleanup was not authorized. In particular:

- `scripts/audit_database_v2.py`
- `scripts/audit_full_coverage_v2.py` and `scripts/audit_full_coverage_v3.py`
- `scripts/debug_live_planner_v2.py`
- `scripts/live_query_plan_smoke_v2.py`
- `scripts/rebuild_academic_database_v2.py`
- `scripts/repair_fullbook_scope_labels_v2.py`
- historical root documents such as `QUERY_PLAN_V15.md` and
  `RAG_V16_IMPROVEMENT_PLAN.md`

These files are not part of the canonical production or CI paths and should be
treated as historical until an explicitly authorized cleanup removes or
archives them.

## Current limitations

- Curriculum data describes plans, not live course offerings, seats, grades, or
  an official graduation decision.
- `review_required` structured facts are not promoted silently to definitive
  policy rules.
- The service refuses when authoritative, in-scope, version-resolved evidence
  is missing or conflicts; it does not guess.
- External model use requires the explicit endpoint/model configuration above;
  there is intentionally no implicit provider fallback.
- Larger retrieval/model benchmarks require their optional data/model
  dependencies; their reports are generated under `eval/reports/` when those
  dependencies are available.

## Current limitation boundary

This report covers the canonical source and local verification only. Generated
artifacts remain reproducible but untracked, and the historical files listed
above remain outside production and CI paths.

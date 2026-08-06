# Final Engineering Report — SWUFE Academic Agent

**Report date:** 2026-08-07
**Scope:** canonical source tree and the local verification runs recorded below. Generated
course data, SQLite databases, indexes, models, and reports are intentionally not committed.

## Repository shape

The tracked source tree has one production HTTP service (`app.server`) and one
bounded runtime (`agent.orchestrator.AgentRuntime`). Historical runtime chains,
versioned servers, compatibility wrappers, handoff material, prior evaluation
outputs, and generated data packages are removed from the tracked tree. The
remaining implementation is organised as follows:

| Area | Role |
|---|---|
| `academic/` | SQLite projection, entity aliases, typed academic and policy tools |
| `query/` | Structured understanding, normalization, and typed plan construction |
| `agent/` | Explicit dependency composition, bounded state machine, sessions, tracing, registry, and MCP adapter |
| `evidence/` | Facts, derived facts, provenance, coverage, conflicts, and claim spans |
| `generation/` | Constrained synthesis, rendering, and claim/citation validation |
| `ingest/` | Source contracts, document parsing, and chunking |
| `app/server/` | FastAPI HTTP surface and health/readiness lifecycle |
| `scripts/` | Dataset download, SQLite build, and integrity verification |

The request path is:

```text
Raw question → UnderstandingDraft → NormalizedQuery → typed execution plan
             → read-only ToolRegistry → EvidencePacket → validated claims
             → rendered answer with source-page citations
```

No model-generated SQL or arbitrary Python execution is accepted. Validation
can trigger at most one targeted policy retrieval; it cannot enter an unbounded
reflection loop.

## Tool and evidence model

`ExecutionPlan.operations` is a discriminated union of typed operations. The
registry checks that the complete planner operation set is also executable.
Independent read-only operations can run in parallel. A mixed question can plan
`get_module_requirements`, `list_courses`, and `retrieve_policy` together.

The shared MCP adapter exposes schemas generated from those same operation
models. Its standard names include `search_policy`, `list_courses`,
`get_course_detail`, `get_graduation_requirements`, `audit_academic_progress`,
`compare_programs`, and `resolve_source`; it validates inputs before invoking
the same registry used by HTTP.

School claims carry `fact_ids` and `evidence_ids`. Numeric and course-code
references are checked against the claim's facts, derived facts retain their
input graph, and citations must be reachable from that graph and have lexical
support for the claim. Equal-authority source conflicts result in a refusal
rather than an automatic choice.

## Data and reproducibility

`python -m scripts.build_all` creates SQLite and an immutable manifest under
`artifacts/manifests/`; `python -m scripts.verify_dataset` checks duplicate
sources, orphan relations/provenance, course codes, credits, semesters,
program relations, duplicate canonical offerings, and requirement evidence.
Raw and generated data remain ignored by Git.

The final local CI fixture build produced:

| Manifest field | Value |
|---|---:|
| Sources | 1 |
| Physical pages | 2 |
| Programs | 2 |
| Course offerings | 2 |
| Requirements | 2 |
| Knowledge chunks | 2 |
| Dataset version | `canonical-ci-fixture-1-07c179ff3a11` |

These are fixture measurements, not claims about the externally distributed
production dataset.

## Executed verification

| Check | Local result |
|---|---|
| Ruff | Passed for the CI source/test set. |
| Mypy | Passed for 39 production source files. |
| Compile check | Passed for canonical packages, scripts, eval runners, and tests. |
| Fixture build + verification | Passed with every reported integrity count at zero. |
| Canonical tests + coverage gate | Passed: 19 tests, 81.98% total coverage, with the configured 50% gate satisfied. |
| Data-driven generalization | Passed 2/2 fixture programs without program-specific business code. |
| Retrieval ablation smoke | Ran on a two-query fixture: lexical Recall@1 0.500, Recall@5/10 1.000, MRR 0.750, nDCG@10 0.815. Dense/reranker variants were explicitly reported unavailable rather than fabricated. |
| Docker build/import | Image built successfully and imported `app.server` and `agent.mcp`. |
| Docker health | A short-lived container using the fixture database returned `{"status":"live"}` and `{"status":"ready","dataset":"canonical-ci-fixture-1"}`. |

## Security and deployment boundaries

BYOK credentials are request-scoped HTTP headers, never request-body fields.
They are not retained by sessions, provider factories, traces, or safe error
responses. The service applies explicit CORS configuration, request-size and
rate limits, provider timeout/retry/circuit-breaker controls, and a bounded
tool-call policy. Document text is treated as data in both understanding and
synthesis prompts.

The Docker image uses a non-root user and exposes `/health/live`; Compose
provides CPU and GPU profiles with persistent artifact mounts.

## Known limitations

- The public repository contains only a small offline fixture. Full corpus
  retrieval, dense-model ablation, latency percentiles, and external-model cost
  benchmarks require the separately released data/model resources and are not
  represented here as synthetic numbers.
- Curriculum plans are not real-time course offerings, capacity, grades, or an
  official graduation decision.
- `review_required` records are not silently promoted to definitive policy
  facts.
- The in-memory session store is bounded for local deployment; multi-worker
  production should select the optional Redis store.

## Diff summary

Against the prior source layout, the final index removes the historical V2–V16
runtime/server/query/test/documentation paths and tracked generated corpora,
while retaining the canonical packages, documents, build scripts, CI workflow,
and offline fixture. User-owned untracked `backups/`, `deliveries/`, and
`tools/` material is intentionally outside this report and was not staged.
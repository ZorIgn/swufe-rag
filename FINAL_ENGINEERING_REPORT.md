# Final Engineering Report — SWUFE Academic Agent

**Report date:** 2026-08-07

## Scope and architecture

The repository exposes one FastAPI service and one bounded academic-agent
runtime. The canonical request path is:

\`\`\`text
RawQuestion → UnderstandingDraft → NormalizedQuery → ExecutionPlan
            → typed read-only tools → EvidencePacket → ClaimValidation
            → rendered answer with provenance
\`\`\`

Structured curriculum facts are read from SQLite. Policy explanations use the
scoped retrieval service. The runtime passes cohort, college, program, and
effective-date scope explicitly; it does not append scope text to the user's
question. Tool operations use typed arguments, dependency-aware execution, a
single plan deadline, and deterministic failure results. School facts are
bound to facts and evidence before an answer is rendered. Equal-authority
policy conflicts and untrusted evidence fail closed.

The MCP adapter and HTTP runtime share the same registry, schemas, executor,
tracing, and validation path.

## Data and reproducibility

\`python -m scripts.build_all\` creates the SQLite projection, a retrieval
manifest, and a dataset manifest. \`python -m scripts.verify_dataset\` checks
source, relation, course, provenance, and evidence integrity. Generated
databases, indexes, and raw releases remain outside Git.

The fresh canonical fixture run on 2026-08-07 reported:

| Field | Value |
| --- | ---: |
| Sources | 1 |
| Physical pages | 2 |
| Programs | 2 |
| Course offerings | 2 |
| Requirements | 2 |
| Knowledge chunks | 2 |
| Dataset version | \`canonical-ci-fixture-1\` |

## Local verification

| Check | Observed result |
| --- | --- |
| \`python -m pytest -q\` | 32 passed |
| Ruff over production, evaluation, and canonical-test paths | passed |
| \`python -m compileall -q ...\` | passed |
| Fixture build and \`verify_dataset\` | all integrity counters 0 |
| Fixture generalization smoke | 2/2 passed |
| Fixture agent evaluation | 2 questions; intent, plan, tool precision, and tool recall all 1.0 |

The test runner emitted one local cache-permission warning; it did not affect
test execution or assertions.

## Evaluation boundaries

The checked-in fixture is intentionally small. The retrieval ablation runner
reports only measurements it actually executes and rejects missing or
mismatched hybrid artifacts; it does not fill dense, RRF, reranker, or MMR
columns with placeholder values. Dense retrieval and reranking require the
optional model/index artifacts and were not claimed as local measurements here.
Full-corpus latency percentiles, external-model token cost, and holdout scores
therefore remain unreported until those artifacts are supplied.

## Security and operational boundaries

BYOK credentials are request-scoped and excluded from sessions, traces,
metrics, caches, and error bodies. Tool calls are read-only, typed, bounded by
timeouts and call limits, and document text is treated as data rather than
instructions. Curriculum answers describe the published plan; they are not
real-time offering, capacity, grade, or official graduation decisions.

## Known limitations

- The public repository does not contain the full released corpus or optional
  dense-model artifacts.
- The local session implementation is bounded in memory; a multi-worker
  deployment should use the optional Redis store.
- External LLM behavior and GPU performance require a separately configured
  provider and should be evaluated with an untouched holdout set.

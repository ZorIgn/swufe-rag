# Architecture

## System boundary

The repository implements one FastAPI service and one bounded academic-assistant runtime. It is an evidence-constrained SQL + policy RAG prototype, not an autonomous agent platform and not a live student-information system.

~~~text
RawQuestion
  -> QuestionUnderstanding
  -> QueryNormalizer + explicit information scope
  -> ExecutionPlanner
  -> typed read-only DAG operations
  -> EvidencePacket + per-operation coverage
  -> AnswerSynthesizer
  -> ClaimValidator
  -> cited FinalAnswer
~~~

<code>app.server</code> is the HTTP boundary. <code>agent.factory.build_runtime</code> composes explicit dependencies; the runtime does not depend on import-order side effects or mutable global model clients.

## Trust boundaries

### Structured facts

Courses, credits, semesters, modules, requirements and curriculum calculations come from parameter-bound SQLite tools. The runtime never asks a model to produce SQL. Derived values keep their input fact IDs and rule evaluation.

### Policy text

Policy retrieval applies cohort, college, program, topic, effective date and supersession constraints before ranking. Lexical mode is an explicit offline diagnostic path. Artifact-backed hybrid mode requires all index, vector, model and evidence-state contracts to validate before readiness.

### Model output

An optional external model may assist question understanding and expression, but it cannot add tool names, execute arbitrary code or bypass evidence validation. The default synthesizer is deterministic. Provider credentials are request-scoped and are excluded from session state and public errors.

### Released artifacts

A content-addressed release binds the database, retrieval files, dataset manifest, local model snapshot digests, holdout descriptor and Git provenance. A candidate becomes active only through a validated, Ed25519-signed evaluation attestation. Runtime startup verifies the pointer, directory hashes, signature and trusted issuer again.

## Bounded execution

The planner emits typed operations with explicit dependencies and output contracts. The ToolRegistry is the sole authority for names, argument schemas, timeouts and read-only status. It is shared by the HTTP runtime and <code>MCPAdapter</code>; the adapter is not an MCP server or transport.

Each output capability is tied to concrete producer operation IDs. Coverage is evaluated per operation, so one successful operation cannot hide the failure of another operation with the same result kind. The state machine allows at most one targeted retrieval repair after insufficient evidence; it never enters an open-ended reflection loop.

## Evidence and answer semantics

Tool results are normalized into an <code>EvidencePacket</code> containing:

- observed and derived facts;
- citations and source provenance;
- execution results;
- per-operation coverage components;
- conflicts and warnings.

Each factual answer span contains typed ClaimAtoms. Validation binds subject, predicate, value, unit, conditions, exceptions, scope, temporal qualifier and comparator to the supporting fact. Directional comparators such as minimum, maximum, before and after are validated against compatible predicates and units. Missing evidence, a failed producer, a scope mismatch or unresolved equal-authority conflict fails closed.

## HTTP contract

Public endpoints:

- <code>GET /options</code>
- <code>POST /ask</code>
- <code>GET /source/{chunk_id}</code>
- <code>GET /academic-audit/options</code>
- <code>POST /academic-audit</code>

Operational endpoints:

- <code>GET /health/live</code>
- <code>GET /health/ready</code>

Readiness is stricter than liveness. A process can be live while refusing traffic because the database, evidence state, hybrid artifact, model snapshot or signed active release is unavailable.

## Deliberate non-features

- No model-generated SQL or arbitrary Python execution.
- No claim that the evidence graph is GraphRAG.
- No MCP server transport.
- No automatic OCR engine for arbitrary scanned PDFs.
- No live course offering, seat, grade or official graduation-decision integration.
- No retrieval-result cache or distributed rate limiter.
- No claim of production quality without a separately supplied corpus, model snapshots and restricted holdout.

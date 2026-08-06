# Architecture

`app.server` is the only HTTP application. It composes one `AgentRuntime` with
explicit dependencies; import order and module globals cannot change behavior.

```text
RawQuestion → UnderstandingDraft → NormalizedQuery → ExecutionPlan
            → typed read-only tools → EvidencePacket → FinalAnswer
            → claim validation → rendered citations
```

The bounded runtime has exactly one possible repair: a single targeted policy
retrieval after validation reports insufficient evidence. It never executes
model-produced SQL or arbitrary Python.

The public HTTP contract is `GET /options`, `POST /ask`, `GET /source/{chunk_id}`,
`GET /academic-audit/options`, and `POST /academic-audit`; liveness and readiness
are exposed separately at `/health/live` and `/health/ready`.

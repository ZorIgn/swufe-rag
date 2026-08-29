# Retrieval

Policy retrieval evaluates time, supersession, cohort, college, program, and
topic scope before either ranking channel runs. Candidates expose separate
`dense_score`, `bm25_score`, `rrf_score`, `reranker_score`,
`exact_entity_score`, `scope_score`, and `final_score` fields. Production and
local composition default to an artifact-backed hybrid path: BM25 and dense
candidates are fused with RRF, scored by a CrossEncoder, relevance-gated, then
diversified with MMR. `SWUFE_RETRIEVAL_MODE=lexical` is an explicit offline
fallback for unit tests and diagnostics; it is never reported as hybrid.

## Strict dense scoping

Dense ranking treats scope as a correctness boundary. The scoped chunk IDs are
first mapped to their aligned `vectors.npy` rows, then the query embedding is
scored with exact normalized inner products against only those rows. Results are
ordered by `(-dense_score, chunk_id)`, so ties are deterministic and a chunk
outside the supplied scope cannot be returned.

The implementation deliberately does not ask the global FAISS index for an
over-fetched top-N list and filter it afterward: in a narrow scope, globally
higher-scoring rows can otherwise crowd out the only relevant scoped row. The
trade-off is an exact `O(|scope| * embedding_dimension)` dot-product scan plus
an `O(|scope| log |scope|)` deterministic sort per query, which is preferred
while scope correctness matters more than approximate global-index latency.
`vectors.npy` and the one-query encoder result must be finite,
dimension-aligned, and unit-normalized; violations make hybrid retrieval
unavailable instead of silently changing cosine-score semantics.

Hybrid readiness requires the database evidence-state hash, ordered chunk IDs,
embedding dimension, and SHA-256 hashes for `documents.jsonl`, `doc_ids.json`,
`vectors.npy`, and `faiss.index` to match the versioned retrieval manifest.
Missing local model weights or any mismatch makes `/health/ready` fail closed.

Course, credit, semester and module facts use parameter-bound SQLite tools
rather than RAG parsing.

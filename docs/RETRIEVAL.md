# Retrieval

Policy retrieval is scope filtered before ranking. Candidates expose separate
`dense_score`, `bm25_score`, `rrf_score`, `reranker_score`,
`exact_entity_score`, `scope_score`, and `final_score` fields. Production and
local composition default to an artifact-backed hybrid path: BM25 and dense
candidates are fused with RRF, scored by a CrossEncoder, relevance-gated, then
diversified with MMR. `SWUFE_RETRIEVAL_MODE=lexical` is an explicit offline
fallback for unit tests and diagnostics; it is never reported as hybrid.

Hybrid readiness requires the database evidence-state hash, ordered chunk IDs,
embedding dimension, and SHA-256 hashes for `documents.jsonl`, `doc_ids.json`,
`vectors.npy`, and `faiss.index` to match the versioned retrieval manifest.
Missing local model weights or any mismatch makes `/health/ready` fail closed.

Scope cache entries are LRU-bounded, report hits/misses/memory estimates, and are
invalidated when `dataset_version` changes. Course, credit, semester and module
facts use parameter-bound SQLite tools rather than RAG parsing.

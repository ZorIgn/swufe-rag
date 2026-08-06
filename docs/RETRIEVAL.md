# Retrieval

Policy retrieval is scope filtered before ranking. Candidates expose separate
`dense_score`, `bm25_score`, `rrf_score`, `reranker_score`,
`exact_entity_score`, `scope_score`, and `final_score` fields. The default local
implementation is a transparent lexical baseline; deployments may supply dense
and reranker scores without changing the record contract.

Scope cache entries are LRU-bounded, report hits/misses/memory estimates, and are
invalidated when `dataset_version` changes. Course, credit, semester and module
facts use parameter-bound SQLite tools rather than RAG parsing.

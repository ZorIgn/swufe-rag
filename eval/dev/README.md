# Development fixtures

This directory contains public, synthetic development labels. They are bound to the checked-in canonical fixture and use <code>dataset_kind=test_fixture</code>.

- <code>queries.json</code> defines typed intent, operation and answer-containment expectations.
- <code>retrieval_documents.jsonl</code> and <code>retrieval_queries.json</code> define graded relevance, scope and hard-negative labels.

These fixtures are suitable for planner regressions, schema compatibility and local diagnostics. They do not establish performance on official school data, unseen user questions, real Chinese-language ambiguity or production model artifacts.

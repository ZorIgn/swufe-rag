# Public frozen test fixture

This directory is a deliberately tiny, synthetic <code>test_fixture</code> for deterministic CI coverage. It is public and is not a secret or promotion-eligible holdout.

<code>manifest.json</code> and its SHA-256 sidecar freeze the fixture roles, file hashes and counts. The strict loader rejects missing, tampered, unlisted or non-fixture data. The retrieval artifact uses deterministic local encoder and reranker fixtures, so CI can exercise:

- lexical and hybrid report paths;
- scope filtering and leakage detection;
- graded relevance and hard negatives;
- artifact and report provenance;
- missing-artifact failure.

The fixture is not representative of official school documents or user traffic. Its metrics must not be reported as school-corpus accuracy, recall, generalization, latency or cost.

Production promotion uses a separate <code>restricted_holdout</code> outside Git. Its manifest, files and access controls are supplied by an authorized operator.

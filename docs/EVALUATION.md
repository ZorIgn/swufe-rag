# Evaluation

## Two evaluation modes

The runners have separate diagnostic and promotion-eligible modes.

Diagnostic mode accepts explicit database, question, corpus or artifact paths. It is useful for development and public CI fixtures. Its report records the supplied inputs and cannot be used to promote a release.

Promotion mode accepts only:

- an immutable candidate <code>release_manifest.json</code>;
- a restricted frozen holdout manifest;
- an output report path.

Database, retrieval corpus, model snapshots, dataset version, evaluator Git commit and fixed metric configuration are derived from the candidate and holdout. Diagnostic overrides are rejected.

## Public fixtures are not a secret holdout

<code>eval/dev/</code> and <code>eval/holdout/</code> contain deliberately small, synthetic <code>test_fixture</code> datasets. Their full questions are public because they test schemas, hashing, scope, hard negatives, artifact loading and report provenance in CI.

A promotion-eligible holdout has <code>dataset_kind=restricted_holdout</code>, remains outside Git and is supplied by an authorized operator. Its manifest and sidecar freeze every role, additional file, SHA-256 and sample count. A public test fixture is rejected wherever a restricted holdout is required.

## Agent metrics

The agent runner evaluates:

- intent accuracy;
- typed plan exact match;
- tool precision and recall;
- answer containment;
- safe-rejection precision, recall and F1;
- scope pollution rate.

Promotion policy v1 requires all positive agent metrics to equal 1.0 and scope pollution to equal 0.0 on the supplied restricted holdout. These strict thresholds are contract gates for a versioned release, not a claim that the fixture is representative of real user traffic.

## Retrieval metrics

Both lexical and artifact-backed hybrid variants must be measured. The report includes Recall@1/5/10, MRR, nDCG@10, scope violations and labeled hard-negative hits. Promotion policy v1 requires:

| Gate | Requirement |
| --- | ---: |
| Recall@10 | at least 0.80 |
| MRR | at least 0.50 |
| nDCG@10 | at least 0.50 |
| hard-negative rate at cutoff | 0 |
| scope-violation rate | 0 |

Recall@1 and Recall@5 are recorded with non-negative floors. Missing hybrid artifacts, skipped variants, zero hard-negative coverage, invented gates, weakened thresholds, wrong operators, non-finite values or mismatched sample counts fail promotion.

## Provenance

Promotion reports bind:

- candidate release ID and manifest digest;
- database, dataset and retrieval artifact identities;
- restricted holdout ID, bundle digest, file hashes and counts;
- embedding and reranker snapshot digests;
- evaluator Git commit and clean-tree status;
- promotion-policy version and digest;
- the exact evaluation configuration.

The attestation creator revalidates both reports, binds their hashes to the candidate subject and signs the resulting statement with Ed25519. Promotion verifies the signature against an explicit issuer/public-key trust registry.

## What is not measured here

The checked-in workflow does not report representative latency, throughput, GPU performance, external-model token cost or school-corpus quality. It does not contain a scheduled large-model or GPU benchmark. Those measurements require authorized data, pinned model snapshots and a declared hardware/provider environment.

For a real deployment, add a separate benchmark report with p50/p95/p99 latency, scope-size distribution, memory, concurrency, token/cost accounting and an error taxonomy over an untouched, representative query set. Keep performance evidence separate from the correctness gates above.

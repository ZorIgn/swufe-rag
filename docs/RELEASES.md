# Release contract

## Purpose

A runnable knowledge base is one immutable release, not a collection of files that happen to share a directory. The contract prevents:

- partial builds from looking loadable;
- a database from being mixed with another index;
- model or source drift from hiding behind an unchanged label;
- evaluation reports from being attached to a different candidate;
- an untested candidate from becoming the active runtime.

## Layout

<code>scripts.build_all</code> builds in a hidden staging directory on the same filesystem, validates the complete bundle and publishes it by content address:

~~~text
artifacts/releases/
  active.json
  attestations/
    sha256-<attestation-digest>.json
  sha256-<release-content-address>/
    academic.sqlite3
    dataset_manifest.json
    retrieval/
      <dataset-version>/
        documents.jsonl
        retrieval_manifest.json
        doc_ids.json
        vectors.npy
        faiss.index
    release_manifest.json
~~~

Lexical diagnostic candidates may omit dense files. Promotion candidates must be hybrid and bind local embedding and reranker snapshot digests. The manifest records the build-time locator for provenance, while <code>SWUFE_EMBEDDING_MODEL</code> and <code>SWUFE_RERANKER_MODEL</code> may point evaluation or runtime at different host/container paths only when the directory digests still match.

## Candidate build

All builds are candidates. <code>--release-tier production</code> is rejected.

A candidate manifest binds:

- dataset and schema version;
- database SHA-256;
- all source-input SHA-256 values;
- evidence-state and retrieval-manifest SHA-256;
- retrieval mode and model snapshot identities;
- restricted holdout descriptor when supplied;
- Git commit, dirty status and dirty diff SHA-256;
- every file in the published directory.

Optional compatibility outputs are copies from the validated staging tree. They never replace the release manifest as the source of truth.

## Restricted holdout descriptor

A promotion candidate must bind a <code>restricted_holdout</code> manifest. The descriptor freezes role names, paths, hashes, counts, dataset version and all additional files. Its release lock contains no question text or gold answer, but its bundle digest is recomputed from the complete restricted manifest.

The checked-in <code>eval/holdout</code> fixture has <code>dataset_kind=test_fixture</code> and is intentionally ineligible for promotion.

## Evaluation and attestation

Promotion requires two reports produced from the same candidate and restricted holdout:

1. agent contract evaluation;
2. lexical and hybrid retrieval evaluation.

The runners derive their inputs from the candidate, use promotion-policy v1 and require the evaluator to be clean at the candidate Git commit. A report with missing data, skipped hybrid, weakened thresholds, wrong sample counts, scope leakage or hard-negative hits is not promotion-eligible.

<code>scripts.create_eval_attestation</code> validates both reports and signs a redacted subject with an Ed25519 private key. The subject includes the candidate release, manifest, database, dataset, retrieval, holdout, model snapshots, evaluator Git state, policy digest and both report digests.

Private signing keys never enter release artifacts. Promotion receives a public key and explicit issuer trust.

## Atomic promotion

<code>scripts.promote_release</code> performs:

1. immutable release validation;
2. attestation strict-JSON loading;
3. signature and trusted-issuer verification;
4. exact subject comparison;
5. content-addressed attestation publication;
6. atomic replacement of <code>active.json</code>.

The active pointer binds the release manifest digest and promotion attestation digest. Any failure leaves the previous pointer unchanged.

Runtime startup verifies the same chain again. <code>SWUFE_DEPLOYMENT_MODE=production</code> forbids <code>SWUFE_ALLOW_UNATTESTED_ACTIVE</code>.

## Rollback

Published <code>sha256-*</code> directories are immutable. A correction creates a new release. Rollback means atomically selecting a previously verified release and its valid attestation, under an operator-controlled change process; it does not mean editing an existing release directory or manifest.

## Reproducibility limits

Release hashes cover repository-built data and artifacts, and hybrid candidates bind local model directory digests. Complete environment reproducibility additionally requires digest-pinned container bases, OS/native-library provenance, hardware details and an external model-distribution policy. The current container tags and deployment environment are not claimed to be bit-for-bit reproducible.

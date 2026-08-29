# Data model

## Canonical projection

The canonical SQLite projection separates:

- <code>sources</code> and <code>source_sections</code>;
- <code>programs</code>, <code>program_aliases</code>;
- <code>modules</code>, <code>module_aliases</code>;
- <code>courses</code>, <code>course_aliases</code>;
- <code>program_courses</code>;
- <code>requirements</code>.

Aliases are data in SQLite plus <code>config/entity_aliases.json</code>; program-specific conditions are not hard-coded in the planner.

## Provenance and trust

Every structured record retains source identity and enough lineage to find the supporting bytes or page region. Depending on the record, this includes source SHA-256, page, table row, cell, text span, parser version, extraction time, confidence and review status.

Sources record authority, publication and effective dates, current status and optional supersession. Trust is not inferred from a filename or a row's self-declared status. Source and evidence reviewer ledgers are separate inputs, and field materialization verifies their relationship to the observed source hash and chunk.

The reviewer fields provide an auditable data contract. Identity proof, authorization, two-person approval, signatures and append-only compliance storage belong to the external governance system and are not supplied by this repository.

## Generated artifacts

<code>python -m scripts.build_all</code> publishes one immutable candidate under:

~~~text
artifacts/releases/
  sha256-<release-content-address>/
    academic.sqlite3
    dataset_manifest.json
    retrieval/<dataset-version>/
    release_manifest.json
~~~

Optional <code>--database</code>, <code>--retrieval-root</code> and <code>--manifest-dir</code> outputs are compatibility materializations. They are not the authoritative runtime identity. A promoted release is selected only through <code>artifacts/releases/active.json</code> and its signed evaluation attestation.

Generated databases, indexes, model snapshots, raw school documents, restricted holdout files and reports remain outside Git.

## Data classes in this repository

| Class | Checked into Git | Permitted claim |
| --- | --- | --- |
| Synthetic canonical fixture | Yes | Deterministic contract and CI coverage |
| Public test-fixture holdout | Yes | Hash, schema, scope, hybrid and provenance smoke |
| Official school corpus | No | No repository-visible coverage or quality claim |
| Restricted holdout | No | Promotion input supplied by an authorized operator |
| Student or live teaching data | No | Unsupported by the runtime |

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eval import candidate_release
from eval.holdout import (
    HoldoutContractError,
    validate_restricted_release_lock,
)
from storage.json_contract import (
    StrictJSONError,
    canonical_json,
    load_strict_json_snapshot,
    loads_strict_json,
    strict_json_object,
)


def _restricted_lock() -> dict[str, object]:
    inputs = {
        "agent_cases": {
            "path": "agent_cases.json",
            "sha256": "a" * 64,
            "count": 2,
        },
        "retrieval_documents": {
            "path": "retrieval_documents.jsonl",
            "sha256": "b" * 64,
            "count": 4,
        },
        "retrieval_queries": {
            "path": "retrieval_queries.json",
            "sha256": "c" * 64,
            "count": 3,
        },
    }
    additional_files = {"labels.json": "d" * 64}
    bundle_sha256 = hashlib.sha256(
        canonical_json(
            {
                "inputs": inputs,
                "additional_files": additional_files,
            }
        )
    ).hexdigest()
    return {
        "status": "frozen",
        "holdout_contract_version": "2",
        "kind": "restricted_holdout",
        "holdout_id": "restricted-v1",
        "dataset_version": "dataset-v1",
        "access": "restricted",
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": "e" * 64,
        "inputs": inputs,
        "additional_files": additional_files,
    }


@pytest.mark.parametrize(
    "raw, message",
    [
        ('{"release_id":"first","release_id":"second"}', "duplicate JSON key"),
        ('{"metric":NaN}', "non-finite JSON number"),
        ('{"metric":Infinity}', "non-finite JSON number"),
        ('{"metric":-Infinity}', "non-finite JSON number"),
    ],
)
def test_strict_json_rejects_ambiguous_or_nonstandard_values(
    raw: str,
    message: str,
) -> None:
    with pytest.raises(StrictJSONError, match=message):
        loads_strict_json(raw)


def test_canonical_json_and_object_contract_reject_non_json_values() -> None:
    with pytest.raises(StrictJSONError, match="finite canonical JSON"):
        canonical_json({"metric": float("nan")})
    with pytest.raises(StrictJSONError, match="keys must be strings"):
        strict_json_object({1: "not-a-json-object"}, label="signed payload")


def test_strict_snapshot_hashes_the_same_bytes_it_parses(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    raw = b'{"contract":"1","value":3}\n'
    path.write_bytes(raw)

    parsed, digest, observed = load_strict_json_snapshot(path)

    assert parsed == {"contract": "1", "value": 3}
    assert observed == raw
    assert digest == hashlib.sha256(raw).hexdigest()


def test_restricted_release_lock_recomputes_its_bundle_digest() -> None:
    lock = _restricted_lock()
    assert validate_restricted_release_lock(lock) == lock

    tampered = {**lock, "bundle_sha256": "0" * 64}
    with pytest.raises(HoldoutContractError, match="bundle digest does not match"):
        validate_restricted_release_lock(tampered)

    fixture = {**lock, "kind": "test_fixture", "access": "public"}
    with pytest.raises(HoldoutContractError, match="must be restricted"):
        validate_restricted_release_lock(fixture)


def test_candidate_release_path_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "release"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "payload.json").write_text("{}", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable on this Windows host")

    with pytest.raises(
        candidate_release.CandidateEvaluationError,
        match="symlink component",
    ):
        candidate_release._release_path(
            root,
            "linked/payload.json",
            label="candidate payload",
        )

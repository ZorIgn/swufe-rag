"""Rebind an index manifest only after exact semantic chunk equivalence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from retrieval.index import file_sha256


def main() -> None:
    chunks_path = Path("data/chunks.jsonl")
    stored_path = Path("artifacts/chunks.json")
    manifest_path = Path("artifacts/manifest.json")
    current = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    if current != stored:
        raise RuntimeError(
            "artifact chunks differ in content or order; rebuild embeddings instead"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("chunk_count") != len(current):
        raise RuntimeError("artifact manifest row count does not match stored chunks")
    manifest["chunks_sha256"] = file_sha256(chunks_path)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    print(
        json.dumps(
            {
                "semantic_rows_verified": len(current),
                "chunks_sha256": manifest["chunks_sha256"],
                "vectors_reused": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Build the canonical database and immutable manifest from versioned inputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from academic.database import build_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/academic.sqlite3"))
    parser.add_argument("--catalog", type=Path, default=Path("data/curriculum_catalog.json"))
    parser.add_argument("--sources", type=Path, default=Path("data/sources.csv"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--aliases", type=Path, default=Path("config/entity_aliases.json"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    args = parser.parse_args()
    report = build_database(args.database, catalog_path=args.catalog, sources_path=args.sources, chunks_path=args.chunks, aliases_path=args.aliases)
    with sqlite3.connect(args.database) as connection:
        page_count = int(connection.execute("SELECT count(DISTINCT source_id || ':' || physical_page) FROM source_sections WHERE physical_page IS NOT NULL").fetchone()[0])
    version = f"{report['dataset_version']}-{str(report['catalog_sha256'])[:12]}"
    manifest = {
        "dataset_version": version,
        "schema_version": report["schema_version"],
        "parser_version": report["parser_version"],
        "source_count": sum(1 for _ in args.sources.open(encoding="utf-8-sig")) - 1,
        "source_hashes": {"catalog": report["catalog_sha256"], "sources": report["sources_sha256"], "chunks": report["chunks_sha256"]},
        "page_count": page_count,
        "chunk_count": report["chunk_count"],
        "program_count": report["program_count"],
        "course_count": report["offering_count"],
        "requirement_count": report["requirement_count"],
        "embedding_model": None,
        "embedding_dimension": None,
        "index_hash": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest_dir / f"{version}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"database": str(args.database), "manifest": str(path), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()

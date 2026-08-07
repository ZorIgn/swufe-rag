"""Obtain externally distributed data without committing generated artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

REQUIRED = ("sources.csv", "chunks.jsonl", "curriculum_catalog.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--url", default=os.getenv("SWUFE_DATASET_URL"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        for name in REQUIRED:
            source = args.source_dir / name
            if not source.is_file():
                raise SystemExit(f"dataset source is missing {source}")
            shutil.copy2(source, args.data_dir / name)
    elif args.url:
        archive = args.data_dir / "dataset.zip"
        urlretrieve(args.url, archive)  # nosec B310 - user-controlled explicit dataset URL
        shutil.unpack_archive(archive, args.data_dir)
        archive.unlink()
    missing = [name for name in REQUIRED if not (args.data_dir / name).is_file()]
    if missing:
        raise SystemExit(
            "dataset unavailable; configure --source-dir or SWUFE_DATASET_URL: "
            + ", ".join(missing)
        )
    print("dataset ready: " + str(args.data_dir.resolve()))


if __name__ == "__main__":
    main()

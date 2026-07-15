"""CLI for producing the frozen Module A knowledge-chunk contract."""

from __future__ import annotations

import argparse
import json

from ingest.pipeline import ingest_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/chunks.jsonl from reviewed sources")
    parser.add_argument("--sources", default="data/sources.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/chunks.jsonl")
    parser.add_argument("--ocr-dir", default="data/ocr")
    parser.add_argument("--report", default="data/ingest_report.json")
    parser.add_argument("--chunk-max-len", type=int, default=500)
    args = parser.parse_args()
    report = ingest_sources(
        args.sources,
        args.raw_dir,
        args.output,
        ocr_dir=args.ocr_dir,
        report_path=args.report,
        chunk_max_len=args.chunk_max_len,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

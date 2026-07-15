from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from contracts import ContractError
from ingest.sources import SOURCE_FIELDS, load_sources


class SourceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        (self.raw / "school").mkdir(parents=True)
        (self.raw / "school" / "policy.txt").write_text("第一条 测试。", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, **overrides: str) -> Path:
        row = {
            "file": "school/policy.txt",
            "doc_title": "本科测试规定",
            "level": "校级",
            "college": "全校",
            "cohort": "不限",
            "year": "2026",
            "status": "现行",
            "page_url": "https://jwc.swufe.edu.cn/info/1.htm",
            "file_url": "https://jwc.swufe.edu.cn/policy.txt",
            "collected_at": "2026-07-14",
        }
        row.update(overrides)
        path = self.root / "sources.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_valid_registry_resolves_relative_source(self) -> None:
        records = load_sources(self._write(), raw_dir=self.raw)
        self.assertEqual(records[0].file, "school/policy.txt")
        self.assertEqual(records[0].year, 2026)

    def test_old_absolute_windows_path_is_rejected_with_context(self) -> None:
        with self.assertRaisesRegex(ContractError, r"line=2.*field=file"):
            load_sources(
                self._write(file=r"G:\old\raw\policy.txt"),
                raw_dir=self.raw,
            )

    def test_non_school_url_and_invalid_enums_fail_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "official swufe.edu.cn"):
            load_sources(
                self._write(file_url="https://example.com/policy.txt"),
                raw_dir=self.raw,
            )
        with self.assertRaisesRegex(ContractError, "校级, 院级"):
            load_sources(self._write(level="校"), raw_dir=self.raw)

    def test_unconverted_doc_is_rejected_before_parsing(self) -> None:
        with self.assertRaisesRegex(ContractError, "convert DOC/ZIP"):
            load_sources(self._write(file="school/legacy.doc"), raw_dir=self.raw)


if __name__ == "__main__":
    unittest.main()

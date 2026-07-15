from __future__ import annotations

from pathlib import Path
import sqlite3
import unittest

from retrieval.index import load_chunks
from storage.metadata_db import MetadataDB


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class MetadataDBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = load_chunks(FIXTURE_PATH)
        self.db = MetadataDB.from_chunks(
            self.chunks, trusted_by_default=True
        )

    def tearDown(self) -> None:
        self.db.close()

    def test_sql_scope_is_applied_before_embedding_rows_are_returned(self) -> None:
        rows = self.db.candidate_rows(
            college="计算机与人工智能学院",
            cohort="2023",
        )
        selected = [self.chunks[index] for index in rows]
        self.assertTrue(selected)
        self.assertTrue(all(chunk["status"] == "现行" for chunk in selected))
        self.assertTrue(
            all(
                chunk["level"] == "校级"
                or chunk["college"] == "计算机与人工智能学院"
                for chunk in selected
            )
        )
        self.assertTrue(
            all(chunk["cohort"] in {"不限", "2023"} for chunk in selected)
        )
        self.assertNotIn("fixture_fin_recommend_019", {c["chunk_id"] for c in selected})
        self.assertNotIn("fixture_it_recommend_old_014", {c["chunk_id"] for c in selected})

    def test_explicit_policy_year_can_select_historical_version(self) -> None:
        rows = self.db.candidate_rows(
            college="计算机与人工智能学院",
            policy_year=2024,
            topic="promotion",
        )
        ids = {self.chunks[index]["chunk_id"] for index in rows}
        self.assertEqual(ids, {"fixture_it_recommend_old_014"})

    def test_disabled_or_untrusted_source_never_enters_candidate_set(self) -> None:
        stored = self.db.chunk("fixture_it_recommend_013")
        self.assertIsNotNone(stored)
        self.db.set_source_state(stored.source_id, trusted=False)
        rows = self.db.candidate_rows(
            college="计算机与人工智能学院", topic="promotion"
        )
        ids = {self.chunks[index]["chunk_id"] for index in rows}
        self.assertNotIn("fixture_it_recommend_013", ids)
        self.assertIsNone(self.db.chunk("fixture_it_recommend_013"))

    def test_schema_rejects_invalid_level_and_boolean_state(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_key, doc_title, level, college, cohort,
                    year, status, topic, page_url, file_url, trusted, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bad",
                    "bad",
                    "bad",
                    "部门级",
                    "全校",
                    "不限",
                    2026,
                    "现行",
                    "school_policy",
                    "https://jwc.swufe.edu.cn/bad",
                    "https://jwc.swufe.edu.cn/bad",
                    2,
                    1,
                ),
            )

    def test_integrity_report_has_no_orphans(self) -> None:
        report = self.db.integrity_report()
        self.assertEqual(report["chunks"], len(self.chunks))
        self.assertEqual(report["orphan_chunks"], 0)
        self.assertEqual(report["eligible_untrusted"], 0)

    def test_scope_values_are_bound_parameters_not_executable_sql(self) -> None:
        before = self.db.connection.execute(
            "SELECT count(*) FROM sources"
        ).fetchone()[0]
        rows = self.db.candidate_rows(topic="promotion' OR 1=1 --")
        self.assertEqual(rows, [])
        self.assertEqual(
            self.db.connection.execute("SELECT count(*) FROM sources").fetchone()[0],
            before,
        )


if __name__ == "__main__":
    unittest.main()

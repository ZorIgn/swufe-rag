from __future__ import annotations

from pathlib import Path
import unittest


REFERENCE = Path(__file__).parents[1] / "API_REFERENCE.md"


class APIReferenceTests(unittest.TestCase):
    def test_reference_lists_every_exposed_http_and_python_entry(self) -> None:
        text = REFERENCE.read_text(encoding="utf-8")
        required = (
            "retrieve()",
            "retrieve_scoped()",
            "answer()",
            "route_question()",
            "HybridRuntime.handle_question()",
            "app.server.create_app(runtime=None)",
            "`POST` | `/ask`",
            "`GET` | `/options`",
            "`GET` | `/source/{chunk_id}`",
            "`GET` | `/api/debug/health`",
            "`GET` | `/api/debug/options`",
            "`GET` | `/api/debug/examples`",
            "`POST` | `/api/debug/retrieve`",
            "`POST` | `/api/debug/ask`",
            "`GET` | `/api/debug/source/{chunk_id}`",
            "python -m ingest",
            "python -m retrieval.index",
            "python -m app.server",
            "python -m app.debug_server",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()

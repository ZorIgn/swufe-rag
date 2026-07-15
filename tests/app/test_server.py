from __future__ import annotations

import importlib.util
from unittest import mock
import unittest

from contracts import CHUNK_FIELDS, ContractError, KnowledgeBaseNotReadyError


HAS_WEB_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx")
)


@unittest.skipUnless(HAS_WEB_DEPS, "install requirements-web.txt to run HTTP tests")
class ProductServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient
        from app.runtime import build_demo_hybrid_runtime
        from app.server import create_app

        cls.client = TestClient(create_app(build_demo_hybrid_runtime()))

    def test_ask_has_formal_d_shape_without_debug_fields(self) -> None:
        response = self.client.post(
            "/ask",
            json={
                "question": "CS205是什么课，多少学分？",
                "college": "计算机与人工智能学院",
                "cohort": "2023",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {
                "mode",
                "answer_md",
                "citations",
                "refused",
                "retrieved",
                "official_links",
                "latency_ms",
            },
        )
        self.assertEqual(payload["mode"], "school_rag")
        self.assertEqual(payload["retrieved"][0]["chunk_id"], "fixture_it_table_011")

    def test_general_question_bypasses_rag_in_formal_api(self) -> None:
        response = self.client.post(
            "/ask", json={"question": "什么是注意力机制"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "general_chat")
        self.assertFalse(payload["refused"])
        self.assertEqual(payload["citations"], [])
        self.assertEqual(payload["retrieved"], [])

    def test_session_follow_up_stays_in_school_rag(self) -> None:
        first = self.client.post(
            "/ask",
            json={
                "question": "挂科后还能推免吗",
                "college": "计算机与人工智能学院",
                "cohort": "2023",
                "session_id": "api-follow-up",
            },
        )
        self.assertEqual(first.status_code, 200)
        second = self.client.post(
            "/ask",
            json={
                "question": "那重修通过以后呢？",
                "session_id": "api-follow-up",
            },
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["mode"], "school_rag")

    def test_source_returns_exact_knowledge_chunk(self) -> None:
        response = self.client.get("/source/fixture_it_table_011")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), set(CHUNK_FIELDS))
        self.assertNotIn("score", response.json())
        self.assertTrue(response.json()["page_url"].startswith("https://"))
        self.assertTrue(response.json()["file_url"].startswith("https://"))

    def test_unknown_source_is_404(self) -> None:
        response = self.client.get("/source/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_product_chat_page_and_options_are_served(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("西财教务助手", page.text)
        self.assertEqual(self.client.get("/assets/chat.js").status_code, 200)
        options = self.client.get("/options")
        self.assertEqual(options.status_code, 200)
        self.assertEqual(options.json()["mode"], "demo-hybrid")

    def test_request_does_not_accept_debug_top_k(self) -> None:
        response = self.client.post(
            "/ask",
            json={"question": "课程替代需要谁审核？", "top_k": 20},
        )
        self.assertEqual(response.status_code, 422)

    def test_default_app_surfaces_missing_production_data_as_503(self) -> None:
        from fastapi.testclient import TestClient
        from app.server import create_app

        with mock.patch(
            "app.server.build_production_hybrid_runtime",
            side_effect=KnowledgeBaseNotReadyError("production knowledge base missing"),
        ):
            response = TestClient(create_app()).post(
                "/ask", json={"question": "课程替代需要谁审核？"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("production knowledge base missing", response.json()["detail"])

    def test_default_app_surfaces_invalid_production_data_as_503(self) -> None:
        from fastapi.testclient import TestClient
        from app.server import create_app

        with mock.patch(
            "app.server.build_production_hybrid_runtime",
            side_effect=ContractError("invalid chunk", line_number=7, field="cohort"),
        ):
            response = TestClient(create_app()).post(
                "/ask", json={"question": "课程替代需要谁审核？"}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("line=7", response.json()["detail"])
        self.assertIn("field=cohort", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

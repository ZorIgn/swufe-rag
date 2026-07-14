from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


HAS_WEB_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx")
)


@unittest.skipUnless(HAS_WEB_DEPS, "install requirements-web.txt to run HTTP tests")
class DebugServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient
        from app.debug_server import create_app
        from app.runtime import build_demo_runtime

        cls.client = TestClient(create_app(build_demo_runtime()))

    def test_health_and_options(self) -> None:
        response = self.client.get("/api/debug/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "demo")

    def test_ask_and_source(self) -> None:
        response = self.client.post(
            "/api/debug/ask",
            json={
                "question": "CS205是什么课，多少学分？",
                "college": "计算机与人工智能学院",
                "cohort": "2023",
                "top_k": 5,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["refused"])
        chunk_id = payload["citations"][0]["chunk_id"]
        source = self.client.get(f"/api/debug/source/{chunk_id}")
        self.assertEqual(source.status_code, 200)
        self.assertIn("CS205", source.json()["text"])

    def test_static_debug_console_is_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("证据调试台", response.text)


class StaticAssetTests(unittest.TestCase):
    def test_assets_exist_without_web_dependencies(self) -> None:
        static = Path(__file__).parents[2] / "app" / "static"
        for name in ("debug.html", "debug.css", "debug.js"):
            self.assertGreater((static / name).stat().st_size, 500)

    def test_answer_state_and_source_panel_have_explicit_reset_rules(self) -> None:
        static = Path(__file__).parents[2] / "app" / "static"
        css = (static / "debug.css").read_text(encoding="utf-8")
        javascript = (static / "debug.js").read_text(encoding="utf-8")
        self.assertIn("[hidden] { display: none !important; }", css)
        self.assertIn("function resetSourcePanel()", javascript)
        self.assertIn("resetSourcePanel();", javascript)


if __name__ == "__main__":
    unittest.main()

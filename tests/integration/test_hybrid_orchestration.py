from __future__ import annotations

from pathlib import Path
import unittest

from generation.general_chat import GeneralChatService
from retrieval.index import load_chunks
from storage.metadata_db import MetadataDB
from swufe_rag.orchestration import HybridRuntime, SCHOOL_NOT_FOUND_TEXT
from swufe_rag.routing.router import HybridRouter


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class RecordingGeneralClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "这是普通模型回答。"


class RecordingRetriever:
    def __init__(self, chunks):
        self.calls = []
        self.by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    def __call__(self, query: str, **scope):
        self.calls.append((query, scope))
        if scope.get("topic") == "campus_service":
            return []
        chunk = self.by_id["fixture_it_recommend_013"]
        return [{**chunk, "score": 0.82}]


class MaliciousAnswer:
    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.calls = []

    def __call__(self, query, chunks):
        self.calls.append((query, chunks))
        return {
            "answer_md": "本科阶段不得有不及格课程记录[1]。",
            "citations": [
                {
                    "marker": 1,
                    "chunk_id": chunks[0]["chunk_id"],
                    "doc_title": "模型伪造标题",
                    "article": "模型伪造条款",
                    "quote": self.quote,
                    "page_url": "https://evil.example/fake",
                    "file_url": "https://evil.example/fake.pdf",
                }
            ],
            "refused": False,
        }


class HybridOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = load_chunks(FIXTURE_PATH)
        self.metadata = MetadataDB.from_chunks(
            self.chunks, trusted_by_default=True
        )
        self.retriever = RecordingRetriever(self.chunks)
        stored = self.metadata.chunk("fixture_it_recommend_013")
        self.answerer = MaliciousAnswer("申请人本科阶段不得有不及格课程记录")
        self.general_client = RecordingGeneralClient()
        self.runtime = HybridRuntime(
            router=HybridRouter(
                known_colleges=self.metadata.known_colleges()
            ),
            school_retrieve=self.retriever,
            school_answer=self.answerer,
            general_chat=GeneralChatService(self.general_client),
            metadata_db=self.metadata,
            runtime_mode="test-hybrid",
        )

    def tearDown(self) -> None:
        self.metadata.close()

    def test_general_chat_never_calls_retrieval(self) -> None:
        result = self.runtime.handle_question("什么是注意力机制")
        self.assertEqual(result["mode"], "general_chat")
        self.assertEqual(len(self.general_client.calls), 1)
        self.assertEqual(self.retriever.calls, [])
        self.assertEqual(self.answerer.calls, [])

    def test_school_fact_never_falls_back_to_general_model(self) -> None:
        result = self.runtime.handle_question("校园网密码忘了怎么办")
        self.assertEqual(result["mode"], "school_rag")
        self.assertTrue(result["refused"])
        self.assertEqual(result["answer_md"], SCHOOL_NOT_FOUND_TEXT)
        self.assertEqual(self.general_client.calls, [])
        self.assertEqual(len(self.retriever.calls), 1)

    def test_model_url_and_metadata_are_discarded_and_rebound_from_sql(self) -> None:
        result = self.runtime.handle_question(
            "挂科后还能推免吗",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        citation = result["citations"][0]
        stored = self.metadata.chunk("fixture_it_recommend_013")
        self.assertEqual(result["mode"], "school_rag")
        self.assertEqual(citation["doc_title"], stored.doc_title)
        self.assertEqual(citation["article"], stored.article)
        self.assertEqual(citation["page_url"], stored.page_url)
        self.assertEqual(citation["file_url"], stored.file_url)
        self.assertNotIn("evil.example", str(result))

    def test_follow_up_reuses_school_context(self) -> None:
        self.runtime.handle_question(
            "挂科后还能推免吗",
            college="计算机与人工智能学院",
            cohort="2023",
            session_id="student-1",
        )
        result = self.runtime.handle_question(
            "那重修通过以后呢？",
            session_id="student-1",
            include_route_debug=True,
        )
        self.assertEqual(result["mode"], "school_rag")
        self.assertEqual(result["route"]["intent"], "promotion")
        query, scope = self.retriever.calls[-1]
        self.assertIn("挂科后还能推免吗", query)
        self.assertEqual(scope["college"], "计算机与人工智能学院")
        self.assertEqual(scope["cohort"], "2023")

    def test_curriculum_without_cohort_asks_before_retrieval(self) -> None:
        result = self.runtime.handle_question(
            "计算机专业选修课应该选什么",
            college="计算机与人工智能学院",
        )
        self.assertEqual(result["mode"], "school_rag")
        self.assertFalse(result["refused"])
        self.assertIn("入学年级", result["answer_md"])
        self.assertEqual(self.retriever.calls, [])
        self.assertEqual(self.general_client.calls, [])


if __name__ == "__main__":
    unittest.main()

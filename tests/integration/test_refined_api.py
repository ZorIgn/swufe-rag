from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from generation.pipeline import AdvancedGenerationService
from retrieval.embed import HashingEncoder
from retrieval.index import build_index, load_index
from retrieval.pipeline import AdvancedRetriever
from retrieval.retriever import HybridRetriever
from swufe_rag.api import answer, configure, retrieve
from tests.generation.helpers import FakeClient


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class RefinedPublicAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        encoder = HashingEncoder(512)
        build_index(
            FIXTURE_PATH,
            cls.temporary.name,
            encoder,
            allow_test_backend=True,
        )
        bundle = load_index(
            FIXTURE_PATH,
            cls.temporary.name,
            encoder,
            allow_test_backend=True,
        )
        cls.pipeline = AdvancedRetriever(HybridRetriever(bundle, encoder))

    @classmethod
    def tearDownClass(cls) -> None:
        configure(retriever=None, generation=None)
        cls.temporary.cleanup()

    def test_canonical_facade_runs_refined_b_to_c_flow(self) -> None:
        client = FakeClient(
            [
                "本科阶段不得有不及格课程记录[1]。"
                "综合成绩由学业成绩70%和综合素质30%构成[1]。"
            ]
        )
        configure(
            retriever=self.pipeline,
            generation=AdvancedGenerationService(client, refuse_th=0.2),
        )
        chunks = retrieve(
            "挂科还能保研吗",
            college="计算机与人工智能学院",
            cohort="2023",
        )
        self.assertEqual(chunks[0]["chunk_id"], "fixture_it_recommend_013")
        result = answer("挂科还能保研吗", chunks)
        self.assertFalse(result["refused"])
        self.assertEqual(result["citations"][0]["chunk_id"], chunks[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
import unittest

from contracts import KnowledgeBaseNotReadyError
from retrieval.embed import HashingEncoder
from retrieval.index import build_index, load_index


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "chunks.jsonl"


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.chunks_path = root / "chunks.jsonl"
        self.artifacts = root / "artifacts"
        shutil.copyfile(FIXTURE_PATH, self.chunks_path)
        self.encoder = HashingEncoder(256)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_and_load_test_artifacts(self) -> None:
        manifest = build_index(
            self.chunks_path,
            self.artifacts,
            self.encoder,
            allow_test_backend=True,
        )
        self.assertEqual(manifest["chunk_count"], 24)
        self.assertEqual(manifest["backend"], "numpy-test-only")
        bundle = load_index(
            self.chunks_path,
            self.artifacts,
            self.encoder,
            allow_test_backend=True,
        )
        self.assertEqual(len(bundle.chunks), 24)
        self.assertEqual(bundle.embeddings.shape, (24, 256))

    def test_test_backend_is_rejected_by_production_loader(self) -> None:
        build_index(
            self.chunks_path,
            self.artifacts,
            self.encoder,
            allow_test_backend=True,
        )
        with self.assertRaisesRegex(KnowledgeBaseNotReadyError, "test-only"):
            load_index(self.chunks_path, self.artifacts, self.encoder)

    def test_source_hash_mismatch_requires_rebuild(self) -> None:
        build_index(
            self.chunks_path,
            self.artifacts,
            self.encoder,
            allow_test_backend=True,
        )
        with self.chunks_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(KnowledgeBaseNotReadyError, "rebuild"):
            load_index(
                self.chunks_path,
                self.artifacts,
                self.encoder,
                allow_test_backend=True,
            )


if __name__ == "__main__":
    unittest.main()


from types import SimpleNamespace
from pathlib import Path

from app.runtime_v14 import _authoritative_chunks
from generation.promotion_conditions import canonical, complete
from storage.metadata_db import MetadataDB


def test_promotion_conditions_require_all_authoritative_subclauses() -> None:
    root = Path(__file__).parents[2]
    metadata = MetadataDB(root / "data" / "metadata.sqlite3")
    chunks = _authoritative_chunks(SimpleNamespace(metadata_db=metadata))
    answer = canonical(chunks)
    assert answer is not None
    assert complete(answer)
    assert len(answer["citations"]) == 5
    assert "75 分及以上" in answer["answer_md"]
    assert "原文件第2页" in answer["citations"][0]["article"]
    metadata.close()

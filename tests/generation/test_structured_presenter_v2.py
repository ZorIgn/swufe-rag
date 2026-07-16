from __future__ import annotations

from generation.structured_presenter import StructuredAnswerPresenter


class Client:
    def __init__(self, value: str):
        self.value = value

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.value


CANONICAL = "第6学期：CST345 知识图谱与应用，3学分，选修[1]。"
CITATIONS = [
    {
        "marker": 1,
        "chunk_id": "c1",
        "doc_title": "培养方案",
        "article": "原文件第467页",
        "quote": "CST345 知识图谱与应用 3.0 选修 6",
        "page_url": "https://example.edu/page",
        "file_url": "https://example.edu/file.pdf",
    }
]


def test_faithful_wording_is_accepted():
    presenter = StructuredAnswerPresenter(
        Client("知识图谱与应用（CST345）在第6学期开设，为3学分选修课[1]。")
    )
    answer, called, error = presenter.present("什么时候开？", CANONICAL, CITATIONS)
    assert called is True
    assert error is None
    assert answer != CANONICAL


def test_changed_credit_falls_back_to_canonical():
    presenter = StructuredAnswerPresenter(
        Client("知识图谱与应用（CST345）在第6学期开设，为2学分选修课[1]。")
    )
    answer, called, error = presenter.present("什么时候开？", CANONICAL, CITATIONS)
    assert called is True
    assert error == "fact_validation_failed"
    assert answer == CANONICAL


def test_generated_url_is_rejected():
    presenter = StructuredAnswerPresenter(
        Client("第6学期 CST345，3学分[1]。https://fake.example")
    )
    answer, _, error = presenter.present("什么时候开？", CANONICAL, CITATIONS)
    assert error == "fact_validation_failed"
    assert answer == CANONICAL

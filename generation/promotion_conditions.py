"""Completeness guard and canonical evidence packet for promotion conditions."""

from __future__ import annotations

from typing import Any


RULE_TITLE = "西南财经大学推荐免试研究生管理办法（2024年修订）"
REQUIRED_ARTICLES = ("第四条", "第1项", "第2项", "第3项")


def complete(answer: dict[str, Any]) -> bool:
    articles = [str(item.get("article") or "") for item in answer.get("citations") or []]
    return all(any(term in article for article in articles) for term in REQUIRED_ARTICLES)


def canonical(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected: list[tuple[int, dict[str, Any]]] = []
    for marker, chunk in enumerate(chunks, 1):
        if chunk["doc_title"] != RULE_TITLE or "第四条" not in chunk["article"]:
            continue
        body = str(chunk["text"]).split("\n", 1)[-1].strip()
        if not body or body == "外语条件":
            continue
        selected.append((marker, chunk))
    articles = [chunk["article"] for _, chunk in selected]
    if not all(any(term in article for article in articles) for term in REQUIRED_ARTICLES):
        return None

    lines = ["根据《西南财经大学推荐免试研究生管理办法（2024年修订）》第四条，申请条件如下："]
    citations = []
    for marker, chunk in selected:
        body = str(chunk["text"]).split("\n", 1)[-1].strip()
        lines.append(f"\n{body}[{marker}]")
        citations.append(
            {
                "marker": marker,
                "chunk_id": chunk["chunk_id"],
                "doc_title": chunk["doc_title"],
                "article": chunk["article"],
                "quote": chunk["text"],
                "page_url": chunk["page_url"],
                "file_url": chunk["file_url"],
            }
        )
    return {"answer_md": "".join(lines), "citations": citations, "refused": False}


__all__ = ["canonical", "complete", "RULE_TITLE"]

"""Safe renderer that adds provenance outside claim text."""

from __future__ import annotations

from evidence.models import FinalAnswer


def render(answer: FinalAnswer) -> FinalAnswer:
    body = answer.answer_md.strip()
    if answer.citations:
        sources = ["### 来源"]
        for index, evidence in enumerate(answer.citations, start=1):
            page = (
                f"，第 {evidence.provenance.physical_page} 页"
                if evidence.provenance.physical_page
                else ""
            )
            link = evidence.page_url or evidence.file_url
            title = f"《{evidence.title}》{page}"
            sources.append(f"- [{index}] [{title}]({link})" if link else f"- [{index}] {title}")
        body = body + "\n\n" + "\n".join(sources)
    return answer.model_copy(update={"answer_md": body})

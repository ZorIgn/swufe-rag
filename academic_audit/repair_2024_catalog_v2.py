"""Second-pass 2024 category parser that excludes repeated statistics rows."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import academic_audit.full_catalog as base
import academic_audit.repair_2024_catalog as repair_module


CODE_RE = re.compile(r"(?<!\d)(\d{6}[A-Z]{0,3})\s*[-—]\s*", re.I)
LABEL_RE = re.compile(r"学科门类|授予学位|专业代码")


def article_majors(article: str, chunks: Iterable[dict[str, Any]]) -> list[str]:
    if article == repair_module.GENERIC_ARTICLE:
        return []
    stem = repair_module._stem(article)
    is_category = bool(re.search(r"类(?:[（(][^）)]*[）)])?$", stem))
    if not is_category:
        stem = re.sub(
            r"^西南财经大学[—-]+电子科技大学联合学士学位",
            "",
            stem,
        )
        return [base._canonical_major(stem)]
    text = " ".join(
        chunk["text"]
        for chunk in chunks
        if base._article_root(chunk["article"]) == article
        and "专业类基本信息" in chunk["article"]
    )
    start = re.search(r"专业代码\s*[:：]", text)
    block = text[start.end() :] if start else text
    stop = re.search(r"标准学制|计划学制|毕业(?:最低)?学分|学分统计", block)
    if stop:
        block = block[: stop.start()]
    matches = list(CODE_RE.finditer(block))
    names: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        value = block[match.end() : end]
        label = LABEL_RE.search(value)
        if label:
            value = value[: label.start()]
        for part in re.split(r"[/、，,]", value):
            part = part.strip(" ,，、;/；")
            if part and re.search(r"[\u4e00-\u9fff]", part):
                names.append(base._canonical_major(part))
    return list(dict.fromkeys(names)) or [base._canonical_major(stem)]


def main() -> None:
    repair_module.article_majors = article_majors
    print(json.dumps(repair_module.repair(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["article_majors"]

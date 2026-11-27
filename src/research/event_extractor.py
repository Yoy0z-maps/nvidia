"""사용자 입력(뉴스/자연어) → 구조화 이벤트."""
from __future__ import annotations

import re

from src.llm import LLM, load_prompt
from src.research.validator import quote_in_text
from src.schemas import Event, ExtractedEvent

_DATE_RE = re.compile(r"(\d{4})\s*[\.\-/년]\s*(\d{1,2})\s*[\.\-/월]\s*(\d{1,2})")


def dates_in_text(text: str) -> set[str]:
    return {f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in _DATE_RE.findall(text)}


def extract_event(llm: LLM, text: str, resolver=None) -> tuple[Event, list[str]]:
    """반환: (이벤트, 경고 목록). 입력에서 확인되지 않는 날짜/인용은 제거한다."""
    raw: ExtractedEvent = llm.parse(load_prompt("event_extraction"), text, ExtractedEvent, purpose="event_extraction")
    warnings: list[str] = []
    data = raw.model_dump()
    if data["event_date"] and data["event_date"] not in dates_in_text(text):
        warnings.append(f"입력에서 확인되지 않는 날짜 제거: {data['event_date']}")
        data["event_date"] = None
    if data["input_quote"] and not quote_in_text(data["input_quote"], text):
        warnings.append("input_quote가 입력 원문에 없어 제거")
        data["input_quote"] = ""
    names = []
    for n in data["entity_names"]:
        if n and n.replace(" ", "") in text.replace(" ", ""):
            names.append(n)
        else:
            warnings.append(f"입력에 없는 기업명 제거: {n}")
    data["entity_names"] = names
    ev = Event(**data)
    if resolver is not None:
        for n in names:
            r, _ = resolver.resolve(n)
            if r:
                ev.entity_ids.append(f"company_{r.stock_code}")
    return ev, warnings

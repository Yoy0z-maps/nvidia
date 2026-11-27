"""탐색 노드별 검색 질문 생성 (캐싱)."""
from __future__ import annotations

import hashlib

from src.llm import LLM, load_prompt
from src.schemas import SearchQueries
from src.storage import Storage


def generate_queries(llm: LLM, storage: Storage, node_name: str, node_type: str, event_summary: str,
                     used: list[str], max_queries: int = 3) -> list[str]:
    key = hashlib.sha256(f"{llm.model}|{node_type}|{node_name}|{event_summary}|{max_queries}".encode()).hexdigest()
    cached = storage.cache_get("query_generation", key)
    if cached is None:
        prompt = load_prompt("query_generation", max_queries=max_queries, used=", ".join(used[-20:]) or "없음")
        data = f"현재 노드: {node_name} ({node_type})\n이벤트: {event_summary}"
        cached = llm.parse(prompt, data, SearchQueries, purpose="query_generation").queries
        storage.cache_set("query_generation", key, cached)
    out = []
    used_set = {u.strip() for u in used}
    for q in cached:
        q = q.strip()
        if q and q not in used_set and q not in out:
            out.append(q)
    return out[:max_queries]

"""검색된 Chunk에서 관계 후보 추출."""
from __future__ import annotations

from src.llm import LLM, load_prompt
from src.schemas import CandidateRelation, CandidateRelationList, SearchResult


def format_chunks(results: list[SearchResult]) -> str:
    parts = []
    for r in results:
        parts.append(f"<chunk id=\"{r.chunk_id}\" company=\"{r.corp_name}\" stock_code=\"{r.stock_code}\" "
                     f"doc_type=\"{r.document_type}\" published=\"{r.published_at}\" section=\"{r.section_path}\">\n"
                     f"{r.raw_text}\n</chunk>")
    return "\n\n".join(parts)


def extract_relations(llm: LLM, node_name: str, event_summary: str, results: list[SearchResult],
                      max_relations: int = 6) -> list[CandidateRelation]:
    if not results:
        return []
    prompt = load_prompt("relationship_extraction", node=node_name, event=event_summary, max_relations=max_relations)
    out = llm.parse(prompt, format_chunks(results), CandidateRelationList, purpose="relationship_extraction")
    allowed = {r.chunk_id for r in results}
    rels = []
    for rel in out.relations[:max_relations]:
        # 제공되지 않은 Chunk를 근거로 든 경우 근거를 무효화 (validator에서 다시 확인)
        if rel.chunk_id is not None and rel.chunk_id not in allowed:
            rel = rel.model_copy(update={"chunk_id": None, "quote": None,
                                         "claim_type": "HYPOTHESIS" if rel.claim_type == "FACT" else rel.claim_type})
        rels.append(rel)
    return rels

"""실험군 A(Vanilla LLM), B(Standard RAG), C(Multi-hop RAG + Graph) - 통일된 출력 형식."""
from __future__ import annotations

import time

from src.llm import LLM
from src.research.relationship_extractor import format_chunks
from src.research.ripple_engine import RippleEngine
from src.research.validator import quote_in_text
from src.retrieval.retriever import Retriever
from src.schemas import CompanyImpactList
from src.storage import Storage

BASE_PROMPT = """당신은 한국 주식시장 리서치 애널리스트다. 입력 이벤트가 국내 상장 기업(KOSPI/KOSDAQ)에 미칠 직접·간접 파급효과를 분석한다.
- 최대 {max_companies}개 기업. 기업명은 한글 정식 명칭, 종목코드를 알면 기재한다.
- economic_impact는 POSITIVE/NEGATIVE/NEUTRAL/UNCERTAIN 중 하나.
- revenue_exposure는 근거가 없으면 "UNKNOWN".
- 종합 투자 점수나 상승 확률을 만들지 않는다.
{extra}"""


def _unify(arm: str, out: CompanyImpactList, elapsed_ms: int, usage: dict, extra: dict | None = None,
           max_companies: int = 7) -> dict:
    return {"arm": arm, "summary": out.summary, "elapsed_ms": elapsed_ms, "usage": usage,
            "companies": [c.model_dump() for c in out.companies[:max_companies]], **(extra or {})}


def run_vanilla(llm: LLM, event_text: str, max_companies: int = 7, as_of_date: str | None = None) -> dict:
    llm.reset_budget(None)
    t0 = time.time()
    extra = f"- 분석 시점은 {as_of_date}이다. 그 이후 정보는 사용하지 않는다." if as_of_date else ""
    out = llm.parse(BASE_PROMPT.format(max_companies=max_companies, extra=extra + "\n- cited_chunk_ids는 빈 목록으로 둔다."),
                    event_text, CompanyImpactList, purpose="baseline_vanilla")
    return _unify("A_vanilla_llm", out, int((time.time() - t0) * 1000), llm.usage.to_dict(),
                  max_companies=max_companies)


def run_standard_rag(llm: LLM, retriever: Retriever, storage: Storage, event_text: str, top_k: int = 10,
                     max_companies: int = 7, as_of_date: str | None = None) -> dict:
    llm.reset_budget(None)
    retriever.reset_budget(None)
    t0 = time.time()
    results = retriever.search(event_text, top_k=top_k, as_of_date=as_of_date)
    extra = ("- 아래 공시 Chunk만 근거로 사용하고, 근거로 쓴 chunk id를 cited_chunk_ids에 넣는다.\n"
             "- 공시에서 확인할 수 없는 관계는 UNCERTAIN으로 둔다.")
    data = f"[이벤트]\n{event_text}\n\n[공시 Chunk]\n{format_chunks(results)}"
    out = llm.parse(BASE_PROMPT.format(max_companies=max_companies, extra=extra), data, CompanyImpactList,
                    purpose="baseline_rag")
    allowed = {r.chunk_id for r in results}
    citation_checks = []
    for c in out.companies:
        bad = [i for i in c.cited_chunk_ids if i not in allowed]
        citation_checks.append({"corp_name": c.corp_name, "cited": c.cited_chunk_ids, "invalid_citations": bad})
    return _unify("B_standard_rag", out, int((time.time() - t0) * 1000), llm.usage.to_dict(),
                  {"retrieved_chunk_ids": sorted(allowed), "retrieval_calls": retriever.calls,
                   "citation_checks": citation_checks}, max_companies)


def run_multihop(engine: RippleEngine, event_text: str, seed: dict, max_companies: int = 7) -> dict:
    res = engine.run(event_text, seed)
    companies = []
    for r in res.company_results[:max_companies]:
        companies.append({
            "corp_name": r["corp_name"], "stock_code": r["stock_code"], "path_description": r["path"],
            "relationship_types": r["relationship_types"], "impact_scope": r["impact_scope"],
            "economic_impact": r.get("economic_impact", "UNCERTAIN"), "conditions": r.get("conditions", []),
            "counter_scenario": r.get("counter_scenario", ""), "technical_relevance": r.get("technical_relevance", ""),
            "revenue_exposure": r.get("revenue_exposure", "UNKNOWN"), "cited_chunk_ids": r.get("cited_chunk_ids", []),
            "verification_status": r["verification_status"], "evidence": r["evidence"],
        })
    return {"arm": "C_multihop_graph", "summary": res.summary, "elapsed_ms": res.elapsed_ms,
            "usage": res.budget["llm"], "retrieval_calls": res.budget["retrieval_calls"], "companies": companies,
            "graph": res.graph, "stopped_reason": res.stopped_reason, "used_chunk_ids": res.used_chunk_ids}


def quote_accuracy(storage: Storage, arm_output: dict) -> float | None:
    """C: evidence 인용이 원문 Chunk에 존재하는 비율. B: 인용 chunk id가 검색 결과 안에 있는 비율."""
    if arm_output["arm"] == "C_multihop_graph":
        checks = []
        for e in arm_output["graph"]["edges"]:
            for ev in e.get("evidence", []):
                ch = storage.get_chunk(ev["chunk_id"])
                checks.append(bool(ch) and quote_in_text(ev["quote"], ch["raw_text"]))
        return sum(checks) / len(checks) if checks else None
    if arm_output["arm"] == "B_standard_rag":
        cited = [(i, i in set(arm_output["retrieved_chunk_ids"])) for c in arm_output["companies"]
                 for i in c["cited_chunk_ids"]]
        return sum(ok for _, ok in cited) / len(cited) if cited else None
    return None

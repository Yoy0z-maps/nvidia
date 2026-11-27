import re
from src.discovery.company_discovery import CompanyDiscovery
from src.discovery.entity_resolution import EntityResolver
from src.graph.graph_store import GraphStore
from src.ingestion.corp_registry import CorpRegistry
from src.research.ripple_engine import RippleConfig, RippleEngine
from src.schemas import CorpInfo, VerificationStatus as VS
from tests.fake_llm import FakeLLM
from tests.test_retriever import build, seed_docs

SEED = {"corp_code": "00000001", "corp_name": "가상메모리", "stock_code": "100001"}
DOCS = [
    ("00000001", "가상메모리", "100001", "d1", "2024-03-10",
     ["당사는 HBM 메모리를 생산합니다.", "주요 장비인 TC 본더는 가상장비로부터 공급받고 있습니다."]),
    ("00000002", "가상장비", "100002", "d2", "2024-04-10", ["당사는 TC 본더를 생산하여 메모리 제조사에 공급합니다."]),
    ("00000003", "가상소재", "100003", "d3", "2024-04-11", ["당사는 반도체 패키징 소재를 생산합니다."]),
]


class StubMarket:
    def __init__(self, m):
        self.m = m

    def market_type(self, corp_code):
        return self.m.get(corp_code, "OTHER")


def make_resolver(storage):
    reg = CorpRegistry([CorpInfo(corp_code="00000002", corp_name="가상장비", stock_code="100002"),
                        CorpInfo(corp_code="00000003", corp_name="가상소재", stock_code="100003"),
                        CorpInfo(corp_code="00000004", corp_name="가상코넥스", stock_code="100004")])
    return EntityResolver(storage, reg, StubMarket({"00000002": "KOSDAQ", "00000003": "KOSPI",
                                                    "00000004": "KONEX"}))


def setup(settings, storage, only_seed=True):
    seed_docs(storage, DOCS)
    if only_seed:  # 관련 기업은 아직 지식 기반에 없음 (company 테이블에서 제거된 상태로 시작)
        storage.conn.execute("UPDATE companies SET ingestion_status='PENDING' WHERE corp_code!='00000001'")
        storage.conn.commit()
    r, vs, es, _ = build(settings, storage)
    storage.conn.execute("UPDATE companies SET ingestion_status='PENDING' WHERE corp_code!='00000001'")
    storage.conn.commit()
    return r


def chunk_id(storage, text_prefix):
    return storage.one("SELECT chunk_id FROM chunks WHERE raw_text LIKE ?", (text_prefix + "%",))["chunk_id"]


def ok_verdict(data):
    one = {"supports_claim": True, "entities_correct": True, "direction_correct": True, "direct_relation": True,
           "additional_conditions": [], "contradicting_points": [], "verdict": "VERIFIED_FACT", "explanation": ""}
    idx = [int(i) for i in re.findall(r"### 주장 (\d+)", data)]
    return {"verdicts": [{"index": i, **one} for i in idx]} if idx else one


def test_discovery_resolves_ingests_and_limits(settings, storage):
    retriever = setup(settings, storage)
    c_supply = chunk_id(storage, "주요 장비인")
    c_equip = chunk_id(storage, "당사는 TC 본더를")
    ingested = []

    def ingest(r, hop, src, reason):
        ingested.append(r.corp_name)
        storage.conn.execute("UPDATE companies SET ingestion_status='READY', discovery_hop=? WHERE corp_code=?",
                             (hop, r.corp_code))
        storage.conn.commit()
        return True

    llm = FakeLLM({
        "discovery_grounded": [{"entities": [
            {"name": "가상장비", "entity_kind": "COMPANY", "relation_to_seed": "장비 공급사", "direction": "SUPPLIER",
             "chunk_id": c_supply, "quote": "TC 본더는 가상장비로부터 공급받고"},
            {"name": "허구회사", "entity_kind": "COMPANY", "relation_to_seed": "?", "direction": "UNKNOWN",
             "chunk_id": c_supply, "quote": "원문에 없는 인용"}]}],
        "discovery_hypothesis": [{"candidates": [
            {"company_name": "가상코넥스", "exploration_direction": "공급망", "rationale": "가설"},
            {"company_name": "가상소재", "exploration_direction": "원자재", "rationale": "패키징 소재 가설"}]}],
        "discovery_verify": [{"relations": [
            {"source_name": "가상장비", "source_type": "COMPANY", "target_name": "TC 본더", "target_type": "PRODUCT",
             "relationship": "PRODUCES", "claim_type": "FACT", "chunk_id": c_equip, "quote": "TC 본더를 생산하여",
             "reasoning": "", "conditions": []}]},
            {"relations": [
            {"source_name": "가상소재", "source_type": "COMPANY", "target_name": "가상메모리", "target_type": "COMPANY",
             "relationship": "SUPPLIES_TO", "claim_type": "HYPOTHESIS", "chunk_id": None, "quote": None,
             "reasoning": "가설", "conditions": []}]}],
        "verification": ok_verdict,
    })
    gs = GraphStore()
    disc = CompanyDiscovery(storage, llm, retriever, make_resolver(storage), ingest, gs,
                            max_companies=3, max_hops=1, max_per_node=3)
    rep = disc.run(SEED)
    assert ingested == ["가상장비", "가상소재"]          # 코넥스·허구 기업 제외
    statuses = {c["candidate_name"]: c["status"] for c in storage.list_candidates()}
    assert statuses["허구회사"] == "REJECTED"
    assert statuses["가상코넥스"] == "UNRESOLVED"
    assert statuses["가상장비"] == "VERIFIED_RELATED"
    assert statuses["가상소재"] == "UNVERIFIED"         # 가설 관계는 검증 실패
    # 미검증 공급 관계를 사실로 확정하지 않음
    for e in gs.edges():
        if e.source_id == "company_100003":
            assert e.verification_status == VS.HYPOTHESIS
    # 문서 근거: 가상장비 → 가상메모리 공급 관계 (원문 인용)
    sup = [e for e in gs.edges() if e.source_id == "company_100002" and e.target_id == "company_100001"]
    assert sup and sup[0].verification_status == VS.VERIFIED_FACT


def test_discovery_respects_max_companies(settings, storage):
    retriever = setup(settings, storage)
    c_supply = chunk_id(storage, "주요 장비인")
    llm = FakeLLM({
        "discovery_grounded": [{"entities": []}],
        "discovery_hypothesis": [{"candidates": [
            {"company_name": "가상장비", "exploration_direction": "공급망", "rationale": "a"},
            {"company_name": "가상소재", "exploration_direction": "원자재", "rationale": "b"}]}],
        "discovery_verify": [{"relations": []}],
        "verification": ok_verdict,
    })
    done = []

    def ingest(r, hop, src, reason):
        done.append(r.corp_name)
        storage.conn.execute("UPDATE companies SET ingestion_status='READY' WHERE corp_code=?", (r.corp_code,))
        storage.conn.commit()
        return True

    rep = CompanyDiscovery(storage, llm, retriever, make_resolver(storage), ingest, GraphStore(),
                           max_companies=2, max_hops=2).run(SEED)
    assert done == ["가상장비"] and "최대 기업 수" in rep.stopped_reason


def test_ripple_engine_multihop_with_evidence(settings, storage):
    seed_docs(storage, DOCS)
    retriever, *_ = build(settings, storage)
    c_hbm = chunk_id(storage, "당사는 HBM")
    c_supply = chunk_id(storage, "주요 장비인")
    c_equip = chunk_id(storage, "당사는 TC 본더를")
    rels = {
        "가상메모리": [{"source_name": "가상메모리", "source_type": "COMPANY", "target_name": "HBM", "target_type": "PRODUCT",
                    "relationship": "PRODUCES", "claim_type": "FACT", "chunk_id": c_hbm, "quote": "HBM 메모리를 생산",
                    "reasoning": "", "conditions": []}],
        "HBM": [{"source_name": "HBM", "source_type": "PRODUCT", "target_name": "TC 본더", "target_type": "PRODUCT",
                 "relationship": "REQUIRES", "claim_type": "INFERENCE", "chunk_id": c_supply,
                 "quote": "주요 장비인 TC 본더는", "reasoning": "", "conditions": []}],
        "TC 본더": [{"source_name": "가상장비", "source_type": "COMPANY", "target_name": "TC 본더", "target_type": "PRODUCT",
                   "relationship": "PRODUCES", "claim_type": "FACT", "chunk_id": c_equip, "quote": "TC 본더를 생산하여",
                   "reasoning": "", "conditions": []},
                  {"source_name": "TC 본더", "source_type": "PRODUCT", "target_name": "가상메모리", "target_type": "COMPANY",
                   "relationship": "SUPPLIES_TO", "claim_type": "FACT", "chunk_id": c_supply,
                   "quote": "원문에 없는 문장", "reasoning": "", "conditions": []}],
    }

    calls = {"n": 0}

    def extract(data):
        # 현재 노드는 프롬프트(instructions)에 있으므로 검색된 chunk로 판별
        calls["n"] += 1
        if "HBM 메모리를 생산" in data and calls["n"] == 1:
            return {"relations": rels["가상메모리"]}
        if calls["n"] == 2:
            return {"relations": rels["HBM"]}
        if calls["n"] == 3:
            return {"relations": rels["TC 본더"]}
        return {"relations": []}

    BASIS_OK = c_hbm
    llm = FakeLLM({
        "event_extraction": {"event_type": "DEMAND_INCREASE", "is_hypothetical": True, "entity_names": [],
                             "industries": [], "products": [], "event_date": "2030-01-01",
                             "summary": "HBM 수요 증가", "assumptions": [], "input_quote": "HBM 수요가 증가할 경우"},
        "query_generation": lambda d: {"queries": [d.split("\n")[0].replace("현재 노드: ", "") + " 생산 장비"]},
        "relationship_extraction": extract,
        "verification": ok_verdict,
        "impact_analysis": {"summary": "요약", "caveat": "재고 수준에 따라 다름", "tiers": [
            {"tier": 1, "title": "핵심 제품", "items": [{"name": "HBM", "effect": "수요 증가 가능성"},
                                                       {"name": "NVIDIA", "effect": "예시에서 유입된 이름"}]}],
            "companies": [
            {"corp_name": "가상장비", "stock_code": "100002", "path_description": "", "relationship_types": [],
             "impact_scope": "INDIRECT", "economic_impact": "POSITIVE", "conditions": ["HBM 증설 시"],
             "counter_scenario": "수요 둔화", "technical_relevance": "TC 본더", "revenue_exposure": "매출 30%",
             "cited_chunk_ids": [999], "role": "공급사", "impact_factor": "장비 수요 증가"}]},
        "hypothesis_insights": {"insights": [
            {"title": "패키징 소재 수요 증가 가능성", "tier": 2, "kind": "원자재", "direction": "POSITIVE",
             "rationale": "적층 수 증가", "related_listed_companies": ["가상소재", "존재하지않는회사"],
             "what_to_check": "주요 원재료 매입처", "basis_chunk_ids": [BASIS_OK, 99999]}]},
    })
    engine = RippleEngine(storage, llm, retriever, make_resolver(storage),
                          RippleConfig(max_hops=3, max_llm_calls=25, max_retrieval_calls=20))
    res = engine.run("HBM 수요가 증가할 경우 관련 산업 파급효과는?", SEED)
    assert res.event["event_date"] is None                       # 입력에 없는 날짜 제거
    statuses = {(e["source_id"], e["relationship"], e["target_id"]): e["verification_status"] for e in res.graph["edges"]}
    assert statuses[("company_100001", "PRODUCES", "product_hbm")] == "VERIFIED_FACT"
    assert statuses[("product_hbm", "REQUIRES", "product_tc본더")] == "SUPPORTED_INFERENCE"
    assert statuses[("company_100002", "PRODUCES", "product_tc본더")] == "VERIFIED_FACT"
    # 원문에 없는 인용 → 확정하지 않음
    assert statuses[("product_tc본더", "SUPPLIES_TO", "company_100001")] == "INSUFFICIENT_EVIDENCE"
    names = {r["corp_name"]: r for r in res.company_results}
    assert "가상메모리" in names
    # 근거 없는 인용 chunk id는 제거되고 매출 노출도는 UNKNOWN
    if "가상장비" in names:
        assert names["가상장비"]["revenue_exposure"] == "UNKNOWN"
        assert names["가상장비"]["role"] == "공급사"
    ins = res.insights[0]
    # 참고 공시: 제공한 목록에 있는 Chunk만, 문서·섹션 정보 포함
    assert [x["chunk_id"] for x in ins["sources"]] == [c_hbm]
    assert ins["sources"][0]["report_name"] == "사업보고서" and ins["sources"][0]["section_path"] == "II. 사업의 내용"
    ev0 = next(r for r in res.company_results if r["evidence"])["evidence"][0]
    assert ev0["section_path"] and ev0["report_name"] and ev0["relation"]
    assert ins["verification_status"] == "HYPOTHESIS"
    assert ins["companies"][0]["stock_code"] == "100003" and ins["companies"][1]["resolved"] is False
    tier_names = [i["name"] for t in res.tiers for i in t["items"]]
    assert "HBM" in tier_names and "NVIDIA" not in tier_names       # 그래프에 없는 예시 기업은 제외
    hbm = next(i for t in res.tiers for i in t["items"] if i["name"] == "HBM")
    assert hbm["effect"] == "수요 증가 가능성" and res.caveat
    assert res.budget["llm"]["calls"] <= 25


def test_ripple_engine_budget_stop(settings, storage):
    seed_docs(storage, DOCS)
    retriever, *_ = build(settings, storage)
    llm = FakeLLM({
        "event_extraction": {"event_type": "DEMAND_INCREASE", "is_hypothetical": True, "entity_names": [],
                             "industries": ["메모리"], "products": ["HBM"], "event_date": None, "summary": "s",
                             "assumptions": [], "input_quote": ""},
        "query_generation": lambda d: {"queries": ["HBM"]},
        "relationship_extraction": lambda d: {"relations": []},
        "impact_analysis": {"summary": "부분 결과 요약", "companies": [], "tiers": [], "caveat": ""},
        "hypothesis_insights": {"insights": []},
    })
    res = RippleEngine(storage, llm, retriever, make_resolver(storage),
                       RippleConfig(max_llm_calls=5)).run("HBM 수요 증가", SEED)
    assert "예산" in res.stopped_reason
    assert res.summary == "부분 결과 요약"            # 최종 요약용 호출은 남겨 둔다
    assert res.budget["llm"]["calls"] <= 5


def test_discovery_rerun_reuses_existing_companies(settings, storage):
    retriever = setup(settings, storage)
    c_equip = chunk_id(storage, "당사는 TC 본더를")

    def responses():
        return {
            "discovery_grounded": lambda d: {"entities": []},
            "discovery_hypothesis": lambda d: {"candidates": [
                {"company_name": "가상장비", "exploration_direction": "공급망", "rationale": "a"}]},
            "discovery_verify": lambda d: {"relations": [
                {"source_name": "가상장비", "source_type": "COMPANY", "target_name": "TC 본더", "target_type": "PRODUCT",
                 "relationship": "PRODUCES", "claim_type": "FACT", "chunk_id": c_equip, "quote": "TC 본더를 생산하여",
                 "reasoning": "", "conditions": []}]},
            "verification": ok_verdict,
        }
    ingested = []

    def ingest(r, hop, src, reason):
        ingested.append(r.corp_name)
        storage.conn.execute("UPDATE companies SET ingestion_status='READY' WHERE corp_code=?", (r.corp_code,))
        storage.conn.commit()
        return True

    CompanyDiscovery(storage, FakeLLM(responses()), retriever, make_resolver(storage), ingest, GraphStore(),
                     max_companies=2, max_hops=1).run(SEED)
    assert ingested == ["가상장비"]
    llm2 = FakeLLM(responses())
    rep = CompanyDiscovery(storage, llm2, retriever, make_resolver(storage), ingest, GraphStore(),
                           max_companies=20, max_hops=1).run(SEED)
    assert ingested == ["가상장비"]                                   # 재수집 없음
    assert not any(p == "discovery_verify" for p, _ in llm2.calls)    # 재검증 없음
    assert rep.processed[0]["method"] == "REUSED"



def test_ripple_finds_upstream_supplier_via_reverse_edge(settings, storage):
    """공급사 -SUPPLIES_TO-> 출발기업 처럼 출발점을 향하는 관계도 따라가 공급사를 결과에 포함한다."""
    from src.graph.traversal import best_company_paths, describe_path
    from src.schemas import Edge, Evidence
    gs = GraphStore()
    a = gs.ensure_company_node("가상메모리", "100001")
    b = gs.ensure_company_node("가상장비", "100002")
    ev = [Evidence(chunk_id=1, document_id="d", quote="장비 공급", published_at="2024-01-01")]
    gs.add_edge(Edge(edge_id="e1", source_id=b, target_id=a, relationship="SUPPLIES_TO", claim_type="FACT",
                     verification_status="VERIFIED_FACT", evidence=ev, created_at="2024-01-01", hop=1))
    paths = best_company_paths(gs, a)
    assert b in paths
    assert describe_path(gs, paths[b][0], a).startswith("가상메모리 <-[SUPPLIES_TO]- 가상장비")
    assert best_company_paths(gs, a, undirected=False) == {}

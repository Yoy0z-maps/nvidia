import pytest
from pydantic import ValidationError

from src.graph.graph_store import GraphStore, path_status
from src.research.validator import RelationValidator, quote_in_text
from src.schemas import Edge, Evidence, SemanticVerdict, VerificationStatus as VS
from tests.fake_llm import FakeLLM
from tests.test_retriever import seed_docs

DOCS = [("00000001", "가상메모리", "100001", "d1", "2024-03-10",
         ["당사는 HBM 메모리를 생산하며 주요 장비는 가상장비로부터 공급받고 있습니다."]),
        ("00000002", "가상장비", "100002", "d2", "2025-02-01", ["당사는 TC 본더를 가상메모리에 납품합니다."])]


def resolver(name):
    table = {"가상메모리": {"corp_code": "00000001", "corp_name": "가상메모리", "stock_code": "100001"},
             "가상장비": {"corp_code": "00000002", "corp_name": "가상장비", "stock_code": "100002"}}
    return table.get(name)


def verdict(v="VERIFIED_FACT", supports=True, direct=True, direction=True):
    return SemanticVerdict(supports_claim=supports, entities_correct=True, direction_correct=direction,
                           direct_relation=direct, additional_conditions=[], contradicting_points=[], verdict=v,
                           explanation="")


def cand(**kw):
    base = dict(source_name="가상장비", source_type="COMPANY", target_name="가상메모리", target_type="COMPANY",
                relationship="SUPPLIES_TO", claim_type="FACT", chunk_id=1,
                quote="주요 장비는 가상장비로부터 공급받고", reasoning="", conditions=[])
    base.update(kw)
    return base


@pytest.fixture
def chunk_ids(storage):
    seed_docs(storage, DOCS)
    return {r["raw_text"][:6]: r["chunk_id"] for r in storage.query("SELECT chunk_id, raw_text FROM chunks")}


def test_quote_normalization():
    assert quote_in_text("HBM  메모리를\n생산", "당사는 HBM 메모리를 생산하며")
    assert not quote_in_text("HBM 메모리를 판매", "당사는 HBM 메모리를 생산하며")


def test_rejects_quote_not_in_source(storage, chunk_ids):
    v = RelationValidator(storage, FakeLLM({"verification": verdict()}), resolver)
    o = v.validate(cand(quote="가상장비가 독점 공급한다"))
    assert o.quote_ok is False and o.status == VS.INSUFFICIENT_EVIDENCE and not o.evidence


def test_rejects_unknown_company_and_chunk(storage, chunk_ids):
    v = RelationValidator(storage, FakeLLM({"verification": verdict()}), resolver)
    assert v.validate(cand(source_name="없는회사")).status == VS.INSUFFICIENT_EVIDENCE
    assert v.validate(cand(chunk_id=99999)).status == VS.INSUFFICIENT_EVIDENCE


def test_quote_and_semantic_are_separate(storage, chunk_ids):
    # 인용은 존재하지만 의미 검증이 뒷받침하지 않음 → 사실 확정 불가
    v = RelationValidator(storage, FakeLLM({"verification": verdict("HYPOTHESIS", supports=False)}), resolver)
    o = v.validate(cand())
    assert o.quote_ok is True and o.status == VS.HYPOTHESIS
    v2 = RelationValidator(storage, FakeLLM({"verification": verdict()}), resolver)
    o2 = v2.validate(cand())
    assert o2.status == VS.VERIFIED_FACT and o2.evidence[0].chunk_id == 1
    # 의미 검증 미실행이면 인용만으로 사실 확정하지 않음
    assert v2.validate(cand(), run_semantic=False).status == VS.HYPOTHESIS


def test_indirect_not_promoted_to_supply(storage, chunk_ids):
    v = RelationValidator(storage, FakeLLM({"verification": verdict(direct=False)}), resolver)
    assert v.validate(cand()).status == VS.HYPOTHESIS


def test_inference_capped(storage, chunk_ids):
    v = RelationValidator(storage, FakeLLM({"verification": verdict("VERIFIED_FACT")}), resolver)
    assert v.validate(cand(claim_type="INFERENCE")).status == VS.SUPPORTED_INFERENCE


def test_future_document_blocked(storage, chunk_ids):
    v = RelationValidator(storage, FakeLLM({"verification": verdict()}), resolver, as_of_date="2024-12-31")
    o = v.validate(cand(chunk_id=2, quote="TC 본더를 가상메모리에 납품"))
    assert o.status == VS.INSUFFICIENT_EVIDENCE and any("기준일" in r for r in o.reasons)


def test_edge_requires_evidence_for_verified():
    with pytest.raises(ValidationError):
        Edge(edge_id="e", source_id="a", target_id="b", relationship="PRODUCES", claim_type="FACT",
             verification_status="VERIFIED_FACT", evidence=[], created_at="2024-01-01")
    with pytest.raises(ValidationError):  # 평가 기준일 이후 근거
        Edge(edge_id="e", source_id="a", target_id="b", relationship="PRODUCES", claim_type="FACT",
             verification_status="VERIFIED_FACT", created_at="2024-01-01", as_of_date="2024-01-01",
             evidence=[Evidence(chunk_id=1, document_id="d", quote="원문 인용", published_at="2024-05-01")])


def test_path_status_weakest_and_graph_separation(tmp_path):
    gs = GraphStore()
    a = gs.ensure_company_node("가상메모리", "100001")
    b = gs.ensure_concept_node("HBM", "PRODUCT")
    c = gs.ensure_concept_node("TC 본더", "PRODUCT")
    ev = [Evidence(chunk_id=1, document_id="d1", quote="HBM 메모리를 생산", published_at="2024-03-10")]
    e1 = Edge(edge_id="e1", source_id=a, target_id=b, relationship="PRODUCES", claim_type="FACT",
              verification_status="VERIFIED_FACT", evidence=ev, created_at="2024-01-01")
    e2 = Edge(edge_id="e2", source_id=b, target_id=c, relationship="REQUIRES", claim_type="HYPOTHESIS",
              verification_status="HYPOTHESIS", created_at="2024-01-01")
    gs.add_edge(e1)
    gs.add_edge(e2)
    assert path_status([e1, e2]) == VS.HYPOTHESIS
    vp, hp = gs.save(tmp_path)
    loaded = GraphStore.load(tmp_path)
    assert {e.edge_id for e in loaded.edges()} == {"e1", "e2"}
    import json
    assert [e["edge_id"] for e in json.loads(vp.read_text())["edges"]] == ["e1"]
    assert [e["edge_id"] for e in json.loads(hp.read_text())["edges"]] == ["e2"]


def test_validate_many_uses_single_llm_call(storage, chunk_ids):
    from tests.test_discovery import ok_verdict
    llm = FakeLLM({"verification": ok_verdict})
    v = RelationValidator(storage, llm, resolver)
    outs = v.validate_many([cand(), cand(claim_type="INFERENCE"), cand(quote="원문에 없는 인용"),
                            cand(claim_type="HYPOTHESIS", chunk_id=None, quote=None)])
    assert [o.status for o in outs] == [VS.VERIFIED_FACT, VS.SUPPORTED_INFERENCE, VS.INSUFFICIENT_EVIDENCE,
                                        VS.HYPOTHESIS]
    assert llm.usage.calls == 1          # 의미 검증은 묶어서 1회


def test_validate_many_missing_verdict_not_promoted(storage, chunk_ids):
    llm = FakeLLM({"verification": lambda d: {"verdicts": []}})
    outs = RelationValidator(storage, llm, resolver).validate_many([cand()])
    assert outs[0].status == VS.HYPOTHESIS and "누락" in outs[0].reasons[-1]

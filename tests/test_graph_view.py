import json
import re

from src.graph.d3_view import graph_payload, to_d3_html
from src.graph.graph_store import GraphStore
from src.schemas import Edge, Evidence


def _graph():
    gs = GraphStore()
    a = gs.ensure_company_node("가상메모리", "100001")
    b = gs.ensure_concept_node("HBM", "PRODUCT")
    c = gs.ensure_company_node("가상장비</script>", "100002")   # 이스케이프 확인
    d = gs.ensure_concept_node("고립 노드", "TECHNOLOGY")
    e2 = gs.ensure_concept_node("다른 섬", "INDUSTRY")
    ev = [Evidence(chunk_id=1, document_id="d", quote="HBM 생산", published_at="2024-01-01")]
    gs.add_edge(Edge(edge_id="e1", source_id=a, target_id=b, relationship="PRODUCES", claim_type="FACT",
                     verification_status="VERIFIED_FACT", evidence=ev, created_at="2024-01-01"))
    gs.add_edge(Edge(edge_id="e2", source_id=b, target_id=c, relationship="REQUIRES", claim_type="HYPOTHESIS",
                     verification_status="HYPOTHESIS", created_at="2024-01-01"))
    gs.add_edge(Edge(edge_id="e3", source_id=a, target_id=b, relationship="EXPOSED_TO", claim_type="HYPOTHESIS",
                     verification_status="HYPOTHESIS", created_at="2024-01-01"))
    gs.add_edge(Edge(edge_id="e4", source_id=d, target_id=e2, relationship="OPERATES_IN", claim_type="HYPOTHESIS",
                     verification_status="HYPOTHESIS", created_at="2024-01-01"))   # 분리된 컴포넌트
    return gs


def test_payload_and_weak_filter():
    p = graph_payload(_graph())
    assert len(p["links"]) == 4 and len(p["nodes"]) == 5
    strong = graph_payload(_graph(), include_weak=False)
    assert [l["id"] for l in strong["links"]] == ["e1"] and len(strong["nodes"]) == 2
    assert next(l for l in p["links"] if l["id"] == "e2")["dash"] == "6,4"


def test_html_embeds_data_safely():
    html = to_d3_html(_graph())
    assert "forceX()" in html and "forceY()" in html and "alphaTarget(0.3)" in html
    assert "가상메모리" in html                      # 한글 이름이 그대로 들어감
    script = html.split("const data = ")[1].split(";\nconst legend")[0]
    assert "</script>" not in script
    assert json.loads(script)["nodes"]
    assert "<script src=" not in html and "https://d3js.org v7.9.0" in html   # d3를 내장 (인터넷 불필요)

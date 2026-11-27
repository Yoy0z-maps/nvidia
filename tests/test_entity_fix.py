from src.discovery.entity_resolution import EntityResolver
from src.graph.graph_store import GraphStore
from src.ingestion.corp_registry import CorpRegistry
from src.schemas import CorpInfo
from tests.test_retriever import build, seed_docs

REG = CorpRegistry([
    CorpInfo(corp_code="01238169", corp_name="오리온", stock_code="271560"),
    CorpInfo(corp_code="00652043", corp_name="한솔오리온텍", stock_code=""),
    CorpInfo(corp_code="00164779", corp_name="SK하이닉스", stock_code="000660"),
])


class Market:
    def market_type(self, corp_code):
        return "KOSPI"


def test_registry_name_inside_candidate_is_not_matched():
    corp, reason = REG.resolve_name("주식회사 한솔오리온텍")
    assert corp is None                                   # 상장사 '오리온'으로 잘못 연결하지 않음
    assert REG.resolve_name("(주)오리온코포레이션")[0] is None
    assert REG.resolve_name("SK하이닉")[0].stock_code == "000660"   # 앞부분 줄임말은 허용


def test_unlisted_company_becomes_graph_node(storage):
    res = EntityResolver(storage, REG, Market())
    assert res.resolve("주식회사 한솔오리온텍")[0] is None          # 수집 대상(상장사)은 아님
    r, reason = res.resolve_for_graph("주식회사 한솔오리온텍")
    assert r.corp_code == "00652043" and r.stock_code == "" and r.market_type == "UNLISTED"
    gs = GraphStore()
    nid = gs.ensure_company_node(r.corp_name, r.stock_code, r.corp_code)
    assert nid == "company_unlisted_00652043" and gs.g.nodes[nid]["stock_code"] is None


def test_excluded_company_not_searched(settings, storage):
    seed_docs(storage, [("00000001", "가상메모리", "100001", "d1", "2024-03-10", ["HBM 메모리 생산"]),
                        ("00000002", "가상식품", "100002", "d2", "2024-03-10", ["HBM 과자 생산"])])
    r, *_ = build(settings, storage)
    assert {x.corp_code for x in r.search("HBM 생산", top_k=5)} == {"00000001", "00000002"}
    storage.set_company_status("00000002", "EXCLUDED")
    r.storage.conn.execute("DELETE FROM kv_cache WHERE namespace='search'")
    r.storage.conn.commit()
    assert {x.corp_code for x in r.search("HBM 생산", top_k=5)} == {"00000001"}


def test_group_abbreviation_and_rename():
    reg = CorpRegistry([CorpInfo(corp_code="00164779", corp_name="SK하이닉스", stock_code="000660"),
                        CorpInfo(corp_code="00000009", corp_name="이닉스", stock_code="452400"),
                        CorpInfo(corp_code="00000010", corp_name="두산테스나", stock_code="131970")])
    assert reg.resolve_name("에스케이하이닉스(주)")[0].stock_code == "000660"   # '이닉스'가 아님
    assert reg.resolve_name("테스나")[0].stock_code == "131970"               # 그룹명 + 회사명 개명

"""파이프라인 조립 통합 테스트 (모든 외부 API는 Mock)."""
import httpx

from src.pipeline import build_app, collect_company, embed_and_index, seed_company
from tests.conftest import CharTokenizer, FakeEmbedder
from tests.fake_llm import FakeLLM
from tests.test_dart_client import _client, _disc, _zip

CORP_XML = ("<result><list><corp_code>00000001</corp_code><corp_name>테스트전자</corp_name>"
            "<stock_code>999999</stock_code><modify_date>20240101</modify_date></list></result>").encode()


def test_build_collect_embed_rerun(settings, sample_xml):
    settings.seed_stock_code = "999999"
    listing = [_disc("20240315000001", "사업보고서 (2023.12)", "20240315")]
    counts = {"document": 0}

    def handler(req):
        p = req.url.path
        if p.endswith("/corpCode.xml"):
            return httpx.Response(200, content=_zip({"CORPCODE.xml": CORP_XML}))
        if p.endswith("/company.json"):
            return httpx.Response(200, json={"status": "000", "corp_cls": "Y"})
        if p.endswith("/list.json"):
            return httpx.Response(200, json={"status": "000", "total_page": 1, "list": listing})
        if p.endswith("/document.xml"):
            counts["document"] += 1
            return httpx.Response(200, content=_zip({"20240315000001.xml": sample_xml}))
        return httpx.Response(404)

    emb = FakeEmbedder()
    app = build_app(settings, need_openai=False, embedder=emb, llm=FakeLLM({}), tokenizer=CharTokenizer(),
                    dart_client=_client(handler))
    seed = seed_company(app)
    assert seed == {"corp_code": "00000001", "corp_name": "테스트전자", "stock_code": "999999"}
    collect_company(app, seed["corp_code"], seed["corp_name"], seed["stock_code"], progress=lambda m: None)
    out = embed_and_index(app, lambda m: None)
    assert out["embedded"] > 0 and out["index_total"] == out["embedded"]
    assert app.storage.get_company("00000001")["market_type"] == "KOSPI"
    assert app.storage.get_company("00000001")["ingestion_status"] == "READY"
    res = app.retriever.search("테스트메모리 매출액", top_k=3)
    assert res and res[0].corp_name == "테스트전자"
    # 재실행: 재다운로드·재임베딩 없음
    calls = emb.calls
    collect_company(app, seed["corp_code"], seed["corp_name"], seed["stock_code"], progress=lambda m: None)
    out2 = embed_and_index(app, lambda m: None)
    assert counts["document"] == 1 and emb.calls == calls and out2["embedded"] == 0



def test_existing_interim_reports_skipped_and_not_searched(settings, storage):
    from src.pipeline import App, apply_document_policy
    from src.discovery.entity_resolution import EntityResolver
    from tests.test_retriever import build, seed_docs
    seed_docs(storage, [("00000001", "가상메모리", "100001", "a1", "2024-03-10", ["HBM 연간 생산"]),
                        ("00000001", "가상메모리", "100001", "q1", "2024-11-10", ["HBM 분기 생산"])])
    storage.conn.execute("UPDATE documents SET document_type='분기보고서' WHERE document_id='q1'")
    storage.conn.commit()
    r, vs, es, _ = build(settings, storage)
    app = App(settings, storage, None, None, None, None, es, vs, r, EntityResolver(storage), None)
    assert apply_document_policy(app, lambda m: None) == 1
    storage.conn.execute("DELETE FROM kv_cache"); storage.conn.commit()
    assert [x.raw_text for x in r.search("HBM 생산", top_k=5)] == ["HBM 연간 생산"]
    settings.collect_interim_reports = True
    assert apply_document_policy(app, lambda m: None) == 1     # 다시 켜면 복구

import datetime as dt
import io
import json
import zipfile

import httpx
import pytest

from src.ingestion.chunker import SectionChunker
from src.ingestion.collector import DisclosureCollector, classify_report
from src.ingestion.corp_registry import CorpRegistry, normalize_name, parse_corp_code_xml
from src.ingestion.dart_client import (DartAuthError, DartClient, DartRateLimit, UnsafeZipError, safe_extract_zip)
from src.ingestion.section_selector import SectionSelector
from src.schemas import DocumentType
from tests.conftest import CharTokenizer


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def _client(handler) -> DartClient:
    return DartClient("TESTKEY", transport=httpx.MockTransport(handler), min_interval=0, max_retries=1)


def test_error_xml_instead_of_zip_auth():
    c = _client(lambda r: httpx.Response(200, content=b"<result><status>010</status><message>key</message></result>"))
    with pytest.raises(DartAuthError):
        c.download_document("20240101000001")


def test_rate_limit_json(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _client(lambda r: httpx.Response(200, json={"status": "020", "message": "limit"}))
    with pytest.raises(DartRateLimit):
        c.company_info("00000001")


def test_no_data_returns_empty_list():
    c = _client(lambda r: httpx.Response(200, json={"status": "013", "message": "조회된 데이타가 없습니다."}))
    assert c.list_disclosures("00000001", "20240101", "20241231") == []


def test_pagination():
    def handler(req):
        page = int(req.url.params["page_no"])
        return httpx.Response(200, json={"status": "000", "page_no": page, "total_page": 2,
                                         "list": [{"rcept_no": f"r{page}"}]})
    c = _client(handler)
    assert [x["rcept_no"] for x in c.list_disclosures("00000001", "20240101", "20241231")] == ["r1", "r2"]


def test_api_key_not_in_exception_message():
    c = _client(lambda r: httpx.Response(200, json={"status": "100", "message": "bad"}))
    with pytest.raises(Exception) as ei:
        c.company_info("00000001")
    assert "TESTKEY" not in str(ei.value)


def test_unsafe_zip_paths_rejected():
    with pytest.raises(UnsafeZipError):
        safe_extract_zip(_zip({"../evil.xml": b"x"}))
    with pytest.raises(UnsafeZipError):
        safe_extract_zip(_zip({"/abs.xml": b"x"}))
    with pytest.raises(UnsafeZipError):
        safe_extract_zip(b"not a zip")


def test_corp_code_parsing_keeps_leading_zeros():
    xml = ("<result><list><corp_code>00164779</corp_code><corp_name>테스트하이닉스</corp_name>"
           "<stock_code>000660</stock_code><modify_date>20240101</modify_date></list>"
           "<list><corp_code>00000002</corp_code><corp_name>비상장(주)</corp_name><stock_code> </stock_code>"
           "<modify_date>20240101</modify_date></list></result>").encode()
    corps = parse_corp_code_xml(xml)
    assert corps[0].corp_code == "00164779" and corps[0].stock_code == "000660"
    reg = CorpRegistry(corps)
    assert reg.get_by_stock_code("660").corp_name == "테스트하이닉스"
    found, reason = reg.resolve_name("(주)테스트하이닉스")
    assert found and found.corp_code == "00164779"
    missing, reason = reg.resolve_name("비상장")
    assert missing is None  # 상장 종목코드 없는 기업은 제외
    assert normalize_name("주식회사 ABC Co., Ltd.") == "abc"


def test_classify_report():
    assert classify_report("사업보고서 (2024.12)") == DocumentType.ANNUAL
    assert classify_report("[기재정정]사업보고서 (2024.12)") == DocumentType.ANNUAL
    assert classify_report("반기보고서 (2025.06)") == DocumentType.HALF
    assert classify_report("단일판매ㆍ공급계약체결") == DocumentType.SUPPLY_CONTRACT
    assert classify_report("주요사항보고서(신규시설투자등)") == DocumentType.NEW_FACILITY
    assert classify_report("임원ㆍ주요주주특정증권등소유상황보고서") is None


def _disc(rcept_no, nm, dt_):
    return {"corp_code": "00000001", "corp_name": "테스트전자", "stock_code": "999999", "corp_cls": "Y",
            "report_nm": nm, "rcept_no": rcept_no, "rcept_dt": dt_, "rm": ""}


def test_collector_plan_and_idempotent_ingest(settings, storage, sample_xml):
    listing = [
        _disc("20240315000001", "사업보고서 (2023.12)", "20240315"),
        _disc("20240401000002", "[기재정정]사업보고서 (2023.12)", "20240401"),
        _disc("20240814000003", "반기보고서 (2024.06)", "20240814"),
        _disc("20240901000004", "단일판매ㆍ공급계약체결", "20240901"),
        _disc("20250101000005", "단일판매ㆍ공급계약체결", "20250101"),  # 기준일 이후
    ]
    downloads = []

    def handler(req):
        path = req.url.path
        if path.endswith("/list.json"):
            return httpx.Response(200, json={"status": "000", "total_page": 1, "list": listing})
        if path.endswith("/document.xml"):
            rn = req.url.params["rcept_no"]
            downloads.append(rn)
            return httpx.Response(200, content=_zip({f"{rn}.xml": sample_xml, f"{rn}_00760.xml": b"<x/>"}))
        return httpx.Response(404)

    storage.upsert_company("00000001", "테스트전자", "999999", "KOSPI")
    col = DisclosureCollector(settings, storage, _client(handler), SectionSelector(),
                              SectionChunker(CharTokenizer(), settings.chunk_size, settings.chunk_overlap))
    plan = col.plan_documents("00000001", as_of=dt.date(2024, 12, 31))
    rcepts = [p[0].rcept_no for p in plan]
    assert rcepts[0] == "20240401000002"           # 정정본이 최신 버전
    assert "20240315000001" not in rcepts
    assert "20240814000003" in rcepts and "20240901000004" in rcepts
    assert "20250101000005" not in rcepts           # 미래 공시 제외

    r1 = col.ingest_company("00000001", "테스트전자", "999999", as_of=dt.date(2024, 12, 31))
    assert not r1.failed, r1.failed
    assert (settings.raw_dir / "999999" / "20240401000002" / "20240401000002.xml").exists()
    n_downloads = len(downloads)
    r2 = col.ingest_company("00000001", "테스트전자", "999999", as_of=dt.date(2024, 12, 31))
    assert len(downloads) == n_downloads              # 재다운로드 없음
    assert set(r2.skipped_existing) == set(r1.ingested)
    report = json.loads(storage.get_document("20240401000002")["section_report_json"])
    assert report["selected"] and report["excluded"]

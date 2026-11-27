"""company_doc 추출 테스트.

기본 테스트는 가상 문서(tests/fixtures/sample_annual.xml)와 가짜 DART 클라이언트만 쓴다.
실제 DART 호출은 `uv run pytest -m live tests/test_company_doc.py`로 따로 실행한다.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.contracts import SECTION_TITLES
from src.ingestion.corp_registry import CorpRegistry
from src.ingestion.dart_client import DartNoData
from src.ingestion.xml_parser import ParagraphBlock, ParsedDocument, Section, parse_dart_xml
from src.news_agent.company_doc import (CompanyDocFetcher, extract_sections, find_field_highlights,
                                        pick_latest_annual)
from src.schemas import CorpInfo

TODAY = dt.date(2099, 6, 1)
RCEPT = "20990317000001"


def _item(report_nm: str, rcept_no: str, rcept_dt: str) -> dict:
    return {"corp_code": "00164779", "corp_name": "SK하이닉스", "stock_code": "000660",
            "report_nm": report_nm, "rcept_no": rcept_no, "rcept_dt": rcept_dt}


class FakeDart:
    def __init__(self, xml: bytes, items: list[dict] | None = None, missing_doc: bool = False):
        self.xml = xml
        self.items = items if items is not None else [_item("사업보고서 (2098.12)", RCEPT, "20990317")]
        self.missing_doc = missing_doc
        self.downloads = 0

    def list_disclosures(self, corp_code, bgn_de, end_de, pblntf_ty=None, **_):
        return self.items

    def download_document(self, rcept_no):
        self.downloads += 1
        if self.missing_doc:
            raise DartNoData("013", "조회된 데이타가 없습니다.")
        return {f"{rcept_no}.xml": self.xml}


@pytest.fixture
def registry() -> CorpRegistry:
    return CorpRegistry([CorpInfo(corp_code="00164779", corp_name="SK하이닉스", stock_code="000660")])


def test_pick_latest_annual_prefers_newest_and_correction():
    items = [
        _item("사업보고서 (2097.12)", "20980315000001", "20980315"),
        _item("반기보고서 (2098.06)", "20980814000001", "20980814"),
        _item("사업보고서 (2098.12)", "20990317000001", "20990317"),
        _item("[기재정정]사업보고서 (2098.12)", "20990420000001", "20990420"),
        _item("분기보고서 (2099.03)", "20990515000001", "20990515"),
    ]
    assert pick_latest_annual(items).rcept_no == "20990420000001"
    assert pick_latest_annual([items[1], items[4]]) is None


def test_extract_sections_from_sample(sample_xml):
    sections = extract_sections(parse_dart_xml(sample_xml))
    ids = [s.section_id for s in sections]
    # 가상 문서에는 I-1, I-2, I-5, II-1, II-2, II-3만 있다. III 이후는 제외된다.
    assert ids == ["I-1", "I-2", "I-5", "II-1", "II-2", "II-3"]
    for s in sections:
        assert s.part == SECTION_TITLES[s.section_id][0]
        assert s.title.startswith(s.section_id.split("-")[1] + ".")
        assert s.text and s.chars == len(s.text)


def test_extract_sections_through_library_wrapper():
    """실제 공시처럼 하위 목차가 번호 없는 묶음 노드(LIBRARY) 안에 있어도 찾는다."""
    def sec(title, level, children=(), text=""):
        s = Section(title=title, level=level, path=[title], children=list(children))
        if text:
            s.blocks.append(ParagraphBlock(text=text))
        return s

    library = sec("LIBRARY", 2, [sec("1. 사업의 개요", 3, text="가상 본문 A"),
                                 sec("2. 주요 제품 및 서비스", 3, text="가상 본문 B")])
    part2 = sec("II. 사업의 내용", 1, [library])
    root = sec("", 0, [part2])
    doc = ParsedDocument(document_name="가상", company_name="가상", root=root, raw_char_count=0, content_hash="x")
    assert [(s.section_id, s.text) for s in extract_sections(doc)] == [("II-1", "가상 본문 A"),
                                                                       ("II-2", "가상 본문 B")]


def test_field_highlights_are_original_lines(sample_xml):
    sections = extract_sections(parse_dart_xml(sample_xml))
    word = next(w for w in sections[3].text.split() if len(w) >= 2)  # II-1 본문에 실제로 있는 단어
    hl = find_field_highlights(sections, [word])
    assert hl
    for h in hl:
        sec = next(s for s in sections if s.section_id == h.section_id)
        assert h.text in sec.text and word.lower() in h.text.lower()
    assert find_field_highlights(sections, ["존재하지않는분야명XYZ"]) == []
    assert find_field_highlights(sections, ["", "  "]) == []


def test_fetch_builds_valid_company_doc_and_caches(settings, registry, sample_xml):
    dart = FakeDart(sample_xml)
    fetcher = CompanyDocFetcher(settings, dart, registry)
    doc = fetcher.fetch("HBM", today=TODAY)
    assert doc.status == "ok"
    assert (doc.corp_code, doc.stock_code, doc.corp_name) == ("00164779", "000660", "SK하이닉스")
    assert doc.rcept_no == RCEPT and doc.rcept_dt == "2099-03-17"
    assert RCEPT in doc.source_url
    assert doc.missing_section_ids == ["I-3", "I-4", "II-4", "II-5", "II-6", "II-7"]
    # 두 번째 호출은 원문을 다시 받지 않는다
    fetcher.fetch("DRAM", today=TODAY)
    assert dart.downloads == 1
    assert (settings.processed_dir / "company_doc" / f"{RCEPT}.json").exists()


def test_fetch_without_annual_report_is_not_found(settings, registry, sample_xml):
    dart = FakeDart(sample_xml, items=[_item("분기보고서 (2099.03)", "20990515000001", "20990515")])
    doc = CompanyDocFetcher(settings, dart, registry).fetch("HBM", today=TODAY)
    assert doc.status == "not_found" and doc.error and doc.sections == []


def test_fetch_missing_document_is_not_found(settings, registry, sample_xml):
    doc = CompanyDocFetcher(settings, FakeDart(sample_xml, missing_doc=True), registry).fetch("HBM", today=TODAY)
    assert doc.status == "not_found"


def test_fetch_unparseable_document_is_parse_error(settings, registry):
    doc = CompanyDocFetcher(settings, FakeDart(b"<DOCUMENT><BODY/></DOCUMENT>"), registry).fetch("HBM", today=TODAY)
    assert doc.status == "parse_error" and doc.error


@pytest.mark.live
def test_live_sk_hynix_company_doc():
    """실제 DART에서 SK하이닉스 최신 사업보고서 I·II를 가져온다 (DART_API_KEY 필요)."""
    from config.settings import Settings
    from src.ingestion.dart_client import DartClient

    s = Settings.from_env()
    if not s.dart_api_key:
        pytest.skip("DART_API_KEY 없음")
    s.ensure_dirs()
    dart = DartClient(s.dart_api_key)
    try:
        doc = CompanyDocFetcher(s, dart).fetch("HBM")
    finally:
        dart.close()
    assert doc.status == "ok", doc.error
    assert doc.missing_section_ids == [], doc.missing_section_ids
    assert doc.field_highlights

"""입력 기업(SK하이닉스) 최신 사업보고서에서 I. 회사의 개요, II. 사업의 내용을 추출해 C1.company_doc을 만든다.

흐름: 종목코드 → corp_code → 최신 사업보고서 rcept_no → 원문 XML → 하위 목차 12개 → CompanyDoc.
원문은 요약하지 않는다. [4]가 RAG 근거로 그대로 인용하기 때문이다.

캐시: 원문 XML은 data/raw/{종목코드}/{rcept_no}/ (수집기와 같은 위치),
추출한 섹션은 data/processed/company_doc/{rcept_no}.json. 사업보고서는 1년에 한 번 바뀐다.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from src.contracts import ANCHOR_COMPANY, ANCHOR_STOCK_CODE, SECTION_TITLES, CompanyDoc, DocSection, FieldHighlight
from src.ingestion.collector import classify_report
from src.ingestion.corp_registry import CorpRegistry
from src.ingestion.dart_client import DartClient, DartError, DartNoData, UnsafeZipError, save_document_files
from src.ingestion.xml_parser import ParsedDocument, Section, parse_dart_xml
from src.schemas import DisclosureItem, DocumentType

log = logging.getLogger(__name__)

# 대목차 판별용 (공백 제거 후 비교). 로마숫자 접두어로 I/II를 구분한다.
_PARTS = {"I": "회사의개요", "II": "사업의내용"}
_ROMAN_PREFIX = re.compile(r"^\s*(I{1,3}|IV|V|VI{0,3}|IX|X)\s*\.")
_NUMBER_PREFIX = re.compile(r"^\s*(\d+)\s*\.")
LOOKBACK_DAYS = 550   # 사업보고서는 결산 후 3월에 나오므로 1년 반을 본다
MAX_HIGHLIGHTS = 30


class CompanyDocError(Exception):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# 순수 함수 (네트워크 없음, 테스트 대상)
# ---------------------------------------------------------------------------
def pick_latest_annual(items: Iterable[dict | DisclosureItem]) -> DisclosureItem | None:
    """공시 목록에서 가장 최근 사업보고서(정정본이 있으면 정정본)를 고른다."""
    annual = []
    for x in items:
        item = x if isinstance(x, DisclosureItem) else DisclosureItem(**x)
        if classify_report(item.report_nm) == DocumentType.ANNUAL:
            annual.append(item)
    return max(annual, key=lambda i: (i.rcept_dt, i.rcept_no), default=None)


def _compact(title: str) -> str:
    return re.sub(r"\s+", "", title)


def _part_key(sec: Section) -> str | None:
    m = _ROMAN_PREFIX.match(sec.title)
    if not m or m.group(1) not in _PARTS:
        return None
    return m.group(1) if _PARTS[m.group(1)] in _compact(sec.title) else None


def _section_text(sec: Section) -> str:
    """하위 목차(가., 나. 등)의 제목과 본문까지 순서대로 이어 붙인다."""
    parts: list[str] = []
    for s in sec.walk():
        if s is not sec and s.title:
            parts.append(s.title)
        own = s.own_text().strip()
        if own:
            parts.append(own)
    return "\n".join(parts).strip()


def _numbered_children(sec: Section):
    """번호 붙은 하위 목차를 순서대로 돌려준다.

    실제 DART XML은 하위 목차를 'LIBRARY' 같은 묶음 노드로 한 번 더 감싸는 경우가 있다
    (예: SK하이닉스 2025 사업보고서의 II. 사업의 내용). 번호도 본문도 없는 묶음 노드는 건너뛰고 안을 본다.
    """
    for child in sec.children:
        if _NUMBER_PREFIX.match(child.title):
            yield child
        elif not child.own_text().strip():
            yield from _numbered_children(child)


def extract_sections(doc: ParsedDocument) -> list[DocSection]:
    """I·II 대목차 아래 번호 붙은 하위 목차를 section_id(I-1 … II-7) 순서로 돌려준다."""
    found: dict[str, DocSection] = {}
    for top in doc.sections():
        key = _part_key(top)
        if key is None:
            continue
        part_name = SECTION_TITLES[f"{key}-1"][0]
        for child in _numbered_children(top):
            m = _NUMBER_PREFIX.match(child.title)
            sid = f"{key}-{int(m.group(1))}"
            if sid not in SECTION_TITLES or sid in found:
                continue
            text = _section_text(child)
            found[sid] = DocSection(section_id=sid, part=part_name, title=child.title.strip(),
                                    text=text, chars=len(text))
    return [found[sid] for sid in SECTION_TITLES if sid in found]


def find_field_highlights(sections: list[DocSection], terms: Iterable[str],
                          limit: int = MAX_HIGHLIGHTS) -> list[FieldHighlight]:
    """분야명·키워드가 들어간 줄(문단 또는 표의 행)을 원문 그대로 뽑는다. 대소문자는 구분하지 않는다."""
    needles = [t.strip().lower() for t in terms if t and t.strip()]
    if not needles:
        return []
    out: list[FieldHighlight] = []
    seen: set[str] = set()
    for sec in sections:
        for line in sec.text.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            low = line.lower()
            if any(n in low for n in needles):
                out.append(FieldHighlight(section_id=sec.section_id, text=line))
                seen.add(line)
                if len(out) >= limit:
                    return out
    return out


# ---------------------------------------------------------------------------
# DART 연동
# ---------------------------------------------------------------------------
class CompanyDocFetcher:
    def __init__(self, settings, dart: DartClient, registry: CorpRegistry | None = None):
        self.s = settings
        self.dart = dart
        self._registry = registry

    @property
    def registry(self) -> CorpRegistry:
        if self._registry is None:
            self._registry = CorpRegistry.load_or_download(self.s.processed_dir / "corp_codes.json", self.dart)
        return self._registry

    @property
    def cache_dir(self) -> Path:
        return self.s.processed_dir / "company_doc"

    # ---- 단계별 ----
    def latest_annual(self, corp_code: str, today: dt.date) -> DisclosureItem:
        start = (today - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
        items = self.dart.list_disclosures(corp_code, start, today.strftime("%Y%m%d"), "A")
        item = pick_latest_annual(items)
        if item is None:
            raise CompanyDocError("not_found", f"최근 {LOOKBACK_DAYS}일 안에 사업보고서가 없습니다.")
        return item

    def load_sections(self, item: DisclosureItem) -> list[DocSection]:
        cache = self.cache_dir / f"{item.rcept_no}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            return [DocSection(**d) for d in data["sections"]]

        raw_dir = self.s.raw_dir / ANCHOR_STOCK_CODE / item.rcept_no
        main = raw_dir / f"{item.rcept_no}.xml"
        if not main.exists():
            try:
                files = self.dart.download_document(item.rcept_no)
            except DartNoData as exc:
                raise CompanyDocError("not_found", f"공시 원본이 없습니다: {exc}") from exc
            main, _ = save_document_files(files, raw_dir, item.rcept_no)
        try:
            parsed = parse_dart_xml(main.read_bytes())
        except Exception as exc:
            raise CompanyDocError("parse_error", f"XML 파싱 실패: {exc}") from exc
        sections = extract_sections(parsed)
        if not sections:
            raise CompanyDocError("parse_error", "I. 회사의 개요 / II. 사업의 내용 목차를 찾지 못했습니다.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"rcept_no": item.rcept_no, "report_nm": item.report_nm,
                                     "sections": [s.model_dump() for s in sections]},
                                    ensure_ascii=False), encoding="utf-8")
        return sections

    # ---- 전체 ----
    def fetch(self, field: str, keywords: Iterable[str] = (), today: dt.date | None = None) -> CompanyDoc:
        """실패해도 예외를 던지지 않고 status가 not_found / parse_error인 CompanyDoc을 돌려준다."""
        today = today or dt.date.today()
        corp = self.registry.get_by_stock_code(ANCHOR_STOCK_CODE)
        if corp is None:
            return CompanyDoc(status="not_found", corp_name=ANCHOR_COMPANY, stock_code=ANCHOR_STOCK_CODE,
                              error=f"기업 사전에 종목코드 {ANCHOR_STOCK_CODE}가 없습니다.")
        base = dict(corp_code=corp.corp_code, stock_code=ANCHOR_STOCK_CODE, corp_name=ANCHOR_COMPANY)
        try:
            item = self.latest_annual(corp.corp_code, today)
            sections = self.load_sections(item)
        except CompanyDocError as exc:
            log.warning("company_doc 실패(%s): %s", exc.status, exc)
            return CompanyDoc(status=exc.status, error=str(exc), **base)  # type: ignore[arg-type]
        except (DartError, UnsafeZipError) as exc:
            log.warning("company_doc DART 오류: %s", exc)
            return CompanyDoc(status="not_found", error=f"DART 오류: {exc}", **base)

        missing = [sid for sid in SECTION_TITLES if sid not in {s.section_id for s in sections}]
        if missing:
            log.warning("company_doc: 찾지 못한 하위 목차 %s (rcept_no=%s)", missing, item.rcept_no)
        return CompanyDoc(
            status="ok", **base,
            report_name=item.report_nm.strip(),
            rcept_no=item.rcept_no,
            rcept_dt=item.published_at,
            source_url=DartClient.viewer_url(item.rcept_no),
            sections=sections,
            field_highlights=find_field_highlights(sections, [field, *keywords]),
        )

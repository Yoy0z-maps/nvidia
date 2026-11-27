"""기업별 공시 수집 → 전처리 → Chunk 저장 파이프라인.

각 단계는 독립적으로 재실행 가능하며, 이미 완료된 문서는 다시 내려받거나 Chunk를 만들지 않는다.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.chunker import SectionChunker
from src.ingestion.dart_client import DartClient, DartError, DartNoData, UnsafeZipError, save_document_files
from src.ingestion.section_selector import SectionSelector
from src.ingestion.xml_parser import parse_dart_xml
from src.schemas import PERIODIC_TYPES, DisclosureItem, DocumentType
from src.storage import Storage

log = logging.getLogger(__name__)

_TYPE_KEYS = [
    (DocumentType.ANNUAL, "사업보고서"),
    (DocumentType.HALF, "반기보고서"),
    (DocumentType.QUARTER, "분기보고서"),
    (DocumentType.SUPPLY_CONTRACT, "단일판매공급계약체결"),
    (DocumentType.NEW_FACILITY, "신규시설투자등"),
    (DocumentType.INVESTMENT_DECISION, "투자판단관련주요경영사항"),
    (DocumentType.OUTLOOK, "영업실적등에대한전망"),
    (DocumentType.SUSPENSION, "영업정지"),
]


def _norm_report(name: str) -> str:
    s = re.sub(r"^\s*\[[^\]]*\]\s*", "", name)      # [기재정정] 등 접두어 제거
    return re.sub(r"[\s·ㆍ・\.\-]", "", s)


def is_attachment_only(report_nm: str) -> bool:
    head = report_nm.strip().split("]")[0] if report_nm.strip().startswith("[") else ""
    return "첨부" in head


def classify_report(report_nm: str) -> DocumentType | None:
    n = _norm_report(report_nm)
    for dtype, key in _TYPE_KEYS:
        if key in n:
            # '사업보고서' 키는 '반기/분기'와 겹치지 않음. 첨부 정정 등은 그대로 분류
            return dtype
    return None


def version_group(item: DisclosureItem, dtype: DocumentType) -> str:
    """정정공시 버전 묶음 키. 정기보고서는 보고서명(기간 포함)으로 묶는다.

    주요사항/거래소 공시는 보고서명만으로 원공시를 식별할 수 없어 접수번호 단위로 관리한다.
    """
    if dtype in PERIODIC_TYPES:
        return f"{item.corp_code}:{_norm_report(item.report_nm)}"
    return f"{item.corp_code}:{item.rcept_no}"


def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


@dataclass
class CompanyIngestResult:
    corp_code: str
    corp_name: str
    selected: list[dict] = field(default_factory=list)
    ingested: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    chunks_created: int = 0


class DisclosureCollector:
    def __init__(self, settings, storage: Storage, dart: DartClient, selector: SectionSelector,
                 chunker: SectionChunker):
        self.s = settings
        self.storage = storage
        self.dart = dart
        self.selector = selector
        self.chunker = chunker

    # ------------------------------------------------------------------
    def plan_documents(self, corp_code: str, as_of: dt.date | None = None) -> list[tuple[DisclosureItem, DocumentType]]:
        """수집할 문서 선정: 최근 사업보고서 1, (설정 시) 그 이후 최신 반기/분기 1, 최근 365일 주요 공시 최대 N.

        반기·분기보고서는 '기업공시서식 작성기준에 따라 분기보고서에는 본 항목을 기재하지 아니하였습니다'처럼
        경영진단·계열회사 등 항목이 생략되는 경우가 많아 기본값으로 수집하지 않는다 (COLLECT_INTERIM_REPORTS).
        """
        end = as_of or dt.date.today()
        periodic = [DisclosureItem(**x) for x in
                    self.dart.list_disclosures(corp_code, _ymd(end - dt.timedelta(days=550)), _ymd(end), "A")]
        recent = [DisclosureItem(**x) for x in
                  self.dart.list_disclosures(corp_code, _ymd(end - dt.timedelta(days=365)), _ymd(end))]
        # 평가 기준일 이후 공시(정정 포함) 제외
        cutoff = _ymd(end)
        # [첨부정정]/[첨부추가]는 첨부서류(감사보고서 등)만 바꾼 공시라 본문(document.xml)이 없다 → 원본/기재정정본 사용
        periodic = [x for x in periodic if x.rcept_dt <= cutoff and not is_attachment_only(x.report_nm)]
        recent = [x for x in recent if x.rcept_dt <= cutoff]

        def latest_version(items: list[tuple[DisclosureItem, DocumentType]]):
            groups: dict[str, tuple[DisclosureItem, DocumentType]] = {}
            for it, t in items:
                g = version_group(it, t)
                if g not in groups or (it.rcept_dt, it.rcept_no) > (groups[g][0].rcept_dt, groups[g][0].rcept_no):
                    groups[g] = (it, t)
            return sorted(groups.values(), key=lambda x: (x[0].rcept_dt, x[0].rcept_no), reverse=True)

        typed = [(x, classify_report(x.report_nm)) for x in periodic]
        annual = latest_version([(x, t) for x, t in typed if t == DocumentType.ANNUAL])
        interim = latest_version([(x, t) for x, t in typed if t in (DocumentType.HALF, DocumentType.QUARTER)])
        collect_interim = getattr(self.s, "collect_interim_reports", False)
        if not annual:
            # 최근 1년 반 안에 사업보고서가 없으면 3년까지 넓혀 찾는다
            older = [DisclosureItem(**x) for x in
                     self.dart.list_disclosures(corp_code, _ymd(end - dt.timedelta(days=1100)), _ymd(end), "A")]
            older = [x for x in older if x.rcept_dt <= cutoff and not is_attachment_only(x.report_nm)]
            annual = latest_version([(x, t) for x in older if (t := classify_report(x.report_nm)) == DocumentType.ANNUAL])
        plan: list[tuple[DisclosureItem, DocumentType]] = []
        if annual:
            plan.append(annual[0])
            later = [x for x in interim if x[0].rcept_dt >= annual[0][0].rcept_dt]
            if collect_interim and later:
                plan.append(later[0])
        elif collect_interim and interim:
            plan.append(interim[0])
        else:
            self.storage.log("collect", "WARNING", "최근 3년 안에 사업보고서가 없습니다 (정기보고서 미수집).", corp_code)

        seen = {p[0].rcept_no for p in plan}
        major = []
        for x in recent:
            t = classify_report(x.report_nm)
            if t is None or t in PERIODIC_TYPES or x.rcept_no in seen:
                continue
            major.append((x, t))
            seen.add(x.rcept_no)
        major.sort(key=lambda m: (m[0].rcept_dt, m[0].rcept_no), reverse=True)
        plan.extend(major[: self.s.max_major_disclosures])
        return plan

    # ------------------------------------------------------------------
    def ingest_company(self, corp_code: str, corp_name: str, stock_code: str,
                       as_of: dt.date | None = None) -> CompanyIngestResult:
        res = CompanyIngestResult(corp_code, corp_name)
        try:
            plan = self.plan_documents(corp_code, as_of)
        except DartError as exc:
            self.storage.log("collect", "ERROR", f"공시 목록 조회 실패: {exc}", corp_code)
            self.storage.set_company_status(corp_code, "FAILED")
            res.failed.append({"stage": "list", "error": str(exc)})
            return res
        if not plan:
            self.storage.log("collect", "WARNING", "수집 대상 보고서가 없습니다.", corp_code)
        res.selected = [{"rcept_no": i.rcept_no, "report_nm": i.report_nm, "rcept_dt": i.rcept_dt,
                         "type": t.value} for i, t in plan]
        for item, dtype in plan:
            try:
                status = self.ingest_document(item, dtype, corp_name, stock_code)
                (res.skipped_existing if status == "EXISTING" else res.ingested).append(item.rcept_no)
            except Exception as exc:  # 문서 단위 실패는 기록 후 계속
                log.exception("문서 처리 실패 %s", item.rcept_no)
                self.storage.log("collect", "ERROR", f"문서 처리 실패: {exc}", corp_code, item.rcept_no)
                res.failed.append({"rcept_no": item.rcept_no, "error": str(exc)})
        n = self.storage.one("SELECT COUNT(*) n FROM chunks JOIN documents USING(document_id) WHERE corp_code=?",
                             (corp_code,))
        res.chunks_created = int(n["n"]) if n else 0
        ok = any(d["ingestion_status"] == "CHUNKED" or d["ingestion_status"] == "READY"
                 for d in self.storage.list_documents(corp_code))
        self.storage.set_company_status(corp_code, "CHUNKED" if ok else "FAILED")
        return res

    def ingest_document(self, item: DisclosureItem, dtype: DocumentType, corp_name: str, stock_code: str,
                        force: bool = False) -> str:
        """force=True 이면 이미 처리된 문서도 저장된 원본 XML로 다시 파싱·청킹한다 (다운로드 없음)."""
        doc_id = item.rcept_no
        existing = self.storage.get_document(doc_id)
        if not force and existing and existing["ingestion_status"] in ("CHUNKED", "READY"):
            return "EXISTING"
        raw_dir: Path = self.s.raw_dir / (stock_code or item.corp_code) / item.rcept_no
        main_xml = raw_dir / f"{item.rcept_no}.xml"
        base = dict(document_id=doc_id, corp_code=item.corp_code, rcept_no=item.rcept_no,
                    report_name=item.report_nm, document_type=dtype.value, published_at=item.published_at,
                    source_url=DartClient.viewer_url(item.rcept_no), is_correction=int(item.is_correction),
                    version_group=version_group(item, dtype))
        # 1) 원본 다운로드 (이미 있으면 재사용)
        candidates = sorted(raw_dir.glob("*.xml")) if raw_dir.exists() else []
        if not candidates:
            try:
                files = self.dart.download_document(item.rcept_no)
            except DartNoData as exc:
                self.storage.upsert_document(**base, ingestion_status="NO_DATA")
                raise RuntimeError(f"공시 원본 없음: {exc}") from exc
            except (DartError, UnsafeZipError) as exc:
                self.storage.upsert_document(**base, ingestion_status="DOWNLOAD_FAILED")
                raise
            main_path, attachments = save_document_files(files, raw_dir, item.rcept_no)
            if attachments:
                self.storage.log("collect", "INFO", f"첨부 문서 {len(attachments)}개는 본문 분석에서 제외",
                                 item.corp_code, item.rcept_no, [p.name for p in attachments])
        else:
            main_path = main_xml if main_xml.exists() else min(candidates, key=lambda p: len(p.name))
        xml_bytes = main_path.read_bytes()

        # 2) 파싱 + 섹션 선택
        try:
            parsed = parse_dart_xml(xml_bytes)
        except Exception as exc:
            self.storage.upsert_document(**base, raw_path=str(main_path), ingestion_status="PARSE_FAILED")
            raise RuntimeError(f"XML 파싱 실패: {exc}") from exc
        selected, report = self.selector.select(parsed, periodic=dtype in PERIODIC_TYPES)
        try:
            report.tokens_before = self.chunker.n_tokens(" ".join(s.own_text() for s in parsed.sections()))
        except Exception:
            pass
        report_json = json.dumps(report.to_dict(), ensure_ascii=False)
        if report.failed_rules:
            self.storage.log("section", "WARNING", f"섹션 규칙 매칭 실패: {report.failed_rules}",
                             item.corp_code, item.rcept_no, report.to_dict())
        if not selected:
            self.storage.upsert_document(**base, raw_path=str(main_path), content_hash=parsed.content_hash,
                                         ingestion_status="SECTION_FAILED", section_report_json=report_json)
            raise RuntimeError("선택된 섹션이 없습니다 (문서 전체 임베딩은 하지 않음).")

        # 3) Chunking
        chunks = self.chunker.chunk_sections(selected, corp_name=corp_name, stock_code=stock_code,
                                             document_type=dtype.value)
        self.storage.upsert_document(**base, raw_path=str(main_path), content_hash=parsed.content_hash,
                                     ingestion_status="PARSED", section_report_json=report_json)
        self.storage.replace_chunks(doc_id, [c.to_row() for c in chunks])
        self.storage.upsert_document(document_id=doc_id, corp_code=item.corp_code, rcept_no=item.rcept_no,
                                     report_name=item.report_nm, document_type=dtype.value,
                                     published_at=item.published_at, ingestion_status="CHUNKED")
        self.storage.log("collect", "INFO", f"{dtype.value} 처리 완료: Chunk {len(chunks)}개",
                         item.corp_code, item.rcept_no,
                         {"chars_before": report.chars_before, "chars_after": report.chars_after,
                          "tokens_before": report.tokens_before, "tokens_after": report.tokens_after})
        return "INGESTED"


def mark_embedded_documents_ready(storage: Storage) -> int:
    """모든 Chunk가 임베딩된 문서를 READY로 전환한다."""
    rows = storage.query(
        """SELECT d.document_id FROM documents d
           WHERE d.ingestion_status='CHUNKED'
             AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id=d.document_id AND c.embedding_id IS NULL)
             AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id=d.document_id)""")
    with storage.tx() as c:
        for r in rows:
            c.execute("UPDATE documents SET ingestion_status='READY' WHERE document_id=?", (r["document_id"],))
        c.execute("""UPDATE companies SET ingestion_status='READY' WHERE corp_code IN
                     (SELECT corp_code FROM documents WHERE ingestion_status='READY')""")
    return len(rows)

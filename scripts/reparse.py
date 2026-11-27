"""이미 수집한 공시를 저장된 원본 XML로 다시 파싱·청킹·임베딩한다 (DART 재다운로드 없음).

파서가 고쳐졌을 때 사용한다. 기본 대상은 본문에 이스케이프되지 않은 '<'(예: <당기>, <부칙신설>)가 있어
섹션 구조가 깨졌던 문서다. 바뀌지 않은 Chunk는 기존 임베딩을 그대로 재사용하고, 새 텍스트만 임베딩한다.

uv run python -m scripts.reparse --dry-run          # 대상 문서와 섹션 변화만 확인
uv run python -m scripts.reparse                    # 재처리 + 임베딩 + 인덱스 재구축
uv run python -m scripts.reparse --stock 005290     # 특정 종목만
uv run python -m scripts.reparse --all              # 모든 문서
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402

STRAY_LT = re.compile(rb"<(?![A-Za-z/!?])")
STATUSES = ("READY", "CHUNKED", "SKIPPED_INTERIM", "SECTION_FAILED", "PARSED")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="모든 문서 재처리")
    ap.add_argument("--stock", action="append", default=[], help="종목코드 (여러 번 지정 가능)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    setup_logging(s)
    from src.ingestion.xml_parser import parse_dart_xml
    from src.pipeline import build_app, embed_and_index, make_collector
    from src.schemas import DisclosureItem, DocumentType

    app = build_app(s, need_dart=False, need_openai=not args.dry_run)
    collector = app.collector or make_collector(s, app.storage, None, app.llm)   # 원본은 로컬 파일만 사용
    rows = app.storage.query(
        f"""SELECT d.*, c.corp_name, c.stock_code, c.ingestion_status AS company_status
            FROM documents d JOIN companies c USING(corp_code)
            WHERE d.ingestion_status IN ({",".join("?" * len(STATUSES))}) ORDER BY c.corp_name, d.published_at""",
        STATUSES)
    targets = []
    for r in rows:
        if args.stock and r["stock_code"] not in {x.zfill(6) for x in args.stock}:
            continue
        raw = Path(r["raw_path"] or "")
        if not raw.exists():   # 다른 PC에서 받은 스냅샷 등 절대경로가 다를 때
            raw = s.raw_dir / (r["stock_code"] or r["corp_code"]) / r["rcept_no"] / f"{r['rcept_no']}.xml"
        r["raw_path"] = str(raw)
        if not raw.exists():
            print(f"  원본 없음, 건너뜀: {r['corp_name']} {r['report_name']} ({raw})")
            continue
        if args.all or args.stock or STRAY_LT.search(raw.read_bytes()):
            targets.append(r)
    print(f"재처리 대상 {len(targets)}개 문서")

    changed = 0
    for r in targets:
        before = json.loads(r["section_report_json"] or "{}")
        if args.dry_run:
            parsed = parse_dart_xml(Path(r["raw_path"]).read_bytes())
            _, rep = collector.selector.select(parsed, periodic=r["document_type"] in ("사업보고서", "반기보고서", "분기보고서"))
            after = rep.to_dict()
        else:
            item = DisclosureItem(corp_code=r["corp_code"], corp_name=r["corp_name"], stock_code=r["stock_code"] or "",
                                  report_nm=r["report_name"], rcept_no=r["rcept_no"],
                                  rcept_dt=r["published_at"].replace("-", ""))
            collector.ingest_document(item, DocumentType(r["document_type"]), r["corp_name"], r["stock_code"] or "",
                                          force=True)
            after = json.loads(app.storage.get_document(r["document_id"])["section_report_json"] or "{}")
        b, a = before.get("chars_after", 0), after.get("chars_after", 0)
        changed += b != a
        print(f"  {r['corp_name']:<10} {r['report_name']:<22} 본문 {b:>7,} → {a:>7,}자 · "
              f"실패 규칙 {before.get('failed_rules', [])} → {after.get('failed_rules', [])}")
    print(f"섹션 선택이 바뀐 문서 {changed}개")
    if args.dry_run or not targets:
        return 0

    print("\n임베딩 (바뀌지 않은 Chunk는 기존 벡터 재사용)")
    out = embed_and_index(app, print)
    app.vector_store.rebuild()   # 교체된 Chunk의 옛 벡터를 인덱스에서 정리 (벡터 파일 재사용, API 호출 없음)
    print(json.dumps({k: v for k, v in out.items() if k != "failed_chunk_ids"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""실제 공시 XML 구조 조사 (Section Selector 규칙 점검용).

uv run python -m scripts.inspect_xml --rcept-no 20250318000xxx   # OpenDART에서 내려받아 조사
uv run python -m scripts.inspect_xml --file data/raw/000660/<rcept_no>/<rcept_no>.xml
uv run python -m scripts.inspect_xml --latest-annual 000660        # 최신 사업보고서 자동 선택
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--rcept-no")
    g.add_argument("--latest-annual", metavar="STOCK_CODE")
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()
    s = Settings.from_env()
    setup_logging(s)

    from lxml import etree

    from src.ingestion.section_selector import SectionSelector
    from src.ingestion.xml_parser import outline, parse_dart_xml

    if args.file:
        xml = Path(args.file).read_bytes()
    else:
        from src.ingestion.corp_registry import CorpRegistry
        from src.ingestion.dart_client import DartClient, save_document_files
        dart = DartClient(s.dart_api_key)
        rcept = args.rcept_no
        stock = "unknown"
        if args.latest_annual:
            reg = CorpRegistry.load_or_download(s.processed_dir / "corp_codes.json", dart)
            corp = reg.get_by_stock_code(args.latest_annual)
            end = dt.date.today()
            items = [i for i in dart.list_disclosures(corp.corp_code, (end - dt.timedelta(days=550)).strftime("%Y%m%d"),
                                                      end.strftime("%Y%m%d"), "A") if "사업보고서" in i["report_nm"]]
            rcept, stock = items[0]["rcept_no"], corp.stock_code
            print(f"최신 사업보고서: {items[0]['report_nm']} ({rcept})")
        files = dart.download_document(rcept)
        main_path, att = save_document_files(files, s.raw_dir / stock / rcept, rcept)
        print(f"저장: {main_path} (첨부 {len(att)}개)")
        xml = main_path.read_bytes()

    root = etree.fromstring(xml, parser=etree.XMLParser(recover=True, huge_tree=True))
    tags = Counter(el.tag for el in root.iter() if isinstance(el.tag, str))
    print("\n[태그 빈도 상위 25]")
    for t, n in tags.most_common(25):
        print(f"  {t}: {n}")
    doc = parse_dart_xml(xml)
    print(f"\n[문서] {doc.document_name} / {doc.company_name} / 본문 {doc.raw_char_count:,}자")
    print("\n[목차]")
    for o in outline(doc, args.depth):
        print(f"  {'  ' * (o['level'] - 1)}{o['title']}  (id={o['id']}, 블록 {o['blocks']}, 표 {o['tables']}, {o['chars']:,}자)")
    selected, report = SectionSelector().select(doc, periodic=True)
    print("\n[선택 결과]")
    print(json.dumps({k: v for k, v in report.to_dict().items() if k != "excluded"}, ensure_ascii=False, indent=1)[:4000])
    print(f"\n제외 섹션 {len(report.excluded)}개, 실패 규칙: {report.failed_rules}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

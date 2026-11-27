from src.ingestion.section_selector import SectionSelector, normalize_title
from src.ingestion.xml_parser import parse_dart_xml


def test_normalize_title_variants():
    assert normalize_title("II. 사업의 내용") == "사업의내용"
    assert normalize_title("Ⅱ.사업의  내용") == "사업의내용"
    assert normalize_title("2. 주요 제품 및 서비스") == "주요제품및서비스"
    assert normalize_title("가. 원재료") == "원재료"
    assert normalize_title("(1) 개요") == "개요"
    assert normalize_title("【 대표이사 등의 확인 】") == "대표이사등의확인"


def test_select_periodic(sample_xml):
    doc = parse_dart_xml(sample_xml)
    selected, report = SectionSelector().select(doc, periodic=True)
    paths = [s.path_str for s in selected]
    assert "II. 사업의 내용 > 1. 사업의 개요" in paths
    assert "II. 사업의 내용 > 2. 주요 제품 및 서비스" in paths
    assert "I. 회사의 개요 > 1. 회사의 개요" in paths
    assert "IV. 이사의 경영진단 및 분석의견 > 2. 개요" in paths
    joined = " ".join(paths)
    assert "재무에 관한 사항" not in joined
    assert "이사회" not in joined
    assert "정관" not in joined
    assert "대표이사" not in joined
    assert "예측정보" not in joined  # MD&A는 지정된 하위 섹션만
    assert report.chars_after < report.chars_before
    assert any("재무에 관한 사항" in e["path"] for e in report.excluded)


def test_failed_rule_reported_without_full_embedding():
    xml = '<DOCUMENT><BODY><PART><TITLE ATOC="Y">III. 재무에 관한 사항</TITLE><P>숫자</P></PART></BODY></DOCUMENT>'
    doc = parse_dart_xml(xml.encode())
    selected, report = SectionSelector().select(doc, periodic=True)
    assert selected == []
    assert "business" in report.failed_rules


def test_llm_title_fallback():
    xml = ('<DOCUMENT><BODY><PART><TITLE ATOC="Y">2. 영업의 현황</TITLE><P>당사 제품 설명</P></PART>'
           '</BODY></DOCUMENT>')
    doc = parse_dart_xml(xml.encode())
    sel = SectionSelector(title_classifier=lambda titles: [t for t in titles if "영업" in t])
    selected, report = sel.select(doc, periodic=True)
    assert report.used_fallback and selected[0].section_type == "business"


def test_major_disclosure_body():
    xml = ('<DOCUMENT><BODY><P>단일판매ㆍ공급계약체결</P><TABLE><TR><TD>계약상대</TD><TD>가상고객</TD></TR></TABLE>'
           '</BODY></DOCUMENT>')
    doc = parse_dart_xml(xml.encode())
    selected, report = SectionSelector().select(doc, periodic=False)
    assert selected and "가상고객" in selected[0].section.own_text()

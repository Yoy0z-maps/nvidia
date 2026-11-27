from src.ingestion.xml_parser import TableBlock, parse_dart_xml


def test_parse_structure(sample_xml):
    doc = parse_dart_xml(sample_xml)
    assert doc.document_name == "사업보고서"
    assert doc.company_name == "테스트전자"
    titles = [s.title for s in doc.sections()]
    assert "II. 사업의 내용" in titles
    assert "2. 주요 제품 및 서비스" in titles
    # COVER는 섹션/본문에 포함되지 않음
    assert all("홍길동" not in s.own_text() for s in doc.sections())


def test_table_keeps_title_unit_columns(sample_xml):
    doc = parse_dart_xml(sample_xml)
    sec = next(s for s in doc.sections() if s.title == "2. 주요 제품 및 서비스")
    table = next(b for b in sec.blocks if isinstance(b, TableBlock))
    assert table.unit.startswith("백만원")
    assert table.header[0] == ["사업부문", "품목", "매출액", "비율"]
    assert ["반도체", "테스트메모리", "1,000", "80.0"] in table.rows
    md = table.to_markdown()
    assert "(단위: 백만원" in md and "| 사업부문 | 품목 | 매출액 | 비율 |" in md


def test_table_group_title(sample_xml):
    doc = parse_dart_xml(sample_xml)
    sec = next(s for s in doc.sections() if s.title == "3. 원재료 및 생산설비")
    table = next(b for b in sec.blocks if isinstance(b, TableBlock))
    assert table.title == "주요 원재료 가격변동 추이"


def test_malformed_xml_recovers():
    bad = "<DOCUMENT><BODY><PART><TITLE ATOC=\"Y\">II. 사업의 내용</TITLE><P>본문 &cr; 내용<P></PART></BODY>".encode()
    doc = parse_dart_xml(bad)
    assert any("사업의 내용" in s.title for s in doc.sections())


def test_euc_kr_document():
    xml = '<?xml version="1.0" encoding="euc-kr"?><DOCUMENT><BODY><PART><TITLE ATOC="Y">II. 사업의 내용</TITLE><P>한글 본문</P></PART></BODY></DOCUMENT>'
    doc = parse_dart_xml(xml.encode("euc-kr"))
    assert "한글 본문" in doc.sections()[0].own_text()


def test_stray_angle_brackets_do_not_swallow_following_sections():
    # 정관 본문의 "<부칙신설>", "<당기>" 같은 꺾쇠가 이후 섹션 전체를 삼키면 안 된다
    xml = """<?xml version="1.0" encoding="utf-8"?><DOCUMENT><BODY>
<SECTION-1><TITLE ATOC="Y">I. 회사의 개요</TITLE>
<SECTION-2><TITLE ATOC="Y">5. 정관에 관한 사항</TITLE><P>제1조 개정 <부칙신설> 및 <당기> 손익</P></SECTION-2>
</SECTION-1>
<SECTION-1><TITLE ATOC="Y">II. 사업의 내용</TITLE><P>당사는 반도체 소재를 생산합니다.</P></SECTION-1>
</BODY></DOCUMENT>""".encode("utf-8")
    doc = parse_dart_xml(xml)
    tops = [s.title for s in doc.root.children]
    assert "II. 사업의 내용" in tops
    art = next(s for s in doc.sections() if s.title == "5. 정관에 관한 사항")
    assert "<부칙신설>" in art.own_text() and "반도체 소재" not in art.own_text()

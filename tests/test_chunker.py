from src.ingestion.chunker import SectionChunker, content_hash
from src.ingestion.section_selector import SectionSelector, SelectedSection
from src.ingestion.xml_parser import ParagraphBlock, Section, TableBlock, parse_dart_xml
from tests.conftest import CharTokenizer


def _chunker(size=200, overlap=20):
    return SectionChunker(CharTokenizer(), size, overlap)


def test_chunks_respect_size_and_header(sample_xml):
    doc = parse_dart_xml(sample_xml)
    selected, _ = SectionSelector().select(doc, periodic=True)
    chunks = _chunker().chunk_sections(selected, corp_name="테스트전자", stock_code="999999", document_type="사업보고서")
    assert chunks
    for c in chunks:
        assert c.token_count <= 200
        assert c.embedding_text.startswith("기업명: 테스트전자\n종목코드: 999999\n문서 유형: 사업보고서\n섹션: ")
        assert c.raw_text in c.embedding_text
    # 섹션 경계를 넘어 합치지 않음
    assert len({c.section_path for c in chunks}) == len({s.path_str for s in selected})


def test_long_paragraph_split_with_overlap():
    text = " ".join(f"문장{i}입니다." for i in range(200))
    sec = Section("긴 섹션", 1, ["긴 섹션"], blocks=[ParagraphBlock(text)])
    chunks = _chunker(100, 20).chunk_sections([SelectedSection(sec, "business", "t")], corp_name="A",
                                               stock_code="000001", document_type="사업보고서")
    assert len(chunks) > 3
    assert all(c.token_count <= 100 for c in chunks)


def test_large_table_split_keeps_header_unit_title():
    rows = [[f"부문{i}", f"{i * 1000:,}", "원"] for i in range(60)]
    t = TableBlock(title="부문별 매출", unit="백만원", header=[["부문", "매출액", "통화"]], rows=rows)
    sec = Section("매출", 1, ["매출"], blocks=[t])
    chunks = _chunker(200, 20).chunk_sections([SelectedSection(sec, "business", "t")], corp_name="A",
                                               stock_code="000001", document_type="사업보고서")
    assert len(chunks) > 1
    for c in chunks:
        assert "[표] 부문별 매출" in c.raw_text
        assert "(단위: 백만원)" in c.raw_text
        assert "| 부문 | 매출액 | 통화 |" in c.raw_text
    all_text = "\n".join(c.raw_text for c in chunks)
    assert all(f"부문{i} " in all_text for i in range(60))


def test_hash_ignores_whitespace_only():
    assert content_hash("매출  1,000\n백만원") == content_hash("매출 1,000 백만원")
    assert content_hash("매출 1,000 백만원") != content_hash("매출 1,100 백만원")

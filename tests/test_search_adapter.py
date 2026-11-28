"""[4] 검색 어댑터 테스트. 기업·접수번호·본문은 모두 가상이다."""
import pytest
from pydantic import ValidationError

from src.ingestion.chunker import content_hash
from src.ingestion.collector import mark_embedded_documents_ready
from src.insight.search_adapter import SearchRequest, item_ids_by_corp, make_c4_search, retrieve
from src.retrieval.embedding import EmbeddingStore
from src.retrieval.retriever import BudgetExceeded, Retriever
from src.retrieval.vector_store import VectorStore
from tests.conftest import FakeEmbedder

A, B = "00000001", "00000002"
RCEPT = "20990101000001"


def chunk(corp, order, text, score=0.5):
    return {"chunk_id": f"{corp}-{RCEPT}-{order:04d}", "corp_code": corp, "item_ids": [f"COM-{corp}"],
            "company": "가상기업", "report": "사업보고서 (2098.12)", "rcept_no": RCEPT, "section": "II. 사업의 내용",
            "source_url": "https://example.com/doc", "text": text, "score": score}


def fake_search(table):
    """table: {(corp_code, query): [청크] 또는 예외}"""
    calls = []

    def search(query, corp_codes=None, k=5):
        calls.append((query, corp_codes, k))
        found = table.get((corp_codes[0], query), [])
        if isinstance(found, Exception):
            raise found
        return found[:k]

    search.calls = calls
    return search


def ids(result, corp):
    return [int(c.chunk_id.rsplit("-", 1)[1]) for c in result.chunks[corp]]


def test_real_retriever_is_mapped_to_c4(settings, storage):
    storage.upsert_company(A, "가상장비", "100001", "KOSPI")
    storage.upsert_document(document_id="d1", corp_code=A, rcept_no=RCEPT, report_name="사업보고서 (2098.12)",
                            document_type="사업보고서", published_at="2099-01-01",
                            source_url="https://example.com/doc", ingestion_status="CHUNKED", version_group="d1")
    text = "TC 본더 장비 공급"
    storage.replace_chunks("d1", [{"section_path": "II. 사업의 내용", "section_type": "business", "chunk_order": 7,
                                   "raw_text": text, "embedding_text": text, "token_count": len(text),
                                   "content_hash": content_hash(text)}])
    store = EmbeddingStore(storage, settings.embeddings_dir, FakeEmbedder())
    store.embed_pending_chunks()
    mark_embedded_documents_ready(storage)
    retriever = Retriever(storage, VectorStore(settings.index_path, storage, store), store)

    search = make_c4_search(retriever, storage, {A: [f"COM-{A}", "FLD-001"]})
    result = retrieve(search, [SearchRequest(corp_code=A, query="본더 장비")])

    assert result.status == "ok" and result.errors == []
    got = result.chunks[A][0].model_dump()
    score = got.pop("score")
    assert score > 0
    assert got == {"chunk_id": f"{A}-{RCEPT}-0007", "corp_code": A, "item_ids": [f"COM-{A}", "FLD-001"],
                   "company": "가상장비", "report": "사업보고서 (2098.12)", "rcept_no": RCEPT,
                   "section": "II. 사업의 내용", "source_url": "https://example.com/doc", "text": text}


def test_item_ids_include_fields_represented_by_company():
    pool = {"items": [
        {"item_id": f"COM-{A}", "item_type": "company", "corp_code": A},
        {"item_id": f"COM-{B}", "item_type": "company", "corp_code": B},
        {"item_id": "FLD-001", "item_type": "field", "rep_companies": [f"COM-{A}"]},
    ]}
    assert item_ids_by_corp(pool) == {A: [f"COM-{A}", "FLD-001"], B: [f"COM-{B}"]}


def test_duplicates_removed_and_queries_interleaved():
    search = fake_search({
        (A, "q1"): [chunk(A, 1, "가"), chunk(A, 2, "나")],
        # 3번은 1번과 공백만 다른 같은 본문, 2번은 q1에서도 나온 같은 청크
        (A, "q2"): [chunk(A, 3, " 가 "), chunk(A, 2, "나"), chunk(A, 4, "다")],
    })
    result = retrieve(search, [SearchRequest(corp_code=A, query=q) for q in ("q1", "q2", "q1")])
    assert ids(result, A) == [1, 2, 4]
    assert [c[0] for c in search.calls] == ["q1", "q2"]  # 같은 검색어는 한 번만


def test_quota_is_applied_per_company_and_rank_is_kept():
    search = fake_search({
        # 키워드로만 찾은 청크(score 0.0)가 1등이어도 검색기의 순서를 유지한다
        (A, "q"): [chunk(A, 1, "a1", 0.0), chunk(A, 2, "a2", 0.9), chunk(A, 3, "a3", 0.8)],
        (B, "q"): [chunk(B, 1, "b1", 0.1)],
    })
    result = retrieve(search, [SearchRequest(corp_code=A, query="q"), SearchRequest(corp_code=B, query="q")], quota=2)
    assert ids(result, A) == [1, 2] and ids(result, B) == [1]
    assert search.calls == [("q", [A], 4), ("q", [B], 4)]
    assert result.status == "ok"


LONG = ("당사는 반도체 제조용 장비를 개발하여 국내외 고객사에 공급하고 있으며 "
        "자체 기술로 설계와 제작, 조립, 검사를 수행하는 수직통합 제조 체계를 갖추고 있습니다.")


def run_one(*chunks_by_rank, quota=5):
    search = fake_search({(A, "q"): list(chunks_by_rank)})
    return retrieve(search, [SearchRequest(corp_code=A, query="q")], quota=quota)


def test_broken_edges_trimmed_and_library_removed():
    raw = "�" + LONG + " 중간�글자�"
    found = chunk(A, 1, raw) | {"section": "II. 사업의 내용 > LIBRARY > 1. 사업의 개요"}
    got = run_one(found).chunks[A][0]
    assert got.text == LONG + " 중간�글자"  # 가장자리만 자른다
    assert got.text in raw  # 인용 검증을 위해 원문의 일부여야 한다
    assert got.section == "II. 사업의 내용 > 1. 사업의 개요"


@pytest.mark.parametrize("text", [
    "나. 연결대상회사의 변동내용",
    "2-3. 판매방법 및 조건",
    "(단위: 백만원)\n| (단위 : 백만원) |\n|---|",
    "[표] 가. 재무상태\n(단위: 백만원)\n|  | (단위 : 백만원) |\n|---|---|",
    "| - 수행 업무 |\n|---|",
])
def test_heading_only_chunk_is_dropped(text):
    result = run_one(chunk(A, 1, text), chunk(A, 2, LONG))
    assert ids(result, A) == [2]


@pytest.mark.parametrize("text", [
    "TC 본더 장비 공급",  # 제목 모양이 아니다
    "2. 파생상품 거래 현황 해당사항 없습니다.",  # 문장
    "1) 매출액 3,000억원",  # 수치
    "(1) 점유율\n| 제품 | 비율 |\n|---|---|\n| TV | 29% |",  # 데이터 행이 있는 표
])
def test_short_chunk_with_content_is_kept(text):
    assert ids(run_one(chunk(A, 1, text)), A) == [1]


def test_only_heading_chunks_means_no_result():
    result = run_one(chunk(A, 1, "가. 주요 제품 등의 현황"))
    assert result.status == "error" and any("검색 결과가 없습니다" in e for e in result.errors)


def test_contained_chunk_is_dropped_and_longer_one_replaces_shorter():
    longer = LONG + " 또한 신규 장비를 병행 개발하고 있습니다."
    assert ids(run_one(chunk(A, 1, longer), chunk(A, 2, LONG)), A) == [1]
    other = "원재료는 국내 협력사에서 매입하며 가격 변동에 대비해 복수의 공급처를 유지하고 있습니다."
    result = run_one(chunk(A, 1, LONG), chunk(A, 2, other), chunk(A, 3, longer))
    assert ids(result, A) == [3, 2]  # 1등 자리는 그대로, 내용이 많은 청크로 바뀐다


def test_same_template_with_different_numbers_is_kept():
    form = "단일판매 공급계약 체결. 계약상대방은 가상고객이며 계약내용은 반도체 제조장비 공급입니다. 계약금액 {}원."
    result = run_one(chunk(A, 1, form.format("1,000,000")), chunk(A, 2, form.format("2,500,000")))
    assert ids(result, A) == [1, 2]


def test_removed_chunks_are_backfilled_from_lower_ranks():
    other = "원재료는 국내 협력사에서 매입하며 가격 변동에 대비해 복수의 공급처를 유지하고 있습니다."
    result = run_one(chunk(A, 1, LONG), chunk(A, 2, "가. 개요"), chunk(A, 3, LONG + " "), chunk(A, 4, other), quota=2)
    assert ids(result, A) == [1, 4]


def test_same_text_in_different_companies_is_kept():
    search = fake_search({(A, "q"): [chunk(A, 1, "공통 문구")], (B, "q"): [chunk(B, 1, "공통 문구")]})
    result = retrieve(search, [SearchRequest(corp_code=c, query="q") for c in (A, B)])
    assert set(result.chunks) == {A, B}


@pytest.mark.parametrize("failure, message", [
    ([], "검색 결과가 없습니다"),
    (BudgetExceeded("x"), "예산"),
    (RuntimeError("sk-secret"), "RuntimeError"),
])
def test_one_failing_company_makes_partial(failure, message):
    search = fake_search({(A, "q"): [chunk(A, 1, "a")], (B, "q"): failure})
    result = retrieve(search, [SearchRequest(corp_code=c, query="q") for c in (A, B)])
    assert result.status == "partial" and list(result.chunks) == [A]
    assert any(message in e for e in result.errors)
    assert not any("sk-secret" in e for e in result.errors)


def test_other_company_chunk_is_dropped():
    search = fake_search({(A, "q"): [chunk(B, 1, "남의 문서"), chunk(A, 2, "a")]})
    result = retrieve(search, [SearchRequest(corp_code=A, query="q")])
    assert ids(result, A) == [2] and result.status == "partial"


def test_nothing_found_is_error():
    assert retrieve(fake_search({}), []).status == "error"
    result = retrieve(fake_search({}), [SearchRequest(corp_code=A, query="q")])
    assert result.status == "error" and result.errors


def test_request_requires_company_code():
    with pytest.raises(ValidationError):
        SearchRequest(corp_code="COM-00000001", query="q")  # 항목 ID는 corp_code가 아니다

import numpy as np

from src.ingestion.collector import mark_embedded_documents_ready
from src.retrieval.embedding import EmbeddingStore
from src.retrieval.retriever import BudgetExceeded, Retriever
from src.retrieval.vector_store import VectorStore
from tests.conftest import FakeEmbedder


def seed_docs(storage, docs):
    """docs: list of (corp_code, corp_name, stock, doc_id, published, [texts])"""
    for corp, name, stock, doc_id, pub, texts in docs:
        storage.upsert_company(corp, name, stock, "KOSPI")
        storage.upsert_document(document_id=doc_id, corp_code=corp, rcept_no=doc_id, report_name="사업보고서",
                                document_type="사업보고서", published_at=pub, source_url=f"u/{doc_id}",
                                ingestion_status="CHUNKED", version_group=doc_id)
        from src.ingestion.chunker import content_hash
        storage.replace_chunks(doc_id, [
            {"section_path": "II. 사업의 내용", "section_type": "business", "chunk_order": i, "raw_text": t,
             "embedding_text": f"기업명: {name}\n\n{t}", "token_count": len(t), "content_hash": content_hash(t)}
            for i, t in enumerate(texts)])


def build(settings, storage, embedder=None):
    embedder = embedder or FakeEmbedder()
    es = EmbeddingStore(storage, settings.embeddings_dir, embedder, batch_size=2)
    stats = es.embed_pending_chunks()
    mark_embedded_documents_ready(storage)
    vs = VectorStore(settings.index_path, storage, es)
    vs.load()
    return Retriever(storage, vs, es), vs, es, stats


DOCS = [
    ("00000001", "가상메모리", "100001", "d1", "2024-03-10", ["HBM 메모리 생산 확대", "서버용 DRAM 수요 증가"]),
    ("00000002", "가상장비", "100002", "d2", "2024-05-01", ["TC 본더 장비 공급", "HBM 적층 장비 수주"]),
    ("00000003", "가상소재", "100003", "d3", "2025-03-01", ["HBM 언더필 소재 개발"]),
]


def test_search_mapping_and_filters(settings, storage):
    seed_docs(storage, DOCS)
    r, vs, es, stats = build(settings, storage)
    assert stats.embedded == 5 and vs.index.ntotal == 5
    res = r.search("HBM 적층 장비", top_k=3)
    assert res and res[0].raw_text == "HBM 적층 장비 수주"
    for x in res:
        ch = storage.get_chunk(x.chunk_id)
        assert ch["raw_text"] == x.raw_text and ch["corp_code"] == x.corp_code
    only = r.search("HBM", top_k=5, corp_codes=["00000001"])
    assert only and {x.corp_code for x in only} == {"00000001"}
    assert r.search("HBM", section_types=["mdna"]) == []


def test_index_save_reload_and_rebuild(settings, storage):
    seed_docs(storage, DOCS)
    r, vs, es, _ = build(settings, storage)
    vs2 = VectorStore(settings.index_path, storage, es)
    idx = vs2.load()
    assert idx.ntotal == 5
    # 불일치 상황: 인덱스 파일 손상 → 재구축
    settings.index_path.write_bytes(b"")
    settings.index_path.unlink()
    vs3 = VectorStore(settings.index_path, storage, es)
    assert vs3.load().ntotal == 5


def test_no_duplicate_embedding_and_reuse_on_rerun(settings, storage):
    docs = DOCS + [("00000001", "가상메모리", "100001", "d1b", "2024-08-10", ["HBM 메모리 생산 확대"])]
    seed_docs(storage, docs)
    emb = FakeEmbedder()
    r, vs, es, stats = build(settings, storage, emb)
    assert stats.reused_same_text == 1          # 같은 기업, 같은 본문
    assert vs.index.ntotal == 5
    calls = emb.calls
    stats2 = EmbeddingStore(storage, settings.embeddings_dir, emb).embed_pending_chunks()
    assert stats2.chunks_total == 0 and emb.calls == calls
    # 출처는 두 문서 모두 유지
    rows = storage.query("SELECT document_id FROM chunks WHERE raw_text='HBM 메모리 생산 확대'")
    assert {x["document_id"] for x in rows} == {"d1", "d1b"}


def test_retrieval_budget(settings, storage):
    seed_docs(storage, DOCS)
    r, *_ = build(settings, storage)
    r.reset_budget(1)
    r.search("HBM")
    r.search("HBM")  # 동일 질문은 캐시
    try:
        r.search("DRAM")
        raised = False
    except BudgetExceeded:
        raised = True
    assert raised and r.cache_hits == 1


def test_model_change_does_not_mix_vectors(settings, storage):
    seed_docs(storage, DOCS)
    r, vs, es, _ = build(settings, storage, FakeEmbedder(dims=64, model="fake-a"))
    es_b = EmbeddingStore(storage, settings.embeddings_dir, FakeEmbedder(dims=32, model="fake-b"))
    vs_b = VectorStore(settings.index_path, storage, es_b)
    idx = vs_b.load()
    assert idx.d == 32 and idx.ntotal == 0   # 다른 모델 벡터를 섞지 않고 새로 구성

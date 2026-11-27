from tests.test_retriever import DOCS, build, seed_docs


def test_future_documents_blocked(settings, storage):
    seed_docs(storage, DOCS)
    r, vs, es, _ = build(settings, storage)
    res = r.search("HBM 언더필 소재", top_k=5, as_of_date="2024-12-31")
    assert res
    assert all(x.published_at <= "2024-12-31" for x in res)
    assert "HBM 언더필 소재 개발" not in [x.raw_text for x in res]


def test_temporal_search_uses_subset_index(settings, storage, monkeypatch):
    seed_docs(storage, DOCS)
    r, vs, es, _ = build(settings, storage)
    seen = {}
    orig = vs.search

    def spy(q, k, allowed_ids=None):
        seen["allowed"] = allowed_ids
        return orig(q, k, allowed_ids)

    monkeypatch.setattr(vs, "search", spy)
    r.search("HBM", as_of_date="2024-04-01")
    future_ids = {c["embedding_id"] for c in storage.query(
        "SELECT ch.embedding_id FROM chunks ch JOIN documents d USING(document_id) WHERE d.published_at > '2024-04-01'")}
    assert seen["allowed"] is not None
    assert not (seen["allowed"] & future_ids)   # 미래 벡터는 검색 대상 인덱스에 들어가지 않음


def test_correction_version_as_of(settings, storage):
    seed_docs(storage, [("00000009", "가상정정", "100009", "orig", "2024-03-10", ["원본 매출 100"]),
                        ("00000009", "가상정정", "100009", "corr", "2024-06-10", ["정정 매출 120"])])
    storage.conn.execute("UPDATE documents SET version_group='g1' WHERE corp_code='00000009'")
    storage.conn.commit()
    r, *_ = build(settings, storage)
    before = r.search("매출", corp_codes=["00000009"], as_of_date="2024-04-01")
    after = r.search("매출", corp_codes=["00000009"])
    assert [x.raw_text for x in before] == ["원본 매출 100"]
    assert [x.raw_text for x in after] == ["정정 매출 120"]

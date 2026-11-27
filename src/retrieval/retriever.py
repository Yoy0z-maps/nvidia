"""RAG 검색 엔진 (Vector + 선택적 Keyword, RRF 결합).

유사도 점수는 검색 순위용이며 사실 여부나 투자 성공 확률을 의미하지 않는다.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np

from src.retrieval.embedding import EmbeddingStore, cache_key_for
from src.retrieval.vector_store import VectorStore
from src.schemas import SearchResult
from src.storage import Storage

log = logging.getLogger(__name__)
RRF_K = 60


class BudgetExceeded(Exception):
    pass


class Retriever:
    def __init__(self, storage: Storage, vector_store: VectorStore, emb_store: EmbeddingStore,
                 max_calls: int | None = None):
        self.storage = storage
        self.vs = vector_store
        self.emb_store = emb_store
        self.max_calls = max_calls
        self.calls = 0
        self.cache_hits = 0

    def reset_budget(self, max_calls: int | None) -> None:
        self.max_calls, self.calls, self.cache_hits = max_calls, 0, 0

    # ------------------------------------------------------------------
    def _eligible_chunks(self, corp_codes, section_types, as_of_date) -> list[dict]:
        sql = ["""SELECT ch.chunk_id, ch.embedding_id, ch.section_path, ch.section_type, ch.raw_text,
                         d.document_id, d.document_type, d.published_at, d.source_url, d.corp_code,
                         c.corp_name, c.stock_code
                  FROM chunks ch JOIN documents d USING(document_id) JOIN companies c USING(corp_code)
                  WHERE ch.embedding_id IS NOT NULL AND d.ingestion_status='READY'
                    AND c.ingestion_status != 'EXCLUDED'"""]
        params: list = []
        if corp_codes:
            sql.append(f"AND d.corp_code IN ({','.join('?' * len(corp_codes))})")
            params += list(corp_codes)
        if section_types:
            sql.append(f"AND ch.section_type IN ({','.join('?' * len(section_types))})")
            params += list(section_types)
        if as_of_date:
            sql.append("AND d.published_at <= ?")
            params.append(as_of_date)
        sql[0] = sql[0].replace("d.document_id,", "d.document_id, d.version_group, d.rcept_no,")
        rows = self.storage.query(" ".join(sql), params)
        # 정정공시: 기준일 시점에 유효한 최신 버전만 사용
        latest: dict[str, tuple[str, str]] = {}
        for r in rows:
            g = r["version_group"] or r["document_id"]
            key = (r["published_at"], r["rcept_no"])
            if g not in latest or key > latest[g]:
                latest[g] = key
        return [r for r in rows if (r["published_at"], r["rcept_no"]) == latest[r["version_group"] or r["document_id"]]]

    def _query_vector(self, query: str) -> np.ndarray:
        emb = self.emb_store.embedder
        key = cache_key_for(emb.model, emb.dimensions, "QUERY::" + query)
        cached = self.storage.cache_get("query_embedding", key)
        if cached is not None:
            return np.array(cached, dtype="float32")
        vec = emb.embed([query])[0]
        self.storage.cache_set("query_embedding", key, vec.tolist())
        return vec

    @staticmethod
    def _keyword_rank(query: str, rows: list[dict], limit: int) -> list[int]:
        terms = [t for t in re.split(r"[\s,\.\?\!·/]+", query) if len(t) >= 2]
        if not terms:
            return []
        scored = []
        for r in rows:
            text = r["raw_text"]
            score = sum(text.count(t) for t in terms)
            if score:
                scored.append((score, r["chunk_id"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scored[:limit]]

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, corp_codes: list[str] | None = None,
               section_types: list[str] | None = None, as_of_date: str | None = None,
               hybrid: bool = True) -> list[SearchResult]:
        cache_key = repr((query, top_k, sorted(corp_codes or []), sorted(section_types or []), as_of_date, hybrid))
        cached = self.storage.cache_get("search", cache_key)
        if cached is not None:
            self.cache_hits += 1
            # 캐시된 결과도 현재 데이터 기준으로 존재하는지 확인
            valid = [SearchResult(**r) for r in cached if self.storage.get_chunk(r["chunk_id"])]
            if len(valid) == len(cached):
                return valid
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"검색 호출 예산({self.max_calls}) 소진")
        self.calls += 1

        rows = self._eligible_chunks(corp_codes, section_types, as_of_date)
        if not rows:
            return []
        by_emb: dict[int, list[dict]] = defaultdict(list)
        by_chunk = {r["chunk_id"]: r for r in rows}
        for r in rows:
            by_emb[r["embedding_id"]].append(r)
        has_excluded = self.storage.one("SELECT 1 FROM companies WHERE ingestion_status='EXCLUDED' LIMIT 1")
        # 검색 대상이 아닌 벡터(제외 기업·분기보고서 등)가 인덱스에 남아 있으면 허용 벡터만으로 검색한다
        index = self.vs.index if self.vs.index is not None else self.vs.load()
        filtered = bool(corp_codes or section_types or as_of_date or has_excluded or len(by_emb) < index.ntotal)
        allowed = set(by_emb) if filtered else None

        qv = self._query_vector(query)
        hits = self.vs.search(qv, top_k * 3, allowed_ids=allowed)
        vector_rank: list[int] = []
        vec_score: dict[int, float] = {}
        for emb_id, score in hits:
            cands = by_emb.get(emb_id)
            if not cands:
                continue
            # 동일 본문이 여러 보고서에 있으면 가장 최근 공개 문서를 대표 출처로 사용
            best = max(cands, key=lambda r: r["published_at"])
            vector_rank.append(best["chunk_id"])
            vec_score[best["chunk_id"]] = score

        if hybrid:
            kw_rank = self._keyword_rank(query, rows, top_k * 3)
            fused: dict[int, float] = defaultdict(float)
            for rank, cid in enumerate(vector_rank):
                fused[cid] += 1.0 / (RRF_K + rank + 1)
            for rank, cid in enumerate(kw_rank):
                fused[cid] += 1.0 / (RRF_K + rank + 1)
            ordered = sorted(fused, key=lambda c: -fused[c])[:top_k]
            methods = {c: ("hybrid" if c in vec_score and c in kw_rank else
                           "vector" if c in vec_score else "keyword") for c in ordered}
        else:
            ordered = vector_rank[:top_k]
            methods = {c: "vector" for c in ordered}

        results = []
        for cid in ordered:
            r = by_chunk[cid]
            results.append(SearchResult(
                chunk_id=cid, corp_code=r["corp_code"], corp_name=r["corp_name"], stock_code=r["stock_code"] or "",
                document_id=r["document_id"], document_type=r["document_type"], section_path=r["section_path"],
                section_type=r["section_type"], published_at=r["published_at"], source_url=r["source_url"] or "",
                raw_text=r["raw_text"], similarity_score=round(vec_score.get(cid, 0.0), 4),
                retrieval_method=methods[cid]))
        self.storage.cache_set("search", cache_key, [x.model_dump() for x in results])
        return results

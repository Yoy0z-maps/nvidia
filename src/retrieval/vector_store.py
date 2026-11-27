"""FAISS 인덱스 (IndexIDMap2 + IndexFlatIP, L2 정규화 → Cosine Similarity).

FAISS 벡터 ID = SQLite embeddings.embedding_id
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.embedding import EmbeddingStore, l2_normalize
from src.storage import Storage

log = logging.getLogger(__name__)


def new_index(dims: int) -> faiss.IndexIDMap2:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dims))


class VectorStore:
    def __init__(self, index_path: Path, storage: Storage, emb_store: EmbeddingStore):
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.json")
        self.storage = storage
        self.emb_store = emb_store
        self.model = emb_store.embedder.model
        self.dims = emb_store.embedder.dimensions
        self.index: faiss.IndexIDMap2 | None = None

    # ---- 로딩 / 일관성 ----
    def _expected_ids(self) -> set[int]:
        return {e["embedding_id"] for e in self.storage.list_embeddings(self.model, self.dims)}

    def _index_ids(self, index: faiss.IndexIDMap2) -> set[int]:
        return set(faiss.vector_to_array(index.id_map).tolist())

    def load(self, rebuild_if_inconsistent: bool = True) -> faiss.IndexIDMap2:
        index = None
        if self.index_path.exists() and self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if meta.get("model") != self.model or meta.get("dimensions") != self.dims:
                log.warning("인덱스 모델/차원 불일치 (%s/%s) - 다른 종류의 벡터를 섞지 않도록 재구축합니다.",
                            meta.get("model"), meta.get("dimensions"))
            else:
                index = faiss.read_index(str(self.index_path))
        if index is not None and rebuild_if_inconsistent and self._index_ids(index) != self._expected_ids():
            log.warning("FAISS 인덱스와 SQLite 메타데이터가 불일치합니다 - 재구축합니다.")
            index = None
        if index is None:
            index = self.rebuild()
        self.index = index
        return index

    def rebuild(self) -> faiss.IndexIDMap2:
        index = new_index(self.dims)
        rows = self.storage.list_embeddings(self.model, self.dims)
        missing = []
        for i in range(0, len(rows), 1000):
            part = rows[i:i + 1000]
            vecs, ids = [], []
            for r in part:
                try:
                    vecs.append(self.emb_store.load_vector(r["vector_path"]))
                    ids.append(r["embedding_id"])
                except FileNotFoundError:
                    missing.append(r["embedding_id"])
            if vecs:
                index.add_with_ids(l2_normalize(np.stack(vecs)), np.array(ids, dtype="int64"))
        if missing:
            log.error("벡터 파일 누락 %d개 (embedding_id=%s...)", len(missing), missing[:10])
        self.index = index
        self.save()
        log.info("FAISS 인덱스 재구축: %d vectors", index.ntotal)
        return index

    def add_missing(self) -> int:
        """SQLite에는 있지만 인덱스에 없는 벡터만 추가."""
        if self.index is None:
            self.load()
        assert self.index is not None
        have = self._index_ids(self.index)
        rows = [r for r in self.storage.list_embeddings(self.model, self.dims) if r["embedding_id"] not in have]
        if rows:
            vecs = np.stack([self.emb_store.load_vector(r["vector_path"]) for r in rows])
            self.index.add_with_ids(l2_normalize(vecs), np.array([r["embedding_id"] for r in rows], dtype="int64"))
            self.save()
        return len(rows)

    def save(self) -> None:
        assert self.index is not None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps({"model": self.model, "dimensions": self.dims,
                                              "ntotal": int(self.index.ntotal)}))

    # ---- 검색 ----
    def search(self, query_vec: np.ndarray, top_k: int, allowed_ids: set[int] | None = None
               ) -> list[tuple[int, float]]:
        """allowed_ids가 주어지면 해당 벡터만으로 임시 인덱스를 구성해 검색한다.

        (전체 검색 후 사후 필터링하지 않는다 - 과거 시점 평가의 미래 문서 누출 방지)
        """
        if self.index is None:
            self.load()
        assert self.index is not None
        q = l2_normalize(query_vec)
        index = self.index
        if allowed_ids is not None:
            ids = sorted(allowed_ids & self._index_ids(self.index))
            if not ids:
                return []
            index = new_index(self.dims)
            vecs = np.stack([self.index.reconstruct(int(i)) for i in ids])
            index.add_with_ids(vecs, np.array(ids, dtype="int64"))
        if index.ntotal == 0:
            return []
        scores, ids = index.search(q, min(top_k, index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

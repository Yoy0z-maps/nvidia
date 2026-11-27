"""임베딩 생성 및 캐싱."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from config.settings import PREPROCESSING_VERSION
from src.storage import Storage

log = logging.getLogger(__name__)

# 비용 추정용 (USD / 1M tokens). 실제 청구액은 OpenAI 대시보드 기준.
EMBEDDING_PRICE_PER_M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    if v.ndim == 1:
        v = v[None, :]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, dimensions: int, max_retries: int = 5, client=None):
        if client is None:
            from openai import OpenAI
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.total_tokens = 0
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                resp = self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dimensions)
                self.total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
                vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype="float32")
                if vecs.shape != (len(texts), self.dimensions):
                    raise RuntimeError(f"임베딩 차원 불일치: {vecs.shape}")
                return vecs
            except Exception as exc:  # openai.RateLimitError, APIConnectionError 등
                last = exc
                status = getattr(exc, "status_code", None)
                if status in (400, 401, 403, 404):
                    raise
                wait = min(60, 2 ** attempt)
                log.warning("임베딩 API 실패 (%s) - %ss 후 재시도 [%d/%d]", type(exc).__name__, wait,
                            attempt + 1, self.max_retries)
                time.sleep(wait)
        raise RuntimeError(f"임베딩 API 재시도 초과: {last}")


def cache_key_for(model: str, dims: int, embedding_text: str) -> str:
    h = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{model}|{dims}|{h}|{PREPROCESSING_VERSION}".encode()).hexdigest()


@dataclass
class EmbeddingStats:
    chunks_total: int = 0
    reused_same_text: int = 0
    reused_cache: int = 0
    embedded: int = 0
    failed_chunk_ids: list[int] = field(default_factory=list)
    tokens: int = 0
    api_calls: int = 0

    def estimated_cost_usd(self, model: str) -> float | None:
        price = EMBEDDING_PRICE_PER_M.get(model)
        return None if price is None else round(self.tokens / 1_000_000 * price, 6)


class EmbeddingStore:
    """임베딩 벡터 파일(.npy) + SQLite embeddings 테이블."""

    def __init__(self, storage: Storage, embeddings_dir: Path, embedder: Embedder, batch_size: int = 64):
        self.storage = storage
        self.dir = embeddings_dir
        self.embedder = embedder
        self.batch_size = batch_size

    def _vector_relpath(self, key: str) -> str:
        return f"{self.embedder.model}-{self.embedder.dimensions}/{key[:2]}/{key}.npy"

    def load_vector(self, vector_path: str) -> np.ndarray:
        return np.load(self.dir / vector_path).astype("float32")

    def _save(self, key: str, vec: np.ndarray) -> int:
        rel = self._vector_relpath(key)
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, l2_normalize(vec)[0])
        return self.storage.insert_embedding(key, self.embedder.model, self.embedder.dimensions, rel)

    def embed_pending_chunks(self) -> EmbeddingStats:
        model, dims = self.embedder.model, self.embedder.dimensions
        stats = EmbeddingStats()
        pending = self.storage.chunks_without_embedding()
        stats.chunks_total = len(pending)
        to_embed: list[tuple[dict, str]] = []
        first_key_by_text: dict[tuple[str, str], str] = {}  # 이번 실행 안의 동일 기업·동일 본문
        for ch in pending:
            text_key = (ch["corp_code"], ch["content_hash"])
            if text_key in first_key_by_text:
                to_embed.append((ch, first_key_by_text[text_key]))
                stats.reused_same_text += 1
                continue
            reuse = self.storage.find_reusable_embedding(ch["corp_code"], ch["content_hash"], model, dims)
            if reuse is not None:
                self.storage.set_chunk_embedding(ch["chunk_id"], reuse)
                stats.reused_same_text += 1
                continue
            key = cache_key_for(model, dims, ch["embedding_text"])
            existing = self.storage.get_embedding_by_key(key)
            if existing:
                self.storage.set_chunk_embedding(ch["chunk_id"], existing["embedding_id"])
                stats.reused_cache += 1
                continue
            first_key_by_text[text_key] = key
            to_embed.append((ch, key))

        # 같은 배치 안의 동일 key 중복 제거
        seen: dict[str, list[dict]] = {}
        for ch, key in to_embed:
            seen.setdefault(key, []).append(ch)
        keys = list(seen)
        tokens_before = getattr(self.embedder, "total_tokens", 0)
        calls_before = getattr(self.embedder, "calls", 0)
        for i in range(0, len(keys), self.batch_size):
            batch = keys[i:i + self.batch_size]
            texts = [seen[k][0]["embedding_text"] for k in batch]
            try:
                vecs = self.embedder.embed(texts)
            except Exception as exc:
                ids = [c["chunk_id"] for k in batch for c in seen[k]]
                stats.failed_chunk_ids.extend(ids)
                self.storage.log("embedding", "ERROR", f"임베딩 실패: {exc}", detail={"chunk_ids": ids})
                log.error("임베딩 배치 실패 (%d chunks): %s", len(ids), exc)
                continue
            for k, v in zip(batch, vecs):
                emb_id = self._save(k, v)
                stats.embedded += 1
                for c in seen[k]:
                    self.storage.set_chunk_embedding(c["chunk_id"], emb_id)
        stats.tokens = getattr(self.embedder, "total_tokens", 0) - tokens_before
        stats.api_calls = getattr(self.embedder, "calls", 0) - calls_before
        log.info("임베딩 완료: 대상 %d, 신규 %d, 동일본문 재사용 %d, 캐시 재사용 %d, 실패 %d, 토큰 %d, 추정비용 $%s",
                 stats.chunks_total, stats.embedded, stats.reused_same_text, stats.reused_cache,
                 len(stats.failed_chunk_ids), stats.tokens, stats.estimated_cost_usd(model))
        return stats

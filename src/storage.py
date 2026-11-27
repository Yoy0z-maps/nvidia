"""SQLite 메타데이터 저장소."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    corp_code TEXT PRIMARY KEY,
    stock_code TEXT,
    corp_name TEXT NOT NULL,
    market_type TEXT,
    ingestion_status TEXT NOT NULL DEFAULT 'PENDING',
    updated_at TEXT,
    discovery_hop INTEGER,
    discovered_from TEXT,
    discovery_reason TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    corp_code TEXT NOT NULL REFERENCES companies(corp_code) ON DELETE CASCADE,
    rcept_no TEXT NOT NULL UNIQUE,
    report_name TEXT NOT NULL,
    document_type TEXT NOT NULL,
    published_at TEXT NOT NULL,
    source_url TEXT,
    raw_path TEXT,
    content_hash TEXT,
    ingestion_status TEXT NOT NULL DEFAULT 'PENDING',
    is_correction INTEGER NOT NULL DEFAULT 0,
    version_group TEXT,
    section_report_json TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL UNIQUE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL,
    section_type TEXT NOT NULL DEFAULT 'other',
    chunk_order INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_id INTEGER REFERENCES embeddings(embedding_id) ON DELETE SET NULL,
    UNIQUE(document_id, chunk_order)
);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_emb ON chunks(embedding_id);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    input_text TEXT,
    seed_stock_code TEXT,
    as_of_date TEXT,
    model TEXT,
    configuration_json TEXT,
    result_path TEXT,
    elapsed_ms INTEGER,
    token_usage_json TEXT,
    status TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    stage TEXT NOT NULL,
    level TEXT NOT NULL,
    corp_code TEXT,
    rcept_no TEXT,
    message TEXT NOT NULL,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS discovery_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_corp_code TEXT,
    candidate_name TEXT NOT NULL,
    method TEXT NOT NULL,
    hop INTEGER NOT NULL,
    direction TEXT,
    rationale TEXT,
    evidence_json TEXT,
    resolved_corp_code TEXT,
    status TEXT NOT NULL,
    status_reason TEXT
);

CREATE TABLE IF NOT EXISTS kv_cache (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, cache_key)
);
"""


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        r = self.conn.execute(sql, tuple(params)).fetchone()
        return dict(r) if r else None

    # ---- companies ----
    def upsert_company(self, corp_code: str, corp_name: str, stock_code: str | None,
                       market_type: str | None = None, status: str | None = None, **discovery: Any) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO companies(corp_code, stock_code, corp_name, market_type, ingestion_status, updated_at,
                                         discovery_hop, discovered_from, discovery_reason)
                   VALUES(?,?,?,?,COALESCE(?, 'PENDING'),?,?,?,?)
                   ON CONFLICT(corp_code) DO UPDATE SET
                     stock_code=excluded.stock_code, corp_name=excluded.corp_name,
                     market_type=COALESCE(excluded.market_type, companies.market_type),
                     ingestion_status=COALESCE(?, companies.ingestion_status),
                     updated_at=excluded.updated_at,
                     discovery_hop=COALESCE(companies.discovery_hop, excluded.discovery_hop),
                     discovered_from=COALESCE(companies.discovered_from, excluded.discovered_from),
                     discovery_reason=COALESCE(companies.discovery_reason, excluded.discovery_reason)""",
                (corp_code, stock_code, corp_name, market_type, status, now_iso(),
                 discovery.get("discovery_hop"), discovery.get("discovered_from"),
                 discovery.get("discovery_reason"), status),
            )

    def set_company_status(self, corp_code: str, status: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE companies SET ingestion_status=?, updated_at=? WHERE corp_code=?",
                      (status, now_iso(), corp_code))

    def get_company(self, corp_code: str) -> dict | None:
        return self.one("SELECT * FROM companies WHERE corp_code=?", (corp_code,))

    def get_company_by_stock(self, stock_code: str) -> dict | None:
        return self.one("SELECT * FROM companies WHERE stock_code=?", (stock_code,))

    def list_companies(self) -> list[dict]:
        return self.query("SELECT * FROM companies ORDER BY COALESCE(discovery_hop, 0), corp_name")

    # ---- documents ----
    def document_exists(self, rcept_no: str) -> bool:
        return self.one("SELECT 1 FROM documents WHERE rcept_no=?", (rcept_no,)) is not None

    def upsert_document(self, **d: Any) -> None:
        cols = ["document_id", "corp_code", "rcept_no", "report_name", "document_type", "published_at",
                "source_url", "raw_path", "content_hash", "ingestion_status", "is_correction", "version_group",
                "section_report_json"]
        defaults = {"is_correction": 0, "ingestion_status": "PENDING"}
        vals = [d.get(k, defaults.get(k)) if d.get(k) is not None else defaults.get(k) for k in cols]
        # 호출자가 명시한 컬럼만 갱신한다
        updates = ", ".join(f"{k}=COALESCE(excluded.{k}, documents.{k})" for k in cols[1:]
                            if d.get(k) is not None) or "document_id=documents.document_id"
        with self.tx() as c:
            c.execute(
                f"INSERT INTO documents({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
                f"ON CONFLICT(document_id) DO UPDATE SET {updates}", vals)

    def get_document(self, document_id: str) -> dict | None:
        return self.one("SELECT * FROM documents WHERE document_id=?", (document_id,))

    def list_documents(self, corp_code: str | None = None) -> list[dict]:
        if corp_code:
            return self.query("SELECT * FROM documents WHERE corp_code=? ORDER BY published_at DESC", (corp_code,))
        return self.query("SELECT d.*, c.corp_name FROM documents d JOIN companies c USING(corp_code) "
                          "ORDER BY c.corp_name, d.published_at DESC")

    # ---- chunks ----
    def replace_chunks(self, document_id: str, chunks: list[dict]) -> list[int]:
        """문서의 Chunk를 교체한다 (전처리 버전 변경 시 재실행 가능)."""
        ids: list[int] = []
        with self.tx() as c:
            c.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            for ch in chunks:
                cur = c.execute(
                    """INSERT INTO chunks(document_id, section_path, section_type, chunk_order, raw_text,
                                          embedding_text, token_count, content_hash)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (document_id, ch["section_path"], ch.get("section_type", "other"), ch["chunk_order"],
                     ch["raw_text"], ch["embedding_text"], ch["token_count"], ch["content_hash"]))
                ids.append(int(cur.lastrowid))
        return ids

    def chunks_without_embedding(self) -> list[dict]:
        return self.query(
            """SELECT ch.*, d.corp_code FROM chunks ch JOIN documents d USING(document_id)
               JOIN companies c ON c.corp_code = d.corp_code
               WHERE ch.embedding_id IS NULL AND d.ingestion_status NOT IN ('SKIPPED_INTERIM')
                 AND c.ingestion_status != 'EXCLUDED'
               ORDER BY ch.chunk_id""")

    def find_reusable_embedding(self, corp_code: str, content_hash: str, model: str, dims: int) -> int | None:
        """동일 기업·동일 본문의 기존 임베딩 재사용 (중복 임베딩 방지)."""
        r = self.one(
            """SELECT ch.embedding_id FROM chunks ch
               JOIN documents d USING(document_id)
               JOIN embeddings e ON e.embedding_id = ch.embedding_id
               WHERE d.corp_code=? AND ch.content_hash=? AND e.model=? AND e.dimensions=? LIMIT 1""",
            (corp_code, content_hash, model, dims))
        return int(r["embedding_id"]) if r else None

    def set_chunk_embedding(self, chunk_id: int, embedding_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE chunks SET embedding_id=? WHERE chunk_id=?", (embedding_id, chunk_id))

    def get_chunk(self, chunk_id: int) -> dict | None:
        return self.one(
            """SELECT ch.*, d.corp_code, d.published_at, d.source_url, d.document_type, d.rcept_no,
                      d.report_name, c.corp_name, c.stock_code
               FROM chunks ch JOIN documents d USING(document_id) JOIN companies c USING(corp_code)
               WHERE ch.chunk_id=?""", (chunk_id,))

    # ---- embeddings ----
    def get_embedding_by_key(self, cache_key: str) -> dict | None:
        return self.one("SELECT * FROM embeddings WHERE cache_key=?", (cache_key,))

    def insert_embedding(self, cache_key: str, model: str, dims: int, vector_path: str) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO embeddings(cache_key, model, dimensions, vector_path) VALUES(?,?,?,?)",
                            (cache_key, model, dims, vector_path))
            return int(cur.lastrowid)

    def list_embeddings(self, model: str, dims: int) -> list[dict]:
        return self.query("SELECT * FROM embeddings WHERE model=? AND dimensions=? ORDER BY embedding_id",
                          (model, dims))

    # ---- logs ----
    def log(self, stage: str, level: str, message: str, corp_code: str | None = None,
            rcept_no: str | None = None, detail: Any = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO ingestion_logs(created_at, stage, level, corp_code, rcept_no, message, detail_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (now_iso(), stage, level, corp_code, rcept_no, message,
                 json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else None))

    def list_logs(self, level: str | None = None, limit: int = 500) -> list[dict]:
        if level:
            return self.query("SELECT * FROM ingestion_logs WHERE level=? ORDER BY log_id DESC LIMIT ?",
                              (level, limit))
        return self.query("SELECT * FROM ingestion_logs ORDER BY log_id DESC LIMIT ?", (limit,))

    # ---- discovery ----
    def add_candidate(self, **c: Any) -> int:
        cols = ["created_at", "source_corp_code", "candidate_name", "method", "hop", "direction", "rationale",
                "evidence_json", "resolved_corp_code", "status", "status_reason"]
        c.setdefault("created_at", now_iso())
        with self.tx() as conn:
            cur = conn.execute(f"INSERT INTO discovery_candidates({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                               [c.get(k) for k in cols])
            return int(cur.lastrowid)

    def list_candidates(self) -> list[dict]:
        return self.query("SELECT * FROM discovery_candidates ORDER BY candidate_id")

    # ---- kv cache ----
    def cache_get(self, namespace: str, key: str) -> Any | None:
        r = self.one("SELECT value_json FROM kv_cache WHERE namespace=? AND cache_key=?", (namespace, key))
        return json.loads(r["value_json"]) if r else None

    def cache_set(self, namespace: str, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO kv_cache(namespace, cache_key, value_json, created_at) VALUES(?,?,?,?)",
                      (namespace, key, json.dumps(value, ensure_ascii=False, default=str), now_iso()))

    # ---- runs ----
    def save_run(self, **r: Any) -> None:
        cols = ["run_id", "input_text", "seed_stock_code", "as_of_date", "model", "configuration_json",
                "result_path", "elapsed_ms", "token_usage_json", "status", "created_at"]
        r.setdefault("created_at", now_iso())
        with self.tx() as c:
            c.execute(f"INSERT OR REPLACE INTO research_runs({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                      [r.get(k) for k in cols])

    def list_runs(self) -> list[dict]:
        return self.query("SELECT * FROM research_runs ORDER BY created_at DESC")

    # ---- 통계 ----
    def stats(self) -> dict:
        def n(sql: str) -> int:
            return int(self.conn.execute(sql).fetchone()[0])
        return {
            "companies": n("SELECT COUNT(*) FROM companies"),
            "companies_ready": n("SELECT COUNT(*) FROM companies WHERE ingestion_status='READY'"),
            "documents": n("SELECT COUNT(*) FROM documents"),
            "chunks": n("SELECT COUNT(*) FROM chunks"),
            "chunks_embedded": n("SELECT COUNT(*) FROM chunks WHERE embedding_id IS NOT NULL"),
            "embeddings": n("SELECT COUNT(*) FROM embeddings"),
            "errors": n("SELECT COUNT(*) FROM ingestion_logs WHERE level='ERROR'"),
        }

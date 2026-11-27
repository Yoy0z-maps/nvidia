"""구성요소 조립 및 단계별(재실행 가능) 파이프라인 함수."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable

from config.settings import Settings
from src.discovery.entity_resolution import EntityResolver, ResolvedCompany
from src.graph.graph_store import GraphStore
from src.ingestion.chunker import SectionChunker, get_tokenizer
from src.ingestion.collector import DisclosureCollector, mark_embedded_documents_ready
from src.ingestion.corp_registry import CorpRegistry, MarketTypeResolver
from src.ingestion.dart_client import DartClient
from src.ingestion.section_selector import SectionSelector
from src.llm import OpenAILLM
from src.retrieval.embedding import EmbeddingStore, OpenAIEmbedder
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.schemas import SectionTitleClassification
from src.storage import Storage

log = logging.getLogger(__name__)


@dataclass
class App:
    settings: Settings
    storage: Storage
    dart: DartClient | None
    registry: CorpRegistry | None
    market: MarketTypeResolver | None
    collector: DisclosureCollector | None
    emb_store: EmbeddingStore | None
    vector_store: VectorStore | None
    retriever: Retriever | None
    resolver: EntityResolver
    llm: OpenAILLM | None

    def load_graph(self) -> GraphStore:
        return GraphStore.load(self.settings.graph_dir)


def build_app(settings: Settings, *, need_dart: bool = True, need_openai: bool = True,
              embedder=None, llm=None, tokenizer=None, dart_client=None) -> App:
    settings.ensure_dirs()
    storage = Storage(settings.db_path)
    dart = dart_client or (DartClient(settings.dart_api_key) if need_dart and settings.dart_api_key else None)
    if need_dart and dart is None:
        raise RuntimeError("DART_API_KEY가 설정되지 않았습니다 (.env 확인).")
    registry = None
    cache = settings.processed_dir / "corp_codes.json"
    if dart is not None or cache.exists():
        registry = CorpRegistry.load_or_download(cache, dart)
    market = MarketTypeResolver(dart, storage) if dart is not None else None

    if llm is None and need_openai:
        llm = OpenAILLM(settings.openai_api_key, settings.llm_model, settings.max_llm_calls)
    if embedder is None and need_openai:
        embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.embedding_dimensions)

    emb_store = vs = retriever = None
    if embedder is not None:
        emb_store = EmbeddingStore(storage, settings.embeddings_dir, embedder, settings.embedding_batch_size)
        vs = VectorStore(settings.index_path, storage, emb_store)
        retriever = Retriever(storage, vs, emb_store, settings.max_retrieval_calls)

    collector = None
    if dart is not None:
        tok = tokenizer or get_tokenizer(settings.embedding_model)
        chunker = SectionChunker(tok, settings.chunk_size, settings.chunk_overlap)

        classifier = None
        if llm is not None:
            def classifier(titles: list[str]) -> list[str]:
                out = llm.parse(
                    "공시 문서의 목차 중 회사의 사업 내용(제품, 원재료, 생산설비, 매출, 수주, 주요계약, 연구개발)에 해당하는 "
                    "항목만 그대로 골라라. 재무제표, 감사, 이사회, 주주, 배당 관련 항목은 제외한다.",
                    "\n".join(titles), SectionTitleClassification, purpose="section_fallback")
                return out.selected_titles
        selector = SectionSelector(token_counter=lambda t: len(tok.encode(t)), title_classifier=classifier)
        collector = DisclosureCollector(settings, storage, dart, selector, chunker)
    resolver = EntityResolver(storage, registry, market)
    return App(settings, storage, dart, registry, market, collector, emb_store, vs, retriever, resolver, llm)


# ----------------------------------------------------------------------
def collect_company(app: App, corp_code: str, corp_name: str, stock_code: str, as_of: dt.date | None = None,
                    progress: Callable[[str], None] = log.info, **discovery) -> dict:
    if app.collector is None:
        raise RuntimeError("DART 클라이언트가 없어 공시를 수집할 수 없습니다.")
    market = app.market.market_type(corp_code) if app.market else None
    app.storage.upsert_company(corp_code, corp_name, stock_code, market, **discovery)
    res = app.collector.ingest_company(corp_code, corp_name, stock_code, as_of)
    progress(f"{corp_name}: 신규 {len(res.ingested)} / 기존 {len(res.skipped_existing)} / 실패 {len(res.failed)} 문서, "
             f"Chunk {res.chunks_created}개")
    return res.__dict__


def apply_document_policy(app: App, progress: Callable[[str], None] = log.info) -> int:
    """COLLECT_INTERIM_REPORTS=false 이면 이미 수집된 반기·분기보고서를 검색 대상에서 뺀다 (파일·벡터는 보존).

    다시 true로 바꾸면 검색 대상으로 복구된다. 반환: 상태가 바뀐 문서 수.
    """
    interim = ("반기보고서", "분기보고서")
    with app.storage.tx() as c:
        if not app.settings.collect_interim_reports:
            n = c.execute("UPDATE documents SET ingestion_status='SKIPPED_INTERIM' WHERE document_type IN (?,?) "
                          "AND ingestion_status IN ('READY','CHUNKED')", interim).rowcount
            msg = f"반기·분기보고서 {n}건을 검색 대상에서 제외 (COLLECT_INTERIM_REPORTS=false)"
        else:
            n = c.execute("UPDATE documents SET ingestion_status='CHUNKED' WHERE ingestion_status='SKIPPED_INTERIM'"
                          ).rowcount
            msg = f"반기·분기보고서 {n}건을 검색 대상으로 복구"
    if n:
        progress(msg)
    return n


def embed_and_index(app: App, progress: Callable[[str], None] = log.info) -> dict:
    if app.emb_store is None or app.vector_store is None:
        raise RuntimeError("임베딩 구성요소가 없습니다 (OPENAI_API_KEY 확인).")
    apply_document_policy(app, progress)
    # 검색 대상에서 빠진 문서(분기보고서 등)의 Chunk는 임베딩하지 않는다
    stats = app.emb_store.embed_pending_chunks()
    app.vector_store.load()
    added = app.vector_store.add_missing()
    ready = mark_embedded_documents_ready(app.storage)
    out = {**stats.__dict__, "estimated_cost_usd": stats.estimated_cost_usd(app.emb_store.embedder.model),
           "index_added": added, "index_total": int(app.vector_store.index.ntotal), "documents_ready": ready}
    progress(f"임베딩: 신규 {stats.embedded}, 재사용 {stats.reused_same_text + stats.reused_cache}, "
             f"실패 {len(stats.failed_chunk_ids)}, 토큰 {stats.tokens}, 추정비용 ${out['estimated_cost_usd']}, "
             f"인덱스 {out['index_total']} vectors")
    return out


def seed_company(app: App) -> dict:
    s = app.settings
    corp = app.registry.get_by_stock_code(s.seed_stock_code) if app.registry else None
    if corp is None:
        c = app.storage.get_company_by_stock(s.seed_stock_code)
        if c:
            return {"corp_code": c["corp_code"], "corp_name": c["corp_name"], "stock_code": c["stock_code"]}
        raise RuntimeError(f"Seed 종목코드 {s.seed_stock_code}를 기업 사전에서 찾을 수 없습니다.")
    return {"corp_code": corp.corp_code, "corp_name": corp.corp_name, "stock_code": corp.stock_code}


def make_ingest_fn(app: App, progress: Callable[[str], None] = log.info):
    """Discovery/Ripple에서 신규 기업 공시를 수집·임베딩하는 함수."""
    def _ingest(r: ResolvedCompany, hop: int, discovered_from: str, reason: str) -> bool:
        res = collect_company(app, r.corp_code, r.corp_name, r.stock_code, progress=progress,
                              discovery_hop=hop, discovered_from=discovered_from, discovery_reason=reason)
        if not res["ingested"] and not res["skipped_existing"]:
            return False
        embed_and_index(app, progress)
        c = app.storage.get_company(r.corp_code)
        return bool(c and c["ingestion_status"] == "READY")
    return _ingest

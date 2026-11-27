"""Related Company Discovery: Seed 공시 → 관련 기업 후보 → 상장사 식별 → 공시 수집/임베딩 → 관계 검증."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable

from src.discovery.entity_resolution import EntityResolver, ResolvedCompany
from src.graph.graph_store import GraphStore
from src.llm import LLM, LLMBudgetExceeded, load_prompt
from src.research.relationship_extractor import format_chunks
from src.research.validator import RelationValidator, quote_in_text
from src.retrieval.retriever import BudgetExceeded, Retriever
from src.schemas import (VERIFIED_STATUSES, CandidateRelation, CandidateRelationList, Edge, HypothesisCompanyList,
                         MentionedEntityList, NodeType, RelationshipType, VerificationStatus)
from src.storage import Storage

log = logging.getLogger(__name__)

GROUNDED_QUERIES = [
    "주요 매출처 및 고객",
    "주요 원재료 매입처 및 공급업체",
    "생산설비 및 장비 투자",
    "경쟁 기업 및 시장 점유율",
    "공급계약 계약상대방",
    "계열회사 및 종속회사 사업",
]
VERIFY_QUERIES = ["주요 제품 및 서비스", "주요 매출처 및 고객", "사업의 개요"]


@dataclass
class DiscoveryReport:
    processed: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    edges_added: int = 0
    stopped_reason: str = "탐색 완료"


class CompanyDiscovery:
    def __init__(self, storage: Storage, llm: LLM, retriever: Retriever, resolver: EntityResolver,
                 ingest_company: Callable[[ResolvedCompany, int, str, str], bool], graph: GraphStore,
                 max_companies: int = 20, max_hops: int = 2, max_per_node: int = 5, top_k: int = 5,
                 progress_cb: Callable[[str], None] | None = None):
        self.storage = storage
        self.llm = llm
        self.retriever = retriever
        self.resolver = resolver
        self.ingest_company = ingest_company
        self.graph = graph
        self.max_companies = max_companies
        self.max_hops = max_hops
        self.max_per_node = max_per_node
        self.top_k = top_k
        self.validator = RelationValidator(storage, llm, resolver.as_dict_resolver())
        self.progress_cb = progress_cb
        self.today = dt.date.today().isoformat()

    def _p(self, msg: str) -> None:
        log.info(msg)
        if self.progress_cb:
            self.progress_cb(msg)

    def _prior_result(self, corp_code: str) -> dict | None:
        c = self.storage.get_company(corp_code)
        if not c or c["ingestion_status"] != "READY":
            return None
        return self.storage.one(
            "SELECT status FROM discovery_candidates WHERE resolved_corp_code=? "
            "AND status IN ('VERIFIED_RELATED','UNVERIFIED') ORDER BY candidate_id DESC LIMIT 1", (corp_code,))

    def _ready_count(self) -> int:
        return sum(1 for c in self.storage.list_companies() if c["ingestion_status"] == "READY")

    # ------------------------------------------------------------------
    def _retrieve(self, corp_code: str, queries: list[str]) -> list:
        res = []
        for q in queries:
            res += self.retriever.search(q, top_k=self.top_k, corp_codes=[corp_code])
        return list({r.chunk_id: r for r in res}.values())

    def document_grounded(self, company: dict, chunks: list) -> list[dict]:
        out = self.llm.parse(load_prompt("company_discovery", "document_grounded", seed_name=company["corp_name"]),
                             format_chunks(chunks), MentionedEntityList, purpose="discovery_grounded")
        by_id = {c.chunk_id: c for c in chunks}
        cands = []
        for ent in out.entities:
            ch = by_id.get(ent.chunk_id)
            quote_ok = bool(ch) and quote_in_text(ent.quote, ch.raw_text)
            item = {"name": ent.name, "kind": ent.entity_kind, "direction": ent.direction,
                    "relation": ent.relation_to_seed, "chunk_id": ent.chunk_id, "quote": ent.quote,
                    "quote_ok": quote_ok, "method": "DOCUMENT"}
            if not quote_ok:
                self.storage.add_candidate(source_corp_code=company["corp_code"], candidate_name=ent.name,
                                           method="DOCUMENT", hop=0, direction=ent.direction,
                                           rationale=ent.relation_to_seed, status="REJECTED",
                                           status_reason="인용이 원문에 없음",
                                           evidence_json=json.dumps(item, ensure_ascii=False))
                continue
            cands.append(item)
        return cands

    def hypothesis_driven(self, company: dict, chunks: list, known: list[str]) -> list[dict]:
        prompt = load_prompt("company_discovery", "hypothesis", seed_name=company["corp_name"],
                             max_candidates=self.max_per_node + 2, known=", ".join(known))
        out = self.llm.parse(prompt, format_chunks(chunks[:8]), HypothesisCompanyList, purpose="discovery_hypothesis")
        return [{"name": c.company_name, "kind": "COMPANY", "direction": c.exploration_direction,
                 "relation": c.rationale, "method": "HYPOTHESIS"} for c in out.candidates]

    def verify_relation(self, source: dict, target: ResolvedCompany, hop: int) -> list[Edge]:
        """후보 기업 공시에서 실제 제품·사업 구조와 Seed 측과의 관계를 검증."""
        chunks = self._retrieve(target.corp_code, VERIFY_QUERIES + [f"{source['corp_name']} 관련 제품"])
        if not chunks:
            return []
        prompt = load_prompt("company_discovery", "verify_company", candidate=target.corp_name,
                             seed_name=source["corp_name"])
        prompt += "\n\n" + load_prompt("relationship_extraction", node=target.corp_name,
                                       event=f"{source['corp_name']}와의 산업 관계 검증", max_relations=5)
        rels = self.llm.parse(prompt, format_chunks(chunks), CandidateRelationList, purpose="discovery_verify").relations
        edges = []
        for o in self.validator.validate_many(rels):
            e = self._add(o, hop)
            if e:
                edges.append(e)
        return edges

    def _add(self, o, hop: int) -> Edge | None:
        if o.status == VerificationStatus.CONTRADICTED:
            return None
        c: CandidateRelation = o.candidate
        ids = []
        for side, name, t in (("source", c.source_name, c.source_type), ("target", c.target_name, c.target_type)):
            if t == NodeType.COMPANY:
                comp = getattr(o, f"{side}_company")
                if not comp:
                    return None
                ids.append(self.graph.ensure_company_node(comp["corp_name"], comp["stock_code"], comp["corp_code"]))
            else:
                ids.append(self.graph.ensure_concept_node(name, t))
        if ids[0] == ids[1]:
            return None
        e = Edge(edge_id=f"edge_{uuid.uuid4().hex[:10]}", source_id=ids[0], target_id=ids[1],
                 relationship=c.relationship, claim_type=c.claim_type, verification_status=o.status,
                 evidence=o.evidence, conditions=o.conditions, reasoning=c.reasoning, created_at=self.today, hop=hop)
        self.graph.add_edge(e)
        return e

    def _grounded_to_relation(self, company: dict, item: dict) -> CandidateRelation:
        rel_map = {"CUSTOMER": RelationshipType.SUPPLIES_TO, "SUPPLIER": RelationshipType.SUPPLIES_TO,
                   "COMPETITOR": RelationshipType.COMPETES_WITH, "PARTNER": RelationshipType.PARTNERS_WITH,
                   "AFFILIATE": RelationshipType.PARTNERS_WITH}
        kind = NodeType(item["kind"])
        rel = rel_map.get(item["direction"], RelationshipType.EXPOSED_TO if kind != NodeType.PRODUCT else RelationshipType.PRODUCES)
        src, tgt = (company["corp_name"], item["name"])
        st, tt = NodeType.COMPANY, kind
        if item["direction"] == "SUPPLIER":
            src, tgt, st, tt = item["name"], company["corp_name"], kind, NodeType.COMPANY
        if kind == NodeType.RESOURCE and item["direction"] not in rel_map:
            rel = RelationshipType.REQUIRES
        return CandidateRelation(source_name=src, source_type=st, target_name=tgt, target_type=tt, relationship=rel,
                                 claim_type="FACT", chunk_id=item["chunk_id"], quote=item["quote"],
                                 reasoning=item["relation"], conditions=[])

    # ------------------------------------------------------------------
    def run(self, seed: dict) -> DiscoveryReport:
        rep = DiscoveryReport()
        frontier: list[tuple[dict, int]] = [(seed, 0)]
        processed: set[str] = {seed["corp_code"]}
        try:
            while frontier:
                company, hop = frontier.pop(0)
                if hop >= self.max_hops:
                    continue
                self._p(f"[Discovery Hop {hop + 1}] {company['corp_name']} 기준 관련 기업 탐색")
                chunks = self._retrieve(company["corp_code"], GROUNDED_QUERIES)
                if not chunks:
                    self._p("  검색 가능한 공시 Chunk가 없어 건너뜀")
                    continue
                grounded = self.document_grounded(company, chunks)
                # 문서 근거 관계(산업/제품 포함)를 그래프에 반영
                rels = [self._grounded_to_relation(company, item) for item in grounded
                        if item["kind"] != "COMPANY" or self.resolver.resolve_for_graph(item["name"])[0]]
                for o in self.validator.validate_many(rels):
                    if self._add(o, hop + 1):
                        rep.edges_added += 1
                known = [c["corp_name"] for c in self.storage.list_companies()]
                hyp = self.hypothesis_driven(company, chunks, known)
                ranked = [g for g in grounded if g["kind"] == "COMPANY"] + hyp
                picked = 0
                for cand in ranked:
                    if picked >= self.max_per_node:
                        break
                    r, reason = self.resolver.resolve(cand["name"])
                    rec = dict(source_corp_code=company["corp_code"], candidate_name=cand["name"], method=cand["method"],
                               hop=hop + 1, direction=cand["direction"], rationale=cand["relation"],
                               evidence_json=json.dumps({k: cand.get(k) for k in ("chunk_id", "quote")}, ensure_ascii=False))
                    if r is None:
                        u, _ = self.resolver.resolve_for_graph(cand["name"])
                        if u is not None:
                            # DART에 등록된 비상장 기업: 공시 수집 대상은 아니지만 그래프 노드로 남김
                            self.storage.add_candidate(**rec, resolved_corp_code=u.corp_code, status="UNLISTED",
                                                       status_reason=f"비상장 기업 ({u.corp_name}) - 그래프 노드로만 반영")
                            rep.candidates.append({**rec, "status": "UNLISTED", "reason": u.corp_name})
                            continue
                        self.storage.add_candidate(**rec, status="UNRESOLVED", status_reason=reason)
                        rep.candidates.append({**rec, "status": "UNRESOLVED", "reason": reason})
                        continue
                    prev = self.storage.get_company(r.corp_code)
                    if prev and prev["ingestion_status"] == "EXCLUDED":
                        self.storage.add_candidate(**rec, resolved_corp_code=r.corp_code, status="SKIPPED_EXCLUDED",
                                                   status_reason="사용자가 제외한 기업")
                        continue
                    if r.corp_code in processed:
                        self.storage.add_candidate(**rec, resolved_corp_code=r.corp_code, status="DUPLICATE",
                                                   status_reason="이미 처리한 기업")
                        continue
                    if self._ready_count() >= self.max_companies:
                        rep.stopped_reason = f"최대 기업 수({self.max_companies}) 도달"
                        self.storage.add_candidate(**rec, resolved_corp_code=r.corp_code, status="SKIPPED_BUDGET",
                                                   status_reason=rep.stopped_reason)
                        frontier.clear()
                        break
                    processed.add(r.corp_code)
                    picked += 1
                    prior = self._prior_result(r.corp_code)
                    if prior:
                        # 이전 실행에서 이미 수집·임베딩·검증한 기업은 재사용 (재수집·재검증 없음)
                        status = prior["status"]
                        rep.processed.append({"corp_name": r.corp_name, "stock_code": r.stock_code, "status": status,
                                              "method": "REUSED", "hop": hop + 1})
                        self._p(f"  기존 기업 재사용: {r.corp_name}({r.stock_code}) → {status}")
                        if status == "VERIFIED_RELATED":
                            frontier.append(({"corp_code": r.corp_code, "corp_name": r.corp_name,
                                              "stock_code": r.stock_code}, hop + 1))
                        continue
                    self._p(f"  후보 {r.corp_name}({r.stock_code}) [{cand['method']}] 공시 수집")
                    ok = self.ingest_company(r, hop + 1, company["corp_code"], f"{cand['method']}: {cand['relation'][:200]}")
                    if not ok:
                        self.storage.add_candidate(**rec, resolved_corp_code=r.corp_code, status="INGEST_FAILED",
                                                   status_reason="공시 수집/임베딩 실패")
                        continue
                    edges = self.verify_relation(company, r, hop + 1)
                    verified = [e for e in edges if e.verification_status in VERIFIED_STATUSES]
                    rep.edges_added += len(edges)
                    status = "VERIFIED_RELATED" if verified else "UNVERIFIED"
                    self.storage.add_candidate(**rec, resolved_corp_code=r.corp_code, status=status,
                                               status_reason=f"검증 관계 {len(verified)}개 / 전체 {len(edges)}개")
                    rep.processed.append({"corp_name": r.corp_name, "stock_code": r.stock_code, "status": status,
                                          "method": cand["method"], "hop": hop + 1})
                    self._p(f"    → {status} (검증 관계 {len(verified)}개)")
                    if verified:
                        frontier.append(({"corp_code": r.corp_code, "corp_name": r.corp_name,
                                          "stock_code": r.stock_code}, hop + 1))
        except (LLMBudgetExceeded, BudgetExceeded) as exc:
            rep.stopped_reason = f"탐색 예산 소진: {exc}"
            self._p(f"⚠ {rep.stopped_reason}")
        return rep

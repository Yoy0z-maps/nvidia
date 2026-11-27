"""Multi-hop Ripple Engine: 제한된 BFS 기반 산업 파급효과 탐색."""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.discovery.entity_resolution import EntityResolver
from src.graph.graph_store import GraphStore, company_node_id, path_status
from src.graph.graph_store import slug
from src.graph.traversal import _neighbors, best_company_paths, describe_path, node_tiers
from src.llm import LLM, LLMBudgetExceeded, load_prompt
from src.research.event_extractor import extract_event
from src.research.query_generator import generate_queries
from src.research.relationship_extractor import extract_relations
from src.research.validator import RelationValidator, ValidationOutcome
from src.retrieval.retriever import BudgetExceeded, Retriever
from src.schemas import (STATUS_STRENGTH, VERIFIED_STATUSES, CandidateRelation, CompanyImpactList, Edge, Event,
                         EventType, Node, NodeType, RelationshipType, VerificationStatus)
from src.storage import Storage

log = logging.getLogger(__name__)

PREMISE_REL = {
    EventType.DEMAND_INCREASE: RelationshipType.INCREASES_DEMAND_FOR,
    EventType.DEMAND_DECREASE: RelationshipType.DECREASES_DEMAND_FOR,
    EventType.CAPACITY_EXPANSION: RelationshipType.INCREASES_DEMAND_FOR,
    EventType.SUPPLY_DISRUPTION: RelationshipType.CONSTRAINS,
    EventType.PRICE_INCREASE: RelationshipType.EXPOSED_TO,
    EventType.PRICE_DECREASE: RelationshipType.EXPOSED_TO,
}


@dataclass
class RippleConfig:
    max_hops: int = 3
    max_branches_per_hop: int = 3
    max_retrieval_calls: int = 20
    max_llm_calls: int = 25
    top_k: int = 5
    queries_per_node: int = 2
    max_relations_per_node: int = 6
    as_of_date: str | None = None
    use_existing_graph: bool = True
    # 인사이트 모드: 가설 관계도 확장하고, 근거 없는 가설 인사이트를 별도로 생성 (결과에 명확히 구분 표시)
    insight_mode: bool = True
    max_result_companies: int = 15


@dataclass
class RippleResult:
    run_id: str
    input_text: str
    event: dict
    event_warnings: list[str]
    graph: dict
    company_results: list[dict]
    summary: str
    rejected: list[dict]
    progress: list[str]
    budget: dict
    elapsed_ms: int
    stopped_reason: str
    as_of_date: str | None
    used_chunk_ids: list[int] = field(default_factory=list)
    tiers: list[dict] = field(default_factory=list)
    insights: list[dict] = field(default_factory=list)
    caveat: str = ""
    root_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RippleEngine:
    def __init__(self, storage: Storage, llm: LLM, retriever: Retriever, resolver: EntityResolver,
                 config: RippleConfig, base_graph: GraphStore | None = None,
                 progress_cb: Callable[[str], None] | None = None,
                 company_expander: Callable[[dict], bool] | None = None):
        self.storage = storage
        self.llm = llm
        self.retriever = retriever
        self.resolver = resolver
        self.cfg = config
        self.base_graph = base_graph if config.use_existing_graph and not config.as_of_date else None
        self.progress_cb = progress_cb
        self.company_expander = company_expander
        self.validator = RelationValidator(storage, llm, resolver.as_dict_resolver(), config.as_of_date)
        self.progress: list[str] = []

    def _p(self, msg: str) -> None:
        self.progress.append(msg)
        log.info(msg)
        if self.progress_cb:
            self.progress_cb(msg)

    # ------------------------------------------------------------------
    def _node_for(self, gs: GraphStore, name: str, ntype: NodeType, outcome: ValidationOutcome | None,
                  side: str) -> str | None:
        if ntype == NodeType.COMPANY:
            comp = getattr(outcome, f"{side}_company", None) if outcome else None
            if comp is None:
                r, _ = self.resolver.resolve_for_graph(name)
                comp = r.to_dict() if r else None
            if comp is None:
                return None
            return gs.ensure_company_node(comp["corp_name"], comp["stock_code"], comp["corp_code"])
        return gs.ensure_concept_node(name, ntype)

    def _add_outcome(self, gs: GraphStore, o: ValidationOutcome, hop: int, today: str) -> Edge | None:
        c = o.candidate
        src = self._node_for(gs, c.source_name, c.source_type, o, "source")
        tgt = self._node_for(gs, c.target_name, c.target_type, o, "target")
        if not src or not tgt or src == tgt:
            return None
        edge = Edge(edge_id=f"edge_{uuid.uuid4().hex[:10]}", source_id=src, target_id=tgt,
                    relationship=c.relationship,
                    claim_type=c.claim_type if o.status != VerificationStatus.VERIFIED_FACT else "FACT",
                    verification_status=o.status, evidence=o.evidence, conditions=o.conditions,
                    reasoning=c.reasoning + (f" | 검증 메모: {'; '.join(o.reasons)}" if o.reasons else ""),
                    created_at=today, as_of_date=self.cfg.as_of_date, hop=hop)
        gs.add_edge(edge)
        return edge

    def _expandable(self) -> set:
        return set(VERIFIED_STATUSES) | ({VerificationStatus.HYPOTHESIS} if self.cfg.insight_mode else set())

    def _seed_graph(self, gs: GraphStore) -> None:
        if self.base_graph is None:
            return
        for nid, d in self.base_graph.g.nodes(data=True):
            gs.add_node(Node(**d))
        allowed = None if self.cfg.insight_mode else VERIFIED_STATUSES
        for e in self.base_graph.edges(allowed):
            if e.verification_status != VerificationStatus.CONTRADICTED:
                gs.add_edge(e)

    # ------------------------------------------------------------------
    def run(self, input_text: str, seed: dict) -> RippleResult:
        t0 = time.time()
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        today = (self.cfg.as_of_date or dt.date.today().isoformat())
        self.llm.reset_budget(self.cfg.max_llm_calls)
        self.retriever.reset_budget(self.cfg.max_retrieval_calls)
        gs = GraphStore()
        self._seed_graph(gs)
        rejected: list[dict] = []
        used_queries: list[str] = []
        used_chunks: set[int] = set()
        stopped = "탐색 완료"

        # 1. 이벤트 구조화
        self._p("이벤트 구조화 중")
        event, warnings = extract_event(self.llm, input_text, self.resolver)
        for w in warnings:
            self._p(f"⚠ {w}")
        event_node = gs.add_node(Node(node_id=f"event_{run_id}", node_type=NodeType.EVENT,
                                      name=event.summary[:60] or "입력 이벤트"))
        seed_node = gs.ensure_company_node(seed["corp_name"], seed["stock_code"], seed["corp_code"])
        roots = [seed_node]
        premise_rel = PREMISE_REL.get(event.event_type, RelationshipType.EXPOSED_TO)
        for name, t in [(p, NodeType.PRODUCT) for p in event.products] + [(i, NodeType.INDUSTRY) for i in event.industries]:
            roots.append(gs.ensure_concept_node(name, t))
        roots += [nid for nid in event.entity_ids if nid in gs.g]
        roots = list(dict.fromkeys(roots))
        for r in roots:
            # 사용자 입력 전제 (근거 검증 대상이 아님, hop=0)
            gs.add_edge(Edge(edge_id=f"premise_{uuid.uuid4().hex[:8]}", source_id=event_node, target_id=r,
                             relationship=premise_rel, claim_type="HYPOTHESIS",
                             verification_status=VerificationStatus.HYPOTHESIS,
                             reasoning="사용자 입력 이벤트 전제 (검증 대상 아님)", created_at=today,
                             as_of_date=self.cfg.as_of_date, hop=0))

        # 2. BFS
        frontier = [(r, 1) for r in roots[: max(self.cfg.max_branches_per_hop, 1) + 1]]
        visited: set[str] = {event_node}
        try:
            while frontier:
                node_id, hop = frontier.pop(0)
                if node_id in visited:
                    continue  # 순환 감지
                visited.add(node_id)
                if hop > self.cfg.max_hops:
                    continue
                nd = gs.g.nodes[node_id]
                self._p(f"[Hop {hop}] {nd['name']} ({nd['node_type']}) 탐색")
                next_nodes: list[tuple[str, int]] = []

                # 2-1. 기존 검증 그래프 우선
                expandable = self._expandable()
                for nb, e in _neighbors(gs, node_id, include_weak=True, undirected=True):
                    if nb not in visited and e.verification_status in expandable:
                        next_nodes.append((nb, STATUS_STRENGTH[e.verification_status]))

                # 2-2. 부족하면 RAG + LLM 확장 (질문 생성·관계 추출·검증 3회 + 최종 요약 1회 보존)
                remaining = (self.cfg.max_llm_calls - self.llm.usage.calls) if self.cfg.max_llm_calls else 99
                reserve = 5 if self.cfg.insight_mode else 4   # 질문·추출·검증 3회 + 요약(+인사이트)
                if len(next_nodes) < self.cfg.max_branches_per_hop and remaining < reserve:
                    stopped = (f"LLM 호출 예산({self.cfg.max_llm_calls}회) 도달 - 최종 요약용 호출을 남기고 탐색 중단. "
                               "MAX_LLM_CALLS를 늘리면 더 깊이 탐색합니다.")
                    self._p(f"⚠ {stopped}")
                    break
                if len(next_nodes) < self.cfg.max_branches_per_hop:
                    queries = generate_queries(self.llm, self.storage, nd["name"], nd["node_type"], event.summary,
                                               used_queries, self.cfg.queries_per_node)
                    results = []
                    corp_filter = [nd["corp_code"]] if nd.get("corp_code") and hop == 1 and node_id == seed_node else None
                    for q in queries:
                        used_queries.append(q)
                        results += self.retriever.search(q, top_k=self.cfg.top_k, as_of_date=self.cfg.as_of_date,
                                                         corp_codes=corp_filter)
                    uniq = list({r.chunk_id: r for r in results}.values())
                    used_chunks |= {r.chunk_id for r in uniq}
                    self._p(f"  검색 {len(queries)}건 → Chunk {len(uniq)}개")
                    cands = extract_relations(self.llm, nd["name"], event.summary, uniq,
                                              self.cfg.max_relations_per_node)
                    self._p(f"  관계 후보 {len(cands)}개 추출")
                    for c, o in zip(cands, self.validator.validate_many(cands)):
                        if o.status == VerificationStatus.CONTRADICTED:
                            rejected.append(o.to_dict())
                            self._p(f"  ✗ 반박됨: {c.source_name} -{c.relationship.value}-> {c.target_name}")
                            continue
                        edge = self._add_outcome(gs, o, hop, today)
                        if edge is None:
                            rejected.append({**o.to_dict(), "reasons": o.reasons + ["노드 식별 실패"]})
                            continue
                        mark = "✓" if o.status in VERIFIED_STATUSES else "?"
                        self._p(f"  {mark} {c.source_name} -{c.relationship.value}-> {c.target_name} [{o.status.value}]")
                        # 근거부족/반박은 확장하지 않는다 (가설은 인사이트 모드에서만, 낮은 우선순위로 확장)
                        if o.status in expandable:
                            nxt = edge.target_id if edge.source_id == node_id else (
                                edge.source_id if edge.target_id == node_id else edge.target_id)
                            if nxt not in visited:
                                next_nodes.append((nxt, STATUS_STRENGTH[o.status]))
                            self._maybe_expand_company(gs, nxt)
                # 강한 근거 순으로 최대 N개 분기
                seen_next = []
                for nid, _ in sorted(next_nodes, key=lambda x: -x[1]):
                    if nid not in seen_next:
                        seen_next.append(nid)
                for nid in seen_next[: self.cfg.max_branches_per_hop]:
                    if hop + 1 <= self.cfg.max_hops:
                        frontier.append((nid, hop + 1))
        except (LLMBudgetExceeded, BudgetExceeded) as exc:
            stopped = f"예산 소진으로 중단 (내부 호출 상한이며 OpenAI 잔액과는 무관): {exc}"
            self._p(f"⚠ {stopped}")

        # 3. 기업별 결과
        company_results, summary, extra = self._company_results(gs, roots, event)
        insights: list[dict] = []
        if self.cfg.insight_mode:
            try:
                insights = self._hypothesis_insights(gs, event, company_results)
            except (LLMBudgetExceeded, BudgetExceeded) as exc:
                self._p(f"⚠ 가설 인사이트 생략 (예산 소진): {exc}")
        result = RippleResult(
            run_id=run_id, input_text=input_text, event=event.model_dump(mode="json"), event_warnings=warnings,
            graph=gs.to_dict(), company_results=company_results, summary=summary, rejected=rejected,
            progress=self.progress, elapsed_ms=int((time.time() - t0) * 1000), stopped_reason=stopped,
            as_of_date=self.cfg.as_of_date, used_chunk_ids=sorted(used_chunks),
            tiers=extra.get("tiers", []), caveat=extra.get("caveat", ""), root_ids=roots, insights=insights,
            budget={"llm": self.llm.usage.to_dict(), "retrieval_calls": self.retriever.calls,
                    "retrieval_cache_hits": self.retriever.cache_hits,
                    "limits": {"llm": self.cfg.max_llm_calls, "retrieval": self.cfg.max_retrieval_calls}})
        return result

    def _maybe_expand_company(self, gs: GraphStore, node_id: str) -> None:
        if self.company_expander is None or self.cfg.as_of_date:
            return
        d = gs.g.nodes[node_id]
        if d.get("node_type") != NodeType.COMPANY.value or not d.get("corp_code"):
            return
        if self.storage.get_company(d["corp_code"]) and self.storage.get_company(d["corp_code"])["ingestion_status"] == "READY":
            return
        try:
            if self.company_expander(d):
                self._p(f"  + 신규 기업 공시 수집: {d['name']}")
        except Exception as exc:
            self._p(f"  ⚠ 신규 기업 수집 실패 ({d['name']}): {exc}")

    # ------------------------------------------------------------------
    def _company_results(self, gs: GraphStore, roots: list[str], event: Event) -> tuple[list[dict], str, dict]:
        best: dict[str, tuple[list[Edge], list[str], str]] = {}
        for r in roots:
            if gs.g.nodes[r].get("node_type") == NodeType.COMPANY.value:
                best[r] = ([], [r], r)  # 루트 자체가 기업 (이벤트 직접 연결)
            for cid, (edges, nodes) in best_company_paths(gs, r, self.cfg.max_hops + 1, include_weak=True,
                                                          undirected=True).items():
                old = best.get(cid)
                if old is None or (old[0] and (len(edges), -STATUS_STRENGTH[path_status(edges)])
                                   < (len(old[0]), -STATUS_STRENGTH[path_status(old[0])])):
                    best[cid] = (edges, nodes, r)
        rows = []
        for cid, (path, nodes, start) in best.items():
            d = gs.g.nodes[cid]
            evidence = [ev.model_dump() for e in path for ev in e.evidence]
            rows.append({
                "node_id": cid, "corp_name": d["name"], "stock_code": d.get("stock_code"), "corp_code": d.get("corp_code"),
                "path": describe_path(gs, path, start) if path else "(이벤트 직접 연결 기업)",
                "path_edge_ids": [e.edge_id for e in path], "path_nodes": nodes, "hops": len(path),
                "relationship_types": [e.relationship.value for e in path],
                "impact_scope": "DIRECT" if len(path) <= 1 else "INDIRECT",
                "verification_status": path_status(path).value if path else "PREMISE",
                "evidence": evidence, "n_paths": 1,
            })
        rows.sort(key=lambda r: (r["hops"], -STATUS_STRENGTH.get(VerificationStatus(r["verification_status"]), 5)
                                 if r["verification_status"] != "PREMISE" else -5))
        rows = rows[: self.cfg.max_result_companies]
        tiers = node_tiers(gs, roots, self.cfg.max_hops + 1)
        for r in rows:
            r["tier"] = tiers.get(r["node_id"], r["hops"])
        summary, extra = "", {"tiers": self._fallback_tiers(gs, tiers), "caveat": ""}
        try:
            summary, extra = self._impact_analysis(gs, rows, event, tiers)
        except (LLMBudgetExceeded, BudgetExceeded) as exc:
            summary = f"(LLM 예산 소진으로 기업별 영향 분석을 생성하지 못했습니다: {exc})"
            for r in rows:
                r.update(economic_impact="UNCERTAIN", revenue_exposure="UNKNOWN", conditions=[], counter_scenario="",
                         technical_relevance="", role="", impact_factor="")
        return rows, summary, extra

    @staticmethod
    def _fallback_tiers(gs: GraphStore, tiers: dict[str, int]) -> list[dict]:
        out: dict[int, list] = {}
        for nid, t in tiers.items():
            d = gs.g.nodes[nid]
            if d.get("node_type") == NodeType.EVENT.value:
                continue
            out.setdefault(t, []).append({"name": d["name"], "node_id": nid, "node_type": d["node_type"], "effect": ""})
        return [{"tier": t, "title": "직접 영향" if t == 0 else f"{t}차 파급효과", "items": items}
                for t, items in sorted(out.items())]

    def _hypothesis_insights(self, gs: GraphStore, event: Event, rows: list[dict], max_insights: int = 8
                             ) -> list[dict]:
        """근거 미확인 가설 인사이트 (LLM 1회). 상장사 이름은 기업 사전으로 식별 가능한 것만 종목코드를 붙인다."""
        from src.schemas import HypothesisInsightList
        self._p("가설 인사이트 생성 중 (근거 미확인)")
        data = (f"이벤트: {event.summary} (유형: {event.event_type.value})\n"
                f"입력 원문: {event.input_quote}\n\n" + _insight_context(gs, rows))
        out = self.llm.parse(load_prompt("hypothesis_insights", max_insights=max_insights), data,
                             HypothesisInsightList, purpose="hypothesis_insights")
        known = {r.get("stock_code") for r in rows}
        result = []
        for ins in out.insights[:max_insights]:
            companies = []
            for name in ins.related_listed_companies[:5]:
                r, reason = self.resolver.resolve(name)
                companies.append({"name": r.corp_name if r else name, "stock_code": r.stock_code if r else None,
                                  "corp_code": r.corp_code if r else None,
                                  "market_type": r.market_type if r else None,
                                  "resolved": bool(r), "in_graph": bool(r and r.stock_code in known),
                                  "note": "" if r else reason})
            result.append({**ins.model_dump(), "companies": companies,
                           "verification_status": VerificationStatus.HYPOTHESIS.value})
        return result

    def _impact_analysis(self, gs: GraphStore, rows: list[dict], event: Event, tiers: dict[str, int]
                         ) -> tuple[str, dict]:
        fallback = {"tiers": self._fallback_tiers(gs, tiers), "caveat": ""}
        if not rows:
            return "근거로 연결된 기업을 찾지 못했습니다.", fallback
        lines = [f"이벤트: {event.summary} (유형: {event.event_type.value}, 가상 시나리오: {event.is_hypothetical})",
                 "\n파급 단계별 노드 (tiers에는 이 이름만 사용):"]
        for t in fallback["tiers"]:
            lines.append(f"- {t['tier']}차 ({t['title']}): " + ", ".join(f"{i['name']}[{i['node_type']}]" for i in t["items"]))
        allowed_chunks: dict[str, set[int]] = {}
        for r in rows:
            lines.append(f"\n## {r['corp_name']} ({r['stock_code']}) - 경로: {r['path']}")
            allowed_chunks[r["corp_name"]] = {e["chunk_id"] for e in r["evidence"]}
            for ev in r["evidence"][:4]:
                ch = self.storage.get_chunk(ev["chunk_id"])
                lines.append(f"- [chunk {ev['chunk_id']}] 인용: {ev['quote']}")
                if ch:
                    lines.append(f"  (원문 일부: {ch['raw_text'][:400]})")
        prompt = load_prompt("verification", "impact") + "\n\n" + load_prompt("examples/impact_example")
        out: CompanyImpactList = self.llm.parse(prompt, "\n".join(lines), CompanyImpactList, purpose="impact_analysis")
        by_name = {c.corp_name: c for c in out.companies}
        for r in rows:
            c = by_name.get(r["corp_name"])
            if c is None:
                r.update(economic_impact="UNCERTAIN", revenue_exposure="UNKNOWN", conditions=[], counter_scenario="",
                         technical_relevance="", cited_chunk_ids=[])
                continue
            cited = [i for i in c.cited_chunk_ids if i in allowed_chunks.get(r["corp_name"], set())]
            exposure = c.revenue_exposure if cited else "UNKNOWN"  # 근거 없는 매출 노출도는 UNKNOWN
            r.update(economic_impact=c.economic_impact, conditions=c.conditions, counter_scenario=c.counter_scenario,
                     technical_relevance=c.technical_relevance, revenue_exposure=exposure, cited_chunk_ids=cited,
                     role=c.role, impact_factor=c.impact_factor)
        # tiers: 그래프에 있는 노드 이름만 남기고, 단계는 그래프 거리로 다시 정한다 (예시 기업 유입 방지)
        by_slug = {slug(gs.g.nodes[n]["name"]): n for n in tiers}
        llm_titles = {t.tier: t.title for t in out.tiers}
        effects: dict[str, str] = {}
        dropped = []
        for t in out.tiers:
            for it in t.items:
                nid = by_slug.get(slug(it.name))
                if nid is None:
                    dropped.append(it.name)
                else:
                    effects[nid] = it.effect
        result_tiers = []
        for t in fallback["tiers"]:
            items = [{**i, "effect": effects.get(i["node_id"], "")} for i in t["items"]]
            result_tiers.append({"tier": t["tier"], "title": llm_titles.get(t["tier"], t["title"]) if t["tier"] else "직접 영향",
                                 "items": items})
        if dropped:
            self._p(f"⚠ 그래프에 없는 이름 제외: {', '.join(dropped[:10])}")
        return out.summary, {"tiers": result_tiers, "caveat": out.caveat}


def _insight_context(gs: GraphStore, rows: list[dict], limit: int = 40) -> str:
    lines = ["[공시 기반 파급 그래프에서 찾은 기업 경로]"]
    lines += [f"- {r['corp_name']}: {r['path']}" for r in rows[:15]]
    names = sorted({d["name"] for _, d in gs.g.nodes(data=True) if d.get("node_type") not in ("EVENT",)})[:limit]
    lines.append("[그래프의 주요 노드] " + ", ".join(names))
    return "\n".join(lines)


def save_result(result: RippleResult, runs_dir: Path, storage: Storage, model: str, config: dict) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    p = runs_dir / f"{result.run_id}.json"
    p.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    storage.save_run(run_id=result.run_id, input_text=result.input_text, seed_stock_code=config.get("seed_stock_code"),
                     as_of_date=result.as_of_date, model=model, configuration_json=json.dumps(config, ensure_ascii=False, default=str),
                     result_path=str(p), elapsed_ms=result.elapsed_ms,
                     token_usage_json=json.dumps(result.budget, ensure_ascii=False),
                     status="COMPLETED" if result.stopped_reason == "탐색 완료" else "PARTIAL")
    return p

"""그래프 경로 탐색 및 시각화."""
from __future__ import annotations

from src.graph.graph_store import GraphStore, path_status
from src.schemas import STATUS_STRENGTH
from src.schemas import VERIFIED_STATUSES, Edge, NodeType

NODE_COLORS = {
    "COMPANY": "#2563eb", "INDUSTRY": "#16a34a", "PRODUCT": "#f59e0b", "TECHNOLOGY": "#7c3aed",
    "RESOURCE": "#0891b2", "EVENT": "#dc2626", "BOTTLENECK": "#be185d",
}
EDGE_STYLES = {
    "VERIFIED_FACT": {"dashes": False, "width": 3, "color": "#111827"},
    "SUPPORTED_INFERENCE": {"dashes": False, "width": 2, "color": "#6b7280"},
    "HYPOTHESIS": {"dashes": True, "width": 1, "color": "#9ca3af"},
    "INSUFFICIENT_EVIDENCE": {"dashes": [2, 6], "width": 1, "color": "#d1d5db"},
    "CONTRADICTED": {"dashes": True, "width": 1, "color": "#ef4444"},
}


def company_paths(gs: GraphStore, start: str, max_hops: int = 5, include_weak: bool = False
                  ) -> dict[str, list[list[Edge]]]:
    """start 노드에서 각 기업 노드까지의 단순 경로(순환 제외)."""
    out: dict[str, list[list[Edge]]] = {}

    def dfs(node: str, path: list[Edge], visited: set[str]) -> None:
        if len(path) >= max_hops:
            return
        for e in gs.out_edges(node, verified_only=not include_weak):
            if e.target_id in visited:
                continue  # 순환 경로 방지
            new = path + [e]
            if gs.g.nodes[e.target_id].get("node_type") == NodeType.COMPANY.value:
                out.setdefault(e.target_id, []).append(new)
            dfs(e.target_id, new, visited | {e.target_id})

    if start in gs.g:
        dfs(start, [], {start})
    return out


def _neighbors(gs: GraphStore, node: str, include_weak: bool, undirected: bool):
    """(이웃 노드, Edge) - 사용자 전제(hop 0)와 반박된 관계는 제외."""
    from src.schemas import VerificationStatus as VS
    allowed = None if include_weak else VERIFIED_STATUSES
    for _, t, d in gs.g.out_edges(node, data=True):
        e = Edge(**d)
        if e.hop == 0 and not e.evidence or e.verification_status == VS.CONTRADICTED:
            continue
        if allowed is None or e.verification_status in allowed:
            yield t, e
    if undirected:
        for s_, _, d in gs.g.in_edges(node, data=True):
            e = Edge(**d)
            if e.hop == 0 and not e.evidence or e.verification_status == VS.CONTRADICTED:
                continue
            if allowed is None or e.verification_status in allowed:
                yield s_, e


def best_company_paths(gs: GraphStore, start: str, max_hops: int = 5, include_weak: bool = True,
                       undirected: bool = True) -> dict[str, tuple[list[Edge], list[str]]]:
    """start에서 각 기업까지의 최단 경로 (같은 길이면 가장 약한 관계가 더 강한 경로).

    undirected=True 이면 관계 방향과 반대로도 따라간다 (예: 공급사 → 출발 기업 관계로 공급사를 찾음).
    반환: {기업 노드: (경로 Edge 목록, 경로 노드 순서)}
    """
    if start not in gs.g:
        return {}
    best: dict[str, tuple[list[Edge], list[str]]] = {start: ([], [start])}
    frontier = [start]
    for _ in range(max_hops):
        cand: dict[str, tuple[list[Edge], list[str]]] = {}
        for n in frontier:
            p_edges, p_nodes = best[n]
            for nb, e in _neighbors(gs, n, include_weak, undirected):
                if nb in best:
                    continue
                new = (p_edges + [e], p_nodes + [nb])
                old = cand.get(nb)
                if old is None or STATUS_STRENGTH[path_status(new[0])] > STATUS_STRENGTH[path_status(old[0])]:
                    cand[nb] = new
        if not cand:
            break
        best.update(cand)
        frontier = list(cand)
    return {n: v for n, v in best.items()
            if n != start and gs.g.nodes[n].get("node_type") == NodeType.COMPANY.value}


def describe_path(gs: GraphStore, path: list[Edge], start: str | None = None) -> str:
    """경로를 실제 관계 방향대로 표시 (역방향으로 따라간 관계는 <-[REL]- 로 표시)."""
    if not path:
        return ""
    cur = start or path[0].source_id
    names = [gs.g.nodes[cur]["name"]]
    for e in path:
        if e.source_id == cur:
            cur = e.target_id
            names.append(f"-[{e.relationship.value}]-> {gs.g.nodes[cur]['name']}")
        else:
            cur = e.source_id
            names.append(f"<-[{e.relationship.value}]- {gs.g.nodes[cur]['name']}")
    return " ".join(names) + f"  (경로 상태: {path_status(path).value})"


def node_tiers(gs: GraphStore, roots: list[str], max_hops: int = 5, undirected: bool = True) -> dict[str, int]:
    """루트(이벤트가 직접 닿는 노드)=0, 이후 관계를 (양방향으로) 따라간 거리. 반박된 관계·사용자 전제 제외."""
    tiers = {r: 0 for r in roots if r in gs.g}
    frontier = list(tiers)
    for depth in range(1, max_hops + 1):
        nxt = []
        for n in frontier:
            for nb, _ in _neighbors(gs, n, include_weak=True, undirected=undirected):
                if nb not in tiers:
                    tiers[nb] = depth
                    nxt.append(nb)
        frontier = nxt
    return tiers

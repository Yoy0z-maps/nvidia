"""Knowledge Graph (NetworkX MultiDiGraph).

- 기업 Node ID: company_{종목코드}
- 제품/산업 등: {type}_{정규화 명칭}, 별칭 관리
- 검증 상태는 Edge 속성으로 분리하며, 저장 시 verified_graph.json / hypotheses.json 으로 나눈다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import networkx as nx

from src.schemas import (STATUS_STRENGTH, VERIFIED_STATUSES, Edge, Node, NodeType, VerificationStatus)


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower()
    s = re.sub(r"\(주\)|주식회사|㈜", "", s)
    s = re.sub(r"[^\w가-힣]+", "", s)
    return s or "unnamed"


def company_node_id(stock_code: str) -> str:
    return f"company_{stock_code}"


class GraphStore:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self._alias: dict[tuple[str, str], str] = {}  # (type, slug) -> node_id

    # ---- 노드 ----
    def add_node(self, node: Node) -> str:
        if node.node_id in self.g:
            data = self.g.nodes[node.node_id]
            data["aliases"] = sorted(set(data.get("aliases", [])) | set(node.aliases) | {node.name})
        else:
            self.g.add_node(node.node_id, **node.model_dump(mode="json"))
        for n in [node.name, *node.aliases]:
            self._alias[(node.node_type.value, slug(n))] = node.node_id
        return node.node_id

    def find_node(self, name: str, node_type: NodeType | None = None) -> str | None:
        types = [node_type.value] if node_type else [t.value for t in NodeType]
        for t in types:
            nid = self._alias.get((t, slug(name)))
            if nid:
                return nid
        return None

    def ensure_concept_node(self, name: str, node_type: NodeType) -> str:
        """기업이 아닌 산업/제품/기술 등의 노드."""
        node_type = NodeType(node_type)
        if node_type == NodeType.COMPANY:
            raise ValueError("기업 노드는 종목코드로 생성해야 합니다 (ensure_company_node).")
        nid = self.find_node(name, node_type)
        if nid:
            return nid
        return self.add_node(Node(node_id=f"{node_type.value.lower()}_{slug(name)}", node_type=node_type, name=name))

    def ensure_company_node(self, corp_name: str, stock_code: str | None, corp_code: str | None = None) -> str:
        if stock_code:
            nid = company_node_id(stock_code)
        elif corp_code:
            nid = f"company_unlisted_{corp_code}"   # 비상장 기업 (DART 고유번호 기준)
        else:
            raise ValueError("기업 노드에는 종목코드 또는 DART 고유번호가 필요합니다.")
        return self.add_node(Node(node_id=nid, node_type=NodeType.COMPANY, name=corp_name,
                                  stock_code=stock_code or None, corp_code=corp_code))

    # ---- 엣지 ----
    def add_edge(self, edge: Edge) -> None:
        for nid in (edge.source_id, edge.target_id):
            if nid not in self.g:
                raise KeyError(f"존재하지 않는 노드: {nid}")
        # 동일 (source, target, relationship) 엣지는 더 강한 상태로 병합
        for k, d in list(self.g.get_edge_data(edge.source_id, edge.target_id, default={}).items()):
            if d["relationship"] == edge.relationship.value:
                old = Edge(**d)
                if STATUS_STRENGTH[edge.verification_status] > STATUS_STRENGTH[old.verification_status]:
                    self.g.remove_edge(edge.source_id, edge.target_id, k)
                    break
                # 근거만 병합
                ev_ids = {(e.chunk_id, e.quote) for e in old.evidence}
                merged = old.evidence + [e for e in edge.evidence if (e.chunk_id, e.quote) not in ev_ids]
                d["evidence"] = [e.model_dump() for e in merged]
                return
        self.g.add_edge(edge.source_id, edge.target_id, key=edge.edge_id, **edge.model_dump(mode="json"))

    def edges(self, statuses: set[VerificationStatus] | None = None) -> list[Edge]:
        out = []
        for _, _, d in self.g.edges(data=True):
            e = Edge(**d)
            if statuses is None or e.verification_status in statuses:
                out.append(e)
        return out

    def out_edges(self, node_id: str, verified_only: bool = True) -> list[Edge]:
        out = []
        for _, _, d in self.g.out_edges(node_id, data=True):
            e = Edge(**d)
            if not verified_only or e.verification_status in VERIFIED_STATUSES:
                out.append(e)
        return out

    def get_edge(self, edge_id: str) -> Edge | None:
        for _, _, k, d in self.g.edges(keys=True, data=True):
            if k == edge_id:
                return Edge(**d)
        return None

    # ---- 저장 ----
    def to_dict(self, statuses: set[VerificationStatus] | None = None) -> dict:
        edges = self.edges(statuses)
        node_ids = {e.source_id for e in edges} | {e.target_id for e in edges}
        if statuses is None:
            node_ids = set(self.g.nodes)
        return {"nodes": [self.g.nodes[n] for n in sorted(node_ids)],
                "edges": [e.model_dump(mode="json") for e in edges]}

    def save(self, graph_dir: Path, prefix: str = "") -> tuple[Path, Path]:
        graph_dir.mkdir(parents=True, exist_ok=True)
        vp = graph_dir / f"{prefix}verified_graph.json"
        hp = graph_dir / f"{prefix}hypotheses.json"
        weak = {s for s in VerificationStatus if s not in VERIFIED_STATUSES}
        vp.write_text(json.dumps(self.to_dict(VERIFIED_STATUSES), ensure_ascii=False, indent=2), encoding="utf-8")
        hp.write_text(json.dumps(self.to_dict(weak), ensure_ascii=False, indent=2), encoding="utf-8")
        return vp, hp

    @classmethod
    def load(cls, graph_dir: Path, prefix: str = "") -> "GraphStore":
        gs = cls()
        for name in (f"{prefix}verified_graph.json", f"{prefix}hypotheses.json"):
            p = graph_dir / name
            if not p.exists():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for n in data["nodes"]:
                gs.add_node(Node(**n))
            for e in data["edges"]:
                gs.add_edge(Edge(**e))
        return gs

    @classmethod
    def from_dict(cls, data: dict) -> "GraphStore":
        gs = cls()
        for n in data["nodes"]:
            gs.add_node(Node(**n))
        for e in data["edges"]:
            gs.add_edge(Edge(**e))
        return gs


def path_status(edges: list[Edge]) -> VerificationStatus:
    """경로 전체 상태 = 가장 약한 Edge의 상태."""
    if not edges:
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    return min((e.verification_status for e in edges), key=lambda s: STATUS_STRENGTH[s])

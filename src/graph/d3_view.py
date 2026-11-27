"""Knowledge Graph 시각화: d3-force (Disjoint force-directed graph).

참고: https://observablehq.com/@d3/disjoint-force-directed-graph/2
- forceCenter 대신 forceX/forceY를 써서 서로 연결되지 않은 컴포넌트도 화면 안에 머문다.
- 드래그 시 alphaTarget(0.3).restart() → 놓으면 alphaTarget(0)으로 다시 안정화된다.
그래프 데이터(GraphStore)와 저장 형식은 바뀌지 않으며, 표시 방법만 바뀐다.
"""
from __future__ import annotations

import json

from src.graph.graph_store import GraphStore
from src.graph.traversal import NODE_COLORS
from src.schemas import VERIFIED_STATUSES

from pathlib import Path

D3_CDN = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
# d3 v7.9.0 (ISC License) 를 프로젝트에 포함해 인터넷 없이도 그래프가 그려지게 한다
D3_LOCAL = Path(__file__).parent / "static" / "d3.v7.9.0.min.js"


def _d3_script_tag() -> str:
    if D3_LOCAL.exists():
        return "<script>" + D3_LOCAL.read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>"
    return f'<script src="{D3_CDN}"></script>'

STATUS_LABELS = {
    "VERIFIED_FACT": "확인된 사실", "SUPPORTED_INFERENCE": "근거 있는 추론", "HYPOTHESIS": "가설",
    "INSUFFICIENT_EVIDENCE": "근거 부족", "CONTRADICTED": "반박됨",
}
# 리서치 탭의 '파급 경로 흐름' 범례와 같은 색 체계를 쓴다
STATUS_STYLES = {
    "PREMISE": {"color": "#8b5cf6", "width": 1.6, "dash": "", "label": "이벤트 전제"},
    "VERIFIED_FACT": {"color": "#2563eb", "width": 2.4, "dash": "", "label": "확인된 사실"},
    "SUPPORTED_INFERENCE": {"color": "#0d9488", "width": 1.8, "dash": "", "label": "근거 있는 추론"},
    "HYPOTHESIS": {"color": "#f59e0b", "width": 1.4, "dash": "6,4", "label": "가설 (미검증)"},
    "INSUFFICIENT_EVIDENCE": {"color": "#9ca3af", "width": 1.1, "dash": "2,5", "label": "근거 부족 (미검증)"},
    "CONTRADICTED": {"color": "#dc2626", "width": 1.2, "dash": "6,4", "label": "반박됨"},
}
REL_LABELS = {
    "PRODUCES": "생산", "REQUIRES": "필요로 함", "SUPPLIES_TO": "공급", "OPERATES_IN": "속한 산업",
    "COMPETES_WITH": "경쟁", "PARTNERS_WITH": "협력", "INCREASES_DEMAND_FOR": "수요 증가",
    "DECREASES_DEMAND_FOR": "수요 감소", "SUBSTITUTES_FOR": "대체", "CONSTRAINS": "제약",
    "SOLVES": "해결", "EXPOSED_TO": "노출",
}
NODE_TYPE_LABELS = {
    "COMPANY": "기업", "EVENT": "이벤트", "PRODUCT": "제품", "INDUSTRY": "산업", "TECHNOLOGY": "기술",
    "RESOURCE": "원자재·자원", "BOTTLENECK": "병목",
}


def graph_payload(gs: GraphStore, include_weak: bool = True) -> dict:
    links, used = [], set()
    for e in gs.edges():
        weak = e.verification_status not in VERIFIED_STATUSES
        if not include_weak and weak:
            continue
        st = e.verification_status.value
        premise = e.edge_id.startswith("premise_") or (
            e.hop == 0 and not e.evidence and gs.g.nodes[e.source_id].get("node_type") == "EVENT")
        style = STATUS_STYLES["PREMISE" if premise else st]
        links.append({
            "id": e.edge_id, "source": e.source_id, "target": e.target_id, "rel": e.relationship.value,
            "relKo": REL_LABELS.get(e.relationship.value, e.relationship.value),
            "status": st, "weak": weak and not premise, "premise": premise,
            "color": style["color"], "width": style["width"], "dash": style["dash"],
            "evidence": len(e.evidence), "reasoning": e.reasoning[:300],
        })
        used |= {e.source_id, e.target_id}
    nodes = []
    for nid in sorted(used):
        d = gs.g.nodes[nid]
        nodes.append({"id": nid, "name": d["name"], "type": d["node_type"], "code": d.get("stock_code") or "",
                      "color": NODE_COLORS.get(d["node_type"], "#999")})
    return {"nodes": nodes, "links": links}


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def to_d3_html(gs: GraphStore, height: int = 680) -> str:
    """노드 크기는 모두 같고, 기본으로는 기업 이름만 보인다.
    노드를 클릭하면 연결된 모든 노드 이름과 관계 이름(배지)이 나타나고 나머지는 흐려진다.
    미검증 관계 표시와 관계 이름 표시 방식은 그래프 위 툴바에서 바꾼다."""
    payload = graph_payload(gs, include_weak=True)
    legend = {
        "nodes": {k: {"color": c, "label": NODE_TYPE_LABELS.get(k, k)} for k, c in NODE_COLORS.items()},
        "edges": STATUS_STYLES,
    }
    return (_TEMPLATE.replace("__HEIGHT__", str(height))
            .replace("__D3__", _d3_script_tag())
            .replace("__DATA__", _json(payload))
            .replace("__LEGEND__", _json(legend)))


_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:transparent}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;background:#fcfcfd}
  svg#g{width:100%;height:100%;cursor:grab}
  .node{stroke:#fff;stroke-width:1.5;cursor:pointer;transition:opacity .2s}
  .node.sel{stroke:#111827;stroke-width:2.5}
  .nlabel{font-size:11.5px;fill:#111827;paint-order:stroke;stroke:#fcfcfd;stroke-width:3.5px;stroke-linejoin:round;pointer-events:none;transition:opacity .2s}
  .nlabel.company{font-weight:600}
  .nlabel.concept{fill:#374151;font-weight:500}
  .link{fill:none;transition:opacity .2s}
  .elabel{pointer-events:none;transition:opacity .2s}
  .elabel text{font-size:10px;font-weight:600}
  .faded{opacity:.08}
  .hidden{display:none}
  #bar{position:absolute;left:10px;right:10px;top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;pointer-events:none}
  #bar > *{pointer-events:auto}
  .chip{background:rgba(255,255,255,.95);border:1px solid #e5e7eb;border-radius:999px;padding:4px 10px;font-size:11.5px;color:#374151;display:flex;align-items:center;gap:6px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .chip input[type=search]{border:none;outline:none;font-size:11.5px;width:130px;background:transparent}
  .seg{display:flex;border:1px solid #e5e7eb;border-radius:999px;overflow:hidden;background:#fff}
  .seg button{border:none;background:transparent;padding:4px 10px;font-size:11.5px;color:#4b5563;cursor:pointer}
  .seg button.on{background:#111827;color:#fff}
  .sw{display:inline-block;width:9px;height:9px;border-radius:50%}
  #legend{position:absolute;left:10px;bottom:10px;background:rgba(255,255,255,.95);border:1px solid #e5e7eb;border-radius:10px;padding:8px 10px;font-size:11px;color:#374151;line-height:1.7;max-width:250px}
  #legend summary{cursor:pointer;font-weight:600;font-size:11px;color:#374151}
  #legend .row{display:flex;flex-wrap:wrap;gap:2px 10px}
  #legend .it{display:flex;align-items:center;gap:4px}
  #legend svg{width:24px;height:8px}
  #legend .t{font-weight:600;color:#6b7280;font-size:10px;margin-top:2px}
  #panel{position:absolute;right:10px;top:52px;width:260px;max-height:calc(100% - 70px);overflow:auto;background:rgba(255,255,255,.97);border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;font-size:12px;color:#111827;display:none;box-shadow:0 4px 14px rgba(0,0,0,.06)}
  #panel h4{margin:0 0 2px;font-size:13.5px}
  #panel .sub{color:#6b7280;font-size:11px;margin-bottom:8px}
  #panel .rel{display:flex;gap:6px;align-items:flex-start;padding:5px 0;border-top:1px solid #f3f4f6;cursor:pointer}
  #panel .rel:hover{background:#f9fafb}
  #panel .badge{flex:none;font-size:10px;font-weight:600;border-radius:999px;padding:1px 7px;border:1px solid;white-space:nowrap}
  #panel .tag{font-size:9.5px;color:#92400e;background:#fef3c7;border-radius:4px;padding:0 4px;margin-left:3px}
  #panel .x{float:right;cursor:pointer;color:#9ca3af;border:none;background:none;font-size:14px}
  #tip{position:absolute;pointer-events:none;background:#111827;color:#f9fafb;font-size:11px;padding:6px 8px;border-radius:6px;max-width:320px;opacity:0;transition:opacity .15s;line-height:1.5}
  #hint{position:absolute;right:12px;bottom:8px;font-size:10px;color:#9ca3af}
</style></head><body>
<div id="wrap"><svg id="g"></svg>
  <div id="bar">
    <label class="chip"><input type="checkbox" id="weak" checked> 미검증 관계 <span id="weakN" style="color:#9ca3af"></span></label>
    <div class="chip" style="padding:3px 4px 3px 10px">관계 이름
      <div class="seg" id="mode"><button data-m="select" class="on">선택 시</button><button data-m="always">항상</button><button data-m="none">숨김</button></div></div>
    <label class="chip">🔍 <input type="search" id="q" placeholder="노드 검색 (Enter)"></label>
  </div>
  <details id="legend" open><summary>범례</summary><div id="lgbody"></div></details><div id="panel"></div><div id="tip"></div>
  <div id="hint">클릭: 연결 보기 · 빈 곳 클릭: 해제 · 드래그 · 휠 확대</div>
</div>
__D3__
<script>
const data = __DATA__;
const legend = __LEGEND__;
const height = __HEIGHT__;
const R = 8;   // 모든 노드 동일 크기
const wrap = document.getElementById("wrap"), tip = document.getElementById("tip"), panel = document.getElementById("panel");
const curWidth = () => wrap.clientWidth || wrap.getBoundingClientRect().width || 900;
const esc = s => String(s ?? "").replace(/[&<>'"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[c]);
const SHORT = {HYPOTHESIS: "가설", INSUFFICIENT_EVIDENCE: "근거 부족", CONTRADICTED: "반박됨"};
const styleOf = l => legend.edges[l.premise ? "PREMISE" : l.status];

document.getElementById("lgbody").innerHTML =
  "<div class='t'>노드</div><div class='row'>" + Object.values(legend.nodes).map(n => `<span class="it"><span class="sw" style="background:${n.color}"></span>${n.label}</span>`).join("") + "</div>"
  + "<div class='t'>관계 (선)</div>" + Object.values(legend.edges).map(s => `<div class="it"><svg><line x1="0" y1="4" x2="24" y2="4" stroke="${s.color}" stroke-width="${Math.max(s.width, 1.5)}" stroke-dasharray="${s.dash}"/></svg>${s.label}</div>`).join("");

if (typeof d3 === "undefined") {
  wrap.innerHTML = "<div style='padding:20px;color:#b91c1c'>d3.js를 불러오지 못했습니다.</div>";
} else if (!data.nodes.length) {
  wrap.innerHTML = "<div style='padding:20px;color:#6b7280'>표시할 관계가 없습니다.</div>";
} else {
  const allNodes = data.nodes.map(d => ({...d}));
  const byId = new Map(allNodes.map(n => [n.id, n]));
  const allLinks = data.links.map(d => ({...d}));
  document.getElementById("weakN").textContent = `(${allLinks.filter(l => l.weak).length})`;
  let showWeak = true, mode = "select", selected = null, nodes = [], links = [];

  const svg = d3.select("#g");
  const defs = svg.append("defs");
  Object.entries(legend.edges).forEach(([k, s]) => {
    defs.append("marker").attr("id", "arrow-" + k).attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0)
      .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 8).attr("markerHeight", 8).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", s.color);
  });
  const root = svg.append("g");
  const zoom = d3.zoom().scaleExtent([0.2, 4]).on("zoom", e => root.attr("transform", e.transform));
  svg.call(zoom).on("dblclick.zoom", null);
  const gLink = root.append("g"), gELabel = root.append("g"), gNode = root.append("g"), gNLabel = root.append("g");

  const simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(d => d.id).distance(l => l.premise ? 70 : 105))
      .force("charge", d3.forceManyBody().strength(-240))
      .force("collide", d3.forceCollide().radius(R + 16))
      .force("x", d3.forceX())
      .force("y", d3.forceY());

  let lastW = 0;
  function fit() {
    const w = curWidth();
    if (w && w !== lastW) { lastW = w; svg.attr("viewBox", [-w / 2, -height / 2, w, height]); simulation.alpha(Math.max(simulation.alpha(), 0.3)).restart(); }
  }
  fit();
  if (window.ResizeObserver) new ResizeObserver(fit).observe(wrap);
  window.addEventListener("resize", fit);
  document.addEventListener("visibilitychange", fit);

  let link, eLabel, node, nLabel;
  function build() {
    links = allLinks.filter(l => showWeak || !l.weak).map(l => ({...l, source: l.source.id || l.source, target: l.target.id || l.target}));
    const used = new Set(links.flatMap(l => [l.source, l.target]));
    nodes = allNodes.filter(n => used.has(n.id));
    if (selected && !used.has(selected.id)) selected = null;
    const pairCount = {};
    links.forEach(l => { const k = [l.source, l.target].sort().join("|"); pairCount[k] = (pairCount[k] || 0) + 1; l.pairIndex = pairCount[k]; });
    links.forEach(l => { l.pairTotal = pairCount[[l.source, l.target].sort().join("|")]; });

    link = gLink.selectAll("path").data(links, d => d.id).join("path").attr("class", "link")
      .attr("stroke", d => d.color).attr("stroke-width", d => d.width).attr("stroke-dasharray", d => d.dash)
      .attr("stroke-opacity", d => d.premise ? 0.55 : 0.8).attr("marker-end", d => `url(#arrow-${d.premise ? "PREMISE" : d.status})`)
      .on("mouseover", (e, d) => showTip(e, edgeTip(d))).on("mousemove", moveTip).on("mouseout", hideTip);

    eLabel = gELabel.selectAll("g.elabel").data(links, d => d.id).join(enter => {
      const g = enter.append("g").attr("class", "elabel");
      g.append("rect").attr("rx", 8).attr("ry", 8);
      g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em");
      return g;
    });
    eLabel.each(function (d) {
      const s = styleOf(d), g = d3.select(this);
      const txt = d.weak ? `? ${d.relKo}` : d.relKo;
      const t = g.select("text").text(txt).attr("fill", d.weak ? "#92400e" : s.color);
      if (d.status === "CONTRADICTED") t.attr("fill", "#b91c1c");
      const w = txt.length * 10.5 + 12;
      g.select("rect").attr("x", -w / 2).attr("y", -9).attr("width", w).attr("height", 18)
        .attr("fill", d.weak ? "#fffbeb" : "#ffffff").attr("stroke", s.color).attr("stroke-width", 1)
        .attr("stroke-dasharray", d.weak ? "3,2" : "");
    });

    node = gNode.selectAll("circle").data(nodes, d => d.id).join("circle").attr("class", "node")
      .attr("r", R).attr("fill", d => d.color)
      .call(drag())
      .on("click", (e, d) => { e.stopPropagation(); select(selected && selected.id === d.id ? null : d); })
      .on("mouseover", (e, d) => showTip(e, `<b>${esc(d.name)}</b>${d.code ? " (" + esc(d.code) + ")" : ""}<br>${esc(legend.nodes[d.type]?.label || d.type)}`))
      .on("mousemove", moveTip).on("mouseout", hideTip);

    nLabel = gNLabel.selectAll("text").data(nodes, d => d.id).join("text")
      .attr("class", d => "nlabel " + (d.type === "COMPANY" ? "company" : "concept"))
      .attr("dx", R + 4).attr("dy", "0.35em").text(d => d.name);

    simulation.nodes(nodes);
    simulation.force("link").links(links);
    simulation.alpha(0.6).restart();
    render();
  }

  // 선택 상태와 표시 모드에 따라 이름/배지/흐림을 갱신
  function render() {
    const near = new Set(), inc = new Set();
    if (selected) {
      near.add(selected.id);
      links.forEach(l => { if (l.source.id === selected.id || l.target.id === selected.id) { inc.add(l.id); near.add(l.source.id); near.add(l.target.id); } });
    }
    node.classed("faded", n => selected && !near.has(n.id)).classed("sel", n => selected && n.id === selected.id);
    link.classed("faded", l => selected && !inc.has(l.id));
    nLabel.classed("hidden", n => selected ? !near.has(n.id) && n.type !== "COMPANY" : n.type !== "COMPANY")
          .classed("faded", n => selected && !near.has(n.id));
    eLabel.classed("hidden", l => mode === "none" || (mode === "select" && !inc.has(l.id)))
          .classed("faded", l => mode === "always" && selected && !inc.has(l.id));
    placeLabels();
    renderPanel(inc);
  }

  function renderPanel(inc) {
    if (!selected) { panel.style.display = "none"; return; }
    const d = selected, rels = links.filter(l => inc.has(l.id));
    const rows = rels.map(l => {
      const out = l.source.id === d.id, other = out ? l.target : l.source, s = styleOf(l);
      return `<div class="rel" data-id="${esc(other.id)}"><span class="badge" style="color:${s.color};border-color:${s.color}${l.weak ? ";border-style:dashed;background:#fffbeb" : ""}">${out ? "→" : "←"} ${esc(l.relKo)}</span>`
        + `<span><b>${esc(other.name)}</b>${l.weak ? `<span class="tag">${esc(SHORT[l.status] || "")}</span>` : ""}<br><span style="color:#9ca3af;font-size:10.5px">${esc(legend.nodes[other.type]?.label || other.type)} · 근거 ${l.evidence}건</span></span></div>`;
    }).join("");
    panel.innerHTML = `<button class="x" title="닫기">✕</button><h4>${esc(d.name)}${d.code ? ` <span style="color:#9ca3af;font-weight:400">${esc(d.code)}</span>` : ""}</h4>`
      + `<div class="sub">${esc(legend.nodes[d.type]?.label || d.type)} · 연결 ${rels.length}개 (검증 ${rels.filter(l => !l.weak).length} · 미검증 ${rels.filter(l => l.weak).length})</div>` + rows;
    panel.style.display = "block";
    panel.querySelector(".x").onclick = () => select(null);
    panel.querySelectorAll(".rel").forEach(el => el.onclick = () => select(byId.get(el.dataset.id)));
  }

  function select(d) { selected = d; render(); }
  svg.on("click", () => select(null));

  document.getElementById("weak").addEventListener("change", e => { showWeak = e.target.checked; build(); });
  document.querySelectorAll("#mode button").forEach(b => b.addEventListener("click", () => {
    mode = b.dataset.m; document.querySelectorAll("#mode button").forEach(x => x.classList.toggle("on", x === b)); render();
  }));
  document.getElementById("q").addEventListener("keydown", e => {
    if (e.key !== "Enter") return;
    const q = e.target.value.trim().toLowerCase(); if (!q) return;
    const hit = nodes.find(n => n.name.toLowerCase().includes(q) || n.code === q);
    if (!hit) return;
    select(hit);
    const w = curWidth(), k = 1.4;
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity.translate(-hit.x * k, -hit.y * k).scale(k));
  });

  // 배지 위치: 선택된 노드의 관계는 상대 노드 쪽(62%)으로 밀어 중심 노드 주변이 붐비지 않게 한다
  function placeLabels() {
    eLabel.attr("transform", d => {
      const sid = selected && selected.id;
      const t = d.source.id === sid ? 0.62 : d.target.id === sid ? 0.38 : 0.5;
      const off = (d.pairIndex - (d.pairTotal + 1) / 2) * 20;   // 같은 쌍의 여러 관계는 위아래로 나눔
      return `translate(${d.source.x + (d.target.x - d.source.x) * t},${d.source.y + (d.target.y - d.source.y) * t + off})`;
    });
  }
  function linkPath(d) {
    const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
    const dx = tx - sx, dy = ty - sy, dist = Math.hypot(dx, dy) || 1, r = R + 3;
    const ex = tx - dx / dist * r, ey = ty - dy / dist * r;
    if (d.pairTotal === 1) return `M${sx},${sy}L${ex},${ey}`;
    const curve = dist * (1.6 - 0.25 * d.pairIndex);
    return `M${sx},${sy}A${curve},${curve} 0 0,1 ${ex},${ey}`;
  }
  simulation.on("tick", () => {
    link.attr("d", linkPath);
    placeLabels();
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    nLabel.attr("x", d => d.x).attr("y", d => d.y);
  });

  function drag() {
    return d3.drag()
      .on("start", e => { if (!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; })
      .on("drag", e => { e.subject.fx = e.x; e.subject.fy = e.y; })
      .on("end", e => { if (!e.active) simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; });
  }
  function edgeTip(d) {
    const s = styleOf(d);
    return `<b>${esc(d.source.name)}</b> → <b>${esc(d.target.name)}</b><br>${esc(d.relKo)} <span style="color:#9ca3af">(${esc(d.rel)})</span> · <span style="color:${s.color}">${esc(s.label)}</span><br>근거 ${d.evidence}건${d.reasoning ? "<br>" + esc(d.reasoning) : ""}`;
  }
  function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1; moveTip(e); }
  function moveTip(e) {
    const b = wrap.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - b.left + 12, b.width - 330) + "px";
    tip.style.top = (e.clientY - b.top + 12) + "px";
  }
  function hideTip() { tip.style.opacity = 0; }
  build();
}
</script></body></html>"""

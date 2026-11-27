"""Knowledge Graph 시각화: d3-force (Disjoint force-directed graph).

참고: https://observablehq.com/@d3/disjoint-force-directed-graph/2
- forceCenter 대신 forceX/forceY를 써서 서로 연결되지 않은 컴포넌트도 화면 안에 머문다.
- 드래그 시 alphaTarget(0.3).restart() → 놓으면 alphaTarget(0)으로 다시 안정화된다.
그래프 데이터(GraphStore)와 저장 형식은 바뀌지 않으며, 표시 방법만 바뀐다.
"""
from __future__ import annotations

import json

from src.graph.graph_store import GraphStore
from src.graph.traversal import EDGE_STYLES, NODE_COLORS
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


def graph_payload(gs: GraphStore, include_weak: bool = True) -> dict:
    links, used = [], set()
    for e in gs.edges():
        if not include_weak and e.verification_status not in VERIFIED_STATUSES:
            continue
        st = e.verification_status.value
        style = EDGE_STYLES[st]
        dash = style["dashes"]
        links.append({
            "id": e.edge_id, "source": e.source_id, "target": e.target_id, "rel": e.relationship.value,
            "status": st, "color": style["color"], "width": style["width"],
            "dash": ("" if dash is False else "6,4" if dash is True else ",".join(str(x) for x in dash)),
            "premise": e.hop == 0 and not e.evidence,
            "evidence": len(e.evidence), "reasoning": e.reasoning[:300],
        })
        used |= {e.source_id, e.target_id}
    nodes = []
    for nid in sorted(used):
        d = gs.g.nodes[nid]
        nodes.append({"id": nid, "name": d["name"], "type": d["node_type"], "code": d.get("stock_code") or "",
                      "color": NODE_COLORS.get(d["node_type"], "#999")})
    return {"nodes": nodes, "links": links}


def to_d3_html(gs: GraphStore, height: int = 660, include_weak: bool = True, show_edge_labels: bool = True) -> str:
    payload = graph_payload(gs, include_weak)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    legend = {"nodes": NODE_COLORS, "edges": {k: {**v, "label": STATUS_LABELS[k]} for k, v in EDGE_STYLES.items()}}
    legend_json = json.dumps(legend, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html,body{{margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:transparent}}
  #wrap{{position:relative;width:100%;height:{height}px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff}}
  svg{{width:100%;height:100%;cursor:grab}}
  .node-label{{font-size:12px;fill:#111827;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round;pointer-events:none}}
  .link-label{{font-size:9px;fill:#6b7280;paint-order:stroke;stroke:#fff;stroke-width:2.5px;pointer-events:none}}
  .faded{{opacity:.12}}
  #legend{{position:absolute;left:10px;top:10px;background:rgba(255,255,255,.92);border:1px solid #e5e7eb;border-radius:8px;padding:8px 10px;font-size:11px;color:#374151;line-height:1.6}}
  #legend .sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}}
  #legend svg{{width:26px;height:8px;vertical-align:0;margin-right:4px;cursor:default}}
  #tip{{position:absolute;pointer-events:none;background:#111827;color:#f9fafb;font-size:11px;padding:6px 8px;border-radius:6px;max-width:320px;opacity:0;transition:opacity .15s;line-height:1.5}}
  #hint{{position:absolute;right:10px;bottom:8px;font-size:10px;color:#9ca3af}}
</style></head><body>
<div id="wrap"><svg id="g"></svg><div id="legend"></div><div id="tip"></div>
<div id="hint">드래그: 노드 이동 · 휠: 확대/축소 · 클릭: 연결 강조 · 더블클릭: 고정 해제</div></div>
{_d3_script_tag()}
<script>
const data = {data};
const legend = {legend_json};
const showEdgeLabels = {str(show_edge_labels).lower()};
const wrap = document.getElementById("wrap");
const height = {height};
// 숨겨진 탭 안에서 로드되면 폭이 0으로 측정되므로, 보이는 순간의 실제 폭으로 viewBox를 다시 맞춘다
const curWidth = () => wrap.clientWidth || wrap.getBoundingClientRect().width || 900;
const tip = document.getElementById("tip");
const esc = s => String(s ?? "").replace(/[&<>'"]/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"}})[c]);

const lg = document.getElementById("legend");
lg.innerHTML = Object.entries(legend.nodes).map(([k,c]) => `<div><span class="sw" style="background:${{c}}"></span>${{k}}</div>`).join("")
  + "<hr style='border:none;border-top:1px solid #e5e7eb;margin:4px 0'>"
  + Object.entries(legend.edges).map(([k,s]) => {{
      const dash = s.dashes === false ? "" : (s.dashes === true ? "6,4" : s.dashes.join(","));
      return `<div><svg><line x1="0" y1="4" x2="26" y2="4" stroke="${{s.color}}" stroke-width="${{s.width}}" stroke-dasharray="${{dash}}"/></svg>${{s.label}}</div>`;
    }}).join("");

if (typeof d3 === "undefined") {{
  wrap.innerHTML = "<div style='padding:20px;color:#b91c1c'>d3.js를 불러오지 못했습니다 (인터넷 연결 확인: cdn.jsdelivr.net)</div>";
}} else if (!data.nodes.length) {{
  wrap.innerHTML = "<div style='padding:20px;color:#6b7280'>표시할 관계가 없습니다.</div>";
}} else {{
  const nodes = data.nodes.map(d => ({{...d}}));
  const links = data.links.map(d => ({{...d}}));
  const radius = d => d.type === "COMPANY" ? 11 : d.type === "EVENT" ? 10 : 7;

  // 같은 두 노드 사이의 여러 관계는 서로 다른 곡률로 그린다
  const pairCount = {{}};
  links.forEach(l => {{
    const k = [l.source, l.target].sort().join("|");
    pairCount[k] = (pairCount[k] || 0) + 1;
    l.pairIndex = pairCount[k];
  }});
  links.forEach(l => {{ l.pairTotal = pairCount[[l.source, l.target].sort().join("|")]; }});

  const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(l => l.premise ? 70 : 110))
      .force("charge", d3.forceManyBody().strength(-260))
      .force("collide", d3.forceCollide().radius(d => radius(d) + 14))
      .force("x", d3.forceX())
      .force("y", d3.forceY());

  const svg = d3.select("#g");
  let lastW = 0;
  function fit() {{
    const w = curWidth();
    if (w && w !== lastW) {{
      lastW = w;
      svg.attr("viewBox", [-w / 2, -height / 2, w, height]);
      simulation.alpha(Math.max(simulation.alpha(), 0.3)).restart();
    }}
  }}
  fit();
  if (window.ResizeObserver) new ResizeObserver(fit).observe(wrap);
  window.addEventListener("resize", fit);
  document.addEventListener("visibilitychange", fit);
  const defs = svg.append("defs");
  const statuses = [...new Set(links.map(l => l.status))];
  statuses.forEach(s => {{
    const c = links.find(l => l.status === s).color;
    defs.append("marker").attr("id", "arrow-" + s).attr("viewBox", "0 -5 10 10")
      .attr("refX", 10).attr("refY", 0).attr("markerUnits", "userSpaceOnUse")
      .attr("markerWidth", 9).attr("markerHeight", 9).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", c);
  }});

  const root = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.2, 4]).on("zoom", e => root.attr("transform", e.transform)));

  const link = root.append("g").attr("fill", "none").selectAll("path").data(links).join("path")
      .attr("stroke", d => d.color).attr("stroke-opacity", d => d.premise ? 0.35 : 0.75)
      .attr("stroke-width", d => d.width).attr("stroke-dasharray", d => d.dash)
      .attr("marker-end", d => `url(#arrow-${{d.status}})`)
      .on("mouseover", (e, d) => showTip(e, `<b>${{d.rel}}</b> · ${{legend.edges[d.status].label}}<br>근거 ${{d.evidence}}개<br>${{esc(d.reasoning)}}`))
      .on("mousemove", moveTip).on("mouseout", hideTip);

  const linkLabel = root.append("g").selectAll("text").data(showEdgeLabels ? links : []).join("text")
      .attr("class", "link-label").attr("text-anchor", "middle").text(d => d.rel);

  const node = root.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", radius).attr("fill", d => d.color).attr("stroke", "#fff").attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .call(drag(simulation))
      .on("click", (e, d) => {{ e.stopPropagation(); focus(d); }})
      .on("dblclick", (e, d) => {{ d.fx = null; d.fy = null; simulation.alpha(0.3).restart(); }})
      .on("mouseover", (e, d) => showTip(e, `<b>${{esc(d.name)}}</b>${{d.code ? " (" + esc(d.code) + ")" : ""}}<br>${{esc(d.type)}}`))
      .on("mousemove", moveTip).on("mouseout", hideTip);

  const label = root.append("g").selectAll("text").data(nodes).join("text")
      .attr("class", "node-label").attr("dx", d => radius(d) + 4).attr("dy", "0.35em")
      .style("font-weight", d => d.type === "COMPANY" ? 600 : 400)
      .text(d => d.code ? `${{d.name}} (${{d.code}})` : d.name);

  svg.on("click", () => focus(null));

  function linkPath(d) {{
    const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
    const dx = tx - sx, dy = ty - sy, dist = Math.hypot(dx, dy) || 1;
    const r = radius(d.target) + 3;
    const ex = tx - dx / dist * r, ey = ty - dy / dist * r;   // 화살표가 노드 가장자리에서 멈추도록
    if (d.pairTotal === 1) return `M${{sx}},${{sy}}L${{ex}},${{ey}}`;
    const curve = dist * (1.6 - 0.25 * d.pairIndex);
    return `M${{sx}},${{sy}}A${{curve}},${{curve}} 0 0,1 ${{ex}},${{ey}}`;
  }}

  simulation.on("tick", () => {{
    link.attr("d", linkPath);
    linkLabel.attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2 - 3);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);
  }});

  function drag(simulation) {{
    function dragstarted(event) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }}
    function dragged(event) {{
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }}
    function dragended(event) {{
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }}
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
  }}

  function focus(d) {{
    if (!d) {{ [node, label, link, linkLabel].forEach(s => s.classed("faded", false)); return; }}
    const near = new Set([d.id]);
    links.forEach(l => {{ if (l.source.id === d.id) near.add(l.target.id); if (l.target.id === d.id) near.add(l.source.id); }});
    node.classed("faded", n => !near.has(n.id));
    label.classed("faded", n => !near.has(n.id));
    link.classed("faded", l => l.source.id !== d.id && l.target.id !== d.id);
    linkLabel.classed("faded", l => l.source.id !== d.id && l.target.id !== d.id);
  }}
  function showTip(e, html) {{ tip.innerHTML = html; tip.style.opacity = 1; moveTip(e); }}
  function moveTip(e) {{
    const b = wrap.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX - b.left + 12, b.width - 330) + "px";
    tip.style.top = (e.clientY - b.top + 12) + "px";
  }}
  function hideTip() {{ tip.style.opacity = 0; }}
}}
</script></body></html>"""

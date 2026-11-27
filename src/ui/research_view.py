"""Research 탭 결과 시각화 (이벤트 요약 · 파급 단계 · 흐름도 · 기업별 영향)."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from src.graph.traversal import EDGE_STYLES, NODE_COLORS

EVENT_LABELS = {
    "DEMAND_INCREASE": "수요 증가", "DEMAND_DECREASE": "수요 감소", "SUPPLY_DISRUPTION": "공급망 장애",
    "CAPACITY_EXPANSION": "생산능력 확대", "TECHNOLOGY_BREAKTHROUGH": "기술 혁신", "REGULATION_CHANGE": "규제 변화",
    "PRICE_INCREASE": "가격 상승", "PRICE_DECREASE": "가격 하락", "CONTRACT_SIGNED": "계약 체결", "UNKNOWN": "미분류",
}
IMPACT = {  # 라벨, 글자색, 배경색
    "POSITIVE": ("긍정적", "#166534", "#dcfce7"), "NEGATIVE": ("부정적", "#991b1b", "#fee2e2"),
    "NEUTRAL": ("중립", "#374151", "#f3f4f6"), "UNCERTAIN": ("불확실", "#92400e", "#fef3c7"),
}
STATUS = {
    "VERIFIED_FACT": ("확인된 사실", "#1e3a8a", "#dbeafe"), "SUPPORTED_INFERENCE": ("근거 있는 추론", "#155e75", "#cffafe"),
    "HYPOTHESIS": ("가설", "#6b7280", "#f3f4f6"), "INSUFFICIENT_EVIDENCE": ("근거 부족", "#9ca3af", "#f9fafb"),
    "CONTRADICTED": ("반박됨", "#991b1b", "#fee2e2"), "PREMISE": ("이벤트 직접", "#5b21b6", "#ede9fe"),
}

CSS = """<style>
.mr-badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.mr-tier-item{padding:6px 0;border-bottom:1px solid rgba(128,128,128,.15)}
.mr-tier-item:last-child{border-bottom:none}
.mr-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0}
.mr-effect{font-size:13px;opacity:.8;margin-left:15px}
.mr-table{width:100%;border-collapse:collapse;font-size:14px}
.mr-table th{text-align:left;font-weight:600;padding:8px;border-bottom:2px solid rgba(128,128,128,.3)}
.mr-table td{padding:8px;border-bottom:1px solid rgba(128,128,128,.15);vertical-align:top}
.mr-cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:6px 0 14px}
.mr-card{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 12px}
.mr-card-k{font-size:12px;opacity:.65}
.mr-card-v{font-size:18px;font-weight:600;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mr-chip{display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;border-radius:6px;background:rgba(128,128,128,.12);font-size:12px}
</style>"""


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def badge(text: str, fg: str, bg: str) -> str:
    return f'<span class="mr-badge" style="color:{fg};background:{bg}">{_e(text)}</span>'


def impact_badge(impact: str | None, status: str | None) -> str:
    label, fg, bg = IMPACT.get(impact or "UNCERTAIN", IMPACT["UNCERTAIN"])
    if impact in ("POSITIVE", "NEGATIVE") and status != "VERIFIED_FACT":
        label += " 가능성"   # 사실로 확인된 경로가 아니면 가능성으로 표시
    return badge(label, fg, bg)


def status_badge(status: str | None) -> str:
    label, fg, bg = STATUS.get(status or "", (status or "-", "#374151", "#f3f4f6"))
    return badge(label, fg, bg)


def _fallback_tiers(res: dict) -> list[dict]:
    """이전 버전 결과(tiers 없음)는 기업별 hop으로 단계를 만든다."""
    out: dict[int, list] = {}
    for r in res.get("company_results", []):
        t = r.get("tier", r.get("hops", 0))
        out.setdefault(t, []).append({"name": r["corp_name"], "node_type": "COMPANY", "effect": r.get("impact_factor", "")})
    return [{"tier": t, "title": "직접 영향" if t == 0 else f"{t}차 파급효과", "items": v} for t, v in sorted(out.items())]


def render_header(res: dict) -> None:
    ev = res.get("event", {})
    rows = res.get("company_results", [])
    verified = sum(1 for r in rows if r.get("verification_status") in ("VERIFIED_FACT", "SUPPORTED_INFERENCE"))
    budget = res.get("budget", {})
    cards = [("이벤트 유형", EVENT_LABELS.get(ev.get("event_type"), ev.get("event_type", "-"))),
             ("시나리오", "가상 시나리오" if ev.get("is_hypothetical") else "실제 뉴스"),
             ("연결된 기업", f"{len(rows)}개"),
             ("근거 확인 경로", f"{verified}개"),
             ("분석", f"{res.get('elapsed_ms', 0) / 1000:.0f}초 · LLM {budget.get('llm', {}).get('calls', '-')}회")]
    st.markdown('<div class="mr-cards">' + "".join(
        f'<div class="mr-card"><div class="mr-card-k">{_e(k)}</div><div class="mr-card-v">{_e(v)}</div></div>'
        for k, v in cards) + "</div>", unsafe_allow_html=True)
    chips = [f'<span class="mr-chip">제품 · {_e(p)}</span>' for p in ev.get("products", [])]
    chips += [f'<span class="mr-chip">산업 · {_e(i)}</span>' for i in ev.get("industries", [])]
    chips += [f'<span class="mr-chip">기업 · {_e(n)}</span>' for n in ev.get("entity_names", [])]
    st.markdown(f"**{_e(ev.get('summary', ''))}**<br>{''.join(chips)}", unsafe_allow_html=True)
    for w in res.get("event_warnings", []):
        st.caption(f"⚠ {w}")
    if res.get("stopped_reason") and res["stopped_reason"] != "탐색 완료":
        st.warning(res["stopped_reason"])


def render_summary(res: dict) -> None:
    with st.container(border=True):
        st.markdown("##### 분석 요약")
        st.write(res.get("summary") or "-")
        if res.get("caveat"):
            st.caption("ⓘ " + res["caveat"])


def render_tiers(res: dict) -> None:
    tiers = res.get("tiers") or _fallback_tiers(res)
    tiers = [t for t in tiers if t.get("items")]
    if not tiers:
        return
    st.markdown("##### 파급효과 단계")
    for start in range(0, len(tiers), 4):
        chunk = tiers[start:start + 4]
        cols = st.columns(len(chunk))
        for col, t in zip(cols, chunk):
            with col.container(border=True):
                head = "직접 영향" if t["tier"] == 0 else f"{t['tier']}차 파급효과"
                sub = "" if t["tier"] == 0 or t["title"] == head else f" — {_e(t['title'])}"
                st.markdown(f"**{head}**{sub}", unsafe_allow_html=True)
                items = []
                for i in t["items"]:
                    color = NODE_COLORS.get(i.get("node_type", ""), "#9ca3af")
                    eff = f'<div class="mr-effect">{_e(i["effect"])}</div>' if i.get("effect") else ""
                    items.append(f'<div class="mr-tier-item"><span class="mr-dot" style="background:{color}"></span>'
                                 f'{_e(i["name"])}{eff}</div>')
                st.markdown("".join(items), unsafe_allow_html=True)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def render_flow(res: dict) -> None:
    """이벤트 → 직접 영향 → 1차 → 2차 ... 흐름 (기업별 최적 경로를 Sankey로)."""
    graph = res.get("graph") or {}
    nodes = {n["node_id"]: n for n in graph.get("nodes", [])}
    edges = {e["edge_id"]: e for e in graph.get("edges", [])}
    used: dict[tuple[str, str], dict] = {}
    for r in res.get("company_results", []):
        pn = r.get("path_nodes")
        for i, eid in enumerate(r.get("path_edge_ids", [])):
            e = edges.get(eid)
            if not e:
                continue
            # 탐색한 순서(이벤트 → 기업)로 흐름을 그린다. 역방향 관계도 흐름 방향으로 표시
            a, b = (pn[i], pn[i + 1]) if pn and len(pn) > i + 1 else (e["source_id"], e["target_id"])
            if (b, a) not in used:
                used[(a, b)] = e
    event_id = next((nid for nid, n in nodes.items() if n.get("node_type") == "EVENT"), None)
    # 이벤트가 직접 닿는 노드 (이전 결과는 premise 관계에서 복원)
    roots = res.get("root_ids") or [e["target_id"] for e in graph.get("edges", []) if e.get("source_id") == event_id]
    if event_id:
        used_nodes = {n for k in used for n in k}
        company_nodes = {r.get("node_id") for r in res.get("company_results", [])}
        for rid in roots:
            if rid in nodes and (rid in used_nodes or rid in company_nodes):
                used.setdefault((event_id, rid), {"verification_status": "PREMISE", "relationship": "이벤트"})
    if not used:
        return
    ids = sorted({n for k in used for n in k}, key=lambda n: (n != event_id, n))
    index = {n: i for i, n in enumerate(ids)}
    labels = [nodes[n]["name"] if n != event_id else "이벤트" for n in ids]
    colors = [NODE_COLORS.get(nodes[n].get("node_type"), "#9ca3af") for n in ids]
    link_colors, link_labels = [], []
    for e in used.values():
        st_ = e.get("verification_status", "HYPOTHESIS")
        base = EDGE_STYLES.get(st_, {"color": "#8b5cf6"})["color"]
        link_colors.append(_rgba(base, 0.45 if st_ != "PREMISE" else 0.25))
        link_labels.append(f"{e.get('relationship', '')} · {STATUS.get(st_, (st_,))[0]}")
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=18, thickness=14, line=dict(width=0)),
        link=dict(source=[index[s] for s, _ in used], target=[index[t] for _, t in used], value=[1] * len(used),
                  color=link_colors, customdata=link_labels, hovertemplate="%{customdata}<extra></extra>"),
    ))
    fig.update_layout(height=max(320, 28 * len(ids)), margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(size=13, color="#111827"))
    st.markdown("##### 파급 경로 흐름")
    st.plotly_chart(fig, width="stretch")
    st.caption("선 색: 검증 상태 (진한 파랑=확인된 사실, 회색=근거 있는 추론, 연회색=가설) · 선 위에 마우스를 올리면 관계가 보입니다.")


def _fin_cols(res: dict, code: str | None) -> tuple[str, str]:
    fin = enrich_financial(res.get("financial")) or {}
    c = (fin.get("companies") or {}).get(code or "", {})
    t = c.get("trailing") or {}
    w = c.get("windows") or {}
    after = (w.get(20) or w.get("20") or {}).get("cumulative_return")
    return _pct(t.get(20) or t.get("20")), _pct(after)


def render_company_table(res: dict) -> None:
    rows = sorted(res.get("company_results", []), key=lambda r: (r.get("tier", r.get("hops", 0)), r["corp_name"]))
    if not rows:
        st.info("근거로 연결된 기업이 없습니다.")
        return
    st.markdown("##### 기업별 영향 분석")
    body = []
    for r in rows:
        t = r.get("tier", r.get("hops", 0))
        body.append(
            "<tr>"
            f"<td><b>{_e(r['corp_name'])}</b><br><span style='opacity:.6;font-size:12px'>"
            f"{_e(r.get('stock_code') or ('비상장' if str(r.get('node_id', '')).startswith('company_unlisted_') else ''))}</span></td>"
            f"<td>{impact_badge(r.get('economic_impact'), r.get('verification_status'))}</td>"
            f"<td>{_e(r.get('role') or ('직접 영향' if t == 0 else '-'))}</td>"
            f"<td>{_e(r.get('impact_factor') or '-')}</td>"
            f"<td>{'직접' if t == 0 else f'{t}차'}</td>"
            f"<td>{status_badge(r.get('verification_status'))}</td>"
            f"<td>{_e(r.get('revenue_exposure') or 'UNKNOWN')}</td>"
            + (f"<td>{_e(_fin_cols(res, r.get('stock_code'))[0])}</td><td>{_e(_fin_cols(res, r.get('stock_code'))[1])}</td>"
               if res.get("financial") else "")
            + "</tr>")
    st.markdown('<table class="mr-table"><thead><tr><th>기업</th><th>영향 방향</th><th>관계</th><th>영향 요인</th>'
                '<th>단계</th><th>검증 상태</th><th>매출 노출도</th>'
                + ("<th>최근 20일</th><th>이후 20일</th>" if res.get("financial") else "")
                + '</tr></thead><tbody>'
                + "".join(body) + "</tbody></table>", unsafe_allow_html=True)


def render_company_details(res: dict) -> None:
    st.markdown("##### 근거 및 조건")
    for r in res.get("company_results", []):
        label = IMPACT.get(r.get("economic_impact") or "UNCERTAIN", IMPACT["UNCERTAIN"])[0]
        with st.expander(f"{r['corp_name']} ({r.get('stock_code') or '-'}) · {label} · "
                         f"{STATUS.get(r.get('verification_status'), ('-',))[0]}"):
            st.markdown(f"**경로**: {r.get('path', '')}")
            if r.get("technical_relevance"):
                st.markdown(f"**기술적 연관성**: {r['technical_relevance']}")
            if r.get("conditions"):
                st.markdown("**영향 발생 조건**\n" + "\n".join(f"- {c}" for c in r["conditions"]))
            if r.get("counter_scenario"):
                st.markdown(f"**반대 시나리오**: {r['counter_scenario']}")
            for ev in r.get("evidence", []):
                st.markdown(f"> {ev['quote']}\n\n— chunk {ev['chunk_id']} · {ev['published_at']} · [공시 원문]({ev['source_url']})")
            if not r.get("evidence"):
                st.caption("원문 인용 근거 없음 (가설 또는 이벤트 직접 연결)")


DIR_LABEL = {"POSITIVE": ("긍정 가능성", "#166534", "#dcfce7"), "NEGATIVE": ("부정 가능성", "#991b1b", "#fee2e2"),
             "UNCERTAIN": ("방향 불확실", "#92400e", "#fef3c7")}


def render_insights(res: dict) -> None:
    ins = res.get("insights") or []
    if not ins:
        return
    st.markdown("##### 가설 인사이트 <span class='mr-badge' style='color:#6b7280;background:#f3f4f6'>근거 미확인</span>",
                unsafe_allow_html=True)
    st.caption("공시 근거로 확인되지 않은 LLM 가설입니다. 아이디어 탐색용이며, '확인할 것'으로 검증이 필요합니다.")
    for start in range(0, len(ins), 2):
        cols = st.columns(2)
        for col, i in zip(cols, ins[start:start + 2]):
            with col.container(border=True):
                lab, fg, bg = DIR_LABEL.get(i.get("direction"), DIR_LABEL["UNCERTAIN"])
                st.markdown(f"{badge(lab, fg, bg)} <span class='mr-chip'>{_e(i.get('tier'))}차 · {_e(i.get('kind'))}</span>"
                            f"<br><b>{_e(i.get('title'))}</b>", unsafe_allow_html=True)
                st.write(i.get("rationale", ""))
                comps = []
                for c in i.get("companies", []):
                    if c.get("resolved"):
                        tag = " · 그래프에 있음" if c.get("in_graph") else ""
                        comps.append(f"<span class='mr-chip'>{_e(c['name'])} ({_e(c['stock_code'])}){tag}</span>")
                    else:
                        comps.append(f"<span class='mr-chip' style='opacity:.55'>{_e(c['name'])} (식별 안 됨)</span>")
                if comps:
                    st.markdown("관련 상장사: " + "".join(comps), unsafe_allow_html=True)
                if i.get("what_to_check"):
                    st.caption("확인할 것: " + i["what_to_check"])


def render_all_relations(res: dict) -> None:
    import pandas as pd
    graph = res.get("graph") or {}
    names = {n["node_id"]: n["name"] for n in graph.get("nodes", [])}
    rows = [{"출발": names.get(e["source_id"], e["source_id"]), "관계": e["relationship"],
             "도착": names.get(e["target_id"], e["target_id"]),
             "검증 상태": STATUS.get(e["verification_status"], (e["verification_status"],))[0],
             "근거 수": len(e.get("evidence", [])), "추론": (e.get("reasoning") or "")[:120]}
            for e in graph.get("edges", []) if not (e.get("hop") == 0 and not e.get("evidence"))]
    if not rows:
        return
    with st.expander(f"그래프의 모든 관계 보기 ({len(rows)}개) - 기업 경로에 포함되지 않은 산업·제품 관계 포함"):
        df = pd.DataFrame(rows)
        opts = sorted(df["검증 상태"].unique())
        pick = st.multiselect("검증 상태", opts, default=opts, key=f"rel_filter_{res.get('run_id')}")
        q = st.text_input("이름 검색", "", key=f"rel_q_{res.get('run_id')}")
        view = df[df["검증 상태"].isin(pick)]
        if q:
            view = view[view["출발"].str.contains(q) | view["도착"].str.contains(q)]
        st.dataframe(view, width="stretch", hide_index=True)


def render_market(res: dict, on_fetch=None) -> None:
    st.markdown("##### 주가 반응 (KRX)")
    fin = res.get("financial")
    if on_fetch is not None:
        c1, c2 = st.columns([2, 1])
        default = (fin or {}).get("event_date") or res.get("event", {}).get("event_date") or ""
        ev_date = c1.text_input("기준일 (비우면 오늘 · 최근 수익률만 표시)", default, key=f"fin_date_{res.get('run_id')}")
        if c2.button("주가 조회" if not fin else "다시 조회", key=f"fin_btn_{res.get('run_id')}"):
            on_fetch(ev_date or None)
            fin = res.get("financial")
    if fin:
        render_financial(fin, compact=True)
    else:
        st.caption("아직 주가를 조회하지 않았습니다. 결과 기업과 가설 인사이트의 상장사 주가를 KRX에서 가져옵니다.")


def render_result(res: dict, on_fetch_financial=None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    render_header(res)
    render_summary(res)
    render_tiers(res)
    render_flow(res)
    render_company_table(res)
    render_insights(res)
    render_market(res, on_fetch_financial)
    render_company_details(res)
    render_all_relations(res)
    st.caption("유사도·검증 상태는 투자 성공 확률이 아니며, 종합 투자 점수는 제공하지 않습니다.")


# ---------------------------------------------------------------- 금융
def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:+.1f}%"


def enrich_financial(fin: dict | None) -> dict | None:
    """이전 버전 결과처럼 최근 수익률이 없으면 저장된 가격(_prices)으로 계산해 채운다."""
    if not fin or fin.get("_enriched"):
        return fin
    import pandas as pd

    from src.financial.metrics import event_index, prepare_prices, trailing_returns
    for code, c in (fin.get("companies") or {}).items():
        rows = (fin.get("_prices") or {}).get(code)
        if c.get("trailing") or not rows:
            continue
        close = prepare_prices(pd.DataFrame(rows))[0]["close"]
        t0 = event_index(close.index, fin.get("event_date"))
        c["trailing"] = trailing_returns(close, (t0 - 1) if t0 else len(close) - 1)
    seed = fin.get("seed_code")
    corr = fin.get("correlation_pre_event") or {}
    if seed and seed in corr:
        for code, c in (fin.get("companies") or {}).items():
            v = corr[seed].get(code)
            if code != seed and v is not None and "corr_with_seed" not in c:
                c["corr_with_seed"] = v
    fin["_enriched"] = True
    return fin


def render_financial(fin: dict, compact: bool = False) -> None:
    """KRX(Korea Stock MCP) 주가 기반 분석 표시."""
    fin = enrich_financial(fin)
    import pandas as pd
    import plotly.express as px

    from src.financial.metrics import normalized_prices, prepare_prices

    status = fin.get("status", {})
    p = status.get("prices", {})
    src_ok = "✓" if p.get("ok") else "✗"
    bench = [f"{k.split('_')[-1]} {'✓' if v.get('ok') else '✗'}" for k, v in status.items()
             if k.startswith("benchmark_") and isinstance(v, dict)]
    st.caption(f"데이터: KRX 일별매매정보 (Korea Stock MCP) {src_ok} · 벤치마크 지수 {', '.join(bench) or '-'} · "
               f"기준일 {fin.get('event_date')} · 기간 {' ~ '.join(fin.get('period', []))}")
    if p.get("message"):
        st.warning(p["message"])
    if status.get("window_note"):
        st.info(status["window_note"])
    names = {code: c.get("corp_name", code) for code, c in fin.get("companies", {}).items()}
    prices = fin.get("_prices", {})
    if prices:
        dfs = {k: prepare_prices(pd.DataFrame(v))[0] for k, v in prices.items()}
        norm = normalized_prices(dfs, fin["event_date"]).rename(columns=lambda c: f"{names.get(c, c)}")
        if not norm.empty:
            fig = px.line(norm, labels={"value": "정규화 주가 (=100)", "date": "", "variable": "기업"})
            fig.update_layout(height=320 if compact else 420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
            st.plotly_chart(fig, width="stretch")
    rows = []
    for code, c in fin.get("companies", {}).items():
        t = c.get("trailing") or {}
        w = c.get("windows") or {}
        def win(k, key):
            return (w.get(k) or w.get(str(k)) or {}).get(key)
        rows.append({"기업": f"{c.get('corp_name')} ({code})", "데이터": "있음" if c.get("available") else "없음",
                     "최근 5일": _pct(t.get(5) or t.get("5")), "최근 20일": _pct(t.get(20) or t.get("20")),
                     "최근 60일": _pct(t.get(60) or t.get("60")),
                     "Seed와 상관계수": "-" if c.get("corr_with_seed") is None else f"{c['corr_with_seed']:.2f}",
                     "이후 5일": _pct(win(5, "cumulative_return")), "이후 20일": _pct(win(20, "cumulative_return")),
                     "이후 20일 초과수익": _pct(win(20, "excess_return")), "CAR 20일": _pct(win(20, "car_market_adjusted"))})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if not compact:
        detail = []
        for code, c in fin.get("companies", {}).items():
            for k, w in (c.get("windows") or {}).items():
                detail.append({"기업": f"{c.get('corp_name')} ({code})", "윈도우(거래일)": k,
                               "누적수익률": _pct(w.get("cumulative_return")), "시장수익률": _pct(w.get("market_return")),
                               "초과수익률": _pct(w.get("excess_return")), "CAR(시장조정)": _pct(w.get("car_market_adjusted")),
                               "CAR(시장모형)": _pct(w.get("car_market_model")), "MDD": _pct(w.get("mdd")),
                               "거래량 변화": _pct(w.get("volume_change")), "비고": w.get("note", "")})
        if detail:
            st.markdown("##### 이벤트 이후 구간 지표")
            st.dataframe(pd.DataFrame(detail), width="stretch", hide_index=True)
        if fin.get("correlation_pre_event"):
            st.markdown("##### 기준일 이전 일별수익률 상관계수")
            corr = pd.DataFrame(fin["correlation_pre_event"]).rename(index=names, columns=names)
            st.dataframe(corr, width="stretch")
    st.caption(fin.get("causality_note", "") + " KRX 일별 종가는 수정주가가 아닙니다.")

"""Market Ripple - Streamlit Dashboard.

uv run streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from config.settings import Settings, setup_logging
from src.graph.graph_store import GraphStore
from src.graph.d3_view import to_d3_html
from src.storage import Storage

st.set_page_config(page_title="Market Ripple", layout="wide")
settings = Settings.from_env()
settings.ensure_dirs()
setup_logging(settings)


@st.cache_resource(show_spinner=False)
def get_app(seed_code: str):
    from src.pipeline import build_app
    settings.seed_stock_code = seed_code
    return build_app(settings, need_dart=bool(settings.dart_api_key))


def storage() -> Storage:
    if "storage" not in st.session_state:
        st.session_state.storage = Storage(settings.db_path)
    return st.session_state.storage


def log_box():
    box = st.empty()
    lines: list[str] = []

    def cb(msg: str):
        lines.append(msg)
        box.code("\n".join(lines[-40:]), language=None)
    return cb


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("Market Ripple")
    seed_code = st.text_input("Seed 종목코드", settings.seed_stock_code)
    missing = [k for k in ("openai_api_key", "dart_api_key") if not getattr(settings, k)]
    if missing:
        st.error(f"미설정 환경변수: {', '.join(m.upper() for m in missing)} (.env 확인)")
    if not settings.krx_api_key:
        st.warning("KRX_API_KEY 미설정 - 주가 조회 불가")
    max_hops = st.slider("분석 최대 Hop", 1, 5, settings.max_analysis_hops)
    max_companies = st.number_input("최대 수집 기업 수", 1, 50, settings.max_companies)
    disc_hops = st.slider("기업 탐색 Hop", 1, 3, settings.max_discovery_hops)

    if st.button("① Seed 데이터 수집·임베딩", width="stretch", disabled=bool(missing)):
        from src.pipeline import collect_company, embed_and_index, seed_company
        app = get_app(seed_code)
        cb = log_box()
        try:
            app.llm.check_model()
            seed = seed_company(app)
            collect_company(app, seed["corp_code"], seed["corp_name"], seed["stock_code"], progress=cb,
                            discovery_hop=0, discovery_reason="seed")
            embed_and_index(app, cb)
            st.success("완료")
        except Exception as exc:
            st.error(f"실패: {exc}")

    if st.button("② 관련 기업 탐색", width="stretch", disabled=bool(missing)):
        from src.discovery.company_discovery import CompanyDiscovery
        from src.pipeline import make_ingest_fn, seed_company
        app = get_app(seed_code)
        cb = log_box()
        try:
            graph = app.load_graph()
            app.llm.reset_budget(int(max_companies) * 8)
            app.retriever.reset_budget(None)
            rep = CompanyDiscovery(app.storage, app.llm, app.retriever, app.resolver, make_ingest_fn(app, cb), graph,
                                   max_companies=int(max_companies), max_hops=disc_hops,
                                   max_per_node=settings.max_discovery_per_node, progress_cb=cb).run(seed_company(app))
            graph.save(settings.graph_dir)
            st.success(f"탐색 종료: {rep.stopped_reason}")
        except Exception as exc:
            st.error(f"실패: {exc}")

    stt = storage().stats()
    st.subheader("데이터 구축 현황")
    st.metric("수집 기업 (READY/전체)", f"{stt['companies_ready']}/{stt['companies']}")
    c1, c2 = st.columns(2)
    c1.metric("문서", stt["documents"])
    c2.metric("Chunk", stt["chunks"])
    c1.metric("임베딩", stt["embeddings"])
    c2.metric("오류 로그", stt["errors"])
    meta = settings.index_path.with_suffix(".meta.json")
    st.caption(f"임베딩: {settings.embedding_model} ({settings.embedding_dimensions}d)")
    if meta.exists():
        st.caption(f"FAISS: {json.loads(meta.read_text())}")
    st.caption(f"LLM: {settings.llm_model}")

tabs = st.tabs(["Research", "Knowledge Graph", "Financial Analysis", "Evaluation", "Data Explorer"])

# ---------------------------------------------------------------- Research
with tabs[0]:
    q = st.text_area("뉴스 본문 또는 자연어 이벤트", "HBM 수요가 증가할 경우 국내 관련 기업과 산업의 파급효과를 분석해줘", height=120)
    col1, col2 = st.columns([1, 1])
    as_of = col1.text_input("기준일 (선택, YYYY-MM-DD)", "")
    insight_mode = col1.checkbox("가설 인사이트 포함 (근거 미확인 경로 확장 + 추가 가설)", True)
    run = col2.button("분석 시작", type="primary", disabled=bool(missing))
    if run:
        from src.pipeline import seed_company
        from src.research.ripple_engine import RippleConfig, RippleEngine, save_result
        app = get_app(seed_code)
        cb = log_box()
        try:
            cfg = RippleConfig(max_hops=max_hops, max_branches_per_hop=settings.max_branches_per_hop,
                               max_retrieval_calls=settings.max_retrieval_calls, max_llm_calls=settings.max_llm_calls,
                               top_k=settings.retrieval_top_k, as_of_date=as_of or None, insight_mode=insight_mode)
            res = RippleEngine(app.storage, app.llm, app.retriever, app.resolver, cfg, base_graph=app.load_graph(),
                               progress_cb=cb).run(q, seed_company(app))
            save_result(res, settings.runs_dir, app.storage, settings.llm_model, {**settings.public_dict(), **cfg.__dict__})
            st.session_state.result = res.to_dict()
            st.session_state.loaded_run = res.run_id
            st.session_state.fin = None
        except Exception as exc:
            st.error(f"분석 실패: {exc}")
    runs = storage().list_runs()
    if runs:
        def _load_run(run_id: str) -> None:
            row = next((r for r in runs if r["run_id"] == run_id), None)
            if row and row["result_path"] and Path(row["result_path"]).exists():
                loaded = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
                st.session_state.result = loaded
                st.session_state.loaded_run = run_id
                # CLI --financial 로 저장된 금융 분석 결과도 함께 표시
                st.session_state.fin = loaded.get("financial")

        # URL ?run=<run_id> 또는 첫 화면에서는 가장 최근 실행 결과를 자동으로 표시
        qp_run = st.query_params.get("run")
        if qp_run and st.session_state.get("loaded_run") != qp_run and "qp_done" not in st.session_state:
            st.session_state.qp_done = True
            _load_run(qp_run)
        elif "result" not in st.session_state:
            _load_run(runs[0]["run_id"])
        labels = {r["run_id"]: f"{r['run_id']} · {(r['input_text'] or '')[:40]}" for r in runs}
        ids = list(labels)
        cur = st.session_state.get("loaded_run")
        pick = st.selectbox("실행 기록", ids, index=ids.index(cur) if cur in ids else 0, format_func=labels.get)
        if pick != cur:
            _load_run(pick)
    res = st.session_state.get("result")
    if res:
        from src.ui.research_view import render_result

        def _fetch_financial(ev_date):
            from src.financial.analysis import run_for_result
            with st.spinner("KRX 주가 조회 중 (처음 조회하는 날짜는 시간이 걸립니다)"):
                fin = run_for_result(res, get_app(seed_code), ev_date or dt.date.today().isoformat())
            res["financial"] = fin
            st.session_state.fin = fin
            # 실행 기록 파일에도 저장 (다시 열어도 유지)
            row = next((r for r in storage().list_runs() if r["run_id"] == res.get("run_id")), None)
            if row and row["result_path"] and Path(row["result_path"]).exists():
                Path(row["result_path"]).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")

        render_result(res, on_fetch_financial=_fetch_financial if not missing else None)

# ---------------------------------------------------------------- Graph
with tabs[1]:
    res = st.session_state.get("result")
    src = st.radio("그래프", ["현재 분석 결과", "지식 기반(탐색 결과)"], horizontal=True)
    gs = GraphStore.from_dict(res["graph"]) if (src == "현재 분석 결과" and res) else GraphStore.load(settings.graph_dir)
    c1, c2 = st.columns(2)
    weak = c1.checkbox("미검증 가설 포함", True)
    edge_labels = c2.checkbox("관계 이름 표시", True)
    if gs.g.number_of_edges():
        graph_html = to_d3_html(gs, height=660, include_weak=weak, show_edge_labels=edge_labels)
        if hasattr(st, "iframe"):
            st.iframe(graph_html, height=680)
        else:
            components.html(graph_html, height=680)
        edges = gs.edges()
        names = {n: d["name"] for n, d in gs.g.nodes(data=True)}
        label = {f"{names[e.source_id]} -{e.relationship.value}-> {names[e.target_id]} [{e.verification_status.value}]": e
                 for e in edges}
        sel = st.selectbox("관계 선택 → 원문 근거", list(label))
        e = label[sel]
        st.write(f"근거 {len(e.evidence)}개 · 조건: {e.conditions} · 추론: {e.reasoning}")
        for ev in e.evidence:
            ch = storage().get_chunk(ev.chunk_id)
            st.markdown(f"**인용**: {ev.quote}")
            if ch:
                st.text_area(f"chunk {ev.chunk_id} · {ch['corp_name']} · {ch['section_path']} · {ch['published_at']}",
                             ch["raw_text"], height=160)
    else:
        st.info("그래프가 비어 있습니다.")

# ---------------------------------------------------------------- Financial
with tabs[2]:
    res = st.session_state.get("result")
    if not res:
        st.info("먼저 Research 탭에서 분석을 실행하거나 결과를 불러오세요.")
    else:
        default_date = res["event"].get("event_date") or (dt.date.today() - dt.timedelta(days=30)).isoformat()
        ev_date = st.text_input("이벤트 기준일", default_date)
        st.caption("벤치마크: data/market/benchmarks/{KOSPI,KOSDAQ}.csv (date,close) 또는 KRX 지수 API")
        if st.button("금융 데이터 조회 (Korea Stock MCP)"):
            from src.financial.analysis import run_for_result
            with st.spinner("KRX 주가 조회 중 (처음 조회하는 날짜는 시간이 걸립니다)"):
                st.session_state.fin = run_for_result(res, get_app(seed_code), ev_date)
                res["financial"] = st.session_state.fin
        fin = st.session_state.get("fin")
        if fin:
            from src.ui.research_view import render_financial
            render_financial(fin)

# ---------------------------------------------------------------- Evaluation
with tabs[3]:
    ev_file = settings.evaluation_dir / "events.json"
    if ev_file.exists():
        events = json.loads(ev_file.read_text(encoding="utf-8"))["events"]
        sel = st.selectbox("평가 이벤트", [e["event_id"] for e in events])
        ev = next(e for e in events if e["event_id"] == sel)
        st.write(ev["input_text"])
        as_of_eval = st.text_input("평가 기준일", ev["as_of_date"])
        if st.button("A/B/C 비교 실행", disabled=bool(missing)):
            from src.evaluation.historical_replay import replay_event
            from src.pipeline import seed_company
            app = get_app(seed_code)
            cb = log_box()
            ev = {**ev, "as_of_date": as_of_eval}
            try:
                st.session_state.eval = replay_event(app, ev, seed_company(app), settings.evaluation_dir / "results",
                                                     progress=cb)
            except Exception as exc:
                st.error(f"평가 실패: {exc}")
        rd = settings.evaluation_dir / "results" / sel / "metrics.json"
        rep = st.session_state.get("eval") or (json.loads(rd.read_text(encoding="utf-8")) if rd.exists() else None)
        if rep:
            rows = []
            for arm, m in rep["metrics"].items():
                if isinstance(m, dict) and "latency_ms" in m:
                    rows.append({"실험군": arm, "Company P": m["company"].get("precision"),
                                 "Company R": m["company"].get("recall"), "Quote Acc": m["quote_accuracy"],
                                 "Path Validity": m.get("path_validity"),
                                 "Novel(verified)": (m.get("valid_novel_discovery") or {}).get("count_verified"),
                                 "Market Dir Acc": (m.get("market_direction") or {}).get("accuracy"),
                                 "Latency(ms)": m["latency_ms"], "Tokens in": m["token_usage"].get("input_tokens"),
                                 "Tokens out": m["token_usage"].get("output_tokens"), "Cost($)": m["estimated_cost_usd"]})
            st.dataframe(pd.DataFrame(rows), width="stretch")
            if not rep.get("ground_truth_labeled"):
                st.warning("정답 라벨이 없어 Precision/Recall 등은 계산되지 않았습니다. events.json의 ground_truth를 사람이 작성하세요.")
            st.caption(rep["leakage_note"])
    else:
        st.info("data/evaluation/events.json 이 없습니다.")

# ---------------------------------------------------------------- Data Explorer
with tabs[4]:
    s = storage()
    st.subheader("수집 기업")
    st.dataframe(pd.DataFrame(s.list_companies()), width="stretch")
    st.subheader("기업 탐색 후보")
    st.dataframe(pd.DataFrame(s.list_candidates()), width="stretch")
    docs = s.list_documents()
    st.subheader("수집 보고서")
    st.dataframe(pd.DataFrame(docs).drop(columns=["section_report_json"], errors="ignore"), width="stretch")
    if docs:
        d = st.selectbox("문서 선택", [f"{x['document_id']} | {x.get('corp_name', '')} | {x['report_name']}" for x in docs])
        doc_id = d.split(" | ")[0]
        doc = s.get_document(doc_id)
        if doc and doc["section_report_json"]:
            rep = json.loads(doc["section_report_json"])
            st.write(f"선택 섹션 {len(rep['selected'])}개 / 제외 {len(rep['excluded'])}개 / 실패 규칙 {rep['failed_rules']} / "
                     f"글자 {rep['chars_before']:,} → {rep['chars_after']:,} / 토큰 {rep.get('tokens_before')} → {rep.get('tokens_after')}")
            st.dataframe(pd.DataFrame(rep["selected"]), width="stretch")
        chunks = s.query("SELECT chunk_id, section_path, token_count, embedding_id, raw_text FROM chunks WHERE document_id=? "
                         "ORDER BY chunk_order", (doc_id,))
        st.write(f"Chunk {len(chunks)}개 · 임베딩 {sum(1 for c in chunks if c['embedding_id'])}개")
        if chunks:
            cid = st.selectbox("Chunk", [c["chunk_id"] for c in chunks])
            c = next(x for x in chunks if x["chunk_id"] == cid)
            st.text_area(c["section_path"], c["raw_text"], height=220)
    st.subheader("수집·전처리 실패 로그")
    st.dataframe(pd.DataFrame(s.list_logs(level="ERROR") + s.list_logs(level="WARNING")), width="stretch")

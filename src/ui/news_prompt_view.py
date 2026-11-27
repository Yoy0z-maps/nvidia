"""[1] 뉴스·공시 → 프롬프트 화면 (사용 흐름 1~2).

분야 입력 → (뉴스 수집·요약·사업보고서·프롬프트) → 뉴스 출처와 사업보고서 확인 → 프롬프트 수정 → "분석 시작"으로 확정.
확정한 C1은 data/runs/c1/에 저장만 한다. [2] 차수 배정·[4] 인사이트로 넘기는 연결은 아직 없다 (파일·JSON으로 전달).
단독 실행: uv run streamlit run news_app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.contracts import ANCHOR_COMPANY, C0Input, C1Bundle

C1_KEY = "news_c1"            # 현재 화면의 C1
SAVED_KEY = "news_c1_saved"   # 확정·저장한 파일 경로
PROMPT_KEY = "news_prompt_edit"
STATUS_TEXT = {"ok": "정상", "partial": "일부만 완료", "error": "실패"}


@st.cache_resource
def _agent(_settings):
    """네이버·DART·OpenAI 클라이언트는 앱이 떠 있는 동안 재사용한다."""
    from src.news_agent.agent import build_agent

    agent, _close = build_agent(_settings)
    return agent


def _set_c1(c1: C1Bundle) -> None:
    st.session_state[C1_KEY] = c1
    st.session_state[PROMPT_KEY] = c1.prompt_final
    st.session_state.pop(SAVED_KEY, None)


def _run(settings, field: str) -> None:
    with st.status(f"{ANCHOR_COMPANY} · {field} 분석 준비 중", expanded=True) as box:
        c1 = _agent(settings).run(C0Input(field=field), progress=box.write)
        state = "complete" if c1.status == "ok" else ("error" if c1.status == "error" else "complete")
        box.update(label=f"완료: {STATUS_TEXT[c1.status]} · 뉴스 {len(c1.news)}건", state=state, expanded=False)
    _set_c1(c1)


def _saved_files(settings) -> list:
    d = settings.runs_dir / "c1"
    return sorted(d.glob("c1_*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []


def _input_section(settings) -> None:
    missing = [k for k, v in (("OPENAI_API_KEY", settings.openai_api_key), ("DART_API_KEY", settings.dart_api_key),
                              ("NAVER_CLIENT_ID", settings.naver_client_id),
                              ("NAVER_CLIENT_SECRET", settings.naver_client_secret)) if not v]
    with st.form("news_input", border=True):
        cols = st.columns([1, 2, 1], vertical_alignment="bottom")
        cols[0].text_input("기업", value=ANCHOR_COMPANY, disabled=True, help="프로토타입에서는 SK하이닉스로 고정합니다.")
        field = cols[1].text_input("분야", key="news_field", placeholder="예: HBM, DRAM, 낸드, 파운드리", max_chars=30)
        go = cols[2].form_submit_button("뉴스·공시 불러오기", type="primary", width="stretch",
                                        icon=":material/search:", disabled=bool(missing))
    if missing:
        st.error(f".env에 다음 키가 없어 실행할 수 없습니다: {', '.join(missing)}", title="API 키 없음")
    if go:
        if not field.strip():
            st.warning("분야를 입력하세요.")
        else:
            _run(settings, field.strip())

    files = _saved_files(settings)
    if files:
        with st.expander("저장된 결과 불러오기 (API를 다시 부르지 않음)", icon=":material/history:"):
            cols = st.columns([3, 1], vertical_alignment="bottom")
            pick = cols[0].selectbox("저장된 C1", files, format_func=lambda p: p.name, key="news_saved_pick")
            if cols[1].button("불러오기", width="stretch", key="news_load"):
                from src.news_agent.agent import load_c1

                _set_c1(load_c1(pick))


def _status_section(c1: C1Bundle) -> None:
    hard = [e for e in c1.errors if "경고" not in e.split(":")[0]]
    warn = [e for e in c1.errors if e not in hard]
    if c1.status == "error":
        st.error("\n".join(f"- {e}" for e in hard), title="뉴스와 사업보고서를 모두 가져오지 못했습니다")
    elif c1.status == "partial":
        st.warning("\n".join(f"- {e}" for e in hard), title="일부만 완료되었습니다")
    if warn:
        with st.expander(f"확인할 점 {len(warn)}건", icon=":material/info:"):
            for w in warn:
                st.caption(w)


def _news_section(c1: C1Bundle) -> None:
    st.subheader("뉴스 출처", divider="gray")
    if not c1.news:
        st.caption("가져온 뉴스가 없습니다.")
        return
    st.caption(f"최근 7일 기사 중 서로 다른 사건 {len(c1.news)}건 · 프롬프트에는 이슈 요약만 들어가고 기사 목록은 들어가지 않습니다.")
    df = pd.DataFrame([{"순번": n.rank, "날짜": n.published_at[:16].replace("T", " "), "제목": n.title,
                        "링크": n.url, "요약": n.summary} for n in c1.news])
    st.dataframe(df, hide_index=True, alt="선택된 뉴스 기사 목록",
                 column_config={"순번": st.column_config.NumberColumn(width="small"),
                                "링크": st.column_config.LinkColumn(display_text="열기", width="small"),
                                "요약": st.column_config.TextColumn(width="large")})
    if c1.issue_summary:
        with st.expander("이슈 요약 (프롬프트에 들어간 내용)", icon=":material/summarize:"):
            st.markdown(c1.issue_summary)


def _report_section(c1: C1Bundle) -> None:
    doc = c1.company_doc
    st.subheader("사업보고서 (I·II 원문)", divider="gray")
    if doc.status != "ok":
        st.caption(f"사업보고서를 가져오지 못했습니다: {doc.error}")
        return
    cols = st.columns([3, 1], vertical_alignment="center")
    cols[0].markdown(f"**{doc.corp_name} {doc.report_name}** · {doc.rcept_dt} 제출 · 섹션 {len(doc.sections)}개 · "
                     f"분야 관련 줄 {len(doc.field_highlights)}개")
    cols[1].link_button("DART 원문", doc.source_url, icon=":material/open_in_new:", width="stretch")
    if doc.field_highlights:
        with st.expander(f"'{c1.field}' 관련 원문 줄", icon=":material/format_quote:"):
            st.dataframe(pd.DataFrame([{"섹션": h.section_id, "원문": h.text} for h in doc.field_highlights]),
                         hide_index=True, alt="사업보고서에서 분야가 언급된 줄")


def _reset_prompt(draft: str) -> None:
    # 위젯이 그려지기 전에 값을 바꿔야 하므로 콜백에서 처리한다
    st.session_state[PROMPT_KEY] = draft
    st.session_state.pop(SAVED_KEY, None)


def _prompt_section(settings, c1: C1Bundle) -> None:
    st.subheader("분석 프롬프트", divider="gray")
    if not c1.prompt_draft:
        st.caption("뉴스 요약이 없어 프롬프트를 만들지 못했습니다. 분야를 바꾸거나 다시 시도하세요.")
        return
    st.caption("필요하면 고친 뒤 '분석 시작'을 누르세요. 사업보고서 원문과 뉴스 목록은 프롬프트와 별도로 함께 넘어갑니다.")
    text = st.text_area("프롬프트", key=PROMPT_KEY, height=420, label_visibility="collapsed")
    edited = text.strip() != c1.prompt_draft.strip()
    cols = st.columns([1, 1, 2], vertical_alignment="center")
    cols[0].button("원래 프롬프트로", icon=":material/undo:", disabled=not edited, width="stretch", key="news_reset",
                   on_click=_reset_prompt, args=(c1.prompt_draft,))
    start = cols[1].button("분석 시작", type="primary", icon=":material/play_arrow:", width="stretch", key="news_start",
                           disabled=not text.strip())
    cols[2].caption("수정됨" if edited else "수정 없음")
    if start:
        from src.news_agent.agent import save_c1

        final = c1.finalize(text)
        path = save_c1(final, settings.runs_dir)
        st.session_state[C1_KEY] = final
        st.session_state[SAVED_KEY] = str(path)

    saved = st.session_state.get(SAVED_KEY)
    if saved:
        cur: C1Bundle = st.session_state[C1_KEY]
        st.success(f"C1을 확정해 저장했습니다: `{saved}`\n\n"
                   "지금은 저장까지만 합니다. [2] 차수 배정, [4] 인사이트와는 아직 연결되어 있지 않으므로 "
                   "필요하면 이 파일이나 아래 JSON을 전달하세요.", title="프롬프트 확정")
        st.download_button("C1 JSON 내려받기", cur.model_dump_json(indent=2), file_name=Path(saved).name,
                           mime="application/json", icon=":material/download:")


def render(settings) -> None:
    st.title("뉴스·공시 프롬프트")
    st.caption(f"{ANCHOR_COMPANY}의 분야를 입력하면 최근 7일 뉴스와 최신 사업보고서로 분석 프롬프트를 만듭니다.")
    _input_section(settings)
    c1: C1Bundle | None = st.session_state.get(C1_KEY)
    if c1 is None:
        return
    st.markdown(f"**{c1.company} · {c1.field}** · 상태: {STATUS_TEXT[c1.status]} · request_id `{c1.request_id}`")
    _status_section(c1)
    _news_section(c1)
    _report_section(c1)
    _prompt_section(settings, c1)

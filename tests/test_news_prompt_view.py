"""7단계 화면 테스트 (Streamlit AppTest, 브라우저·외부 API 없음). C1은 가상 샘플(tests/fixtures/c1_sample.json)."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _app(tmp_path):
    def script(data_dir: str):
        import json
        from pathlib import Path

        import streamlit as st

        from config.settings import Settings
        from src.contracts import C1Bundle
        from src.ui.news_prompt_view import C1_KEY, PROMPT_KEY, render

        s = Settings(data_dir=Path(data_dir), openai_api_key="x", dart_api_key="x",
                     naver_client_id="x", naver_client_secret="x")
        if C1_KEY not in st.session_state:
            raw = json.loads(Path("tests/fixtures/c1_sample.json").read_text(encoding="utf-8"))
            raw.pop("_note")
            c1 = C1Bundle.model_validate(raw)
            st.session_state[C1_KEY] = c1
            st.session_state[PROMPT_KEY] = c1.prompt_final
        render(s)

    return AppTest.from_function(script, args=(str(tmp_path / "data"),), default_timeout=30)


def test_shows_news_report_and_prompt(tmp_path):
    at = _app(tmp_path).run()
    assert not at.exception
    assert [h.value for h in at.subheader] == ["뉴스 출처", "사업보고서 (I·II 원문)", "분석 프롬프트"]
    assert at.dataframe[0].value["링크"].str.startswith("https://example.com").all()
    assert at.text_area(key="news_prompt_edit").value.startswith("[가상]")
    assert at.text_input(key="news_field").value == ""


def test_edit_reset_and_confirm_saves_c1(tmp_path):
    at = _app(tmp_path).run()
    original = at.text_area(key="news_prompt_edit").value
    at.text_area(key="news_prompt_edit").set_value(original + "\n추가 지시: 가상.").run()
    assert "수정됨" in [c.value for c in at.caption]
    at.button(key="news_reset").click().run()
    assert at.text_area(key="news_prompt_edit").value == original

    at.text_area(key="news_prompt_edit").set_value(original + "\n추가 지시: 가상.").run()
    at.button(key="news_start").click().run()
    assert not at.exception and at.success
    saved = list((tmp_path / "data" / "runs" / "c1").glob("c1_*.json"))
    assert len(saved) == 1
    from src.news_agent.agent import load_c1

    c1 = load_c1(saved[0])
    assert c1.edited and c1.prompt_final.endswith("추가 지시: 가상.")

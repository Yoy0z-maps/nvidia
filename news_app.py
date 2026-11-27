"""[1] 뉴스·공시 프롬프트 화면 단독 실행 (공용 app.py와 별도).

uv run streamlit run news_app.py
"""
from __future__ import annotations

import streamlit as st

from config.settings import Settings, setup_logging
from src.ui.news_prompt_view import render

st.set_page_config(page_title="Market Ripple · 뉴스·공시 프롬프트", page_icon=":material/newspaper:", layout="wide")


@st.cache_resource
def _settings() -> Settings:
    s = Settings.from_env()
    s.ensure_dirs()
    setup_logging(s)
    return s


render(_settings())

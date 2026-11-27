"""6단계: agent.run(C0) → C1 테스트. 네이버·LLM·DART는 모두 가짜이고 데이터는 가상이다.

실제 호출은 `uv run pytest -m live tests/test_news_agent.py` (또는 scripts.run_news_agent)로 따로 실행한다.
"""
from __future__ import annotations

import datetime as dt
import re

import pytest

from src.contracts import SECTION_TITLES, C0Input, C1Bundle, CompanyDoc, DocSection
from src.news_agent.agent import NewsAgent, load_c1, save_c1
from tests.fake_llm import FakeLLM

NOW = dt.datetime(2099, 11, 27, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
NAVER = "https://n.news.naver.com/mnews/article/000/"
TOPICS = ["증산", "수율", "단가", "고객사", "패키징", "투자", "인력", "특허", "전력", "냉각", "검증", "출하"]


class FakeNaver:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def search(self, q, display=100, start=1, sort="date"):
        if self.fail:
            raise RuntimeError("네이버 장애(가상)")
        if start > 1:
            return []
        return [{"title": f"SK하이닉스 HBM {t} 가상 소식", "description": f"가상 발췌 {t} HBM",
                 "link": f"{NAVER}{i}", "originallink": f"https://example.com/{i}",
                 "pubDate": (NOW - dt.timedelta(hours=6 * i)).strftime("%a, %d %b %Y %H:%M:%S +0900")}
                for i, t in enumerate(TOPICS)]

    def fetch_body(self, url):
        return f"가상 본문 {url.rsplit('/', 1)[1]}. 매출은 1,234억원이다."


class FakeFetcher:
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.calls = []

    def fetch(self, field, keywords=(), today=None):
        self.calls.append((field, list(keywords), today))
        if not self.ok:
            return CompanyDoc(status="not_found", corp_name="SK하이닉스", stock_code="000660", error="가상 실패")
        part, title = SECTION_TITLES["II-2"]
        text = "가상 주요 제품 HBM"
        sec = DocSection(section_id="II-2", part=part, title=title, text=text, chars=len(text))
        return CompanyDoc(status="ok", corp_code="00164779", stock_code="000660", corp_name="SK하이닉스",
                          report_name="사업보고서 (2098.12)", rcept_no="20990317000001", rcept_dt="2099-03-17",
                          source_url="https://example.com/dart", sections=[sec])


def events(data: str) -> dict:
    n = len(re.findall(r"^\[\d+\]", data, flags=re.M))
    return {"events": [{"event": f"가상 사건 {i}", "ids": [i]} for i in range(n)]}


def digest(data: str) -> dict:
    n = len(re.findall(r"^\[\d+\]", data, flags=re.M))
    return {"articles": [{"id": i, "summary": f"가상 요약 {i}이다. 둘째 문장이다."} for i in range(n)],
            "issue_paragraph_1": "가상 첫 문단이다.", "issue_paragraph_2": "가상 둘째 문단이다.", "issue_paragraph_3": "",
            "keywords": ["가상기업", "HBM4", "증설"]}


def llm(summary=digest) -> FakeLLM:
    return FakeLLM({"news_event_selection": events, "news_summary": summary})


def run(naver=None, fetcher=None, fake_llm=None) -> tuple[C1Bundle, FakeFetcher]:
    fetcher = fetcher or FakeFetcher()
    agent = NewsAgent(fake_llm or llm(), naver or FakeNaver(), fetcher)
    return agent.run(C0Input(field="HBM", request_id="req-1"), now=NOW), fetcher


def test_ok_path_builds_full_c1():
    c1, fetcher = run()
    assert c1.status == "ok" and c1.company == "SK하이닉스" and c1.request_id == "req-1"
    assert len(c1.news) == 10 and all(n.summary for n in c1.news)
    assert c1.keywords == ["가상기업", "HBM4", "증설"]
    assert fetcher.calls == [("HBM", ["가상기업", "HBM4", "증설"], NOW.date())]   # 핵심어로 사업보고서 발췌
    assert c1.issue_summary == "가상 첫 문단이다.\n\n가상 둘째 문단이다."
    assert c1.prompt_draft == c1.prompt_final and not c1.edited
    assert "## 최근 이슈 요약" in c1.prompt_draft and "사업보고서 (2098.12)" in c1.prompt_draft
    assert "example.com" not in c1.prompt_draft and "가상 주요 제품" not in c1.prompt_draft
    assert c1.company_doc.sections[0].text == "가상 주요 제품 HBM"
    assert all("경고" in e for e in c1.errors)   # ok일 때는 경고만 있을 수 있다


def test_news_failure_is_partial_with_report_only():
    c1, _ = run(naver=FakeNaver(fail=True))
    assert c1.status == "partial" and c1.news == [] and c1.prompt_draft == ""
    assert any(e.startswith("news: 수집 실패") for e in c1.errors)
    assert c1.company_doc.status == "ok"


def test_summary_failure_keeps_news_without_summary():
    def broken(data):
        raise RuntimeError("LLM 장애(가상)")

    c1, fetcher = run(fake_llm=llm(summary=broken))
    assert c1.status == "partial" and len(c1.news) == 10 and not any(n.summary for n in c1.news)
    assert any(e.startswith("news_summary: 요약 실패") for e in c1.errors)
    assert fetcher.calls[0][1] == []   # 핵심어 없이 분야만으로 사업보고서 발췌


def test_report_failure_is_partial_and_prompt_says_news_only():
    c1, _ = run(fetcher=FakeFetcher(ok=False))
    assert c1.status == "partial" and c1.prompt_draft
    assert "사업보고서를 가져오지 못했으므로" in c1.prompt_draft
    assert any(e.startswith("company_doc: not_found") for e in c1.errors)


def test_everything_failing_is_error_but_valid_c1():
    c1, _ = run(naver=FakeNaver(fail=True), fetcher=FakeFetcher(ok=False))
    assert c1.status == "error" and len(c1.errors) >= 2
    C1Bundle.model_validate(c1.model_dump())


def test_save_load_and_finalize(tmp_path):
    c1, _ = run()
    path = save_c1(c1, tmp_path)
    assert path.name == "c1_HBM_req-1.json" and path.parent.name == "c1"
    loaded = load_c1(path)
    assert loaded == c1
    edited = loaded.finalize(loaded.prompt_draft + "\n추가 지시: 가상.")
    assert edited.edited and edited.prompt_final.endswith("추가 지시: 가상.")


@pytest.mark.live
def test_live_run_and_save():
    from config.settings import Settings
    from src.news_agent.agent import build_agent

    s = Settings.from_env()
    if not (s.openai_api_key and s.dart_api_key and s.naver_client_id):
        pytest.skip("키 없음")
    agent, close = build_agent(s)
    try:
        c1 = agent.run(C0Input(field="HBM"))
    finally:
        close()
    path = save_c1(c1, s.runs_dir)
    assert c1.status == "ok", c1.errors
    assert len(c1.news) == 10 and len(c1.company_doc.sections) == 12 and c1.company_doc.field_highlights
    assert load_c1(path) == c1
    print(path, c1.errors)

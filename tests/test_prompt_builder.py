"""5단계: 분석 프롬프트(최근 이슈 요약 + 분석 지시) 테스트. 데이터는 모두 가상이다.

실제 DART·네이버·OpenAI 호출은 `uv run pytest -m live tests/test_prompt_builder.py`로 따로 실행한다.
"""
from __future__ import annotations

import datetime as dt
import re

import pytest

from src.contracts import CompanyDoc, DocSection, NewsItem, SECTION_TITLES
from src.news_agent.prompt_builder import build_analysis_prompt, news_period
from src.news_agent.summarizer import mismatched_units


def news(n=3) -> list[NewsItem]:
    return [NewsItem(rank=i + 1, title=f"가상 기사 {i}", url=f"https://example.com/{i}",
                     published_at=f"2099-11-{27 - i:02d}T09:00:00+09:00", summary=f"요약 {i}.") for i in range(n)]


def doc() -> CompanyDoc:
    part, title = SECTION_TITLES["II-2"]
    sec = DocSection(section_id="II-2", part=part, title=title, text="가상 본문", chars=5)
    return CompanyDoc(status="ok", corp_code="00164779", stock_code="000660", corp_name="SK하이닉스",
                      report_name="사업보고서 (2098.12)", rcept_no="20990317000001", rcept_dt="2099-03-17",
                      source_url="https://example.com/dart", sections=[sec])


def test_prompt_is_issue_summary_plus_instructions_only():
    p = build_analysis_prompt("SK하이닉스", "HBM", "이슈1\n\n이슈2", doc(), news(), as_of=dt.date(2099, 11, 27))
    assert re.findall(r"^## (.+)$", p, flags=re.M) == ["최근 이슈 요약 (2099-11-25 ~ 2099-11-27, 뉴스 3건 종합)",
                                                       "분석 지시"]
    assert "기준 기업은 SK하이닉스, 분야는 'HBM', 기준일은 2099-11-27" in p
    assert "이슈1\n\n이슈2" in p
    # 사업보고서는 원문을 옮기지 않고 가리키기만 한다
    assert "사업보고서(사업보고서 (2098.12), 2099-03-17 제출)의 I. 회사의 개요, II. 사업의 내용 원문을 근거로" in p
    assert "가상 본문" not in p
    # 뉴스 목록·URL·핵심어는 프롬프트 밖에 둔다
    assert "example.com" not in p and "가상 기사" not in p and "핵심어" not in p
    assert not re.search(r"\{[a-z_]+\}", p)
    assert news_period(news(1)) == "2099-11-27"


def test_prompt_without_report_says_news_only():
    bad = CompanyDoc(status="not_found", corp_name="SK하이닉스", error="없음")
    p = build_analysis_prompt("SK하이닉스", "HBM", "이슈1\n\n이슈2", bad, news())
    assert "사업보고서를 가져오지 못했으므로" in p and "섹션ID" not in p


def test_mismatched_units():
    src = "투자 현황\n(단위: 십억원)\n| 보완투자 | 30,173 |\n생산능력은 41,216,512백만원입니다."
    assert mismatched_units("투자는 30,173억원이다.", src) == ["30,173억원 (원문 단위: 십억원)"]
    assert mismatched_units("투자는 30,173십억원, 생산능력 41,216,512백만원.", src) == []
    assert mismatched_units("생산능력은 41,216,512억원.", src) == ["41,216,512억원 (원문 단위: 백만원)"]
    assert mismatched_units("가상 수치 123억원.", src) == []   # 원문에 없는 숫자는 숫자 검사에서 따로 잡는다


@pytest.mark.live
def test_live_prompt():
    from config.settings import Settings
    from src.ingestion.dart_client import DartClient
    from src.llm import OpenAILLM
    from src.news_agent.company_doc import CompanyDocFetcher
    from src.news_agent.naver_client import NaverNewsClient
    from src.news_agent.news import collect_news
    from src.news_agent.summarizer import summarize_news

    s = Settings.from_env()
    if not (s.naver_client_id and s.openai_api_key and s.dart_api_key):
        pytest.skip("키 없음")
    s.ensure_dirs()
    llm = OpenAILLM(s.openai_api_key, s.llm_model)
    with NaverNewsClient(s.naver_client_id, s.naver_client_secret) as c:
        arts = collect_news(c, "SK하이닉스", "HBM", llm=llm)
    d = summarize_news(llm, arts, "SK하이닉스", "HBM")
    dart = DartClient(s.dart_api_key)
    try:
        cdoc = CompanyDocFetcher(s, dart).fetch("HBM", d.keywords)
    finally:
        dart.close()
    items = [a.to_news_item(rank=i + 1, summary=d.summaries[i]) for i, a in enumerate(arts)]
    p = build_analysis_prompt("SK하이닉스", "HBM", d.issue_summary, cdoc, items)
    assert re.findall(r"^## (.+)$", p, flags=re.M)[-1] == "분석 지시" and "http" not in p
    assert cdoc.status == "ok" and "섹션ID" in p
    (s.runs_dir / "prompt_sample_HBM.md").write_text(p, encoding="utf-8")
    print({"warnings": d.warnings, "llm": llm.usage.to_dict()})

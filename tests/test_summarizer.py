"""5단계: 뉴스 요약 테스트. 기사와 LLM 응답은 모두 가상 데이터다.

실제 네이버·OpenAI 호출은 `uv run pytest -m live tests/test_summarizer.py`로 따로 실행한다.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.contracts import split_paragraphs
from src.news_agent.news import NewsArticle
from src.news_agent.summarizer import (MAX_ARTICLE_CHARS, NewsDigestOutput, SummaryError, check_digest,
                                       count_sentences, format_articles, summarize_news, unsupported_numbers)
from tests.fake_llm import FakeLLM


def articles(n=3) -> list[NewsArticle]:
    return [NewsArticle(title=f"가상 HBM 기사 {i}", url=f"https://example.com/{i}", origin_url="",
                        published_at=f"2099-11-{27 - i:02d}T09:00:00+09:00", description=f"가상 발췌 {i}",
                        body=("가상 본문 " * 1000) if i == 0 else "")
            for i in range(n)]


def good_output(n=3, **over) -> dict:
    out = {
        "articles": [{"id": i, "summary": f"가상 기사 {i}의 첫 문장이다. 둘째 문장이다."} for i in range(n)],
        "issue_paragraph_1": "가상 첫 문단이다. 사건을 정리한다.",
        "issue_paragraph_2": "가상 둘째 문단이다. 의미를 정리한다.",
        "issue_paragraph_3": "",
        "keywords": ["가상기업", "HBM4", "증설"],
    }
    out.update(over)
    return out


def test_format_articles_marks_body_or_excerpt_and_caps_length():
    text = format_articles(articles())
    assert "[0] 2099-11-27T09:00 가상 HBM 기사 0" in text
    assert "본문: 가상 본문" in text and "발췌: 가상 발췌 1" in text
    first = text.split("\n\n")[0]
    assert len(first.split("본문: ", 1)[1]) == MAX_ARTICLE_CHARS


def test_summarize_news_happy_path():
    llm = FakeLLM({"news_summary": good_output()})
    d = summarize_news(llm, articles(), "SK하이닉스", "HBM")
    assert d.summaries[2].startswith("가상 기사 2")
    assert len(split_paragraphs(d.issue_summary)) == 2
    assert d.keywords == ["가상기업", "HBM4", "증설"]
    instructions_have_data_only = llm.calls[0][1]
    assert "[1]" in instructions_have_data_only


def test_check_digest_repairs_paragraphs_and_keywords():
    out = NewsDigestOutput.model_validate(good_output(
        issue_paragraph_1="문단1 첫 줄\n이어지는 줄", issue_paragraph_2="문단2", issue_paragraph_3="문단3\n\n문단4",
        keywords=[" HBM ", "가상기업", "가상기업", "#증설", "패키징", "a", "b", "c", "d"]))
    d, problems = check_digest(out, 3, "HBM")
    assert problems == []
    assert split_paragraphs(d.issue_summary) == ["문단1 첫 줄 이어지는 줄", "문단2", "문단3 문단4"]
    assert d.keywords == ["가상기업", "증설", "패키징", "a", "b", "c", "d"]   # 분야 제외, 중복 제거, 최대 7개


def test_company_name_not_a_keyword():
    out = NewsDigestOutput.model_validate(good_output(keywords=["SK하이닉스", "하이닉스", "가상기업", "HBM4", "증설"]))
    d, _ = check_digest(out, 3, "HBM")
    assert d.keywords == ["가상기업", "HBM4", "증설"]


def test_numbers_not_in_source_are_flagged():
    assert unsupported_numbers("1,200억원과 17Gbps, 36단", "가상 기사: 1200억원, 17Gbps") == ["36"]
    out = NewsDigestOutput.model_validate(good_output(issue_paragraph_2="가상 둘째 문단. 점유율 99%이다."))
    d, _ = check_digest(out, 3, "HBM", source="[0] 가상 원문 [1] [2] 2099")
    assert any("99" in w for w in d.warnings)


def test_single_field_with_blank_lines_is_split():
    out = NewsDigestOutput.model_validate(good_output(issue_paragraph_1="문단1\n\n문단2", issue_paragraph_2=""))
    d, _ = check_digest(out, 3, "HBM")
    assert split_paragraphs(d.issue_summary) == ["문단1", "문단2"]


def test_missing_article_summary_retries_then_fails():
    bad = good_output(articles=[{"id": 0, "summary": "요약."}, {"id": 7, "summary": "없는 번호."}])
    llm = FakeLLM({"news_summary": [bad, good_output()]})
    assert summarize_news(llm, articles(), "SK하이닉스", "HBM").summaries[1]
    assert len(llm.calls) == 2
    with pytest.raises(SummaryError, match="누락"):
        summarize_news(FakeLLM({"news_summary": bad}), articles(), "SK하이닉스", "HBM")


def test_too_few_paragraphs_or_keywords_fail():
    for over in ({"issue_paragraph_2": ""}, {"keywords": ["HBM", "하나"]}):
        with pytest.raises(SummaryError):
            summarize_news(FakeLLM({"news_summary": good_output(**over)}), articles(), "SK하이닉스", "HBM")


def test_sentence_count_warning_only():
    out = NewsDigestOutput.model_validate(good_output(
        articles=[{"id": 0, "summary": "한 문장."}] + [{"id": i, "summary": "가. 나."} for i in (1, 2)]))
    d, problems = check_digest(out, 3, "HBM")
    assert problems == [] and d.warnings and "[0]" in d.warnings[0]
    assert count_sentences("첫 문장이다. 둘째 문장이다! 셋째?") == 3


def test_empty_articles_rejected():
    with pytest.raises(SummaryError):
        summarize_news(FakeLLM({}), [], "SK하이닉스", "HBM")


@pytest.mark.live
def test_live_summary_and_prompt():
    import json

    from config.settings import Settings
    from src.llm import OpenAILLM
    from src.news_agent.naver_client import NaverNewsClient
    from src.news_agent.news import collect_news

    s = Settings.from_env()
    if not (s.naver_client_id and s.openai_api_key):
        pytest.skip("키 없음")
    llm = OpenAILLM(s.openai_api_key, s.llm_model)
    with NaverNewsClient(s.naver_client_id, s.naver_client_secret) as c:
        arts = collect_news(c, "SK하이닉스", "HBM", llm=llm)
    d = summarize_news(llm, arts, "SK하이닉스", "HBM")
    assert len(d.summaries) == len(arts) and all(d.summaries)
    assert 2 <= len(split_paragraphs(d.issue_summary)) <= 3
    assert 3 <= len(d.keywords) <= 7
    print(json.dumps({"keywords": d.keywords, "warnings": d.warnings, "llm": llm.usage.to_dict()},
                     ensure_ascii=False))

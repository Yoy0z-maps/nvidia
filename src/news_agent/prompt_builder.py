"""분석 프롬프트(C1.prompt_draft) 조립. LLM을 부르지 않는다.

구성: ① 최근 이슈 요약 ② 분석 지시 (prompts/analysis_prompt.md).
- 사업보고서 원문은 C1.company_doc(JSON)으로 함께 넘어가므로 프롬프트에 옮겨 적지 않고, 지시문에서 가리키기만 한다.
- 뉴스 목록·URL은 C1.news로 따로 두고(화면 출처·개선용), 핵심어는 C1.keywords로 따로 둔다.
사용자는 화면에서 이 프롬프트를 고칠 수 있고, 확정본이 C1.prompt_final이 된다.
"""
from __future__ import annotations

import datetime as dt

from src.contracts import CompanyDoc, NewsItem
from src.llm import load_prompt


def news_period(news: list[NewsItem]) -> str:
    days = sorted(n.published_at[:10] for n in news)
    if not days:
        return "기간 없음"
    return days[0] if days[0] == days[-1] else f"{days[0]} ~ {days[-1]}"


def report_instruction(company: str, doc: CompanyDoc) -> str:
    if doc.status != "ok":
        return f"{company} 사업보고서를 가져오지 못했으므로, {company}에 관한 판단은 뉴스에 나온 내용으로만 하고 그렇게 밝힌다."
    return (f"함께 제공되는 {company} 사업보고서({doc.report_name}, {doc.rcept_dt} 제출)의 I. 회사의 개요, "
            f"II. 사업의 내용 원문을 근거로 쓰고, 근거가 되는 섹션ID(예: II-2)를 밝힌다.")


def build_analysis_prompt(company: str, field_name: str, issue_summary: str, company_doc: CompanyDoc,
                          news: list[NewsItem], as_of: dt.date | None = None) -> str:
    """news는 기간·건수 표시에만 쓴다 (제목·URL은 넣지 않는다)."""
    as_of = as_of or dt.date.today()
    return load_prompt(
        "analysis_prompt",
        company=company,
        field=field_name,
        as_of=as_of.isoformat(),
        news_period=news_period(news),
        n_news=len(news),
        issue_summary=issue_summary.strip(),
        report_instruction=report_instruction(company, company_doc),
    ).strip() + "\n"

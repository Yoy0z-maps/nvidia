"""[1] 뉴스·공시 agent: C0(분야) → C1(프롬프트 + 사업보고서 원문 + 뉴스 목록).

흐름
  1. 최근 7일 뉴스 수집·사건 단위 선택 (news.collect_news, LLM 1회)
  2. 기사별 요약·이슈 요약·핵심어 (summarizer.summarize_news, LLM 1회)
  3. SK하이닉스 최신 사업보고서 I·II 원문 (company_doc, 분야·핵심어로 발췌 표시)
  4. 분석 프롬프트 = 최근 이슈 요약 + 분석 지시 (prompt_builder)

어느 단계가 실패해도 예외를 밖으로 던지지 않는다. 남은 결과로 C1을 만들고 status를 partial/error로,
errors에 사유를 적는다. 다른 역할([2]·[4]·화면)은 C1의 status와 errors만 보면 된다.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

from src.contracts import C0Input, C1Bundle, CompanyDoc
from src.news_agent.news import NewsArticle, collect_news
from src.news_agent.prompt_builder import build_analysis_prompt
from src.news_agent.summarizer import NewsDigest, summarize_news

log = logging.getLogger(__name__)


class NewsAgent:
    def __init__(self, llm, naver_client, company_doc_fetcher, *, fetch_body: bool = True):
        """naver_client: NaverNewsClient, company_doc_fetcher: CompanyDocFetcher (테스트에서는 가짜를 넣는다)."""
        self.llm = llm
        self.naver = naver_client
        self.fetcher = company_doc_fetcher
        self.fetch_body = fetch_body

    def run(self, c0: C0Input, now: dt.datetime | None = None) -> C1Bundle:
        started = time.monotonic()
        errors: list[str] = []
        company, field_name = c0.company, c0.field
        now = now or dt.datetime.now(dt.timezone.utc).astimezone()

        # 1. 뉴스
        articles: list[NewsArticle] = []
        try:
            articles = collect_news(self.naver, company, field_name, llm=self.llm, now=now,
                                    fetch_body=self.fetch_body)
            if not articles:
                errors.append(f"news: 최근 7일 안에 '{company} {field_name}' 기사가 없습니다.")
        except Exception as exc:
            log.exception("뉴스 수집 실패")
            errors.append(f"news: 수집 실패 ({type(exc).__name__}: {exc})")

        # 2. 요약
        digest: NewsDigest | None = None
        if articles:
            try:
                digest = summarize_news(self.llm, articles, company, field_name)
                errors += [f"news_summary 경고: {w}" for w in digest.warnings]
            except Exception as exc:
                log.exception("뉴스 요약 실패")
                errors.append(f"news_summary: 요약 실패 ({type(exc).__name__}: {exc})")
        keywords = digest.keywords if digest else []

        # 3. 사업보고서 I·II (fetch는 실패해도 예외 대신 status로 알려 준다)
        try:
            company_doc = self.fetcher.fetch(field_name, keywords, today=now.date())
        except Exception as exc:
            log.exception("사업보고서 수집 실패")
            company_doc = CompanyDoc(status="not_found", corp_name=company, error=f"{type(exc).__name__}: {exc}")
        if company_doc.status != "ok":
            errors.append(f"company_doc: {company_doc.status} ({company_doc.error})")
        elif company_doc.missing_section_ids:
            errors.append(f"company_doc 경고: 찾지 못한 하위 목차 {company_doc.missing_section_ids}")

        # 4. 프롬프트와 C1
        news_items = []
        if digest:
            news_items = [a.to_news_item(rank=i + 1, summary=digest.summaries[i]) for i, a in enumerate(articles)]
        else:
            news_items = [a.to_news_item(rank=i + 1) for i, a in enumerate(articles)]
        issue_summary = digest.issue_summary if digest else ""
        prompt = build_analysis_prompt(company, field_name, issue_summary, company_doc, news_items,
                                       as_of=now.date()) if digest else ""

        # ok: 요약·프롬프트와 사업보고서가 모두 있음 / partial: 둘 중 하나만 / error: 둘 다 없음
        # (errors에는 실패 사유와 함께 '경고'도 들어갈 수 있다. ok여도 errors가 비어 있지 않을 수 있다.)
        if digest and company_doc.status == "ok":
            status = "ok"
        elif digest or company_doc.status == "ok":
            status = "partial"
        else:
            status = "error"
        c1 = C1Bundle(status=status, errors=errors, request_id=c0.request_id, company=company, field=field_name,
                      news=news_items, issue_summary=issue_summary, keywords=keywords,
                      prompt_draft=prompt, prompt_final=prompt, edited=False, company_doc=company_doc)
        log.info("C1 생성: status=%s, 뉴스 %d건, 사업보고서 %s, %.1f초, LLM %s", c1.status, len(news_items),
                 company_doc.status, time.monotonic() - started, getattr(self.llm, "usage", None))
        return c1


# ---------------------------------------------------------------------------
# 저장·불러오기 (다른 역할에 넘기는 파일)
# ---------------------------------------------------------------------------
def c1_path(runs_dir: Path, c1: C1Bundle) -> Path:
    return runs_dir / "c1" / f"c1_{c1.field}_{c1.request_id}.json"


def save_c1(c1: C1Bundle, runs_dir: Path) -> Path:
    path = c1_path(runs_dir, c1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(c1.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_c1(path: Path) -> C1Bundle:
    return C1Bundle.model_validate(json.loads(path.read_text(encoding="utf-8")))


def build_agent(settings, llm=None):
    """실제 클라이언트로 NewsAgent를 만든다. 반환: (agent, close 함수)."""
    from src.ingestion.dart_client import DartClient
    from src.llm import OpenAILLM
    from src.news_agent.company_doc import CompanyDocFetcher
    from src.news_agent.naver_client import NaverNewsClient

    settings.ensure_dirs()
    llm = llm or OpenAILLM(settings.openai_api_key, settings.llm_model)
    naver = NaverNewsClient(settings.naver_client_id, settings.naver_client_secret)
    dart = DartClient(settings.dart_api_key)
    agent = NewsAgent(llm, naver, CompanyDocFetcher(settings, dart))

    def close() -> None:
        naver.close()
        dart.close()

    return agent, close

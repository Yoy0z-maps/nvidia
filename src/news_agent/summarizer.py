"""뉴스 요약: 기사별 요약(2~3문장), 이슈 요약(2~3문단), 핵심어(3~7개)를 LLM 1회 호출로 만든다.

LLM 출력은 그대로 믿지 않고 검사한다. 문단 수·핵심어 수처럼 고칠 수 있는 것은 고치고,
기사 요약이 빠지는 등 고칠 수 없는 것은 한 번 다시 요청한 뒤에도 안 되면 SummaryError를 낸다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from src.contracts import ISSUE_SUMMARY_PARAGRAPHS, split_paragraphs
from src.llm import load_prompt
from src.news_agent.news import COMPANY_TERMS, NewsArticle

log = logging.getLogger(__name__)

KEYWORDS_RANGE = (3, 7)
MAX_ARTICLE_CHARS = 3000   # 기사 하나당 LLM에 보내는 본문 상한 (본문 추출 상한 4,000자보다 작게)
ARTICLE_SENTENCES = (2, 3)


class ArticleSummary(BaseModel):
    id: int
    summary: str


class NewsDigestOutput(BaseModel):
    """LLM 구조화 출력 형식.

    이슈 요약을 문단 목록(list)으로 받으면 모델이 전체를 한 항목에 넣는 일이 있어(2026-11-27 실측),
    문단마다 필드를 따로 둔다. 셋째 문단은 비워도 된다.
    """
    articles: list[ArticleSummary]
    issue_paragraph_1: str
    issue_paragraph_2: str
    issue_paragraph_3: str
    keywords: list[str]

    @property
    def issue_paragraphs(self) -> list[str]:
        return [self.issue_paragraph_1, self.issue_paragraph_2, self.issue_paragraph_3]


class SummaryError(RuntimeError):
    pass


@dataclass
class NewsDigest:
    summaries: list[str]                      # articles와 같은 순서
    issue_summary: str                        # 빈 줄로 구분된 2~3문단
    keywords: list[str]
    warnings: list[str] = field(default_factory=list)


def format_articles(articles: list[NewsArticle]) -> str:
    blocks = []
    for i, a in enumerate(articles):
        text = a.text_for_summary[:MAX_ARTICLE_CHARS]
        kind = "본문" if a.body else "발췌"
        blocks.append(f"[{i}] {a.published_at[:16]} {a.title}\n출처: {a.url}\n{kind}: {text}")
    return "\n\n".join(blocks)


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s.strip()])


def _clean_keywords(raw: list[str], field_name: str, exclude: tuple[str, ...] = COMPANY_TERMS) -> list[str]:
    """공백·기호 정리, 중복 제거, 분야명과 입력 기업명 제외, 최대 7개."""
    skip = {field_name.lower(), *(e.lower() for e in exclude)}
    out: list[str] = []
    for k in raw:
        k = " ".join(k.split()).strip(" ,.#")
        if not k or k.lower() in skip or k.lower() in (x.lower() for x in out):
            continue
        out.append(k)
    return out[: KEYWORDS_RANGE[1]]


def _fit_paragraphs(paragraphs: list[str]) -> list[str]:
    """2~3문단으로 맞춘다. 문단 안의 줄바꿈은 공백으로 바꾸고, 4문단 이상이면 뒤 문단들을 마지막 문단에 합친다."""
    paras: list[str] = []
    for p in paragraphs:
        if p and p.strip():  # 한 필드 안에 빈 줄로 여러 문단을 넣은 경우도 나눈다
            paras += [" ".join(x.split()) for x in split_paragraphs(p)]
    hi = ISSUE_SUMMARY_PARAGRAPHS[1]
    if len(paras) > hi:
        paras = paras[: hi - 1] + [" ".join(paras[hi - 1:])]
    return paras


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def unsupported_numbers(text: str, source: str) -> list[str]:
    """text에 나온 숫자 중 source(기사 원문)에 없는 것. 천 단위 쉼표는 무시하고 비교한다."""
    # 숫자 단위로 비교한다 ('99'가 '2099' 안에 있다고 해서 원문에 있는 것으로 보지 않는다)
    src_numbers = {m.replace(",", "") for m in _NUMBER.findall(source)}
    out = []
    for n in _NUMBER.findall(text):
        if n.replace(",", "") not in src_numbers and n not in out:
            out.append(n)
    return out


# 금액 단위. 긴 것부터 맞춰야 '십억원'이 '억원'으로 잘못 읽히지 않는다.
_MONEY_UNITS = ("백만달러", "천달러", "달러", "십억원", "백만원", "천원", "조원", "억원", "만원", "원")
_UNIT_ALT = "|".join(_MONEY_UNITS)
_NUM_UNIT = re.compile(rf"(\d+(?:[.,]\d+)*)\s*({_UNIT_ALT})")
_UNIT_HEADER = re.compile(rf"단위\s*[:：]\s*({_UNIT_ALT})")


def _source_units(source: str) -> dict[str, set[str]]:
    """원문의 숫자 → 그 숫자에 붙은 금액 단위들. 숫자 바로 뒤 단위가 없으면 앞쪽 가장 가까운 '(단위: …)'를 쓴다."""
    headers = [(m.start(), m.group(1)) for m in _UNIT_HEADER.finditer(source)]
    out: dict[str, set[str]] = {}
    for m in _NUMBER.finditer(source):
        key = m.group(0).replace(",", "")
        tail = _NUM_UNIT.match(source, m.start())
        if tail and tail.group(1) == m.group(0):
            unit = tail.group(2)
        else:
            before = [u for pos, u in headers if pos < m.start()]
            unit = before[-1] if before else None
        if unit:
            out.setdefault(key, set()).add(unit)
    return out


def mismatched_units(text: str, source: str) -> list[str]:
    """text의 '숫자+금액 단위' 중 원문에서 같은 숫자가 다른 단위로만 나오는 것 (예: 십억원 → 억원)."""
    units = _source_units(source)
    out = []
    for num, unit in _NUM_UNIT.findall(text):
        found = units.get(num.replace(",", ""))
        if found and unit not in found:
            label = f"{num}{unit} (원문 단위: {', '.join(sorted(found))})"
            if label not in out:
                out.append(label)
    return out


def check_digest(out: NewsDigestOutput, n_articles: int, field_name: str,
                 source: str | None = None) -> tuple[NewsDigest | None, list[str]]:
    """(결과, 문제 목록). 결과가 None이면 다시 요청해야 하는 문제가 있다."""
    problems: list[str] = []
    by_id: dict[int, str] = {}
    for a in out.articles:
        s = " ".join(a.summary.split())
        if 0 <= a.id < n_articles and s and a.id not in by_id:
            by_id[a.id] = s
    missing = [i for i in range(n_articles) if i not in by_id]
    if missing:
        problems.append(f"기사 요약 누락: {missing}")

    paras = _fit_paragraphs(out.issue_paragraphs)
    lo, hi = ISSUE_SUMMARY_PARAGRAPHS
    if not lo <= len(paras) <= hi:
        problems.append(f"이슈 요약 문단 수 {len(paras)}")

    keywords = _clean_keywords(out.keywords, field_name)
    if len(keywords) < KEYWORDS_RANGE[0]:
        problems.append(f"핵심어 {len(keywords)}개")

    if problems:
        return None, problems

    warnings = []
    lo_s, hi_s = ARTICLE_SENTENCES
    off = [i for i, s in by_id.items() if not lo_s <= count_sentences(s) <= hi_s + 1]
    if off:
        warnings.append(f"2~3문장을 벗어난 기사 요약: {sorted(off)}")
    if source is not None:
        written = "\n".join([*by_id.values(), *paras])
        made_up = unsupported_numbers(written, source)
        if made_up:
            warnings.append(f"기사 원문에 없는 숫자: {made_up}")
        wrong_unit = mismatched_units(written, source)
        if wrong_unit:
            warnings.append(f"단위가 기사 원문과 다른 금액: {wrong_unit}")
    return NewsDigest(summaries=[by_id[i] for i in range(n_articles)], issue_summary="\n\n".join(paras),
                      keywords=keywords, warnings=warnings), []


def summarize_news(llm, articles: list[NewsArticle], company: str, field_name: str,
                   max_attempts: int = 2) -> NewsDigest:
    if not articles:
        raise SummaryError("요약할 기사가 없습니다.")
    instructions = load_prompt("news_summary", company=company, field=field_name, n=len(articles))
    data = format_articles(articles)
    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        out = llm.parse(instructions, data, NewsDigestOutput, purpose="news_summary")
        digest, problems = check_digest(out, len(articles), field_name, source=data)
        if digest is not None:
            for w in digest.warnings:
                log.info("뉴스 요약 경고: %s", w)
            return digest
        log.warning("뉴스 요약 결과 문제 (%d/%d): %s", attempt, max_attempts, problems)
    raise SummaryError(f"뉴스 요약 실패: {problems}")

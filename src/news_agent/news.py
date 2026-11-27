"""뉴스 수집: 최근 7일 기사 → 정제·중복 제거 → 서로 다른 사건 10개 선택 → 본문 확보.

- 같은 사건을 여러 언론사가 다르게 쓴 제목은 글자·의미 유사도로 구분되지 않아(2026-11-27 실측),
  사건 단위 묶기는 LLM 한 번 호출로 한다 (prompts/news_event_selection.md).
- 서로 다른 사건이 10개보다 적으면 남은 자리는 최신 기사로 채운다.
- 결과(NewsArticle)는 5단계 요약의 입력이다. 기사별 summary는 이 단계에서 만들지 않는다.
"""
from __future__ import annotations

import datetime as dt
import html
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime

from pydantic import BaseModel

from src.contracts import MAX_NEWS, NewsItem
from src.llm import load_prompt
from src.news_agent.naver_client import is_naver_news_url

log = logging.getLogger(__name__)

WINDOW_DAYS = 7              # 최근 며칠 기사를 볼지
MAX_PAGES = 10               # 검색어당 최대 페이지 (display 100 기준 1,000건)
MAX_START = 1000             # 네이버 검색 API의 start 최댓값
MAX_EVENT_CANDIDATES = 150   # 사건 묶기에 LLM으로 보낼 최대 기사 수

# 입력 기업 표기 (SK하이닉스 고정). 제목에는 '하이닉스'만 쓰는 경우가 많다.
COMPANY_TERMS = ("SK하이닉스", "하이닉스", "SK hynix", "hynix")
TITLE_SIMILARITY = 0.9    # 정규화한 제목의 글자 유사도가 이 이상이면 같은 사건
WORD_OVERLAP = 0.8        # 또는 제목 단어가 이 비율 이상 겹치면 같은 사건
# 주가·시황 기사 제목 표현. 이런 기사는 분야 사업 분석에 도움이 적어 맨 뒤 순위로 미룬다.
STOCK_TITLE_TERMS = ("특징주", "목표가", "목표주가", "순매수", "순매도", "주가", "증시", "코스피", "코스닥",
                     "시가총액", "시총", "상한가", "하한가", "급등", "급락", "폭등", "폭락", "PER", "수익률", "종목",
                     "매수", "매도", "베팅", "외국인", "개미", "주식", "재테크", "머니팁")
_TAG = re.compile(r"<[^>]+>")
_BRACKET = re.compile(r"^\s*[\[【(<][^\]】)>]{1,15}[\]】)>]\s*")   # [단독], 【속보】 등 머리표


@dataclass
class NewsArticle:
    title: str
    url: str            # 네이버 link (네이버 뉴스가 아니면 언론사 URL과 같다)
    origin_url: str     # 언론사 원문 URL
    published_at: str   # ISO 8601, 시간대 포함
    description: str    # API가 주는 짧은 발췌
    body: str = ""      # 네이버 뉴스 본문. 못 가져오면 빈 문자열
    title_has_field: bool = False     # 제목에 분야(또는 동의어)가 있음
    title_has_company: bool = False   # 제목에 SK하이닉스 표기가 있음
    event: str = ""                   # 사건 단위 선택에서 붙은 사건 이름 (규칙 기반으로 고르면 빈 문자열)

    @property
    def body_available(self) -> bool:
        """네이버 뉴스 기사라 본문을 가져올 수 있는지."""
        return is_naver_news_url(self.url)

    @property
    def is_stock_news(self) -> bool:
        return is_stock_title(self.title)

    @property
    def text_for_summary(self) -> str:
        """요약에 쓸 텍스트. 본문이 없으면 발췌를 쓴다."""
        return self.body or self.description

    def to_news_item(self, rank: int, summary: str = "") -> NewsItem:
        return NewsItem(rank=rank, title=self.title, url=self.url, origin_url=self.origin_url,
                        published_at=self.published_at, summary=summary)


# ---------------------------------------------------------------------------
# 정제
# ---------------------------------------------------------------------------
def strip_html(text: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", text or "")).split())


def to_iso(pub_date: str) -> str | None:
    """'Fri, 27 Nov 2026 12:18:00 +0900' → '2026-11-27T12:18:00+09:00'. 해석 실패 시 None."""
    try:
        d = parsedate_to_datetime(pub_date)
    except (TypeError, ValueError):
        return None
    if d is None or d.tzinfo is None:
        return None
    return d.isoformat()


def _ts(a: NewsArticle) -> dt.datetime:
    return dt.datetime.fromisoformat(a.published_at)


def normalize_title(title: str) -> str:
    t = title
    while True:
        new = _BRACKET.sub("", t)
        if new == t:
            break
        t = new
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", t).lower()


def _contains(text: str, terms: Iterable[str]) -> bool:
    low = text.lower()
    return any(t.lower() in low for t in terms if t)


def parse_items(items: Iterable[dict], field_terms: Iterable[str],
                company_terms: Iterable[str] = COMPANY_TERMS) -> list[NewsArticle]:
    """API items → NewsArticle. 기업과 분야가 제목·발췌 어디에도 없는 기사, 날짜가 없는 기사는 버린다."""
    field_terms, company_terms = list(field_terms), list(company_terms)
    out: list[NewsArticle] = []
    for it in items:
        title = strip_html(it.get("title", ""))
        desc = strip_html(it.get("description", ""))
        link = (it.get("link") or "").strip()
        origin = (it.get("originallink") or "").strip()
        published = to_iso(it.get("pubDate", ""))
        if not title or not (link or origin) or published is None:
            continue
        both = f"{title} {desc}"
        if not (_contains(both, field_terms) and _contains(both, company_terms)):
            continue
        out.append(NewsArticle(title=title, url=link or origin, origin_url=origin, published_at=published,
                               description=desc,
                               title_has_field=_contains(title, field_terms),
                               title_has_company=_contains(title, company_terms)))
    return out


def dedupe(articles: list[NewsArticle]) -> list[NewsArticle]:
    """최신 기사부터 보면서 URL이 같거나 제목이 비슷한 기사를 버린다."""
    kept: list[NewsArticle] = []
    seen_urls: set[str] = set()
    seen_titles: list[tuple[str, set[str]]] = []
    for a in sorted(articles, key=_ts, reverse=True):
        urls = {u for u in (a.url, a.origin_url) if u}
        if urls & seen_urls:
            continue
        norm, words = normalize_title(a.title), title_words(a.title)
        if any(same_story(norm, words, n, w) for n, w in seen_titles):
            continue
        kept.append(a)
        seen_urls |= urls
        seen_titles.append((norm, words))
    return kept


def title_words(title: str) -> set[str]:
    t = title
    while (new := _BRACKET.sub("", t)) != t:
        t = new
    return {w.lower() for w in re.findall(r"[0-9a-zA-Z가-힣]+", t)}


def same_story(norm_a: str, words_a: set[str], norm_b: str, words_b: set[str]) -> bool:
    """같은 사건을 다룬 기사 제목인지. 'SK하이닉스 HBM …'처럼 앞부분이 같은 제목이 많아 기준을 높게 둔다."""
    if norm_a == norm_b:
        return True
    if SequenceMatcher(None, norm_a, norm_b).ratio() >= TITLE_SIMILARITY:
        return True
    union = words_a | words_b
    return len(union) >= 3 and len(words_a & words_b) / len(union) >= WORD_OVERLAP


def is_stock_title(title: str) -> bool:
    return any(t.lower() in title.lower() for t in STOCK_TITLE_TERMS)


def selection_priority(a: NewsArticle) -> tuple[bool, bool, bool, bool]:
    """높을수록 먼저 고른다.

    ① 주가·시황 기사가 아님 ② 제목에 분야가 나옴 ③ 본문을 가져올 수 있는 네이버 뉴스 ④ 제목에 기업이 나옴
    """
    return (not a.is_stock_news, a.title_has_field, a.body_available, a.title_has_company)


def select_latest(articles: list[NewsArticle], limit: int = MAX_NEWS) -> list[NewsArticle]:
    """우선순위(selection_priority)가 높은 기사부터, 같은 순위 안에서는 최신순으로 limit개를 고른다.

    주가·시황 기사는 다른 기사가 모자랄 때만 들어간다. 결과는 최신순으로 정렬한다.
    """
    ranked = sorted(articles, key=lambda a: (selection_priority(a), _ts(a)), reverse=True)
    return sorted(ranked[:limit], key=_ts, reverse=True)


# ---------------------------------------------------------------------------
# 사건 단위 선택 (LLM)
# ---------------------------------------------------------------------------
class NewsEvent(BaseModel):
    event: str
    ids: list[int]


class NewsEventList(BaseModel):
    events: list[NewsEvent]


def event_candidates(articles: list[NewsArticle], cap: int = MAX_EVENT_CANDIDATES) -> list[NewsArticle]:
    """LLM에 보낼 후보: 제목에 분야가 있고 주가 기사가 아닌 기사, 최신순 cap개."""
    good = [a for a in articles if a.title_has_field and not a.is_stock_news]
    return sorted(good, key=_ts, reverse=True)[:cap]


def format_candidates(cands: list[NewsArticle]) -> str:
    return "\n".join(f"[{i}] {a.published_at[:16]} {a.title} | {a.description}" for i, a in enumerate(cands))


def representatives(cands: list[NewsArticle], events: NewsEventList, limit: int) -> list[NewsArticle]:
    """사건마다 대표 기사 1건(네이버 본문 기사 우선, 그다음 최신)을 사건 순서대로 limit개까지 고른다.

    잘못된 번호는 버리고, 앞 사건에 이미 들어간 번호는 뒤 사건에서 뺀다(같은 기사가 두 번 뽑히지 않게).
    """
    used: set[int] = set()
    picked: list[NewsArticle] = []
    for ev in events.events:
        ids = [i for i in dict.fromkeys(ev.ids) if 0 <= i < len(cands) and i not in used]
        if not ids:
            continue
        used |= set(ids)
        rep = cands[max(ids, key=lambda i: (cands[i].body_available, _ts(cands[i])))]
        rep.event = ev.event.strip()
        picked.append(rep)
        if len(picked) >= limit:
            break
    return picked


def select_distinct_events(articles: list[NewsArticle], llm, limit: int = MAX_NEWS,
                           days: int = WINDOW_DAYS) -> list[NewsArticle]:
    """서로 다른 사건의 기사를 최대한 고르고, 모자라면 최신 기사로 채운다. 결과는 최신순.

    llm이 없거나 호출이 실패하면 규칙 기반 선택(select_latest)으로 대신한다.
    """
    picked: list[NewsArticle] = []
    cands = event_candidates(articles)
    if llm is not None and cands:
        try:
            instructions = load_prompt("news_event_selection", days=days, max_events=limit + 2)
            events = llm.parse(instructions, format_candidates(cands), NewsEventList, purpose="news_event_selection")
            picked = representatives(cands, events, limit)
        except Exception as exc:  # 선택 실패가 전체 실패로 번지지 않게 한다
            log.warning("사건 단위 선택 실패 - 규칙 기반 선택으로 대신함: %s", exc)
            picked = []
    if not picked:
        return select_latest(articles, limit)
    if len(picked) < limit:
        # 서로 다른 사건이 모자라면 최신 기사로 채운다 (같은 우선순위 규칙)
        chosen = {id(a) for a in picked}
        picked += select_latest([a for a in articles if id(a) not in chosen], limit - len(picked))
    return sorted(picked, key=_ts, reverse=True)


# ---------------------------------------------------------------------------
# 전체
# ---------------------------------------------------------------------------
def build_queries(company: str, field: str, synonyms: Iterable[str] = ()) -> list[str]:
    queries = [f"{company} {field}"]
    queries += [f"{company} {s}" for s in synonyms if s and s.strip() and s.strip() != field]
    return list(dict.fromkeys(queries))


def search_window(client, query: str, since: dt.datetime, display: int = 100,
                  max_pages: int = MAX_PAGES) -> list[dict]:
    """최신순으로 페이지를 넘기며 since 이후 기사를 모은다. 네이버 API는 start 1000까지만 허용한다."""
    raw: list[dict] = []
    for page in range(max_pages):
        start = 1 + page * display
        if start > MAX_START:
            break
        batch = client.search(query, display=display, start=start, sort="date")
        raw += batch
        oldest = to_iso(batch[-1].get("pubDate", "")) if batch else None
        if len(batch) < display or (oldest and dt.datetime.fromisoformat(oldest) < since):
            break
    return raw


def within(articles: list[NewsArticle], since: dt.datetime, until: dt.datetime) -> list[NewsArticle]:
    return [a for a in articles if since <= _ts(a) <= until]


def collect_news(client, company: str, field: str, synonyms: Iterable[str] = (), limit: int = MAX_NEWS,
                 llm=None, days: int = WINDOW_DAYS, now: dt.datetime | None = None, display: int = 100,
                 max_pages: int = MAX_PAGES, fetch_body: bool = True) -> list[NewsArticle]:
    """최근 days일 뉴스 → 정제·중복 제거 → 서로 다른 사건 limit개(모자라면 최신 기사로 채움) → 본문.

    llm을 주면 사건 단위로 겹치지 않게 고르고, 없으면 규칙 기반(우선순위·최신순)으로 고른다.
    """
    now = now or dt.datetime.now(dt.timezone.utc).astimezone()
    since = now - dt.timedelta(days=days)
    synonyms = list(synonyms)
    raw: list[dict] = []
    for q in build_queries(company, field, synonyms):
        raw += search_window(client, q, since, display=display, max_pages=max_pages)
    pool = within(dedupe(parse_items(raw, [field, *synonyms])), since, now)
    articles = select_distinct_events(pool, llm, limit, days)
    oldest = min((_ts(a) for a in pool), default=None)
    log.info("뉴스 %d건 수집 → %d일 범위 %d건 (가장 오래된 기사 %s) → 선택 %d건, 사건 %d개 (분야=%s)",
             len(raw), days, len(pool), oldest, len(articles), sum(1 for a in articles if a.event), field)
    if fetch_body:
        for a in articles:
            a.body = client.fetch_body(a.url) or ""
    return articles

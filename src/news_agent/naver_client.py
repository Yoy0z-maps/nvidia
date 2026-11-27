"""네이버 뉴스 검색 API와 네이버 뉴스 본문 가져오기.

API: GET https://openapi.naver.com/v1/search/news.json
     헤더 X-Naver-Client-Id / X-Naver-Client-Secret, 파라미터 query, display(최대 100), start, sort(date|sim).
     응답 items[]: title, originallink, link, description, pubDate (RFC 822). title/description에는 <b> 태그와
     HTML 엔티티가 섞여 있다. (2026-11-27 실제 호출로 확인)
본문: link가 n.news.naver.com 기사일 때만 #dic_area 영역을 읽는다. 그 외 언론사 페이지는 가져오지 않는다.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx
from lxml import html as lxml_html

log = logging.getLogger(__name__)

SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"
NAVER_NEWS_HOSTS = {"n.news.naver.com", "news.naver.com", "m.news.naver.com"}
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) market-ripple/0.1"
MAX_DISPLAY = 100
MAX_BODY_CHARS = 4000


class NaverError(Exception):
    def __init__(self, status: int | str, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status


class NaverAuthError(NaverError):
    pass


class NaverRateLimit(NaverError):
    pass


def is_naver_news_url(url: str) -> bool:
    return urlparse(url).netloc.lower() in NAVER_NEWS_HOSTS


def extract_naver_article_body(page_html: str) -> str | None:
    """네이버 뉴스 기사 HTML에서 본문(#dic_area) 텍스트를 뽑는다. 없으면 None."""
    try:
        tree = lxml_html.fromstring(page_html)
    except (ValueError, lxml_html.etree.ParserError):
        return None
    nodes = tree.xpath('//*[@id="dic_area"]') or tree.xpath('//*[@id="newsct_article"]')
    if not nodes:
        return None
    node = nodes[0]
    for bad in node.xpath('.//script|.//style|.//*[contains(@class,"img_desc")]'
                          '|.//*[contains(@class,"end_photo_org")]'):
        bad.drop_tree()  # remove()와 달리 뒤에 이어지는 텍스트(tail)는 남긴다
    lines = [" ".join(t.split()) for t in node.itertext()]
    text = "\n".join(t for t in lines if t)
    return text[:MAX_BODY_CHARS] or None


class NaverNewsClient:
    def __init__(self, client_id: str, client_secret: str, *, timeout: float = 20.0, max_retries: int = 2,
                 transport: httpx.BaseTransport | None = None, min_interval: float = 0.1):
        if not client_id or not client_secret:
            raise NaverAuthError("NO_KEY", "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다.")
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        self.call_count = 0
        self._api = httpx.Client(timeout=timeout, transport=transport,
                                 headers={"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret})
        # 본문 페이지 요청에는 API 키를 보내지 않는다
        self._web = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True,
                                 headers={"User-Agent": BROWSER_UA})

    def close(self) -> None:
        self._api.close()
        self._web.close()

    def __enter__(self) -> "NaverNewsClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def search(self, query: str, display: int = 30, start: int = 1, sort: str = "date") -> list[dict]:
        """검색 결과 items 원본을 돌려준다."""
        if sort not in ("date", "sim"):
            raise ValueError(f"sort는 date 또는 sim이어야 합니다: {sort!r}")
        params = {"query": query, "display": max(1, min(display, MAX_DISPLAY)), "start": start, "sort": sort}
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.call_count += 1
            try:
                resp = self._api.get(SEARCH_URL, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                log.warning("네이버 검색 요청 실패 (%s) - 재시도 [%d/%d]", type(exc).__name__, attempt + 1,
                            self.max_retries)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                return resp.json().get("items", [])
            message = _error_message(resp)
            if resp.status_code in (401, 403):
                raise NaverAuthError(resp.status_code, message)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = (NaverRateLimit if resp.status_code == 429 else NaverError)(resp.status_code, message)
                log.warning("네이버 검색 HTTP %s - 재시도 [%d/%d]", resp.status_code, attempt + 1, self.max_retries)
                time.sleep(2 ** attempt)
                continue
            raise NaverError(resp.status_code, message)
        if isinstance(last, NaverError):
            raise last
        raise NaverError("NETWORK", f"네트워크 오류: {last}")

    def fetch_body(self, url: str) -> str | None:
        """네이버 뉴스 기사면 본문을, 아니거나 실패하면 None을 돌려준다 (예외를 던지지 않는다)."""
        if not is_naver_news_url(url):
            return None
        self._throttle()
        try:
            resp = self._web.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            log.info("기사 본문 요청 실패 (%s): %s", type(exc).__name__, url)
            return None
        if resp.status_code != 200:
            log.info("기사 본문 HTTP %s: %s", resp.status_code, url)
            return None
        return extract_naver_article_body(resp.text)


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        return f"{data.get('errorCode', '')} {data.get('errorMessage', '')}".strip()
    except ValueError:
        return resp.text[:200]

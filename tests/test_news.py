"""뉴스 수집 테스트. API 응답은 실제 형식만 흉내 낸 가상 데이터다.

실제 네이버 호출은 `uv run pytest -m live tests/test_news.py`로 따로 실행한다.
"""
from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from src.news_agent.naver_client import (NaverAuthError, NaverNewsClient, NaverRateLimit,
                                         extract_naver_article_body, is_naver_news_url)
from src.news_agent.news import (build_queries, collect_news, dedupe, event_candidates, is_stock_title,
                                 normalize_title, parse_items, search_window, select_distinct_events,
                                 select_latest, strip_html, to_iso)
from tests.fake_llm import FakeLLM


def item(title, desc="가상 발췌 <b>HBM</b>", link="https://example.com/a", origin="",
         pub="Fri, 27 Nov 2099 12:00:00 +0900"):
    return {"title": title, "description": desc, "link": link, "originallink": origin or link, "pubDate": pub}


def pub(hour: int, day: int = 27) -> str:
    return f"Fri, {day} Nov 2099 {hour:02d}:00:00 +0900"


# ---- 정제 ----
def test_strip_html_and_entities():
    assert strip_html("<b>SK하이닉스</b>, R&amp;D &quot;HBM&quot;") == 'SK하이닉스, R&D "HBM"'


def test_to_iso_keeps_timezone():
    assert to_iso("Fri, 27 Nov 2099 12:18:00 +0900") == "2099-11-27T12:18:00+09:00"
    assert to_iso("엉터리") is None


def test_normalize_title_drops_tags_and_symbols():
    assert normalize_title("[단독] 【속보】 SK하이닉스, HBM 증산!") == normalize_title("SK하이닉스 HBM 증산")


def test_parse_items_requires_company_and_field():
    items = [
        item("<b>SK하이닉스</b> <b>HBM</b> 가상 기사"),
        item("가상 기사 제목", desc="하이닉스가 HBM을 가상으로 언급"),     # 발췌에만 → 남지만 제목 매치 아님
        item("SK하이닉스 가상 기사", desc="분야 언급 없음"),               # 분야 없음 → 제외
        item("가상 기사", desc="HBM만 언급"),                              # 기업 없음 → 제외
        item("SK하이닉스 HBM 날짜 없음", pub="잘못된 날짜"),                # 날짜 없음 → 제외
    ]
    got = parse_items(items, ["HBM"])
    assert [(a.title, a.title_has_field, a.title_has_company) for a in got] == [
        ("SK하이닉스 HBM 가상 기사", True, True), ("가상 기사 제목", False, False)]
    assert got[0].published_at == "2099-11-27T12:00:00+09:00"


def test_dedupe_by_url_and_similar_title():
    arts = parse_items([
        item("SK하이닉스 HBM 가상 증산 발표", link="https://example.com/1", pub=pub(12)),
        item("[단독] SK하이닉스, HBM 가상 증산 발표", link="https://example.com/2", pub=pub(11)),  # 비슷한 제목
        item("SK하이닉스 HBM 전혀 다른 가상 소식", link="https://example.com/1", pub=pub(10)),       # 같은 URL
        item("SK하이닉스 HBM 전혀 다른 가상 소식", link="https://example.com/3", pub=pub(9)),
    ], ["HBM"])
    assert [a.url for a in dedupe(arts)] == ["https://example.com/1", "https://example.com/3"]


def test_titles_sharing_prefix_are_not_merged():
    arts = parse_items([
        item("SK하이닉스 HBM 증산 관련 가상 기사", link="https://example.com/1", pub=pub(12)),
        item("SK하이닉스 HBM 수율 관련 가상 기사", link="https://example.com/2", pub=pub(11)),
        item("SK하이닉스, HBM 증산 관련 가상 기사", link="https://example.com/3", pub=pub(10)),  # 1과 같은 사건
    ], ["HBM"])
    assert [a.url for a in dedupe(arts)] == ["https://example.com/1", "https://example.com/2"]


def test_select_latest_prefers_title_hits_then_fills():
    arts = parse_items(
        [item(f"SK하이닉스 HBM 가상 기사 {i}", link=f"https://example.com/t{i}", pub=pub(i)) for i in range(1, 4)]
        + [item(f"가상 기타 {i}", desc="하이닉스 HBM 언급", link=f"https://example.com/d{i}", pub=pub(20 + i))
           for i in range(1, 4)], ["HBM"])
    picked = select_latest(arts, limit=4)
    assert [a.url for a in picked] == ["https://example.com/d3", "https://example.com/t3",
                                       "https://example.com/t2", "https://example.com/t1"]


def test_is_stock_title():
    assert is_stock_title("[특징주] SK하이닉스, 가상 반등")
    assert is_stock_title("SK하이닉스 목표가 가상 상향")
    assert is_stock_title("가상 亞증시 PER 2배대")
    assert is_stock_title("외국인의 SK하닉 가상 상승 베팅")
    assert is_stock_title("가상 95년생…폭락 전 털어낸 타이밍 [머니팁]")
    assert not is_stock_title("SK하이닉스, HBM 가상 증산 발표")


def test_select_priority_field_title_then_naver_body_and_stock_last():
    naver = "https://n.news.naver.com/mnews/article/000/"
    arts = parse_items([
        item("SK하이닉스 HBM 가상 신제품 공개", link="https://example.com/new", pub=pub(23)),     # 최신이지만 발췌만
        item("[특징주] SK하이닉스 HBM 가상 반등", link=naver + "1", pub=pub(22)),                 # 주가 기사
        item("SK하이닉스 HBM 가상 장비 발주", link=naver + "2", pub=pub(10)),                    # 네이버 본문
        item("가상 메모리 동향", desc="하이닉스 HBM 언급", link=naver + "3", pub=pub(9)),          # 네이버, 제목 미포함
    ], ["HBM"])
    # 제목에 분야가 있는 기사가 먼저, 그 안에서는 네이버 본문 기사가 먼저
    assert [a.url for a in select_latest(arts, limit=1)] == [naver + "2"]
    assert [a.url for a in select_latest(arts, limit=2)] == ["https://example.com/new", naver + "2"]
    assert [a.url for a in select_latest(arts, limit=3)] == ["https://example.com/new", naver + "2", naver + "3"]
    # 다른 기사가 모자랄 때만 주가 기사가 들어간다
    assert naver + "1" in [a.url for a in select_latest(arts, limit=4)]


def test_build_queries_dedupes():
    assert build_queries("SK하이닉스", "HBM", ["고대역폭메모리", "HBM", " "]) == ["SK하이닉스 HBM",
                                                                               "SK하이닉스 고대역폭메모리"]


def test_to_news_item_matches_contract():
    a = parse_items([item("SK하이닉스 HBM 가상 기사")], ["HBM"])[0]
    n = a.to_news_item(rank=1, summary="가상 요약")
    assert n.rank == 1 and n.url == "https://example.com/a" and n.published_at.endswith("+09:00")


# ---- 본문 ----
NAVER_PAGE = """<html><body><div id="newsct_article"><article id="dic_area">
가상 첫 문장입니다.<br><span class="end_photo_org"><em class="img_desc">사진 설명</em></span>
<script>var x=1;</script>가상 둘째 문장입니다.</article></div></body></html>"""


def test_extract_naver_article_body():
    assert extract_naver_article_body(NAVER_PAGE) == "가상 첫 문장입니다.\n가상 둘째 문장입니다."
    assert extract_naver_article_body("<html><body>본문 없음</body></html>") is None


def test_is_naver_news_url():
    assert is_naver_news_url("https://n.news.naver.com/mnews/article/000/0000000001")
    assert not is_naver_news_url("https://example.com/news/1")


# ---- 클라이언트 (가짜 HTTP) ----
def _client(handler) -> NaverNewsClient:
    return NaverNewsClient("id", "secret", transport=httpx.MockTransport(handler), min_interval=0, max_retries=1)


def test_search_sends_keys_in_headers_and_params():
    seen = {}

    def handler(req: httpx.Request):
        seen["headers"], seen["params"] = req.headers, dict(req.url.params)
        return httpx.Response(200, json={"items": [item("가상")]})

    items = _client(handler).search("SK하이닉스 HBM", display=500)
    assert items and seen["headers"]["X-Naver-Client-Id"] == "id"
    assert seen["params"] == {"query": "SK하이닉스 HBM", "display": "100", "start": "1", "sort": "date"}


def test_search_auth_error_is_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401, json={"errorCode": "024", "errorMessage": "Authentication failed"})

    with pytest.raises(NaverAuthError):
        _client(handler).search("x")
    assert len(calls) == 1


def test_search_rate_limit_retries_then_raises(monkeypatch):
    monkeypatch.setattr("src.news_agent.naver_client.time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, json={"errorCode": "012", "errorMessage": "limit"})

    with pytest.raises(NaverRateLimit):
        _client(handler).search("x")
    assert len(calls) == 2


def test_fetch_body_only_for_naver_and_never_sends_api_keys():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, text=NAVER_PAGE)

    c = _client(handler)
    assert c.fetch_body("https://example.com/a") is None and not seen
    assert c.fetch_body("https://n.news.naver.com/mnews/article/000/0000000001").startswith("가상 첫 문장")
    assert "X-Naver-Client-Id" not in seen[0].headers


NOW = dt.datetime(2099, 11, 27, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
NAVER = "https://n.news.naver.com/mnews/article/000/"
TOPICS = ["증산", "수율", "단가", "고객사", "패키징", "투자", "인력", "특허", "전력", "냉각", "검증", "출하", "재고",
          "계약", "공정"]


class FakeClient:
    """i번째 기사는 NOW에서 i*12시간 전. page 크기만큼 잘라 돌려준다."""

    def __init__(self, n=15):
        self.n = n
        self.calls = []

    def search(self, q, display=100, start=1, sort="date"):
        self.calls.append((q, start))
        out = []
        for i in range(start - 1, min(start - 1 + display, self.n)):
            t = NOW - dt.timedelta(hours=12 * i)
            out.append(item(f"SK하이닉스 HBM {TOPICS[i % 15]} 관련 가상 기사 {i}", link=f"{NAVER}{i}",
                            pub=t.strftime("%a, %d %b %Y %H:%M:%S +0900")))
        return out

    def fetch_body(self, url):
        return None if url.endswith("/0") else f"가상 본문 {url.rsplit('/', 1)[1]}"


def test_collect_news_rule_based_without_llm():
    fc = FakeClient()
    arts = collect_news(fc, "SK하이닉스", "HBM", synonyms=["고대역폭메모리"], now=NOW)
    assert [q for q, _ in fc.calls] == ["SK하이닉스 HBM", "SK하이닉스 고대역폭메모리"]
    assert len(arts) == 10 and arts[0].url == f"{NAVER}0"
    assert arts[0].body == "" and arts[0].text_for_summary          # 본문 실패 → 발췌 사용
    assert all(a.body for a in arts[1:]) and all(a.event == "" for a in arts)


def test_window_keeps_only_last_seven_days():
    arts = collect_news(FakeClient(n=30), "SK하이닉스", "HBM", now=NOW, limit=30, fetch_body=False)
    urls = {a.url for a in arts}
    # 12시간 간격: i=14는 168시간(정확히 7일) 전이라 포함, i=15는 180시간 전이라 제외
    assert f"{NAVER}14" in urls and f"{NAVER}15" not in urls
    assert min(dt.datetime.fromisoformat(a.published_at) for a in arts) >= NOW - dt.timedelta(days=7)


def test_search_window_stops_paging_at_window_or_api_limit():
    fc = FakeClient(n=5000)
    search_window(fc, "q", NOW - dt.timedelta(days=7), display=10)
    assert [s for _, s in fc.calls] == [1, 11]          # 두 번째 페이지에서 7일 밖으로 넘어감
    fc2 = FakeClient(n=5000)
    search_window(fc2, "q", NOW - dt.timedelta(days=9999), display=100)
    assert [s for _, s in fc2.calls][-1] == 901          # start는 1000을 넘지 않는다


def _pool(n=12):
    return parse_items([item(f"SK하이닉스 HBM {TOPICS[i % 15]} 가상 {i}",
                             link=(f"{NAVER}{i}" if i % 2 else f"https://example.com/{i}"),
                             pub=(NOW - dt.timedelta(hours=i)).strftime("%a, %d %b %Y %H:%M:%S +0900"))
                        for i in range(n)], ["HBM"])


def test_event_selection_picks_one_per_event_prefers_naver_body():
    pool = _pool()
    cands = event_candidates(pool)   # 최신순: 번호 i = i시간 전 기사
    llm = FakeLLM({"news_event_selection": {"events": [
        {"event": "사건A", "ids": [0, 1, 2]},       # 1번(네이버)이 대표
        {"event": "사건B", "ids": [2, 3, 99]},      # 2는 이미 사용, 99는 없는 번호 → 3번
        {"event": "사건C", "ids": [4]},
    ]}})
    arts = select_distinct_events(pool, llm, limit=3)
    assert [(a.url, a.event) for a in arts] == [(f"{NAVER}1", "사건A"), (f"{NAVER}3", "사건B"),
                                                ("https://example.com/4", "사건C")]
    assert "[0]" in llm.calls[0][1] and "7일" not in llm.calls[0][1]   # 데이터에는 기사 목록만 들어간다
    assert cands[0].url == "https://example.com/0"


def test_event_selection_fills_with_latest_when_few_events():
    pool = _pool()
    llm = FakeLLM({"news_event_selection": {"events": [{"event": "하나뿐인 사건", "ids": list(range(12))}]}})
    arts = select_distinct_events(pool, llm, limit=4)
    assert len(arts) == 4
    assert sum(1 for a in arts if a.event) == 1
    assert len({a.url for a in arts}) == 4


def test_event_selection_falls_back_when_llm_fails():
    class BrokenLLM:
        def parse(self, *a, **k):
            raise RuntimeError("API down")

    arts = select_distinct_events(_pool(), BrokenLLM(), limit=5)
    assert len(arts) == 5 and all(a.event == "" for a in arts)


@pytest.mark.live
def test_live_naver_news():
    from config.settings import Settings

    s = Settings.from_env()
    if not (s.naver_client_id and s.naver_client_secret):
        pytest.skip("네이버 키 없음")
    from src.llm import OpenAILLM

    if not s.openai_api_key:
        pytest.skip("OPENAI_API_KEY 없음")
    llm = OpenAILLM(s.openai_api_key, s.llm_model)
    with NaverNewsClient(s.naver_client_id, s.naver_client_secret) as c:
        arts = collect_news(c, "SK하이닉스", "HBM", llm=llm)
    now = dt.datetime.now(dt.timezone.utc)
    assert len(arts) == 10
    assert len({a.url for a in arts}) == 10
    times = [dt.datetime.fromisoformat(a.published_at) for a in arts]
    assert times == sorted(times, reverse=True)
    assert all(now - t <= dt.timedelta(days=7) for t in times)
    assert sum(1 for a in arts if a.event) >= 7, "서로 다른 사건이 너무 적다"
    assert len({a.event for a in arts if a.event}) == sum(1 for a in arts if a.event)
    assert sum(1 for a in arts if a.body) >= 7, "본문 있는 기사가 너무 적다"
    assert not any(a.is_stock_news for a in arts)
    print(json.dumps([{"t": a.title, "event": a.event, "body": len(a.body)} for a in arts], ensure_ascii=False))

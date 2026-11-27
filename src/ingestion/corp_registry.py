"""기업 고유번호 사전 및 상장 시장 확인."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

from lxml import etree

from src.schemas import CorpInfo

log = logging.getLogger(__name__)

MARKET_BY_CLS = {"Y": "KOSPI", "K": "KOSDAQ", "N": "KONEX", "E": "OTHER"}

_SUFFIX_RE = re.compile(r"(주식회사|\(주\)|㈜|\(株\)|co\.?,?\s*ltd\.?|corporation|corp\.?|inc\.?)", re.I)


# 공시에서 한글로 풀어 쓰는 그룹 약칭 (예: '에스케이하이닉스(주)' = 'SK하이닉스')
_HANGUL_ABBR = [("에스케이", "sk"), ("엘지", "lg"), ("케이티", "kt"), ("씨제이", "cj"), ("지에스", "gs"),
                ("엘에스", "ls"), ("에이치디", "hd"), ("케이씨씨", "kcc"), ("디비", "db"), ("에스디아이", "sdi")]
# 그룹명 + 회사명 형태의 개명 (예: '테스나' → '두산테스나')
GROUP_PREFIXES = ("두산", "한솔", "sk", "lg", "삼성", "현대", "한화", "롯데", "cj", "gs", "ls", "hd", "포스코", "효성",
                  "코오롱", "동부", "db", "kt", "신세계", "한진", "금호", "대림", "영풍", "oci", "솔브레인")


def normalize_name(name: str) -> str:
    """기업명 정규화: 유니코드 NFKC, 법인 표기 제거, 공백/기호 제거, 한글 그룹 약칭 → 영문, 소문자."""
    s = unicodedata.normalize("NFKC", name or "")
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"[\s\.\,\-·&'\"()]", "", s).lower()
    for ko, en in _HANGUL_ABBR:
        if s.startswith(ko):
            s = en + s[len(ko):]
            break
    return s


def parse_corp_code_xml(xml_bytes: bytes) -> list[CorpInfo]:
    root = etree.fromstring(xml_bytes, parser=etree.XMLParser(recover=True, huge_tree=True))
    out: list[CorpInfo] = []
    for el in root.iter("list"):
        code = (el.findtext("corp_code") or "").strip()
        if not code:
            continue
        out.append(CorpInfo(
            corp_code=code.zfill(8),
            corp_name=(el.findtext("corp_name") or "").strip(),
            stock_code=(el.findtext("stock_code") or "").strip(),
            modify_date=(el.findtext("modify_date") or "").strip(),
        ))
    return out


class CorpRegistry:
    """전체 기업 사전은 로컬 JSON으로 캐싱한다 (기업 정보만 저장)."""

    def __init__(self, corps: list[CorpInfo]):
        self.corps = corps
        self.by_code = {c.corp_code: c for c in corps}
        self.by_stock = {c.stock_code: c for c in corps if c.stock_code}
        self._by_norm: dict[str, list[CorpInfo]] = {}
        for c in corps:
            self._by_norm.setdefault(normalize_name(c.corp_name), []).append(c)

    # ---- 로딩 ----
    @classmethod
    def load_or_download(cls, cache_path: Path, dart_client=None, refresh: bool = False) -> "CorpRegistry":
        if cache_path.exists() and not refresh:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return cls([CorpInfo(**d) for d in data["corps"]])
        if dart_client is None:
            raise RuntimeError("기업 사전 캐시가 없고 DART 클라이언트도 없습니다.")
        xml_bytes = dart_client.download_corp_codes()
        corps = parse_corp_code_xml(xml_bytes)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"count": len(corps), "corps": [c.model_dump() for c in corps]},
                                         ensure_ascii=False), encoding="utf-8")
        log.info("기업 고유번호 사전 저장: %d개 (종목코드 보유 %d개)", len(corps),
                 sum(1 for c in corps if c.stock_code))
        return cls(corps)

    # ---- 검색 ----
    def get_by_stock_code(self, stock_code: str) -> CorpInfo | None:
        return self.by_stock.get(stock_code.zfill(6))

    def resolve_name(self, name: str, listed_only: bool = True) -> tuple[CorpInfo | None, str]:
        """기업명 → CorpInfo. 정확히 식별할 수 없으면 (None, 사유)."""
        norm = normalize_name(name)
        if not norm:
            return None, "빈 기업명"
        exact = [c for c in self._by_norm.get(norm, []) if c.stock_code or not listed_only]
        if len(exact) == 1:
            return exact[0], "정규화 명칭 정확 일치"
        if len(exact) > 1:
            return None, f"동일 명칭 기업 {len(exact)}개 - 모호함"
        # 부분 일치는 '후보 이름이 사전 이름의 앞부분'이고 길이 차이가 작을 때만, 유일할 때만 허용
        # (예: 'SK하이닉' → 'SK하이닉스'는 허용 / '한솔오리온텍' → '오리온'처럼 사전 이름이 후보 안에 들어간 경우는 거부)
        partial = [c for k, cs in self._by_norm.items()
                   if len(norm) >= 2 and ((len(norm) >= 3 and k.startswith(norm) and len(norm) / len(k) >= 0.75)
                                          or (k.endswith(norm) and k[:-len(norm)] in GROUP_PREFIXES))
                   for c in cs if c.stock_code or not listed_only]
        if len(partial) == 1:
            return partial[0], "유일한 부분 일치"
        if len(partial) > 1:
            return None, f"부분 일치 후보 {len(partial)}개 - 모호함: " + ", ".join(c.corp_name for c in partial[:5])
        return None, "기업 사전에 없음"


def resolve_unlisted(registry: "CorpRegistry", name: str) -> CorpInfo | None:
    """DART 기업 사전에 정확히 한 건만 있는 비상장 기업 (정규화 명칭 정확 일치만)."""
    cands = [c for c in registry._by_norm.get(normalize_name(name), []) if not c.stock_code]
    return cands[0] if len(cands) == 1 else None


class MarketTypeResolver:
    """OpenDART company.json의 corp_cls로 KOSPI/KOSDAQ 상장 여부를 확인한다 (캐싱)."""

    def __init__(self, dart_client, storage):
        self.dart = dart_client
        self.storage = storage

    def market_type(self, corp_code: str) -> str | None:
        cached = self.storage.cache_get("market_type", corp_code)
        if cached is not None:
            return cached
        info = self.dart.company_info(corp_code)
        market = MARKET_BY_CLS.get(info.get("corp_cls", ""), None)
        self.storage.cache_set("market_type", corp_code, market)
        return market

    def is_kospi_or_kosdaq(self, corp_code: str) -> bool:
        return self.market_type(corp_code) in {"KOSPI", "KOSDAQ"}

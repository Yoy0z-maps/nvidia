"""기업명 → 상장기업 식별."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.ingestion.corp_registry import CorpRegistry, MarketTypeResolver, normalize_name
from src.storage import Storage

log = logging.getLogger(__name__)


@dataclass
class ResolvedCompany:
    corp_code: str
    corp_name: str
    stock_code: str
    market_type: str | None
    in_knowledge_base: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class EntityResolver:
    def __init__(self, storage: Storage, registry: CorpRegistry | None = None,
                 market: MarketTypeResolver | None = None):
        self.storage = storage
        self.registry = registry
        self.market = market
        self._cache: dict[str, tuple[ResolvedCompany | None, str]] = {}

    def resolve(self, name: str, require_listed_market: bool = True) -> tuple[ResolvedCompany | None, str]:
        """정확히 식별할 수 없으면 (None, 사유)를 반환한다."""
        key = f"{normalize_name(name)}|{require_listed_market}"
        if key in self._cache:
            return self._cache[key]
        result = self._resolve(name, require_listed_market)
        self._cache[key] = result
        return result

    def _resolve(self, name: str, require_listed_market: bool) -> tuple[ResolvedCompany | None, str]:
        norm = normalize_name(name)
        # 1) 이미 수집한 기업
        for c in self.storage.list_companies():
            if normalize_name(c["corp_name"]) == norm or c["stock_code"] == name.strip():
                return ResolvedCompany(c["corp_code"], c["corp_name"], c["stock_code"], c["market_type"], True), \
                    "지식 기반 기업"
        if self.registry is None:
            return None, "기업 사전 없음 (지식 기반 외 기업 식별 불가)"
        corp, reason = self.registry.resolve_name(name, listed_only=True)
        if corp is None:
            return None, reason
        market = None
        if self.market is not None:
            try:
                market = self.market.market_type(corp.corp_code)
            except Exception as exc:
                return None, f"상장시장 확인 실패: {exc}"
            if require_listed_market and market not in ("KOSPI", "KOSDAQ"):
                return None, f"KOSPI/KOSDAQ 상장사가 아님 (market={market})"
        elif require_listed_market:
            return None, "상장시장 확인 불가 (MarketTypeResolver 없음)"
        return ResolvedCompany(corp.corp_code, corp.corp_name, corp.stock_code, market, False), reason

    def resolve_for_graph(self, name: str) -> tuple[ResolvedCompany | None, str]:
        """그래프 노드용: 상장사로 식별되면 상장사, 아니면 DART에 정확히 등록된 비상장 기업.

        비상장 기업은 공시를 수집하지 않고 주가도 없지만, 공시 원문에 나온 실제 관계 상대방을 그래프에 남긴다.
        """
        r, reason = self.resolve(name)
        if r is not None or self.registry is None:
            return r, reason
        from src.ingestion.corp_registry import resolve_unlisted
        u = resolve_unlisted(self.registry, name)
        if u is None:
            return None, reason
        return ResolvedCompany(u.corp_code, u.corp_name, "", "UNLISTED", False), "DART 등록 비상장 기업"

    def as_dict_resolver(self):
        """validator용: 이름 → dict | None (비상장 기업 포함)"""
        def _f(name: str) -> dict | None:
            r, _ = self.resolve_for_graph(name)
            return r.to_dict() if r else None
        return _f

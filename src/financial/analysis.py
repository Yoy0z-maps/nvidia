"""리서치 결과 기업들에 대한 실제 금융 데이터 분석."""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.financial import metrics as M
from src.financial.market_data import (ChainBenchmarkProvider, CsvBenchmarkProvider, KrxIndexBenchmarkProvider,
                                       MCPKrxPriceProvider)
from src.financial.mcp_client import make_mcp

log = logging.getLogger(__name__)


def default_providers(settings):
    price = MCPKrxPriceProvider(lambda: make_mcp(settings), settings.market_dir)
    bench = ChainBenchmarkProvider([CsvBenchmarkProvider(settings.market_dir / "benchmarks"),
                                    KrxIndexBenchmarkProvider(settings.krx_api_key, settings.market_dir)])
    return price, bench


def analyze(companies: list[dict], event_date: str, price_provider, bench_provider,
            pre_days: int = 400, post_days: int = 100, seed_code: str | None = None) -> dict:
    """companies: [{stock_code, corp_name, market_type}] → 기업별 이벤트 분석.

    가격을 조회하지 못한 기업은 지표를 계산하지 않고 상태만 표시한다.
    """
    ev = dt.date.fromisoformat(event_date)
    start, end = ev - dt.timedelta(days=pre_days), min(ev + dt.timedelta(days=post_days), dt.date.today())
    codes = {c["stock_code"]: c.get("market_type") for c in companies if c.get("stock_code")}
    out: dict = {"event_date": event_date, "period": [start.isoformat(), end.isoformat()], "companies": {},
                 "seed_code": seed_code,
                 "status": {}, "causality_note": M.CAUSALITY_NOTE}
    unknown_market = [k for k, v in codes.items() if v not in ("KOSPI", "KOSDAQ")]
    if unknown_market:
        out["status"]["market_type"] = f"상장시장 미확인 종목 제외: {unknown_market}"
    raw, st = price_provider.get_prices(codes, start, end)
    out["status"]["prices"] = st.__dict__
    benches = {}
    for m in {v for v in codes.values() if v in ("KOSPI", "KOSDAQ")}:
        bdf, bst = bench_provider.get_index(m, start, end)
        out["status"][f"benchmark_{m}"] = bst.__dict__
        benches[m] = M.prepare_prices(bdf)[0] if bst.ok else None
    prepared = {}
    for c in companies:
        code = c.get("stock_code")
        df = raw.get(code)
        if df is None or df.empty:
            out["companies"][code or c["corp_name"]] = {"corp_name": c["corp_name"], "available": False,
                                                       "note": "금융 데이터 없음"}
            continue
        p, issues = M.prepare_prices(df)
        prepared[code] = p
        res = M.event_study(p, benches.get(codes.get(code)), ev)
        res["notes"] = issues + res["notes"]
        out["companies"][code] = {"corp_name": c["corp_name"], "market": codes.get(code), "available": True, **res}
    if len(prepared) >= 2:
        corr = M.return_correlation(prepared, start, ev)
        out["correlation_pre_event"] = corr.round(4).where(pd.notna(corr), None).to_dict()
        if seed_code and seed_code in corr.columns:
            for code, c in out["companies"].items():
                v = corr.loc[code, seed_code] if code in corr.index else None
                if isinstance(c, dict) and code != seed_code and v is not None and pd.notna(v):
                    c["corr_with_seed"] = round(float(v), 4)
    if ev >= dt.date.today() - dt.timedelta(days=1):
        out["status"]["window_note"] = ("기준일이 오늘(또는 최근)이라 이벤트 이후 수익률은 아직 계산할 수 없습니다. "
                                        "최근 5·20·60거래일 수익률과 상관계수만 표시합니다.")
    out["_prices"] = {k: v.reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
                      for k, v in prepared.items()}
    return out


def companies_from_result(res: dict, app) -> list[dict]:
    """분석 결과의 기업 + 가설 인사이트에서 식별된 상장사 (중복 제거, 상장시장 포함)."""
    seen, out = set(), []

    def add(code, name, corp_code, mt=None):
        if not code or code in seen:
            return
        seen.add(code)
        if not mt and corp_code:
            c = app.storage.get_company(corp_code)
            mt = (c or {}).get("market_type")
            if not mt and app.market is not None:
                try:
                    mt = app.market.market_type(corp_code)
                except Exception:
                    mt = None
        out.append({"stock_code": code, "corp_name": name, "market_type": mt})

    for r in res.get("company_results", []):
        add(r.get("stock_code"), r["corp_name"], r.get("corp_code"))
    for ins in res.get("insights", []):
        for c in ins.get("companies", []):
            if c.get("resolved"):
                add(c.get("stock_code"), c["name"], c.get("corp_code"), c.get("market_type"))
    return out


def run_for_result(res: dict, app, event_date: str) -> dict:
    return analyze(companies_from_result(res, app), event_date, *default_providers(app.settings),
                   seed_code=app.settings.seed_stock_code)

"""금융 데이터 Provider.

- MCPKrxPriceProvider: Korea Stock MCP의 get_stock_trade_info (KRX 일별 매매정보, 원시 종가 - 수정주가 아님)
- KrxIndexBenchmarkProvider: KRX Open API 지수 일별시세 (MCP에 지수 Tool이 없어 별도 Provider로 분리, 미검증)
- CsvBenchmarkProvider: 사용자가 내려받은 지수 CSV (data/market/benchmarks/{KOSPI|KOSDAQ}.csv: date,close)

데이터가 없으면 빈 DataFrame과 상태 메시지를 반환한다. 가상 데이터를 만들지 않는다.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd

from src.financial.mcp_client import KoreaStockMCP

log = logging.getLogger(__name__)


def _num(s) -> float | None:
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def weekdays(start: dt.date, end: dt.date) -> list[str]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    return out


@dataclass
class FetchStatus:
    source: str
    ok: bool
    message: str = ""
    adjusted: bool = False
    warnings: list[str] = field(default_factory=list)


class PriceProvider(Protocol):
    def get_prices(self, stock_codes: dict[str, str], start: dt.date, end: dt.date
                   ) -> tuple[dict[str, pd.DataFrame], FetchStatus]: ...


class BenchmarkProvider(Protocol):
    def get_index(self, market: str, start: dt.date, end: dt.date) -> tuple[pd.DataFrame, FetchStatus]: ...


class MCPKrxPriceProvider:
    """basDd 1개당 KRX API 1회 호출 (MCP Tool 설명 기준). 일자·시장 단위로 로컬 캐시한다."""

    def __init__(self, mcp_factory, cache_dir: Path, max_days: int = 140):
        self.mcp_factory = mcp_factory  # () -> KoreaStockMCP (async context manager)
        self.cache_dir = cache_dir
        self.max_days = max_days

    def _cache_path(self, market: str, bas_dd: str) -> Path:
        return self.cache_dir / "krx_trade" / market / f"{bas_dd}.json"

    async def _fetch(self, by_market: dict[str, list[str]], dates: list[str]) -> tuple[dict, list[str]]:
        rows: dict[str, list[dict]] = {}
        errors: list[str] = []
        async with self.mcp_factory() as mcp:
            for market, codes in by_market.items():
                need = []
                for d in dates:
                    p = self._cache_path(market, d)
                    if p.exists():
                        cached = json.loads(p.read_text())
                        if set(codes) <= set(cached["codes"]):
                            for r in cached["rows"]:
                                rows.setdefault(r["ISU_CD"], []).append(r)
                            continue
                    need.append(d)
                for i in range(0, len(need), 10):
                    batch = need[i:i + 10]
                    try:
                        res = await mcp.call("get_stock_trade_info", {"basDdList": batch, "market": market,
                                                                      "codeList": codes})
                    except Exception as exc:
                        errors.append(f"{market} {batch[0]}~{batch[-1]}: {exc}")
                        continue
                    for d in batch:
                        day_rows = res.get(d, []) if isinstance(res, dict) else []
                        p = self._cache_path(market, d)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(json.dumps({"codes": codes, "rows": day_rows}, ensure_ascii=False))
                        for r in day_rows:
                            rows.setdefault(r["ISU_CD"], []).append(r)
        return rows, errors

    def get_prices(self, stock_codes: dict[str, str], start: dt.date, end: dt.date):
        """stock_codes: {종목코드: 'KOSPI'|'KOSDAQ'}"""
        dates = weekdays(start, end)
        if len(dates) > self.max_days:
            dates = dates[-self.max_days:]
        by_market: dict[str, list[str]] = {}
        for code, m in stock_codes.items():
            if m in ("KOSPI", "KOSDAQ"):
                by_market.setdefault(m, []).append(code)
        try:
            rows, errors = asyncio.run(self._fetch(by_market, dates))
        except Exception as exc:
            return {}, FetchStatus("korea-stock-mcp/get_stock_trade_info", False, f"MCP 조회 실패: {exc}")
        out = {}
        for code, rs in rows.items():
            df = pd.DataFrame({
                "date": pd.to_datetime([r["BAS_DD"] for r in rs]),
                "close": [_num(r.get("TDD_CLSPRC")) for r in rs],
                "volume": [_num(r.get("ACC_TRDVOL")) for r in rs],
                "listed_shares": [_num(r.get("LIST_SHRS")) for r in rs],
            }).drop_duplicates("date").sort_values("date")
            out[code] = df
        warnings = ["KRX 일별 매매정보 종가는 수정주가가 아닙니다 - 액면분할 등 기업행동 구간은 수익률이 왜곡될 수 있습니다."]
        for code, df in out.items():
            sh = df["listed_shares"].dropna()
            if len(sh) > 1 and (sh.pct_change().abs() > 0.2).any():
                warnings.append(f"{code}: 상장주식수 20% 이상 변동 감지 - 기업행동(분할/증자 등) 확인 필요")
        ok = bool(out)
        msg = "" if ok else "조회된 가격 데이터 없음"
        if errors:
            msg = (msg + " / " if msg else "") + f"일부 조회 실패 {len(errors)}건: {errors[:2]}"
            if any("401" in e for e in errors):
                msg += (" → KRX 인증 거부(401): 키 미승인 또는 서비스 이용신청 미승인. "
                        "`uv run python -m scripts.check_krx` 로 원인을 확인하세요.")
        return out, FetchStatus("korea-stock-mcp/get_stock_trade_info", ok, msg, adjusted=False, warnings=warnings)


class CsvBenchmarkProvider:
    def __init__(self, directory: Path):
        self.dir = directory

    def get_index(self, market: str, start: dt.date, end: dt.date):
        p = self.dir / f"{market}.csv"
        if not p.exists():
            return pd.DataFrame(), FetchStatus(f"csv:{p}", False, f"벤치마크 파일 없음: {p}")
        df = pd.read_csv(p)
        if not {"date", "close"} <= set(df.columns):
            return pd.DataFrame(), FetchStatus(f"csv:{p}", False, "CSV에 date, close 열이 필요합니다.")
        df["date"] = pd.to_datetime(df["date"].astype(str))
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return df[["date", "close"]], FetchStatus(f"csv:{p.name}", not df.empty, "" if not df.empty else "기간 내 데이터 없음")


class KrxIndexBenchmarkProvider:
    """KRX Open API 지수 일별시세. (미검증: 실제 응답 필드는 KRX 승인 키로 확인 필요)

    엔드포인트: /svc/apis/idx/kospi_dd_trd, /svc/apis/idx/kosdaq_dd_trd (서비스 이용 신청 필요)
    """
    BASE = "https://data-dbg.krx.co.kr/svc/apis/idx"
    ENDPOINT = {"KOSPI": "kospi_dd_trd", "KOSDAQ": "kosdaq_dd_trd"}
    INDEX_NAME = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}

    def __init__(self, api_key: str, cache_dir: Path, transport: httpx.BaseTransport | None = None):
        self.key = api_key
        self.cache_dir = cache_dir
        self.client = httpx.Client(timeout=30, transport=transport)

    def get_index(self, market: str, start: dt.date, end: dt.date):
        src = f"krx-openapi/idx/{self.ENDPOINT.get(market)}"
        if not self.key:
            return pd.DataFrame(), FetchStatus(src, False, "KRX_API_KEY 미설정")
        rows, errors = [], []
        for d in weekdays(start, end):
            cp = self.cache_dir / "krx_index" / market / f"{d}.json"
            if cp.exists():
                data = json.loads(cp.read_text())
            else:
                try:
                    r = self.client.get(f"{self.BASE}/{self.ENDPOINT[market]}", params={"basDd": d},
                                        headers={"AUTH_KEY": self.key})
                    r.raise_for_status()
                    data = r.json()
                except Exception as exc:
                    detail = ""
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        detail = f" {resp.status_code} {resp.text[:80]}"
                    errors.append(f"{d}: {type(exc).__name__}{detail}")
                    if len(errors) >= 3 and not rows:
                        break
                    continue
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data, ensure_ascii=False))
            for row in data.get("OutBlock_1", []) or []:
                if row.get("IDX_NM") == self.INDEX_NAME[market]:
                    rows.append({"date": pd.to_datetime(row.get("BAS_DD")), "close": _num(row.get("CLSPRC_IDX"))})
        df = pd.DataFrame(rows)
        if df.empty:
            return df, FetchStatus(src, False, f"지수 데이터 없음 (오류 {len(errors)}건: {errors[:3]})")
        return df, FetchStatus(src, True, f"오류 {len(errors)}건" if errors else "")


class ChainBenchmarkProvider:
    def __init__(self, providers: list):
        self.providers = providers

    def get_index(self, market: str, start: dt.date, end: dt.date):
        msgs = []
        for p in self.providers:
            df, st = p.get_index(market, start, end)
            if st.ok:
                return df, st
            msgs.append(f"{st.source}: {st.message}")
        return pd.DataFrame(), FetchStatus("benchmark", False, " | ".join(msgs))

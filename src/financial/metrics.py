"""금융 지표 계산 (LLM이 아닌 Python에서 계산).

이벤트 윈도우 정의:
- t0 = 이벤트일 이후(포함) 첫 거래일
- k 거래일 윈도우 = [t0, t0+k-1], 기준가 = t0 직전 거래일 종가
주가 동조성은 산업 관계의 인과성을 증명하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (1, 5, 20, 60)
CAUSALITY_NOTE = "주가 동조성·초과수익률은 산업 관계의 인과성을 증명하지 않습니다."


def prepare_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """date, close(, volume) 정렬·중복·결측 확인."""
    issues = []
    if df.empty:
        return df, ["가격 데이터 없음"]
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if d["date"].duplicated().any():
        issues.append(f"중복 거래일 {int(d['date'].duplicated().sum())}건 제거")
        d = d.drop_duplicates("date", keep="last")
    if not d["date"].is_monotonic_increasing:
        issues.append("거래일 정렬")
        d = d.sort_values("date")
    na = int(d["close"].isna().sum())
    if na:
        issues.append(f"종가 결측 {na}건 제외")
        d = d.dropna(subset=["close"])
    return d.set_index("date"), issues


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def event_index(index: pd.DatetimeIndex, event_date) -> int | None:
    pos = index.searchsorted(pd.Timestamp(event_date))
    return int(pos) if pos < len(index) else None


def cumulative_return(close: pd.Series, t0: int, k: int) -> float | None:
    if t0 < 1 or t0 + k - 1 >= len(close):
        return None
    return float(close.iloc[t0 + k - 1] / close.iloc[t0 - 1] - 1)


def trailing_returns(close: pd.Series, anchor: int, windows=(5, 20, 60)) -> dict:
    """anchor 거래일 종가 기준 직전 k거래일 수익률."""
    out = {}
    for k in windows:
        out[k] = float(close.iloc[anchor] / close.iloc[anchor - k] - 1) if 0 <= anchor - k < anchor < len(close) else None
    return out


def max_drawdown(close: pd.Series) -> float | None:
    if len(close) < 2:
        return None
    peak = close.cummax()
    return float((close / peak - 1).min())


def market_model(stock_ret: pd.Series, mkt_ret: pd.Series, est_start: int, est_end: int, min_obs: int = 60
                 ) -> tuple[float, float] | None:
    """추정기간 OLS: r = alpha + beta * r_m"""
    s = stock_ret.iloc[est_start:est_end]
    m = mkt_ret.reindex(s.index).dropna()
    s = s.reindex(m.index)
    if len(s) < min_obs:
        return None
    beta, alpha = np.polyfit(m.values, s.values, 1)
    return float(alpha), float(beta)


def event_study(stock: pd.DataFrame, market: pd.DataFrame | None, event_date, windows=WINDOWS,
                est_window: tuple[int, int] = (-250, -11)) -> dict:
    """단일 종목 이벤트 분석. 계산 불가 지표는 None + 사유."""
    out: dict = {"windows": {}, "notes": [], "causality_note": CAUSALITY_NOTE}
    close = stock["close"]
    t0 = event_index(close.index, event_date)
    # 기준일 직전까지의 최근 수익률 (기준일이 오늘이어도 계산 가능)
    anchor = (t0 - 1) if t0 is not None else len(close) - 1
    out["trailing"] = trailing_returns(close, anchor)
    out["last_date"] = close.index[-1].date().isoformat() if len(close) else None
    if t0 is None:
        out["notes"].append("기준일 이후 거래일 데이터가 아직 없어 이벤트 이후 지표는 계산하지 않았습니다 (최근 수익률만 표시)")
        return out
    if t0 < 1:
        out["notes"].append("이벤트일 이전 가격 데이터 부족")
        return out
    out["event_trading_day"] = close.index[t0].date().isoformat()
    r = daily_returns(close)
    mret = None
    if market is not None and not market.empty:
        mret = daily_returns(market["close"]).reindex(r.index)
    else:
        out["notes"].append("벤치마크 데이터 없음 - 시장 대비 지표 미계산")
    # r는 close보다 1개 짧다: close 인덱스 t ↔ r 인덱스 t-1
    mm = None
    if mret is not None:
        es, ee = t0 - 1 + est_window[0], t0 - 1 + est_window[1]
        if es >= 0:
            mm = market_model(r, mret, es, ee)
        if mm is None:
            out["notes"].append("시장모형 추정기간 데이터 부족 - 시장모형 CAR 미계산")
        else:
            out["market_model"] = {"alpha": mm[0], "beta": mm[1]}
    for k in windows:
        w: dict = {"cumulative_return": cumulative_return(close, t0, k)}
        if w["cumulative_return"] is None:
            w["note"] = "윈도우 데이터 부족"
            out["windows"][k] = w
            continue
        seg = r.iloc[t0 - 1: t0 - 1 + k]
        w["mdd"] = max_drawdown(close.iloc[t0 - 1: t0 + k])
        if "volume" in stock and stock["volume"].notna().any() and t0 >= 20:
            pre = stock["volume"].iloc[t0 - 20:t0].mean()
            post = stock["volume"].iloc[t0:t0 + k].mean()
            w["volume_change"] = float(post / pre - 1) if pre else None
        if mret is not None:
            mseg = mret.reindex(seg.index)
            if mseg.notna().all():
                mcum = float((1 + mseg).prod() - 1)
                w["market_return"] = mcum
                w["excess_return"] = w["cumulative_return"] - mcum           # 단순 시장 조정
                w["car_market_adjusted"] = float((seg - mseg).sum())
                if mm is not None:
                    ar = seg - (mm[0] + mm[1] * mseg)
                    w["car_market_model"] = float(ar.sum())                  # 시장모형 비정상수익률
            else:
                w["note"] = "벤치마크 거래일 불일치"
        out["windows"][k] = w
    return out


def return_correlation(prices: dict[str, pd.DataFrame], start=None, end=None, min_obs: int = 20) -> pd.DataFrame:
    rets = {}
    for code, df in prices.items():
        s = df["close"]
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        rets[code] = daily_returns(s)
    frame = pd.DataFrame(rets)
    return frame.corr(min_periods=min_obs)


def normalized_prices(prices: dict[str, pd.DataFrame], base_date, lookback: int = 120) -> pd.DataFrame:
    """기준일 직전 종가=100. 기준일 이후 데이터가 없으면 최근 lookback 거래일의 첫날=100."""
    out = {}
    for code, df in prices.items():
        c = df["close"].dropna()
        if c.empty:
            continue
        i = event_index(c.index, base_date)
        if i is not None and i >= 1:
            out[code] = c / c.iloc[i - 1] * 100
        else:
            tail = c.iloc[-lookback:]
            out[code] = tail / tail.iloc[0] * 100
    return pd.DataFrame(out)

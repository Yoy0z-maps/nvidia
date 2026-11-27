import asyncio
import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.financial import metrics as M
from src.financial.market_data import CsvBenchmarkProvider, FetchStatus, MCPKrxPriceProvider, weekdays
from src.financial.mcp_client import KoreaStockMCP, ToolInfo


def _series(prices, start="2024-01-01", volume=None):
    idx = pd.bdate_range(start, periods=len(prices))
    df = pd.DataFrame({"date": idx, "close": prices})
    if volume is not None:
        df["volume"] = volume
    return M.prepare_prices(df)[0]


def test_cumulative_and_daily_returns():
    s = _series([100, 110, 121, 133.1])
    r = M.daily_returns(s["close"])
    assert np.allclose(r.values, [0.1, 0.1, 0.1])
    assert M.cumulative_return(s["close"], 1, 2) == pytest.approx(0.21)
    assert M.cumulative_return(s["close"], 1, 10) is None  # 데이터 부족은 계산하지 않음


def test_mdd():
    s = _series([100, 120, 90, 110, 80, 130])
    assert M.max_drawdown(s["close"]) == pytest.approx(80 / 120 - 1)


def test_event_study_market_adjusted_and_model():
    rng = np.random.default_rng(0)
    n = 330
    m_ret = rng.normal(0, 0.01, n)
    s_ret = 0.0005 + 1.5 * m_ret + rng.normal(0, 0.001, n)
    s_ret[300:305] += 0.02  # 이벤트 이후 5일 초과수익
    mkt = _series(100 * np.cumprod(1 + np.r_[0, m_ret[1:]]))
    stk = _series(100 * np.cumprod(1 + np.r_[0, s_ret[1:]]), volume=np.r_[np.full(300, 100.0), np.full(30, 200.0)])
    event_date = stk.index[300]
    res = M.event_study(stk, mkt, event_date)
    w5 = res["windows"][5]
    assert res["market_model"]["beta"] == pytest.approx(1.5, abs=0.05)
    assert w5["car_market_model"] == pytest.approx(0.10, abs=0.01)
    assert "car_market_adjusted" in w5 and "excess_return" in w5
    assert w5["volume_change"] == pytest.approx(1.0)
    assert res["windows"][60].get("note") == "윈도우 데이터 부족"
    assert "인과성" in res["causality_note"]


def test_event_study_without_benchmark():
    stk = _series(list(range(100, 140)))
    res = M.event_study(stk, None, stk.index[10])
    assert "car_market_adjusted" not in res["windows"][5]
    assert any("벤치마크" in n for n in res["notes"])


def test_prepare_prices_issues():
    df = pd.DataFrame({"date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"],
                       "close": [2.0, 1.0, 1.0, None]})
    d, issues = M.prepare_prices(df)
    assert list(d["close"]) == [1.0, 2.0] and len(issues) == 3


def test_correlation_and_normalized():
    a = _series([100, 101, 102, 101, 103] * 6)
    b = _series([50, 50.5, 51, 50.5, 51.5] * 6)
    corr = M.return_correlation({"A": a, "B": b}, min_obs=5)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)
    norm = M.normalized_prices({"A": a}, a.index[2])
    assert norm["A"].iloc[1] == pytest.approx(100.0)


def test_weekdays():
    assert weekdays(dt.date(2024, 1, 5), dt.date(2024, 1, 8)) == ["20240105", "20240108"]


class FakeMCP:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def call(self, name, args):
        self.calls.append((name, args))
        return {d: [r for r in self.data.get(d, []) if r["ISU_CD"] in args["codeList"]] for d in args["basDdList"]}


def test_mcp_price_provider_normalizes_and_caches(tmp_path):
    data = {"20240102": [{"BAS_DD": "20240102", "ISU_CD": "000660", "TDD_CLSPRC": "140,000", "ACC_TRDVOL": "1,000",
                          "LIST_SHRS": "728002365"}],
            "20240103": [{"BAS_DD": "20240103", "ISU_CD": "000660", "TDD_CLSPRC": "141,000", "ACC_TRDVOL": "2,000",
                          "LIST_SHRS": "728002365"}]}
    fake = FakeMCP(data)
    prov = MCPKrxPriceProvider(lambda: fake, tmp_path)
    out, st = prov.get_prices({"000660": "KOSPI"}, dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    assert st.ok and not st.adjusted and list(out["000660"]["close"]) == [140000.0, 141000.0]
    n = len(fake.calls)
    prov.get_prices({"000660": "KOSPI"}, dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    assert len(fake.calls) == n  # 캐시 재사용


def test_mcp_provider_failure_is_explicit(tmp_path):
    class Boom(FakeMCP):
        async def call(self, name, args):
            raise RuntimeError("There is no KRX API KEY")
    out, st = MCPKrxPriceProvider(lambda: Boom({}), tmp_path).get_prices({"000660": "KOSPI"},
                                                                         dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    assert out == {} and not st.ok and "KRX" in st.message


def test_csv_benchmark(tmp_path):
    (tmp_path / "KOSPI.csv").write_text("date,close\n2024-01-02,2600\n2024-01-03,2610\n")
    df, st = CsvBenchmarkProvider(tmp_path).get_index("KOSPI", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert st.ok and len(df) == 2
    df, st = CsvBenchmarkProvider(tmp_path).get_index("KOSDAQ", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert not st.ok


def test_mcp_allowlist_and_schema_check():
    m = KoreaStockMCP("node", ["x.js"])
    tools = json.loads((__import__("pathlib").Path(__file__).parent.parent / "docs" /
                        "korea_stock_mcp_1.4.1_tools.json").read_text())
    m.tools = {k: ToolInfo(v["name"], v["description"], v["input_schema"]) for k, v in tools.items()}
    with pytest.raises(PermissionError):
        asyncio.run(m.call("get_disclosure", {"rcept_no": "1"}))       # 허용 목록 외
    with pytest.raises(ValueError):
        asyncio.run(m.call("get_stock_trade_info", {"market": "KOSPI"}))  # 필수 파라미터 누락
    with pytest.raises(ValueError):
        asyncio.run(m.call("get_price_history", {}))                     # 존재하지 않는 Tool
    fn = m.openai_tools()
    assert {t["name"] for t in fn} <= m.allowed_tools and all(t["type"] == "function" for t in fn)


@pytest.mark.live
@pytest.mark.skipif(shutil.which("node") is None, reason="node 필요")
def test_live_mcp_list_tools():
    """실제 MCP 서버 연결 (uv run pytest -m live)."""
    from config.settings import Settings
    from src.financial.mcp_client import list_tools_once
    s = Settings.from_env()
    tools = asyncio.run(list_tools_once(s.mcp_command, s.mcp_args))
    assert {"get_stock_trade_info", "get_corp_code"} <= {t["name"] for t in tools}

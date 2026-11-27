"""환경 점검: API Key, LLM 모델 사용 가능 여부, 임베딩, tiktoken, OpenDART, Korea Stock MCP.

uv run python -m scripts.check_environment [--install-mcp] [--skip-mcp]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402

OK, FAIL, WARN, SKIP = "OK", "FAIL", "WARN", "SKIP"


def check(results: list, name: str, fn):
    try:
        status, detail = fn()
    except Exception as exc:
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    results.append({"check": name, "status": status, "detail": detail})
    print(f"[{status:4}] {name}: {detail}")
    return status


def install_mcp() -> tuple[str, str]:
    from src.financial.mcp_client import MCP_PACKAGE
    if shutil.which("npm") is None:
        return FAIL, "npm이 없습니다. Node.js 18+ 설치 필요"
    r = subprocess.run(["npm", "install", "--prefix", str(ROOT / ".mcp"), MCP_PACKAGE, "--no-audit", "--no-fund"],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return FAIL, r.stderr[-500:]
    return OK, f"{MCP_PACKAGE} → .mcp/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-mcp", action="store_true", help="korea-stock-mcp를 .mcp/에 설치")
    ap.add_argument("--skip-mcp", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    s.ensure_dirs()
    setup_logging(s)
    results: list[dict] = []

    check(results, "Python", lambda: (OK if sys.version_info[:2] == (3, 11) else WARN, sys.version.split()[0]))
    for key in ("OPENAI_API_KEY", "DART_API_KEY", "KRX_API_KEY"):
        val = getattr(s, key.lower())
        check(results, f"env {key}",
              lambda v=val, k=key: (OK, "설정됨") if v else (FAIL if k != "KRX_API_KEY" else WARN, "미설정"))

    # --- OpenAI ---
    if s.openai_api_key:
        from src.llm import ModelUnavailableError, OpenAILLM

        def _llm():
            llm = OpenAILLM(s.openai_api_key, s.llm_model)
            try:
                llm.check_model()
            except ModelUnavailableError as exc:
                return FAIL, str(exc)
            return OK, f"LLM_MODEL={s.llm_model} 사용 가능"
        check(results, "OpenAI LLM 모델", _llm)

        def _emb():
            from src.retrieval.embedding import OpenAIEmbedder
            v = OpenAIEmbedder(s.openai_api_key, s.embedding_model, s.embedding_dimensions, max_retries=1).embed(["테스트"])
            return OK, f"{s.embedding_model} dims={v.shape[1]}"
        check(results, "OpenAI Embedding", _emb)
    else:
        results.append({"check": "OpenAI", "status": SKIP, "detail": "OPENAI_API_KEY 미설정"})

    def _tok():
        from src.ingestion.chunker import get_tokenizer
        t = get_tokenizer(s.embedding_model)
        return OK, f"'SK하이닉스 HBM' = {len(t.encode('SK하이닉스 HBM'))} tokens"
    check(results, "tiktoken", _tok)

    # --- OpenDART ---
    seed_corp = None
    if s.dart_api_key:
        from src.ingestion.corp_registry import CorpRegistry
        from src.ingestion.dart_client import DartClient
        dart = DartClient(s.dart_api_key)

        def _corp():
            nonlocal seed_corp
            reg = CorpRegistry.load_or_download(s.processed_dir / "corp_codes.json", dart)
            seed_corp = reg.get_by_stock_code(s.seed_stock_code)
            if not seed_corp:
                return FAIL, f"Seed 종목코드 {s.seed_stock_code} 없음"
            return OK, f"기업 {len(reg.corps)}개, Seed={seed_corp.corp_name}({seed_corp.corp_code})"
        check(results, "OpenDART corpCode", _corp)

        def _list():
            end = dt.date.today()
            items = dart.list_disclosures(seed_corp.corp_code, (end - dt.timedelta(days=365)).strftime("%Y%m%d"),
                                          end.strftime("%Y%m%d"), "A")
            return OK, f"최근 1년 정기공시 {len(items)}건: " + ", ".join(i["report_nm"] for i in items[:3])
        if seed_corp:
            check(results, "OpenDART list.json", _list)
            check(results, "OpenDART company.json", lambda: (OK, f"corp_cls={dart.company_info(seed_corp.corp_code).get('corp_cls')}"))

    # --- Korea Stock MCP ---
    if args.install_mcp:
        check(results, "MCP 설치", install_mcp)
    if not args.skip_mcp:
        from src.financial.mcp_client import make_mcp

        async def _mcp_probe():
            out = {}
            async with make_mcp(s) as m:
                out["tools"] = {k: v.to_dict() for k, v in m.tools.items()}
                print("    Tool:", ", ".join(sorted(m.tools)))
                steps = [("get_today_date", {})]
                if s.dart_api_key:
                    steps.append(("get_corp_code", {"stock_code": s.seed_stock_code}))
                if s.krx_api_key:
                    d = dt.date.today() - dt.timedelta(days=1)
                    while d.weekday() >= 5:
                        d -= dt.timedelta(days=1)
                    steps.append(("get_stock_trade_info", {"basDdList": [d.strftime("%Y%m%d")], "market": "KOSPI",
                                                           "codeList": [s.seed_stock_code]}))
                for name, a in steps:
                    try:
                        r = await m.call(name, a)
                        out[name] = {"ok": True, "sample": json.dumps(r, ensure_ascii=False)[:300]}
                    except Exception as exc:
                        out[name] = {"ok": False, "error": str(exc)[:300]}
                    print(f"    {name}: {out[name]}")
            (s.logs_dir / "mcp_tools.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            failed = [k for k, v in out.items() if isinstance(v, dict) and v.get("ok") is False]
            return (WARN if failed else OK), f"Tool {len(out['tools'])}개, 실패 호출: {failed or '없음'}"

        from src.financial.mcp_client import make_mcp as _mk
        problems = _mk(s).check_runtime()
        for p in problems:
            print(f"    - {p}")
        check(results, "Korea Stock MCP", lambda: asyncio.run(_mcp_probe()))

    (s.logs_dir / "environment_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    fails = [r for r in results if r["status"] == FAIL]
    print(f"\n결과: {len(results) - len(fails)}/{len(results)} 통과 (상세: {s.logs_dir / 'environment_check.json'})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

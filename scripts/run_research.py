"""이벤트 파급효과 분석.

uv run python -m scripts.run_research --seed 000660 --query "HBM 수요가 증가할 경우 ..."
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed")
    ap.add_argument("--query", required=True)
    ap.add_argument("--hops", type=int, help="분석 Hop (기본 3, 최대 5)")
    ap.add_argument("--as-of", help="YYYY-MM-DD: 해당 날짜까지 공개된 문서만 사용")
    ap.add_argument("--financial", action="store_true", help="결과 기업에 대해 MCP 금융 분석 실행")
    ap.add_argument("--event-date", help="금융 분석 기준일 (기본: 이벤트 추출 날짜 또는 오늘)")
    ap.add_argument("--no-insights", action="store_true", help="가설 확장·가설 인사이트 끄기 (근거 확인된 경로만)")
    ap.add_argument("--open", action="store_true", help="분석 후 Streamlit UI를 열어 결과를 바로 표시")
    ap.add_argument("--expand-companies", action="store_true", help="새로 발견된 기업 공시 추가 수집 (기업 수 제한 적용)")
    args = ap.parse_args()

    s = Settings.from_env()
    if args.seed:
        s.seed_stock_code = args.seed.zfill(6)
    if args.hops:
        s.max_analysis_hops = args.hops
        s.validate()
    setup_logging(s)

    from src.pipeline import build_app, make_ingest_fn, seed_company
    from src.research.ripple_engine import RippleConfig, RippleEngine, save_result

    app = build_app(s, need_dart=bool(s.dart_api_key))
    app.llm.check_model()
    seed = seed_company(app)
    expander = None
    if args.expand_companies and app.collector is not None:
        ingest = make_ingest_fn(app, print)

        def expander(node: dict) -> bool:
            ready = sum(1 for c in app.storage.list_companies() if c["ingestion_status"] == "READY")
            if ready >= s.max_companies:
                return False
            r, _ = app.resolver.resolve(node["name"])
            return bool(r) and ingest(r, 99, seed["corp_code"], "ripple 확장")

    cfg = RippleConfig(max_hops=s.max_analysis_hops, max_branches_per_hop=s.max_branches_per_hop,
                       max_retrieval_calls=s.max_retrieval_calls, max_llm_calls=s.max_llm_calls,
                       top_k=s.retrieval_top_k, as_of_date=args.as_of, insight_mode=not args.no_insights)
    engine = RippleEngine(app.storage, app.llm, app.retriever, app.resolver, cfg,
                          base_graph=app.load_graph(), progress_cb=print, company_expander=expander)
    res = engine.run(args.query, seed)
    path = save_result(res, s.runs_dir, app.storage, s.llm_model, {**s.public_dict(), **cfg.__dict__})

    print("\n=== 이벤트 ===")
    print(json.dumps(res.event, ensure_ascii=False, indent=2))
    print("\n=== 요약 ===\n" + res.summary)
    print("\n=== 기업별 결과 ===")
    for r in res.company_results:
        print(f"- {r['corp_name']}({r['stock_code']}) [{r['verification_status']}] {r.get('economic_impact', '')}"
              f"\n    경로: {r['path']}\n    매출 노출도: {r.get('revenue_exposure', 'UNKNOWN')}")
    print(f"\n중단 사유: {res.stopped_reason} / LLM {res.budget['llm']['calls']}회, 검색 {res.budget['retrieval_calls']}회, "
          f"{res.elapsed_ms / 1000:.1f}s")

    if args.financial:
        from src.financial.analysis import run_for_result
        ev_date = args.event_date or res.event.get("event_date") or dt.date.today().isoformat()
        fin = run_for_result(json.loads(path.read_text(encoding="utf-8")), app, ev_date)
        res_d = json.loads(path.read_text(encoding="utf-8"))
        res_d["financial"] = fin
        path.write_text(json.dumps(res_d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\n=== 금융 분석 상태 ===")
        print(json.dumps(fin["status"], ensure_ascii=False, indent=2, default=str))
    print(f"\n결과 저장: {path}")
    url = f"http://localhost:8501/?run={res.run_id}"
    if args.open:
        open_ui(url)
    else:
        print(f"UI에서 보기: uv run streamlit run app.py 실행 후 {url}  (또는 --open 옵션)")
    return 0


def open_ui(url: str) -> None:
    """Streamlit이 이미 떠 있으면 브라우저만 열고, 아니면 서버를 띄운 뒤 연다 (Ctrl+C로 종료)."""
    import socket
    import subprocess
    import time
    import webbrowser

    def up() -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", 8501)) == 0

    proc = None
    if not up():
        print("Streamlit 서버 시작 중...")
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                                 "--server.port", "8501", "--server.headless", "true"], cwd=ROOT)
        for _ in range(60):
            if up():
                break
            time.sleep(0.5)
    webbrowser.open(url)
    print(f"브라우저에서 열림: {url}")
    if proc is not None:
        print("종료하려면 Ctrl+C")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())

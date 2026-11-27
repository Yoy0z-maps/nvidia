"""과거 시나리오 평가 (A: Vanilla LLM / B: Standard RAG / C: Multi-hop RAG + Graph).

uv run python -m scripts.run_evaluation --event-file data/evaluation/events.json [--ensure-docs] [--skip-financial]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-file", default="data/evaluation/events.json")
    ap.add_argument("--events", nargs="*", help="실행할 event_id (기본: 전체)")
    ap.add_argument("--ensure-docs", action="store_true",
                    help="지식 기반 기업들의 '기준일 시점' 공시를 추가 수집 (기준일 이후 공시는 수집하지 않음)")
    ap.add_argument("--skip-financial", action="store_true")
    ap.add_argument("--max-companies", type=int, default=7)
    args = ap.parse_args()

    s = Settings.from_env()
    setup_logging(s)
    from src.evaluation.historical_replay import LEAKAGE_NOTE, ensure_as_of_documents, replay_event
    from src.pipeline import build_app, seed_company

    app = build_app(s, need_dart=bool(s.dart_api_key))
    app.llm.check_model()
    seed = seed_company(app)
    data = json.loads((ROOT / args.event_file).read_text(encoding="utf-8"))
    events = [e for e in data["events"] if not args.events or e["event_id"] in args.events]
    if len(events) < 3:
        print("⚠ 스펙 기준 최소 3개 이벤트 평가가 필요합니다.")
    fin_fn = None
    if not args.skip_financial:
        from src.financial.analysis import analyze, default_providers
        providers = default_providers(s)

        def fin_fn(companies, event_date):
            for c in companies:
                if not c.get("market_type") and c.get("stock_code") and app.registry and app.market:
                    corp = app.registry.get_by_stock_code(c["stock_code"])
                    c["market_type"] = app.market.market_type(corp.corp_code) if corp else None
            return analyze(companies, event_date, *providers)

    out_dir = s.evaluation_dir / "results"
    reports = []
    for ev in events:
        if args.ensure_docs:
            comps = [c for c in app.storage.list_companies() if c["stock_code"]]
            ensure_as_of_documents(app, ev, comps, print)
        reports.append(replay_event(app, ev, seed, out_dir, args.max_companies, fin_fn, print))

    summary = {"model": s.llm_model, "events": reports, "leakage_note": LEAKAGE_NOTE}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== 평가 요약 ===")
    for r in reports:
        print(f"\n[{r['event_id']}] 기준일 {r['as_of_date']} / 정답 라벨: {'있음' if r['ground_truth_labeled'] else '없음'}")
        for arm, m in r["metrics"].items():
            if not isinstance(m, dict) or "latency_ms" not in m:
                continue
            md = (m.get("market_direction") or {}).get("accuracy")
            print(f"  {arm}: latency={m['latency_ms']}ms, tokens={m['token_usage'].get('input_tokens', 0)}+"
                  f"{m['token_usage'].get('output_tokens', 0)}, cost≈${m['estimated_cost_usd']}, "
                  f"quote_acc={m['quote_accuracy']}, market_dir_acc={md}")
            if "valid_novel_discovery" in m:
                print(f"    path_validity={m['path_validity']}, novel(verified)={m['valid_novel_discovery']['count_verified']}")
    print(f"\n{LEAKAGE_NOTE}\n결과: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

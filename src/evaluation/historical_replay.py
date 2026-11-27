"""Historical Scenario Replay.

1. 평가 기준일(as_of_date)까지 공개된 문서만 사용 (Retriever가 기준일 이하 벡터로 임시 인덱스를 구성)
2. 현재 Knowledge Graph를 사용하지 않음 (RippleConfig.use_existing_graph=False + as_of_date)
3. 세 실험군(A/B/C) 예측을 먼저 저장
4. 그 다음에 기준일 이후 금융 데이터를 조회해 비교
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from src.evaluation import evaluator as E
from src.evaluation.baseline import quote_accuracy, run_multihop, run_standard_rag, run_vanilla
from src.research.ripple_engine import RippleConfig, RippleEngine

log = logging.getLogger(__name__)

LEAKAGE_NOTE = ("LLM 사전학습 데이터에 평가 기준일 이후 정보가 포함되어 있을 수 있습니다. "
                "이 결과를 순수한 미래 예측 성능으로 해석하지 마십시오.")


def ensure_as_of_documents(app, event: dict, companies: list[dict], progress=log.info) -> None:
    """기준일 시점의 정기보고서/주요공시가 없으면 수집한다 (기준일 이후 공시는 수집하지 않음)."""
    from src.pipeline import collect_company, embed_and_index
    as_of = dt.date.fromisoformat(event["as_of_date"])
    for c in companies:
        collect_company(app, c["corp_code"], c["corp_name"], c["stock_code"], as_of=as_of, progress=progress)
    embed_and_index(app, progress)


def replay_event(app, event: dict, seed: dict, out_dir: Path, max_companies: int = 7,
                 financial_fn=None, progress=log.info) -> dict:
    s = app.settings
    as_of = event["as_of_date"]
    ev_dir = out_dir / event["event_id"]
    ev_dir.mkdir(parents=True, exist_ok=True)
    text = event["input_text"]
    progress(f"[{event['event_id']}] 기준일 {as_of} - A: Vanilla LLM")
    a = run_vanilla(app.llm, text, max_companies, as_of)
    progress(f"[{event['event_id']}] B: Standard RAG")
    b = run_standard_rag(app.llm, app.retriever, app.storage, text, top_k=10, max_companies=max_companies,
                         as_of_date=as_of)
    progress(f"[{event['event_id']}] C: Multi-hop RAG + Graph")
    engine = RippleEngine(app.storage, app.llm, app.retriever, app.resolver,
                          RippleConfig(max_hops=s.max_analysis_hops, max_branches_per_hop=s.max_branches_per_hop,
                                       max_retrieval_calls=s.max_retrieval_calls, max_llm_calls=s.max_llm_calls,
                                       top_k=s.retrieval_top_k, as_of_date=as_of, use_existing_graph=False),
                          progress_cb=progress)
    c = run_multihop(engine, text, seed, max_companies)
    for arm in (a, b, c):
        arm["model"] = app.llm.model
        arm["as_of_date"] = as_of
        (ev_dir / f"{arm['arm']}.json").write_text(json.dumps(arm, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
    progress(f"[{event['event_id']}] 예측 저장 완료 → 이후 데이터와 비교")

    # --- 예측 저장 후: 사후 데이터 비교 ---
    gt = event.get("ground_truth", {})
    fin = None
    if financial_fn is not None:
        all_companies = {}
        for arm in (a, b, c):
            for co in arm["companies"]:
                if co.get("stock_code"):
                    all_companies[co["stock_code"]] = co
        try:
            fin = financial_fn(list(all_companies.values()), event["event_date"])
        except Exception as exc:
            fin = {"error": str(exc), "companies": {}}
        (ev_dir / "financial.json").write_text(json.dumps({k: v for k, v in fin.items() if k != "_prices"},
                                                          ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metrics = {}
    for arm in (a, b, c):
        m = {"company": E.company_precision_recall(arm["companies"], gt),
             "latency_ms": arm["elapsed_ms"], "token_usage": arm["usage"],
             "estimated_cost_usd": E.estimate_cost(app.llm.model, arm["usage"]),
             "quote_accuracy": quote_accuracy(app.storage, arm)}
        if fin is not None:
            m["market_direction"] = E.market_direction_accuracy(arm["companies"], fin)
        if arm["arm"] == "C_multihop_graph":
            m["relation"] = E.relation_precision_recall(arm["graph"], gt)
            m["path_validity"] = E.path_validity(arm["companies"])
            m["evidence_entailment_accuracy"] = E.entailment_accuracy(arm["graph"], gt)
            m["valid_novel_discovery"] = E.valid_novel_discoveries(arm, [a, b], gt)
        metrics[arm["arm"]] = m
    metrics["retrieval_recall@5"] = E.retrieval_recall_at_k(app.retriever, gt.get("retrieval_qrels", []), 5, as_of)
    report = {"event_id": event["event_id"], "as_of_date": as_of, "event_date": event["event_date"],
              "model": app.llm.model, "metrics": metrics, "leakage_note": LEAKAGE_NOTE,
              "ground_truth_labeled": bool(gt.get("labeled_by"))}
    (ev_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report

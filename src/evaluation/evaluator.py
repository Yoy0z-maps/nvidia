"""평가 지표 계산. 정답(Ground Truth)은 사람이 원문을 확인해 작성한 라벨만 사용한다."""
from __future__ import annotations

from src.ingestion.corp_registry import normalize_name
from src.schemas import VERIFIED_STATUSES, VerificationStatus

# 비용 추정용 (USD / 1M tokens). 실제 단가는 OpenAI 가격표로 갱신하세요.
LLM_PRICE_PER_M = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60)}


def _company_key(c: dict) -> str:
    return (c.get("stock_code") or "").strip() or normalize_name(c.get("corp_name", ""))


def _gt_company_keys(gt: dict, valid_only: bool = True) -> set[str]:
    keys = set()
    for c in gt.get("companies", []):
        if valid_only and c.get("label") not in (None, "VALID", True):
            continue
        keys.add(_company_key(c))
        keys.add(normalize_name(c.get("corp_name", "")))
    return {k for k in keys if k}


def company_precision_recall(pred: list[dict], gt: dict) -> dict:
    gold = [c for c in gt.get("companies", []) if c.get("label") in (None, "VALID", True)]
    if not gold:
        return {"precision": None, "recall": None, "note": "정답 라벨 없음"}
    gold_keys = [{_company_key(c), normalize_name(c["corp_name"])} for c in gold]
    hits_pred = [any(({_company_key(p), normalize_name(p["corp_name"])} & g) for g in gold_keys) for p in pred]
    hits_gold = [any(({_company_key(p), normalize_name(p["corp_name"])} & g) for p in pred) for g in gold_keys]
    return {"precision": sum(hits_pred) / len(pred) if pred else None,
            "recall": sum(hits_gold) / len(gold)}


def relation_precision_recall(graph: dict, gt: dict) -> dict:
    gold = gt.get("relations", [])
    if not gold:
        return {"precision": None, "recall": None, "note": "관계 정답 라벨 없음"}
    names = {n["node_id"]: normalize_name(n["name"]) for n in graph["nodes"]}
    pred = {(names.get(e["source_id"]), e["relationship"], names.get(e["target_id"]))
            for e in graph["edges"] if VerificationStatus(e["verification_status"]) in VERIFIED_STATUSES}
    gold_set = {(normalize_name(r["source"]), r["relationship"], normalize_name(r["target"])) for r in gold}
    tp = len(pred & gold_set)
    return {"precision": tp / len(pred) if pred else None, "recall": tp / len(gold_set), "tp": tp}


def path_validity(companies: list[dict]) -> float | None:
    st = [c.get("verification_status") for c in companies if c.get("verification_status") not in (None, "PREMISE")]
    if not st:
        return None
    return sum(VerificationStatus(s) in VERIFIED_STATUSES for s in st) / len(st)


def entailment_accuracy(graph: dict, gt: dict) -> float | None:
    """사람이 라벨링한 edge 근거 함의 여부(entailment_labels: {"source|REL|target": true/false})와 비교."""
    labels = gt.get("entailment_labels") or {}
    if not labels:
        return None
    names = {n["node_id"]: n["name"] for n in graph["nodes"]}
    checks = []
    for e in graph["edges"]:
        key = f"{names.get(e['source_id'])}|{e['relationship']}|{names.get(e['target_id'])}"
        if key in labels:
            predicted = VerificationStatus(e["verification_status"]) in VERIFIED_STATUSES
            checks.append(predicted == bool(labels[key]))
    return sum(checks) / len(checks) if checks else None


def valid_novel_discoveries(c_out: dict, baselines: list[dict], gt: dict) -> dict:
    """비교군(A, B)에 없던 기업 중 독립 근거 검증(경로 상태 VERIFIED/SUPPORTED)을 통과한 기업.

    정답 라벨이 있으면 라벨상 VALID 인 것만 최종 카운트한다.
    """
    seen = set()
    for b in baselines:
        for c in b["companies"]:
            seen |= {_company_key(c), normalize_name(c["corp_name"])}
    gold = _gt_company_keys(gt)
    novel = []
    for c in c_out["companies"]:
        keys = {_company_key(c), normalize_name(c["corp_name"])}
        if keys & seen:
            continue
        st = c.get("verification_status")
        if st in (None, "PREMISE") or VerificationStatus(st) not in VERIFIED_STATUSES:
            continue
        novel.append({"corp_name": c["corp_name"], "stock_code": c.get("stock_code"),
                      "label_valid": (bool(keys & gold) if gold else None)})
    return {"candidates": novel, "count_verified": len(novel),
            "count_label_valid": (sum(1 for n in novel if n["label_valid"]) if gold else None)}


def retrieval_recall_at_k(retriever, qrels: list[dict], k: int = 5, as_of_date: str | None = None) -> float | None:
    if not qrels:
        return None
    scores = []
    for q in qrels:
        rel = set(q["relevant_chunk_ids"])
        if not rel:
            continue
        got = {r.chunk_id for r in retriever.search(q["query"], top_k=k, as_of_date=as_of_date)}
        scores.append(len(got & rel) / len(rel))
    return sum(scores) / len(scores) if scores else None


def market_direction_accuracy(companies: list[dict], fin: dict, window: int = 20,
                              metric: str = "excess_return") -> dict:
    """예측 POSITIVE/NEGATIVE 와 이벤트 이후 초과수익률 부호 비교. 계산 가능한 기업만."""
    checks = []
    for c in companies:
        pred = c.get("economic_impact")
        if pred not in ("POSITIVE", "NEGATIVE"):
            continue
        f = (fin.get("companies") or {}).get(c.get("stock_code") or "")
        if not f or not f.get("available"):
            continue
        val = ((f.get("windows") or {}).get(window) or (f.get("windows") or {}).get(str(window)) or {}).get(metric)
        if val is None:
            continue
        checks.append({"corp_name": c["corp_name"], "predicted": pred, metric: val,
                       "correct": (val > 0) == (pred == "POSITIVE")})
    return {"accuracy": (sum(x["correct"] for x in checks) / len(checks)) if checks else None,
            "n": len(checks), "details": checks, "window": window, "metric": metric}


def estimate_cost(model: str, usage: dict) -> float | None:
    p = LLM_PRICE_PER_M.get(model)
    if not p or not usage:
        return None
    return round(usage.get("input_tokens", 0) / 1e6 * p[0] + usage.get("output_tokens", 0) / 1e6 * p[1], 6)

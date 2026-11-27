import json
from pathlib import Path

from src.evaluation import evaluator as E


def test_company_pr_requires_labels():
    assert E.company_precision_recall([{"corp_name": "A"}], {})["precision"] is None
    gt = {"companies": [{"corp_name": "가상장비", "stock_code": "100002", "label": "VALID"},
                        {"corp_name": "가상소재", "stock_code": "100003", "label": "VALID"}]}
    pred = [{"corp_name": "가상장비(주)", "stock_code": "100002"}, {"corp_name": "기타", "stock_code": "999999"}]
    r = E.company_precision_recall(pred, gt)
    assert r == {"precision": 0.5, "recall": 0.5}


def test_novel_discovery_requires_verified_path():
    c = {"companies": [{"corp_name": "가상소재", "stock_code": "100003", "verification_status": "SUPPORTED_INFERENCE"},
                       {"corp_name": "가상부품", "stock_code": "100004", "verification_status": "HYPOTHESIS"},
                       {"corp_name": "가상장비", "stock_code": "100002", "verification_status": "VERIFIED_FACT"}]}
    a = {"companies": [{"corp_name": "가상장비", "stock_code": "100002"}]}
    r = E.valid_novel_discoveries(c, [a], {})
    assert [x["corp_name"] for x in r["candidates"]] == ["가상소재"] and r["count_label_valid"] is None


def test_market_direction():
    fin = {"companies": {"100002": {"available": True, "windows": {20: {"excess_return": 0.05}}},
                         "100003": {"available": True, "windows": {20: {"excess_return": -0.02}}},
                         "100004": {"available": False}}}
    comps = [{"corp_name": "a", "stock_code": "100002", "economic_impact": "POSITIVE"},
             {"corp_name": "b", "stock_code": "100003", "economic_impact": "POSITIVE"},
             {"corp_name": "c", "stock_code": "100004", "economic_impact": "NEGATIVE"},
             {"corp_name": "d", "stock_code": "100005", "economic_impact": "UNCERTAIN"}]
    r = E.market_direction_accuracy(comps, fin)
    assert r["n"] == 2 and r["accuracy"] == 0.5


def test_events_file_has_three_real_events_without_fabricated_labels():
    data = json.loads((Path(__file__).parent.parent / "data/evaluation/events.json").read_text(encoding="utf-8"))
    assert len(data["events"]) >= 3
    for ev in data["events"]:
        assert ev["as_of_date"] < ev["event_date"] and ev["sources"]
        assert ev["ground_truth"]["labeled_by"] is None or ev["ground_truth"]["companies"]


def test_relation_pr():
    graph = {"nodes": [{"node_id": "a", "name": "가상메모리"}, {"node_id": "b", "name": "HBM"}],
             "edges": [{"source_id": "a", "target_id": "b", "relationship": "PRODUCES",
                        "verification_status": "VERIFIED_FACT"}]}
    gt = {"relations": [{"source": "가상메모리", "relationship": "PRODUCES", "target": "HBM"},
                        {"source": "가상메모리", "relationship": "REQUIRES", "target": "웨이퍼"}]}
    r = E.relation_precision_recall(graph, gt)
    assert r["precision"] == 1.0 and r["recall"] == 0.5

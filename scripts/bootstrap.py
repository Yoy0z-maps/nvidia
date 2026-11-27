"""Seed 기업 및 주변 기업 자동 데이터 구축.

uv run python -m scripts.bootstrap --seed 000660 --max-companies 20 --max-hops 2
단계: 기업 사전 → Seed 공시 수집/전처리 → 임베딩/인덱스 → 관련 기업 탐색(수집·임베딩 포함) → 그래프 저장
각 단계는 재실행 시 완료된 데이터를 재사용한다.
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
    ap.add_argument("--seed", help="Seed 종목코드 (기본: SEED_STOCK_CODE)")
    ap.add_argument("--max-companies", type=int)
    ap.add_argument("--max-hops", type=int)
    ap.add_argument("--skip-discovery", action="store_true", help="Seed 데이터만 구축")
    ap.add_argument("--discovery-llm-calls", type=int, help="관련 기업 탐색 LLM 호출 상한 (기본: 기업 수 x 8)")
    ap.add_argument("--per-node", type=int, help="탐색 노드당 후보 기업 수 (기본: MAX_DISCOVERY_PER_NODE)")
    args = ap.parse_args()

    s = Settings.from_env()
    if args.seed:
        s.seed_stock_code = args.seed.zfill(6)
    if args.max_companies:
        s.max_companies = args.max_companies
    if args.per_node:
        s.max_discovery_per_node = args.per_node
    if args.max_hops is not None:
        s.max_discovery_hops = args.max_hops
    setup_logging(s)

    from src.discovery.company_discovery import CompanyDiscovery
    from src.pipeline import build_app, collect_company, embed_and_index, make_ingest_fn, seed_company

    app = build_app(s)
    app.llm.check_model()
    seed = seed_company(app)
    print(f"Seed: {seed['corp_name']} ({seed['stock_code']}, corp_code={seed['corp_code']})")

    print("\n[1/3] Seed 공시 수집 및 전처리")
    res = collect_company(app, seed["corp_code"], seed["corp_name"], seed["stock_code"], progress=print,
                          discovery_hop=0, discovery_reason="seed")
    for d in res["selected"]:
        print(f"  - {d['rcept_dt']} {d['report_nm']} ({d['type']})")
    if res["failed"]:
        print("  실패:", json.dumps(res["failed"], ensure_ascii=False))

    print("\n[2/3] 임베딩 및 FAISS 인덱스")
    stats = embed_and_index(app, print)
    print(json.dumps({k: v for k, v in stats.items() if k != "failed_chunk_ids"}, ensure_ascii=False))

    if args.skip_discovery:
        return 0
    print("\n[3/3] 관련 기업 탐색")
    graph = app.load_graph()
    app.llm.reset_budget(args.discovery_llm_calls or s.max_companies * 8)
    app.retriever.reset_budget(None)
    disc = CompanyDiscovery(app.storage, app.llm, app.retriever, app.resolver, make_ingest_fn(app, print), graph,
                            max_companies=s.max_companies, max_hops=s.max_discovery_hops,
                            max_per_node=s.max_discovery_per_node, top_k=s.retrieval_top_k, progress_cb=print)
    rep = disc.run(seed)
    vp, hp = graph.save(s.graph_dir)
    print(f"\n탐색 종료: {rep.stopped_reason}")
    for p in rep.processed:
        print(f"  - {p['corp_name']}({p['stock_code']}) hop={p['hop']} {p['method']} → {p['status']}")
    print(f"그래프 저장: {vp.name}, {hp.name} / 추가된 관계 {rep.edges_added}개")
    print(f"LLM 사용량: {json.dumps(app.llm.usage.to_dict(), ensure_ascii=False)}")
    print(f"데이터 현황: {app.storage.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

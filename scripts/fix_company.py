"""잘못 식별되어 수집된 기업을 검색 대상에서 제외하고, 원래 가리키던 기업으로 관계를 다시 연결한다.

예) 한솔테크닉스 공시의 '한솔오리온텍'(비상장 종속회사)이 이름 매칭 오류로 식품회사 '오리온'(271560)으로 연결된 경우

uv run python -m scripts.fix_company --exclude 271560 --reason "한솔오리온텍 오매칭" --relink
uv run python -m scripts.fix_company --include 271560        # 되돌리기

- 수집한 원본 XML·Chunk·벡터는 삭제하지 않는다 (검색 대상에서만 제외, --include로 복구 가능)
- 지식 그래프에서 해당 기업과 연결된 관계는 제거하고, 수정 전 파일은 data/graph/backup_<시각>/ 에 남긴다
- --relink: 이 기업으로 잘못 연결됐던 공시 언급을 바뀐 이름 매칭 규칙으로 다시 식별하고 관계를 재검증한다 (LLM 사용)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402


def backup_graph(graph_dir: Path) -> Path | None:
    files = [p for p in (graph_dir / "verified_graph.json", graph_dir / "hypotheses.json") if p.exists()]
    if not files:
        return None
    dst = graph_dir / f"backup_{dt.datetime.now():%Y%m%d-%H%M%S}"
    dst.mkdir(parents=True)
    for p in files:
        shutil.copy2(p, dst / p.name)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exclude", metavar="STOCK_CODE")
    g.add_argument("--include", metavar="STOCK_CODE")
    ap.add_argument("--reason", default="")
    ap.add_argument("--relink", action="store_true", help="잘못 연결됐던 공시 언급을 다시 식별·검증 (LLM 사용)")
    args = ap.parse_args()

    s = Settings.from_env()
    setup_logging(s)
    from src.graph.graph_store import GraphStore, company_node_id
    from src.storage import Storage

    st = Storage(s.db_path)
    code = (args.exclude or args.include).zfill(6)
    comp = st.get_company_by_stock(code)
    if not comp:
        print(f"지식 기반에 종목코드 {code} 기업이 없습니다.")
        return 1

    if args.include:
        has_ready = st.one("SELECT 1 FROM documents WHERE corp_code=? AND ingestion_status='READY'", (comp["corp_code"],))
        st.set_company_status(comp["corp_code"], "READY" if has_ready else "CHUNKED")
        st.log("fix_company", "INFO", f"제외 해제: {comp['corp_name']}({code})", comp["corp_code"])
        print(f"{comp['corp_name']}({code}) 를 다시 검색 대상에 포함했습니다. 그래프 관계는 backup 폴더에서 복구하거나 "
              "bootstrap을 다시 실행하면 재구성됩니다.")
        return 0

    # ---- 제외 ----
    st.set_company_status(comp["corp_code"], "EXCLUDED")
    with st.tx() as c:
        c.execute("UPDATE companies SET discovery_reason = COALESCE(discovery_reason,'') || ? WHERE corp_code=?",
                  (f" | 제외: {args.reason}", comp["corp_code"]))
        n_cand = c.execute("UPDATE discovery_candidates SET status='MISRESOLVED', status_reason=? "
                           "WHERE resolved_corp_code=? AND status IN ('VERIFIED_RELATED','UNVERIFIED','DUPLICATE')",
                           (f"오매칭으로 제외: {args.reason}", comp["corp_code"])).rowcount
    st.log("fix_company", "INFO", f"검색 대상 제외: {comp['corp_name']}({code}) - {args.reason}", comp["corp_code"])
    print(f"[1] {comp['corp_name']}({code}) 검색 대상 제외 (문서·벡터는 보존). 탐색 후보 {n_cand}건을 MISRESOLVED로 표시")

    backup = backup_graph(s.graph_dir)
    gs = GraphStore.load(s.graph_dir)
    nid = company_node_id(code)
    removed = 0
    if nid in gs.g:
        removed = gs.g.degree(nid)
        gs.g.remove_node(nid)
    gs.save(s.graph_dir)
    print(f"[2] 지식 그래프에서 관계 {removed}개 제거 (백업: {backup})")

    if not args.relink:
        print("    --relink 를 붙이면 원래 가리키던 기업으로 다시 연결합니다.")
        return 0

    # ---- 재연결 ----
    from src.discovery.company_discovery import CompanyDiscovery
    from src.pipeline import build_app
    app = build_app(s, need_dart=bool(s.dart_api_key))
    app.llm.check_model()
    disc = CompanyDiscovery(app.storage, app.llm, app.retriever, app.resolver, lambda *a: False, gs)
    rows = app.storage.query("SELECT * FROM discovery_candidates WHERE resolved_corp_code=? AND status='MISRESOLVED' "
                             "AND method='DOCUMENT'", (comp["corp_code"],))
    added = 0
    for r in rows:
        ev = json.loads(r["evidence_json"] or "{}")
        new, reason = app.resolver.resolve_for_graph(r["candidate_name"])
        if new is None or new.corp_code == comp["corp_code"]:
            print(f"    - '{r['candidate_name']}': 다시 식별 불가 ({reason}) → 관계를 만들지 않음")
            continue
        source = app.storage.get_company(r["source_corp_code"])
        item = {"name": new.corp_name, "kind": "COMPANY", "direction": r["direction"] or "UNKNOWN",
                "relation": r["rationale"] or "", "chunk_id": ev.get("chunk_id"), "quote": ev.get("quote") or ""}
        o = disc.validator.validate(disc._grounded_to_relation(source, item))
        e = disc._add(o, r["hop"] or 1)
        label = f"{new.corp_name} ({new.stock_code or '비상장'})"
        if e:
            added += 1
            with app.storage.tx() as c:
                c.execute("INSERT INTO discovery_candidates(created_at, source_corp_code, candidate_name, method, hop, "
                          "direction, rationale, evidence_json, resolved_corp_code, status, status_reason) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (dt.datetime.now().isoformat(timespec="seconds"), r["source_corp_code"], r["candidate_name"],
                           "DOCUMENT", r["hop"], r["direction"], r["rationale"], r["evidence_json"], new.corp_code,
                           "UNLISTED" if not new.stock_code else "RELINKED", f"재식별: {label}"))
        print(f"    - '{r['candidate_name']}' → {label}: {o.status.value}" + ("" if e else " (그래프에 추가 안 됨)"))
    gs.save(s.graph_dir)
    print(f"[3] 재연결 관계 {added}개 추가. 빈 자리는 bootstrap을 다시 실행하면 새 기업으로 채워집니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

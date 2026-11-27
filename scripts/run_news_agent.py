"""[1] 뉴스·공시 agent 실행: 분야를 받아 C1 JSON을 만든다 (기업은 SK하이닉스 고정).

uv run python -m scripts.run_news_agent --field HBM
uv run python -m scripts.run_news_agent --field DRAM --prompt-only   # 프롬프트만 출력

결과: data/runs/c1/c1_{분야}_{request_id}.json  (C1 계약: src/contracts.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="C0(분야) → C1 JSON")
    ap.add_argument("--field", required=True, help="분야 (예: HBM, DRAM, 낸드)")
    ap.add_argument("--prompt-only", action="store_true", help="저장 후 프롬프트 본문만 출력")
    args = ap.parse_args()

    s = Settings.from_env()
    missing = [k for k, v in (("OPENAI_API_KEY", s.openai_api_key), ("DART_API_KEY", s.dart_api_key),
                              ("NAVER_CLIENT_ID", s.naver_client_id),
                              ("NAVER_CLIENT_SECRET", s.naver_client_secret)) if not v]
    if missing:
        print(f".env에 다음 키가 없습니다: {', '.join(missing)}", file=sys.stderr)
        return 2
    setup_logging(s)

    from src.contracts import C0Input
    from src.news_agent.agent import build_agent, save_c1

    agent, close = build_agent(s)
    try:
        c1 = agent.run(C0Input(field=args.field))
    finally:
        close()
    path = save_c1(c1, s.runs_dir)

    if args.prompt_only:
        print(c1.prompt_draft)
    else:
        doc = c1.company_doc
        print(f"status      : {c1.status}")
        print(f"request_id  : {c1.request_id}")
        print(f"뉴스        : {len(c1.news)}건 (본문 요약 {sum(1 for n in c1.news if n.summary)}건)")
        print(f"핵심어      : {', '.join(c1.keywords)}")
        print(f"사업보고서  : {doc.status} {doc.report_name} {doc.rcept_dt} 섹션 {len(doc.sections)}개, "
              f"분야 발췌 {len(doc.field_highlights)}줄")
        print(f"프롬프트    : {len(c1.prompt_draft)}자")
        for e in c1.errors:
            print(f"  - {e}")
        print(f"저장        : {path}")
    return 0 if c1.status != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

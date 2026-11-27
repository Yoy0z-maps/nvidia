"""임베딩 대기 Chunk 처리 및 FAISS 인덱스 동기화/재구축.

uv run python -m scripts.build_index [--rebuild]
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
    ap.add_argument("--rebuild", action="store_true", help="SQLite 메타데이터 기준으로 인덱스 전체 재구축")
    args = ap.parse_args()
    s = Settings.from_env()
    setup_logging(s)
    from src.pipeline import build_app, embed_and_index
    app = build_app(s, need_dart=False)
    if args.rebuild:
        app.vector_store.rebuild()
    out = embed_and_index(app, print)
    print(json.dumps({k: v for k, v in out.items() if k != "failed_chunk_ids"}, ensure_ascii=False, indent=2))
    if out["failed_chunk_ids"]:
        print(f"실패 Chunk {len(out['failed_chunk_ids'])}개 - 재실행 시 다시 시도합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

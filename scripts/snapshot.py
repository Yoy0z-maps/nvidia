"""임베딩·그래프 데이터를 하나의 압축 파일로 내보내고 가져온다 (OpenAI 재호출 없이 공유).

내보내기:  uv run python -m scripts.snapshot export            → dist/market-ripple-data-YYYYMMDD.tar.gz
가져오기:  uv run python -m scripts.snapshot import <파일 또는 URL> [--force]

포함: metadata.db(문서·Chunk·임베딩 메타), embeddings/(벡터 .npy), index/(FAISS), graph/, runs/(분석 결과)
제외: raw/(DART 원본 ZIP), processed/, market/(KRX 시세 캐시), logs/, graph/backup_*, .env(API 키)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PREPROCESSING_VERSION, Settings  # noqa: E402

INCLUDE_DIRS = ["embeddings", "index", "graph", "runs"]
MANIFEST = "snapshot_manifest.json"


def _skip(rel: Path) -> bool:
    return (rel.parts[:1] == ("graph",) and len(rel.parts) > 1 and rel.parts[1].startswith("backup_")) \
        or rel.name in {".DS_Store"} or rel.suffix in {".lock", ".tmp"}


def _counts(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
        return {"companies": q("select count(*) from companies"), "documents": q("select count(*) from documents"),
                "chunks": q("select count(*) from chunks"), "embeddings": q("select count(*) from embeddings")}
    finally:
        con.close()


def export(s: Settings, out: Path | None) -> Path:
    data = s.data_dir
    if not s.db_path.exists():
        raise SystemExit(f"{s.db_path} 가 없습니다. 먼저 bootstrap/build_index 를 실행하세요.")
    out = out or ROOT / "dist" / f"market-ripple-data-{dt.date.today():%Y%m%d}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # 실행 중인 앱이 있어도 일관된 사본이 나오도록 SQLite backup API 사용 (원본은 읽기만 함)
        db_copy = Path(tmp) / "metadata.db"
        src = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
        dst = sqlite3.connect(db_copy)
        src.backup(dst)
        dst.close(); src.close()
        manifest = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "embedding_model": s.embedding_model, "embedding_dimensions": s.embedding_dimensions,
            "preprocessing_version": PREPROCESSING_VERSION, "counts": _counts(db_copy),
        }
        (Path(tmp) / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with tarfile.open(out, "w:gz") as tar:
            tar.add(Path(tmp) / MANIFEST, arcname=MANIFEST)
            tar.add(db_copy, arcname="metadata.db")
            for d in INCLUDE_DIRS:
                base = data / d
                if not base.exists():
                    continue
                for f in sorted(base.rglob("*")):
                    rel = f.relative_to(data)
                    if f.is_file() and not _skip(rel):
                        tar.add(f, arcname=str(rel))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({**manifest, "file": str(out), "size_mb": round(out.stat().st_size / 1e6, 1), "sha256": sha},
                     ensure_ascii=False, indent=2))
    return out


def _download(url: str, dest: Path) -> Path:
    print(f"다운로드: {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def _safe_members(tar: tarfile.TarFile):
    allowed = {MANIFEST, "metadata.db", *INCLUDE_DIRS}
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()) or p.parts[0] not in allowed:
            raise SystemExit(f"허용되지 않는 항목이 들어 있습니다: {m.name}")
        yield m


def import_(s: Settings, source: str, force: bool) -> None:
    data = s.data_dir
    if s.db_path.exists() and not force:
        raise SystemExit(f"{s.db_path} 가 이미 있습니다. 덮어쓰려면 --force (기존 data/는 data_backup_* 으로 옮겨 둡니다).")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(source)
        if source.startswith(("http://", "https://")):
            path = _download(source, Path(tmp) / "snapshot.tar.gz")
        stage = Path(tmp) / "x"
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(stage, members=list(_safe_members(tar)))
        manifest = json.loads((stage / MANIFEST).read_text(encoding="utf-8"))
        if (manifest["embedding_model"], manifest["embedding_dimensions"]) != (s.embedding_model, s.embedding_dimensions):
            print(f"⚠ 스냅샷 임베딩 모델({manifest['embedding_model']}/{manifest['embedding_dimensions']})이 "
                  f".env 설정({s.embedding_model}/{s.embedding_dimensions})과 다릅니다. .env를 스냅샷에 맞추세요.")
        if manifest["preprocessing_version"] != PREPROCESSING_VERSION:
            print("⚠ 전처리 버전이 달라 새로 수집하는 문서는 다시 임베딩될 수 있습니다.")
        data.mkdir(parents=True, exist_ok=True)
        moved = []
        if force:
            backup = data.parent / f"data_backup_{dt.datetime.now():%Y%m%d-%H%M%S}"
            backup.mkdir()
            for name in ["metadata.db", "metadata.db-journal", "metadata.db-wal", "metadata.db-shm", *INCLUDE_DIRS]:
                if (data / name).exists():
                    shutil.move(str(data / name), backup / name)
                    moved.append(name)
            if moved:
                print(f"기존 데이터 {moved} → {backup}")
        for item in stage.iterdir():
            if item.name != MANIFEST:
                shutil.move(str(item), data / item.name)
    print(json.dumps({"imported": manifest["counts"], "created_at": manifest["created_at"]}, ensure_ascii=False))
    print("완료. `uv run streamlit run app.py` 로 바로 확인할 수 있습니다 (OpenAI 임베딩 재호출 없음).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="data/를 압축 파일로 내보내기")
    e.add_argument("--out", type=Path)
    i = sub.add_parser("import", help="압축 파일(경로 또는 URL)을 data/로 가져오기")
    i.add_argument("source")
    i.add_argument("--force", action="store_true", help="기존 data/를 백업 폴더로 옮기고 덮어쓰기")
    args = ap.parse_args()
    s = Settings.from_env()
    if args.cmd == "export":
        export(s, args.out)
    else:
        import_(s, args.source, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

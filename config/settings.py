"""환경변수 기반 설정.

모든 API Key는 환경변수(.env)로만 관리하며 로그나 repr에 노출하지 않는다.
"""
from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_VERSION = "v1"


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # 잘못된 설정은 조용히 무시하지 않는다
        raise ValueError(f"환경변수 {name} 값이 정수가 아닙니다: {raw!r}") from exc


@dataclass
class Settings:
    openai_api_key: str = field(default="", repr=False)
    dart_api_key: str = field(default="", repr=False)
    krx_api_key: str = field(default="", repr=False)

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    seed_company: str = "SK하이닉스"
    seed_stock_code: str = "000660"

    max_companies: int = 20
    max_discovery_hops: int = 2
    max_analysis_hops: int = 3
    max_branches_per_hop: int = 3
    max_discovery_per_node: int = 5   # 기업 탐색 시 노드당 후보 수
    max_retrieval_calls: int = 20
    max_llm_calls: int = 25

    chunk_size: int = 600
    chunk_overlap: int = 80
    embedding_batch_size: int = 64
    retrieval_top_k: int = 5
    max_major_disclosures: int = 10
    collect_interim_reports: bool = False   # 반기·분기보고서 수집 여부 (기본: 사업보고서만)

    data_dir: Path = PROJECT_ROOT / "data"
    log_level: str = "INFO"

    # korea-stock-mcp@1.4.1 의 bin 스크립트 shebang("#!/usr/bin/env node --max-old-space-size=4096")은
    # Linux에서 실행되지 않으므로 기본값은 로컬 설치본을 node로 직접 실행한다 (scripts.check_environment --install-mcp).
    mcp_command: str = "node"
    mcp_args: list[str] = field(default_factory=lambda: [".mcp/node_modules/korea-stock-mcp/dist/index.js"])

    # 분석 Hop 상한 (스펙: 최대 5)
    HARD_MAX_ANALYSIS_HOPS: int = 5

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        if not data_dir.is_absolute():
            data_dir = (PROJECT_ROOT / data_dir).resolve()
        s = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            dart_api_key=os.getenv("DART_API_KEY", ""),
            krx_api_key=os.getenv("KRX_API_KEY", "").strip().strip("\"'"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 1536),
            seed_company=os.getenv("SEED_COMPANY", "SK하이닉스"),
            seed_stock_code=os.getenv("SEED_STOCK_CODE", "000660").zfill(6),
            max_companies=_int("MAX_COMPANIES", 20),
            max_discovery_hops=_int("MAX_DISCOVERY_HOPS", 2),
            max_analysis_hops=_int("MAX_ANALYSIS_HOPS", 3),
            max_branches_per_hop=_int("MAX_BRANCHES_PER_HOP", 3),
            max_discovery_per_node=_int("MAX_DISCOVERY_PER_NODE", 5),
            max_retrieval_calls=_int("MAX_RETRIEVAL_CALLS", 20),
            max_llm_calls=_int("MAX_LLM_CALLS", 25),
            chunk_size=_int("CHUNK_SIZE", 600),
            chunk_overlap=_int("CHUNK_OVERLAP", 80),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 64),
            retrieval_top_k=_int("RETRIEVAL_TOP_K", 5),
            max_major_disclosures=_int("MAX_MAJOR_DISCLOSURES", 10),
            collect_interim_reports=os.getenv("COLLECT_INTERIM_REPORTS", "false").strip().lower() in ("1", "true", "yes"),
            data_dir=data_dir,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            mcp_command=os.getenv("MCP_COMMAND", "node"),
            mcp_args=[a if not a.startswith(".") else str((PROJECT_ROOT / a).resolve())
                      for a in shlex.split(os.getenv("MCP_ARGS", ".mcp/node_modules/korea-stock-mcp/dist/index.js"))],
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")
        if not 1 <= self.max_analysis_hops <= self.HARD_MAX_ANALYSIS_HOPS:
            raise ValueError("MAX_ANALYSIS_HOPS는 1~5 범위여야 합니다.")
        if self.max_companies < 1:
            raise ValueError("MAX_COMPANIES는 1 이상이어야 합니다.")

    # ---- 경로 ----
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index" / "vector.index"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"

    @property
    def market_dir(self) -> Path:
        return self.data_dir / "market"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.embeddings_dir, self.index_path.parent,
                  self.graph_dir, self.market_dir, self.runs_dir, self.evaluation_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """API Key를 제외한 설정값 (실행 기록용)."""
        out = {}
        for f in fields(self):
            if f.name.endswith("api_key"):
                continue
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        return out


class _SecretFilter(logging.Filter):
    """로그 메시지에서 API Key 값을 마스킹한다."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = [s for s in secrets if s and len(s) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secrets:
            msg = record.getMessage()
            for s in self.secrets:
                msg = msg.replace(s, "***")
            record.msg, record.args = msg, ()
        return True


def setup_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if getattr(root, "_market_ripple_configured", False):
        return
    root.setLevel(settings.log_level.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    secret_filter = _SecretFilter([settings.openai_api_key, settings.dart_api_key, settings.krx_api_key])
    for h in (logging.StreamHandler(), logging.FileHandler(settings.logs_dir / "market_ripple.log", encoding="utf-8")):
        h.setFormatter(fmt)
        h.addFilter(secret_filter)
        root.addHandler(h)
    # httpx는 요청 URL(crtfc_key 포함)을 INFO로 남기므로 WARNING으로 올린다
    logging.getLogger("httpx").setLevel(logging.WARNING)
    root._market_ripple_configured = True  # type: ignore[attr-defined]

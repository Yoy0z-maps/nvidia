"""테스트 공용 픽스처. Mock 데이터는 테스트에서만 사용한다."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from src.storage import Storage  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


class CharTokenizer:
    """테스트용 토크나이저 (문자 1개 = 토큰 1개). 실제 실행에서는 tiktoken을 사용한다."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class FakeEmbedder:
    """해시 기반 결정적 bag-of-words 임베더 (테스트 전용)."""

    def __init__(self, dims: int = 64, model: str = "fake-embedding"):
        self.model = model
        self.dimensions = dims
        self.calls = 0
        self.total_tokens = 0
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts.extend(texts)
        out = np.zeros((len(texts), self.dimensions), dtype="float32")
        for i, t in enumerate(texts):
            for w in t.replace("\n", " ").split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dimensions] += 1.0
            self.total_tokens += len(t)
        return out


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=20, embedding_dimensions=64,
                 embedding_model="fake-embedding", dart_api_key="TESTKEY", openai_api_key="")
    s.ensure_dirs()
    return s


@pytest.fixture
def storage(settings) -> Storage:
    st = Storage(settings.db_path)
    yield st
    st.close()


@pytest.fixture
def sample_xml() -> bytes:
    return (FIXTURES / "sample_annual.xml").read_bytes()

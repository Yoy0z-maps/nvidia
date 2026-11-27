"""Section-aware Recursive Chunking (실제 토큰 수 기준)."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from src.ingestion.section_selector import SelectedSection
from src.ingestion.xml_parser import ParagraphBlock, TableBlock

log = logging.getLogger(__name__)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


def get_tokenizer(model: str = "text-embedding-3-small") -> Tokenizer:
    """tiktoken 인코딩. 최초 실행 시 인코딩 파일을 내려받으므로 네트워크가 필요하다."""
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def normalize_for_hash(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", s).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    section_path: str
    section_type: str
    chunk_order: int
    raw_text: str
    embedding_text: str
    token_count: int
    content_hash: str
    kind: str

    def to_row(self) -> dict:
        return self.__dict__.copy()


_SENT_SPLIT = re.compile(r"(?<=[다요음함됨임]\.)\s+|(?<=[\.\?\!])\s+(?=[가-힣A-Z0-9\(])|\n+")


class SectionChunker:
    def __init__(self, tokenizer: Tokenizer, chunk_size: int = 600, overlap: int = 80):
        if overlap >= chunk_size:
            raise ValueError("overlap은 chunk_size보다 작아야 합니다.")
        self.tok = tokenizer
        self.size = chunk_size
        self.overlap = overlap

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))

    # ---- 텍스트 분할 ----
    def _split_long(self, text: str) -> list[str]:
        """chunk_size를 넘는 문단을 문장 → 토큰 단위로 재귀 분할."""
        if self.n_tokens(text) <= self.size:
            return [text]
        sents = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
        if len(sents) > 1:
            out, buf = [], ""
            for s in sents:
                cand = f"{buf} {s}".strip() if buf else s
                if self.n_tokens(cand) <= self.size:
                    buf = cand
                else:
                    if buf:
                        out.append(buf)
                    if self.n_tokens(s) > self.size:
                        out.extend(self._hard_split(s))
                        buf = ""
                    else:
                        buf = s
            if buf:
                out.append(buf)
            return out
        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        ids = self.tok.encode(text)
        step = self.size - self.overlap
        return [self.tok.decode(ids[i:i + self.size]) for i in range(0, len(ids), step)
                if ids[i:i + self.size]]

    def _tail(self, text: str) -> str:
        ids = self.tok.encode(text)
        if len(ids) <= self.overlap:
            return text
        return self.tok.decode(ids[-self.overlap:])

    # ---- 표 분할 ----
    def _split_table(self, table: TableBlock) -> list[str]:
        md = table.to_markdown()
        if self.n_tokens(md) <= self.size:
            return [md]
        rows = table.rows if table.header else table.rows[1:]
        pieces: list[str] = []
        cur: list[list[str]] = []
        for r in rows:
            cand = cur + [r]
            if self.n_tokens(self._table_md(table, cand)) <= self.size or not cur:
                cur = cand
            else:
                pieces.append(self._table_md(table, cur))
                cur = [r]
        if cur:
            pieces.append(self._table_md(table, cur))
        # 한 행이 너무 긴 경우 토큰 단위로 강제 분할
        out: list[str] = []
        for p in pieces:
            out.extend([p] if self.n_tokens(p) <= self.size else self._hard_split(p))
        return out

    @staticmethod
    def _table_md(table: TableBlock, rows: list[list[str]]) -> str:
        """열 이름·단위·제목을 유지한 부분 표."""
        if table.header:
            return table.to_markdown(rows)
        first = table.rows[0] if table.rows else []
        return TableBlock(table.title, table.unit, [first], rows).to_markdown(rows) + "\n(표 일부)"

    # ---- 메인 ----
    def chunk_sections(self, selected: list[SelectedSection], *, corp_name: str, stock_code: str,
                       document_type: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        order = 0
        for sel in selected:
            header = (f"기업명: {corp_name}\n종목코드: {stock_code}\n문서 유형: {document_type}\n"
                      f"섹션: {sel.path_str}\n\n")
            texts: list[tuple[str, str]] = []  # (kind, text)
            buf = ""
            for block in sel.section.blocks:
                if isinstance(block, TableBlock):
                    if buf:
                        texts.append(("text", buf))
                        buf = ""
                    texts.extend(("table", t) for t in self._split_table(block))
                    continue
                assert isinstance(block, ParagraphBlock)
                for piece in self._split_long(block.text):
                    cand = f"{buf}\n{piece}" if buf else piece
                    if self.n_tokens(cand) <= self.size:
                        buf = cand
                    else:
                        texts.append(("text", buf))
                        tail = self._tail(buf)
                        joined = f"{tail}\n{piece}"
                        buf = joined if self.n_tokens(joined) <= self.size else piece
            if buf:
                texts.append(("text", buf))
            for kind, raw in texts:
                raw = raw.strip()
                if not raw:
                    continue
                chunks.append(Chunk(
                    section_path=sel.path_str, section_type=sel.section_type, chunk_order=order,
                    raw_text=raw, embedding_text=header + raw, token_count=self.n_tokens(raw),
                    content_hash=content_hash(raw), kind=kind))
                order += 1
        return chunks

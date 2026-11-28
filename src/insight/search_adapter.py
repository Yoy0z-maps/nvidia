"""[4] 검색 어댑터: [3]의 검색 결과를 C4 형식으로 맞추고, 근거 후보를 정리해 기업별 쿼터를 적용한다.

    SearchRequest 목록 (기업 + 검색어)
      → search(query, corp_codes=[기업], k)      # PRD C4의 검색 함수 모양
      → 정리 (가장자리의 깨진 글자, 섹션명의 LIBRARY) + 제목만 있는 청크 제외
      → 중복·비슷한 청크 제거 + 기업별 최대 quota개
      → AdapterResult (status, errors, chunks)

search()를 함수로 주입받으므로, [3]이 PRD 모양의 search()를 따로 주면 make_c4_search() 대신 그대로 넣으면 된다.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from itertools import zip_longest
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.contracts import Status
from src.ingestion.chunker import content_hash
from src.retrieval.retriever import BudgetExceeded, Retriever
from src.schemas import SearchResult
from src.storage import Storage

DEFAULT_QUOTA = 5  # 기업당 남길 청크 수 (PRD [4] 3단계의 k=5)

CORP_CODE = r"^\d{8}$"

BROKEN = "�"  # 청크를 토큰 단위로 자를 때 한글 한 글자가 잘리면 남는 글자
SHINGLE = 5  # 비슷한 청크를 비교하는 조각 길이 (공백을 뺀 글자 수)
NEAR_DUP = 0.8  # 한 청크의 조각이 이 비율 이상 다른 청크에 있으면 같은 내용으로 본다
HEADING_MAX = 60  # 이보다 짧은 청크만 "제목뿐"인지 검사한다 (공백을 뺀 글자 수)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# 제목·표 머리의 시작 모양: "[표]", "(단위", "| ...", "1.", "2-3)", "(1)", "가.", "①", "ⓙ"
_HEADING_START = re.compile(r"^\s*(?:\[|【|\(단위|\||\d+(?:[-.]\d+)*\s*[.)]|\(\s*[\d가-하]+\s*\)|[가-하]\s*[.)]|[①-⑳ⓐ-ⓩ])")
_SENTENCE = re.compile(r"니다|다\.|음\.|함\.")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class C4Chunk(_Model):
    """search()가 돌려주는 한 건 = chunks.jsonl 한 줄 + score (PRD C4)."""

    chunk_id: str = Field(pattern=r"^\d{8}-\d{14}-\d{4,}$")  # corp_code-rcept_no-문서 안 순번
    corp_code: str = Field(pattern=CORP_CODE)
    item_ids: list[str]
    company: str = Field(min_length=1)
    report: str = Field(min_length=1)
    rcept_no: str = Field(pattern=r"^\d{14}$")
    section: str
    source_url: str = Field(min_length=1)
    # 원문. 인용 검증이 이 문자열을 대조한다. retrieve()는 가장자리의 깨진 글자만 잘라 낸다 (원문의 일부로 남는다).
    text: str = Field(min_length=1)
    score: float  # 벡터 유사도. 키워드로만 찾은 청크는 0.0이므로 이 값으로 다시 정렬하지 않는다.


class SearchRequest(_Model):
    """검색 1건. 1단계(검색 대상·질의 생성)가 만든다. 기업을 비우면 전체 검색이 되므로 필수다."""

    corp_code: str = Field(pattern=CORP_CODE)
    query: str = Field(min_length=1)


class AdapterResult(_Model):
    status: Status
    errors: list[str]
    chunks: dict[str, list[C4Chunk]]  # corp_code -> 관련도 순 청크 (결과가 있는 기업만)


class SearchFn(Protocol):
    def __call__(self, query: str, corp_codes: list[str] | None = None, k: int = 5) -> list[dict]: ...


# ---------------------------------------------------------------------------
# 1) 매핑: Retriever 결과 -> C4
# ---------------------------------------------------------------------------
def item_ids_by_corp(pool: Mapping) -> dict[str, list[str]]:
    """C2 사전 풀에서 corp_code -> item_ids (기업 자신 + 이 기업을 대표 기업으로 둔 분야·카테고리)."""
    code_of = {i["item_id"]: i["corp_code"] for i in pool["items"] if i["item_type"] == "company"}
    out = {code: [item_id] for item_id, code in code_of.items()}
    for item in pool["items"]:
        for rep in item.get("rep_companies") or []:  # rep는 corp_code가 아니라 "COM-..." 항목 ID
            out[code_of[rep]].append(item["item_id"])
    return out


def to_c4_chunk(result: SearchResult, storage: Storage, item_ids: Mapping[str, list[str]]) -> C4Chunk:
    row = storage.get_chunk(result.chunk_id)  # SearchResult에 없는 접수번호·보고서명·순번을 DB에서 채운다
    if row is None:
        raise LookupError(f"DB에 없는 chunk_id입니다: {result.chunk_id}")
    return C4Chunk(
        chunk_id=f"{result.corp_code}-{row['rcept_no']}-{row['chunk_order']:04d}",
        corp_code=result.corp_code,
        item_ids=item_ids.get(result.corp_code, [f"COM-{result.corp_code}"]),
        company=result.corp_name,
        report=row["report_name"],
        rcept_no=row["rcept_no"],
        section=result.section_path,
        source_url=result.source_url,
        text=result.raw_text,
        score=result.similarity_score,
    )


def make_c4_search(retriever: Retriever, storage: Storage,
                   item_ids: Mapping[str, list[str]] | None = None) -> SearchFn:
    """기존 Retriever를 PRD C4의 search(query, corp_codes, k) -> list[dict] 모양으로 감싼다."""

    def search(query: str, corp_codes: list[str] | None = None, k: int = 5) -> list[dict]:
        results = retriever.search(query, top_k=k, corp_codes=corp_codes)
        return [to_c4_chunk(r, storage, item_ids or {}).model_dump() for r in results]

    return search


# ---------------------------------------------------------------------------
# 2) 청크 정리
# ---------------------------------------------------------------------------
def is_heading_only(text: str) -> bool:
    """근거가 될 내용 없이 제목·단위·표 머리만 있는 청크인가. 확실하지 않으면 False (남긴다)."""
    if len("".join(text.split())) >= HEADING_MAX or not _HEADING_START.match(text):
        return False
    if _SENTENCE.search(text) or any(len(n.replace(",", "").replace(".", "")) >= 3 for n in _NUMBER.findall(text)):
        return False
    lines = [line.strip() for line in text.splitlines()]
    separator = next((i for i, line in enumerate(lines) if line.startswith("|---")), None)
    # 마크다운 표는 구분선(|---) 아래가 데이터 행이다
    return separator is None or not any(line.startswith("|") for line in lines[separator + 1:])


def clean_chunk(chunk: C4Chunk) -> C4Chunk | None:
    """깨진 글자와 파서가 남긴 섹션 이름을 정리한다. 근거로 쓸 내용이 없으면 None."""
    text = chunk.text.strip(BROKEN)  # 가장자리만 자른다. 중간을 고치면 인용 검증이 원문과 달라진다.
    if not text.strip() or is_heading_only(text):
        return None
    section = " > ".join(part for part in chunk.section.split(" > ") if part != "LIBRARY")
    return chunk.model_copy(update={"text": text, "section": section})


# ---------------------------------------------------------------------------
# 3) 중복 제거 + 기업별 쿼터
# ---------------------------------------------------------------------------
def _shingles(text: str) -> set[str]:
    compact = "".join(text.split()).replace(BROKEN, "")
    return {compact[i:i + SHINGLE] for i in range(max(1, len(compact) - SHINGLE + 1))}


def covers(a: str, b: str) -> bool:
    """b의 내용이 a에 이미 들어 있는가.

    공시는 같은 서식에 금액·날짜만 다른 경우가 많으므로, b에 a에 없는 숫자가 하나라도 있으면 다른 내용으로 본다.
    """
    if not set(_NUMBER.findall(b)) <= set(_NUMBER.findall(a)):
        return False
    part = _shingles(b)
    return len(part & _shingles(a)) / len(part) >= NEAR_DUP


def _merge(ranked_lists: list[list[C4Chunk]], quota: int) -> list[C4Chunk]:
    """검색어별 결과를 1등끼리, 2등끼리 번갈아 뽑는다. 같은 내용은 한 번만 남긴다.

    뒤에 나온 청크가 이미 뽑은 청크를 포함하면, 순위는 그대로 두고 내용이 더 많은 쪽으로 바꾼다.
    """
    picked: list[C4Chunk] = []
    seen: set[str] = set()
    for same_rank in zip_longest(*ranked_lists):
        for chunk in same_rank:
            if chunk is None:
                continue
            keys = {chunk.chunk_id, content_hash(chunk.text)}  # 해시는 공백 차이를 무시한다
            if seen & keys:
                continue
            seen |= keys
            # 바꾸기를 먼저 확인한다. 조금만 더 긴 청크는 서로를 포함하므로, 버리기를 먼저 하면 긴 쪽이 사라진다.
            smaller = next((i for i, p in enumerate(picked)
                            if len(chunk.text) > len(p.text) and covers(chunk.text, p.text)), None)
            if smaller is not None:
                picked[smaller] = chunk
            elif len(picked) < quota and not any(covers(p.text, chunk.text) for p in picked):
                picked.append(chunk)
    return picked


def retrieve(search: SearchFn, requests: Iterable[SearchRequest], quota: int = DEFAULT_QUOTA) -> AdapterResult:
    """기업마다 따로 검색해 기업당 최대 quota개를 돌려준다. 요청 순서(1차 먼저)대로 검색한다."""
    if quota < 1:
        raise ValueError("quota는 1 이상이어야 합니다.")
    queries: dict[str, dict[str, None]] = {}  # dict를 순서 있는 집합으로 사용 (같은 검색어 반복 제거)
    for req in requests:
        queries.setdefault(req.corp_code, {})[req.query] = None

    errors: list[str] = []
    chunks: dict[str, list[C4Chunk]] = {}
    for corp_code, corp_queries in queries.items():
        ranked: list[list[C4Chunk]] = []
        for query in corp_queries:
            try:
                # 중복을 버린 뒤에도 quota를 채울 수 있게 2배를 받는다
                found = [C4Chunk.model_validate(d) for d in search(query, corp_codes=[corp_code], k=quota * 2)]
            except BudgetExceeded:
                errors.append(f"{corp_code}: 검색 호출 예산이 소진됐습니다.")
                continue
            except Exception as exc:  # 한 건이 실패해도 나머지 기업은 계속 검색한다
                errors.append(f"{corp_code}: 검색 실패 ({type(exc).__name__}).")
                continue
            own = [c for c in found if c.corp_code == corp_code]
            if len(own) != len(found):
                errors.append(f"{corp_code}: 다른 기업의 청크 {len(found) - len(own)}건을 제외했습니다.")
            ranked.append([c for c in map(clean_chunk, own) if c is not None])
        merged = _merge(ranked, quota)
        if merged:
            chunks[corp_code] = merged
        else:
            errors.append(f"{corp_code}: 검색 결과가 없습니다.")

    if not queries:
        errors.append("검색 요청이 없습니다.")
    status: Status = "error" if not chunks else "partial" if errors else "ok"
    return AdapterResult(status=status, errors=list(dict.fromkeys(errors)), chunks=chunks)

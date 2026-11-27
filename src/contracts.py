"""역할 간 데이터 계약 (C0, C1).

[1] 뉴스·공시 agent의 입력(C0)과 출력(C1)을 정의한다. [2] 차수 배정, [4] 인사이트, 화면은
이 모델로 C1을 검증한 뒤 사용한다. 필드를 바꿀 때는 SCHEMA_VERSION을 올리고 팀 리뷰를 받는다.

프로토타입 범위: 기업은 SK하이닉스로 고정하고, 사용자는 분야(HBM, DRAM 등)만 고른다.
"""
from __future__ import annotations

import datetime as dt
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.1"

# 프로토타입 고정 기업 (config/settings.py의 SEED_COMPANY / SEED_STOCK_CODE 기본값과 같다)
ANCHOR_COMPANY = "SK하이닉스"
ANCHOR_STOCK_CODE = "000660"
AnchorCompany = Literal["SK하이닉스"]

Status = Literal["ok", "partial", "error"]
CompanyDocStatus = Literal["ok", "not_found", "parse_error"]

# 사업보고서 I·II 하위 목차. section_id -> (대목차, 소목차)
SECTION_TITLES: dict[str, tuple[str, str]] = {
    "I-1": ("I. 회사의 개요", "1. 회사의 개요"),
    "I-2": ("I. 회사의 개요", "2. 회사의 연혁"),
    "I-3": ("I. 회사의 개요", "3. 자본금 변동사항"),
    "I-4": ("I. 회사의 개요", "4. 주식의 총수 등"),
    "I-5": ("I. 회사의 개요", "5. 정관에 관한 사항"),
    "II-1": ("II. 사업의 내용", "1. 사업의 개요"),
    "II-2": ("II. 사업의 내용", "2. 주요 제품 및 서비스"),
    "II-3": ("II. 사업의 내용", "3. 원재료 및 생산설비"),
    "II-4": ("II. 사업의 내용", "4. 매출 및 수주상황"),
    "II-5": ("II. 사업의 내용", "5. 위험관리 및 파생거래"),
    "II-6": ("II. 사업의 내용", "6. 주요계약 및 연구개발활동"),
    "II-7": ("II. 사업의 내용", "7. 기타 참고사항"),
}

MAX_NEWS = 10
# issue_summary(프롬프트에 들어가는 뉴스 종합 요약)는 빈 줄로 구분된 2~3문단
ISSUE_SUMMARY_PARAGRAPHS = (2, 3)


def split_paragraphs(text: str) -> list[str]:
    """빈 줄(공백만 있는 줄 포함)로 문단을 나눈다."""
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


class _Contract(BaseModel):
    # 계약에 없는 필드(오타 포함)는 받지 않는다
    model_config = ConfigDict(extra="forbid")


def _check_section_id(v: str) -> str:
    if v not in SECTION_TITLES:
        raise ValueError(f"section_id는 {list(SECTION_TITLES)} 중 하나여야 합니다: {v!r}")
    return v


def _check_http_url(v: str) -> str:
    if v and not re.match(r"https?://", v):
        raise ValueError(f"http(s) URL이어야 합니다: {v!r}")
    return v


# ---------------------------------------------------------------------------
# C0 사용자 입력
# ---------------------------------------------------------------------------
class C0Input(_Contract):
    """사용자는 분야만 입력한다. company는 SK하이닉스로 고정이며 생략하면 자동으로 채워진다."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company: AnchorCompany = ANCHOR_COMPANY
    field: str = Field(min_length=1)

    @field_validator("field")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("빈 문자열은 허용하지 않습니다.")
        return v


# ---------------------------------------------------------------------------
# C1 구성 요소
# ---------------------------------------------------------------------------
class NewsItem(_Contract):
    rank: int = Field(ge=1, le=MAX_NEWS)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    origin_url: str = ""
    published_at: str  # ISO 8601, 시간대 포함
    summary: str = ""

    @field_validator("url", "origin_url")
    @classmethod
    def _v_url(cls, v: str) -> str:
        return _check_http_url(v)

    @field_validator("published_at")
    @classmethod
    def _v_published_at(cls, v: str) -> str:
        try:
            parsed = dt.datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"published_at은 ISO 8601이어야 합니다: {v!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"published_at에 시간대가 필요합니다: {v!r}")
        return v


class DocSection(_Contract):
    section_id: str
    part: str
    title: str
    text: str
    chars: int = Field(ge=0)

    @field_validator("section_id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_section_id(v)

    @model_validator(mode="after")
    def _v_consistency(self) -> "DocSection":
        part, _ = SECTION_TITLES[self.section_id]
        if self.part != part:
            raise ValueError(f"{self.section_id}의 part는 {part!r}이어야 합니다: {self.part!r}")
        if self.chars != len(self.text):
            raise ValueError(f"{self.section_id}: chars({self.chars})가 text 길이({len(self.text)})와 다릅니다.")
        return self


class FieldHighlight(_Contract):
    section_id: str
    text: str = Field(min_length=1)

    @field_validator("section_id")
    @classmethod
    def _v_id(cls, v: str) -> str:
        return _check_section_id(v)


class CompanyDoc(_Contract):
    """입력 기업의 최신 사업보고서 I·II 원문 (요약하지 않는다)."""

    status: CompanyDocStatus
    corp_code: str = ""
    stock_code: str = ""
    corp_name: str = ""
    report_name: str = ""
    fiscal_period: str = ""
    rcept_no: str = ""
    rcept_dt: str = ""  # YYYY-MM-DD
    source_url: str = ""
    sections: list[DocSection] = Field(default_factory=list)
    field_highlights: list[FieldHighlight] = Field(default_factory=list)
    error: str = ""

    @field_validator("corp_code")
    @classmethod
    def _v_corp_code(cls, v: str) -> str:
        if v and not re.fullmatch(r"\d{8}", v):
            raise ValueError(f"corp_code는 8자리 숫자 문자열이어야 합니다: {v!r}")
        return v

    @field_validator("rcept_no")
    @classmethod
    def _v_rcept_no(cls, v: str) -> str:
        if v and not re.fullmatch(r"\d{14}", v):
            raise ValueError(f"rcept_no는 14자리 숫자 문자열이어야 합니다: {v!r}")
        return v

    @field_validator("rcept_dt")
    @classmethod
    def _v_rcept_dt(cls, v: str) -> str:
        if v:
            try:
                dt.date.fromisoformat(v)
            except ValueError as exc:
                raise ValueError(f"rcept_dt는 YYYY-MM-DD여야 합니다: {v!r}") from exc
        return v

    @field_validator("source_url")
    @classmethod
    def _v_source_url(cls, v: str) -> str:
        return _check_http_url(v)

    @model_validator(mode="after")
    def _v_status(self) -> "CompanyDoc":
        if self.status == "ok":
            missing = [k for k in ("corp_code", "corp_name", "rcept_no", "rcept_dt", "source_url")
                       if not getattr(self, k)]
            if missing:
                raise ValueError(f"status=ok이면 {missing} 값이 필요합니다.")
            if not self.sections:
                raise ValueError("status=ok이면 sections가 1개 이상 필요합니다.")
            if self.corp_name != ANCHOR_COMPANY or self.stock_code != ANCHOR_STOCK_CODE:
                raise ValueError(f"company_doc은 {ANCHOR_COMPANY}({ANCHOR_STOCK_CODE}) 문서여야 합니다: "
                                 f"{self.corp_name}({self.stock_code})")
        elif not self.error:
            raise ValueError(f"status={self.status}이면 error에 사유를 적어야 합니다.")
        ids = [s.section_id for s in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError(f"section_id가 중복되었습니다: {ids}")
        known = set(ids)
        orphan = [h.section_id for h in self.field_highlights if h.section_id not in known]
        if orphan:
            raise ValueError(f"field_highlights가 sections에 없는 섹션을 가리킵니다: {orphan}")
        return self

    @property
    def missing_section_ids(self) -> list[str]:
        have = {s.section_id for s in self.sections}
        return [sid for sid in SECTION_TITLES if sid not in have]


# ---------------------------------------------------------------------------
# C1 프롬프트·공시 번들
# ---------------------------------------------------------------------------
class C1Bundle(_Contract):
    """[1]의 출력.

    prompt_draft/prompt_final = 최근 이슈 요약 + 분석 지시. 사업보고서 원문은 company_doc으로 함께 넘기고
    프롬프트에는 옮겨 적지 않는다. 뉴스 목록(news)과 핵심어(keywords)도 프롬프트 밖에 따로 둔다.
    """

    schema_version: Literal["1.1"] = SCHEMA_VERSION
    status: Status
    errors: list[str] = Field(default_factory=list)
    request_id: str = Field(min_length=1)
    company: AnchorCompany = ANCHOR_COMPANY
    field: str = Field(min_length=1)
    news: list[NewsItem] = Field(default_factory=list, max_length=MAX_NEWS)
    issue_summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    prompt_draft: str = ""
    prompt_final: str = ""
    edited: bool = False
    company_doc: CompanyDoc

    @model_validator(mode="after")
    def _v_bundle(self) -> "C1Bundle":
        ranks = [n.rank for n in self.news]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"news.rank는 1부터 순서대로여야 합니다: {ranks}")
        if self.issue_summary.strip():
            lo, hi = ISSUE_SUMMARY_PARAGRAPHS
            n = len(split_paragraphs(self.issue_summary))
            if not lo <= n <= hi:
                raise ValueError(f"issue_summary는 빈 줄로 구분된 {lo}~{hi}문단이어야 합니다: {n}문단")
        elif self.status == "ok":
            raise ValueError("status=ok이면 issue_summary가 필요합니다.")
        if self.edited != (self.prompt_final != self.prompt_draft):
            raise ValueError("edited는 prompt_final이 prompt_draft와 다를 때만 true입니다.")
        if self.status == "ok":
            if not self.news or not self.prompt_final:
                raise ValueError("status=ok이면 news와 prompt_final이 필요합니다.")
            if self.company_doc.status != "ok":
                raise ValueError("status=ok이면 company_doc.status도 ok여야 합니다.")
        elif not self.errors:
            raise ValueError(f"status={self.status}이면 errors에 사유를 적어야 합니다.")
        return self

    def finalize(self, edited_prompt: str | None) -> "C1Bundle":
        """화면에서 사용자가 확정한 프롬프트를 반영한 새 번들을 돌려준다."""
        final = self.prompt_draft if edited_prompt is None else edited_prompt.strip()
        if not final:
            raise ValueError("확정 프롬프트가 비어 있습니다.")
        data = self.model_dump()
        data.update(prompt_final=final, edited=final != self.prompt_draft)
        return C1Bundle.model_validate(data)

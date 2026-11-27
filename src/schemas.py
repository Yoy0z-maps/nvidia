"""Pydantic 스키마.

실제 검증 로직이 걸린 스키마(Edge 등)는 여기에서 강제한다.
"""
from __future__ import annotations

import datetime as dt
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 열거형
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    COMPANY = "COMPANY"
    INDUSTRY = "INDUSTRY"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    RESOURCE = "RESOURCE"
    EVENT = "EVENT"
    BOTTLENECK = "BOTTLENECK"


class RelationshipType(str, Enum):
    PRODUCES = "PRODUCES"
    REQUIRES = "REQUIRES"
    SUPPLIES_TO = "SUPPLIES_TO"
    OPERATES_IN = "OPERATES_IN"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    INCREASES_DEMAND_FOR = "INCREASES_DEMAND_FOR"
    DECREASES_DEMAND_FOR = "DECREASES_DEMAND_FOR"
    SUBSTITUTES_FOR = "SUBSTITUTES_FOR"
    CONSTRAINS = "CONSTRAINS"
    SOLVES = "SOLVES"
    EXPOSED_TO = "EXPOSED_TO"


class VerificationStatus(str, Enum):
    VERIFIED_FACT = "VERIFIED_FACT"
    SUPPORTED_INFERENCE = "SUPPORTED_INFERENCE"
    HYPOTHESIS = "HYPOTHESIS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTED = "CONTRADICTED"


# 강한 → 약한 순서. 경로 상태는 가장 약한 Edge를 따른다.
STATUS_STRENGTH = {
    VerificationStatus.VERIFIED_FACT: 4,
    VerificationStatus.SUPPORTED_INFERENCE: 3,
    VerificationStatus.HYPOTHESIS: 2,
    VerificationStatus.INSUFFICIENT_EVIDENCE: 1,
    VerificationStatus.CONTRADICTED: 0,
}
VERIFIED_STATUSES = {VerificationStatus.VERIFIED_FACT, VerificationStatus.SUPPORTED_INFERENCE}


class EventType(str, Enum):
    DEMAND_INCREASE = "DEMAND_INCREASE"
    DEMAND_DECREASE = "DEMAND_DECREASE"
    SUPPLY_DISRUPTION = "SUPPLY_DISRUPTION"
    CAPACITY_EXPANSION = "CAPACITY_EXPANSION"
    TECHNOLOGY_BREAKTHROUGH = "TECHNOLOGY_BREAKTHROUGH"
    REGULATION_CHANGE = "REGULATION_CHANGE"
    PRICE_INCREASE = "PRICE_INCREASE"
    PRICE_DECREASE = "PRICE_DECREASE"
    CONTRACT_SIGNED = "CONTRACT_SIGNED"
    UNKNOWN = "UNKNOWN"


class DocumentType(str, Enum):
    ANNUAL = "사업보고서"
    HALF = "반기보고서"
    QUARTER = "분기보고서"
    SUPPLY_CONTRACT = "단일판매·공급계약체결"
    NEW_FACILITY = "신규시설투자등"
    INVESTMENT_DECISION = "투자판단관련주요경영사항"
    OUTLOOK = "영업실적등에대한전망"
    SUSPENSION = "영업정지"


PERIODIC_TYPES = {DocumentType.ANNUAL, DocumentType.HALF, DocumentType.QUARTER}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_date(v: str | None) -> str | None:
    if v is None:
        return v
    if not _DATE_RE.match(v):
        raise ValueError(f"날짜 형식은 YYYY-MM-DD 이어야 합니다: {v!r}")
    dt.date.fromisoformat(v)
    return v


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------
class CorpInfo(BaseModel):
    corp_code: str
    corp_name: str
    stock_code: str = ""
    modify_date: str = ""

    @field_validator("corp_code")
    @classmethod
    def _corp_code(cls, v: str) -> str:
        if not re.fullmatch(r"\d{8}", v):
            raise ValueError(f"corp_code는 8자리 숫자 문자열이어야 합니다: {v!r}")
        return v


class DisclosureItem(BaseModel):
    corp_code: str
    corp_name: str
    stock_code: str = ""
    corp_cls: str = ""
    report_nm: str
    rcept_no: str
    rcept_dt: str  # YYYYMMDD
    rm: str = ""

    @property
    def published_at(self) -> str:
        d = self.rcept_dt
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    @property
    def is_correction(self) -> bool:
        return self.report_nm.strip().startswith("[") and "정정" in self.report_nm.split("]")[0]


class SearchResult(BaseModel):
    chunk_id: int
    corp_code: str
    corp_name: str
    stock_code: str
    document_id: str
    document_type: str
    section_path: str
    section_type: str
    published_at: str
    source_url: str
    raw_text: str
    similarity_score: float
    retrieval_method: str = "vector"


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    chunk_id: int
    document_id: str
    quote: str = Field(min_length=4)
    source_url: str = ""
    published_at: str

    @field_validator("published_at")
    @classmethod
    def _v_date(cls, v: str) -> str:
        return _check_date(v)  # type: ignore[return-value]


class Node(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    aliases: list[str] = Field(default_factory=list)
    stock_code: str | None = None
    corp_code: str | None = None


class Edge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relationship: RelationshipType
    claim_type: Literal["FACT", "INFERENCE", "HYPOTHESIS"]
    verification_status: VerificationStatus
    evidence: list[Evidence] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    created_at: str
    as_of_date: str | None = None
    hop: int = 0

    @field_validator("created_at", "as_of_date")
    @classmethod
    def _v_dates(cls, v: str | None) -> str | None:
        return _check_date(v)

    @model_validator(mode="after")
    def _evidence_rules(self) -> "Edge":
        if self.verification_status in VERIFIED_STATUSES and not self.evidence:
            raise ValueError(f"{self.verification_status.value} 상태에는 최소 1개의 evidence가 필요합니다.")
        if self.verification_status == VerificationStatus.VERIFIED_FACT and self.claim_type != "FACT":
            raise ValueError("VERIFIED_FACT는 claim_type=FACT 에만 부여할 수 있습니다.")
        if self.as_of_date:
            for ev in self.evidence:
                if ev.published_at > self.as_of_date:
                    raise ValueError(f"평가 기준일({self.as_of_date}) 이후 문서를 근거로 사용할 수 없습니다.")
        return self


# ---------------------------------------------------------------------------
# LLM Structured Outputs
# (OpenAI Structured Outputs 제약으로 기본값 없는 필드 + nullable 위주로 정의)
# ---------------------------------------------------------------------------
class ExtractedEvent(BaseModel):
    event_type: EventType
    is_hypothetical: bool = Field(description="가상의 시나리오면 true, 실제 발생한 뉴스로 서술되면 false")
    entity_names: list[str] = Field(description="입력에 명시된 기업명")
    industries: list[str]
    products: list[str]
    event_date: str | None = Field(description="입력에 명시된 날짜(YYYY-MM-DD). 없으면 null")
    summary: str
    assumptions: list[str]
    input_quote: str = Field(description="이벤트의 핵심을 담은 입력 원문 구절 (그대로 복사)")


class Event(ExtractedEvent):
    entity_ids: list[str] = Field(default_factory=list)


class CandidateRelation(BaseModel):
    source_name: str
    source_type: NodeType
    target_name: str
    target_type: NodeType
    relationship: RelationshipType
    claim_type: Literal["FACT", "INFERENCE", "HYPOTHESIS"]
    chunk_id: int | None = Field(description="근거 Chunk ID. 근거가 없으면 null")
    quote: str | None = Field(description="근거 Chunk 원문에서 그대로 복사한 구절. 없으면 null")
    reasoning: str
    conditions: list[str]


class CandidateRelationList(BaseModel):
    relations: list[CandidateRelation]


class SemanticVerdict(BaseModel):
    supports_claim: bool
    entities_correct: bool
    direction_correct: bool
    direct_relation: bool = Field(description="직접 관계면 true, 간접(산업 수준) 관계면 false")
    additional_conditions: list[str]
    contradicting_points: list[str]
    verdict: Literal["VERIFIED_FACT", "SUPPORTED_INFERENCE", "HYPOTHESIS", "INSUFFICIENT_EVIDENCE", "CONTRADICTED"]
    explanation: str


class IndexedSemanticVerdict(SemanticVerdict):
    index: int = Field(description="'### 주장 N' 의 N")


class SemanticVerdictBatch(BaseModel):
    verdicts: list[IndexedSemanticVerdict]


class SearchQueries(BaseModel):
    queries: list[str]


class MentionedEntity(BaseModel):
    name: str
    entity_kind: Literal["COMPANY", "INDUSTRY", "PRODUCT", "TECHNOLOGY", "RESOURCE"]
    relation_to_seed: str
    direction: Literal["CUSTOMER", "SUPPLIER", "COMPETITOR", "PARTNER", "AFFILIATE", "INDUSTRY_LINK", "UNKNOWN"]
    chunk_id: int
    quote: str


class MentionedEntityList(BaseModel):
    entities: list[MentionedEntity]


class HypothesisCompany(BaseModel):
    company_name: str = Field(description="한국거래소 상장사의 한글 정식 명칭")
    exploration_direction: Literal[
        "주요제품", "원자재", "생산설비", "전방산업", "후방산업", "공급망", "경쟁기업", "기술적병목", "대체기술", "문제해결산업"
    ]
    rationale: str


class HypothesisCompanyList(BaseModel):
    candidates: list[HypothesisCompany]


class CompanyImpact(BaseModel):
    corp_name: str
    stock_code: str | None
    path_description: str
    relationship_types: list[str]
    impact_scope: Literal["DIRECT", "INDIRECT"]
    economic_impact: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "UNCERTAIN"]
    conditions: list[str]
    counter_scenario: str
    technical_relevance: str
    revenue_exposure: str = Field(description="공시로 확인되지 않으면 'UNKNOWN'")
    cited_chunk_ids: list[int]
    role: Literal["직접 영향", "고객사", "공급사", "경쟁사", "다운스트림", "업스트림", "간접 영향"] = Field(
        description="이벤트 대상과의 관계")
    impact_factor: str = Field(description="영향 요인 한 구절 (예: 'GPU 공급 차질', '대체 생산 수요')")


class TierItem(BaseModel):
    name: str = Field(description="그래프에 있는 노드 이름 그대로")
    effect: str = Field(description="'~ 가능성' 형태의 한 줄 영향")


class RippleTier(BaseModel):
    tier: int = Field(description="파급 단계 (1, 2, 3 ...)")
    title: str = Field(description="단계 이름 (예: '고객사 및 경쟁사', '다운스트림 산업')")
    items: list[TierItem]


class CompanyImpactList(BaseModel):
    summary: str
    tiers: list[RippleTier]
    companies: list[CompanyImpact]
    caveat: str = Field(description="실제 영향이 달라질 수 있는 조건 (공정, 제품, 재고, 대체 가능성 등)")


class HypothesisInsight(BaseModel):
    title: str = Field(description="한 줄 인사이트 제목")
    tier: int = Field(description="이벤트로부터 몇 차 파급인지 (1, 2, 3 ...)")
    kind: Literal["기업", "산업", "제품", "기술", "원자재", "병목"]
    direction: Literal["POSITIVE", "NEGATIVE", "UNCERTAIN"]
    rationale: str = Field(description="왜 그렇게 볼 수 있는지 (산업 구조 관점, 2~3문장)")
    related_listed_companies: list[str] = Field(description="관련 국내 상장사 한글 정식 명칭 (모르면 빈 목록)")
    what_to_check: str = Field(description="이 가설을 확인하려면 무엇을 봐야 하는지")


class HypothesisInsightList(BaseModel):
    insights: list[HypothesisInsight]


class SectionTitleClassification(BaseModel):
    selected_titles: list[str]

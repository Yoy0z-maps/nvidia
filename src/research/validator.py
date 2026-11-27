"""관계 후보 검증: Structural → Quote → Semantic.

인용 존재 여부(Quote)와 의미 검증(Semantic)은 분리되어 있으며 각각 결과를 남긴다.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from src.llm import LLM, LLMBudgetExceeded, load_prompt
from src.schemas import (STATUS_STRENGTH, CandidateRelation, Evidence, NodeType, RelationshipType,
                         SemanticVerdict, SemanticVerdictBatch, VerificationStatus)
from src.storage import Storage

log = logging.getLogger(__name__)
VS = VerificationStatus

CLAIM_CAP = {"FACT": VS.VERIFIED_FACT, "INFERENCE": VS.SUPPORTED_INFERENCE, "HYPOTHESIS": VS.HYPOTHESIS}


def normalize_quote_text(s: str) -> str:
    """공백과 유니코드 표기 차이만 정규화한다 (의미·수치는 바꾸지 않음)."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_quote_text(quote)
    return bool(q) and q in normalize_quote_text(text)


def weaker(a: VS, b: VS) -> VS:
    return a if STATUS_STRENGTH[a] <= STATUS_STRENGTH[b] else b


@dataclass
class ValidationOutcome:
    candidate: CandidateRelation
    status: VerificationStatus
    structural_ok: bool
    quote_ok: bool | None
    semantic: SemanticVerdict | None
    evidence: list[Evidence] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    source_company: dict | None = None
    target_company: dict | None = None

    def to_dict(self) -> dict:
        return {"candidate": self.candidate.model_dump(mode="json"), "status": self.status.value,
                "structural_ok": self.structural_ok, "quote_ok": self.quote_ok,
                "semantic": self.semantic.model_dump(mode="json") if self.semantic else None,
                "reasons": self.reasons, "conditions": self.conditions}


class RelationValidator:
    def __init__(self, storage: Storage, llm: LLM | None,
                 company_resolver: Callable[[str], dict | None], as_of_date: str | None = None):
        self.storage = storage
        self.llm = llm
        self.resolve_company = company_resolver
        self.as_of = as_of_date

    # ---- 1. Structural ----
    def structural(self, cand: CandidateRelation | dict) -> tuple[CandidateRelation | None, list[str], dict]:
        reasons: list[str] = []
        ctx: dict = {}
        if isinstance(cand, dict):
            try:
                cand = CandidateRelation(**cand)
            except ValidationError as exc:
                return None, [f"스키마 오류: {exc.errors()[:3]}"], ctx
        for side, name, ntype in (("source", cand.source_name, cand.source_type),
                                  ("target", cand.target_name, cand.target_type)):
            if ntype == NodeType.COMPANY:
                comp = self.resolve_company(name)
                if not comp:
                    reasons.append(f"존재하지 않거나 식별 불가한 기업: {name}")
                ctx[f"{side}_company"] = comp
        if cand.claim_type != "HYPOTHESIS":
            if cand.chunk_id is None or not cand.quote:
                reasons.append("FACT/INFERENCE 주장에 chunk_id 또는 quote가 없음")
            else:
                chunk = self.storage.get_chunk(cand.chunk_id)
                if not chunk:
                    reasons.append(f"존재하지 않는 Chunk ID: {cand.chunk_id}")
                else:
                    ctx["chunk"] = chunk
                    try:
                        dt.date.fromisoformat(chunk["published_at"])
                    except ValueError:
                        reasons.append(f"유효하지 않은 공개일: {chunk['published_at']}")
                    if self.as_of and chunk["published_at"] > self.as_of:
                        reasons.append(f"평가 기준일({self.as_of}) 이후 문서 근거 사용")
        if cand.source_name.strip() == cand.target_name.strip() and cand.source_type == cand.target_type:
            reasons.append("자기 자신으로의 관계")
        return cand, reasons, ctx

    # ---- 2. Quote ----
    @staticmethod
    def quote_check(cand: CandidateRelation, chunk: dict) -> bool:
        return bool(cand.quote) and quote_in_text(cand.quote, chunk["raw_text"])

    # ---- 3. Semantic ----
    @staticmethod
    def _claim_block(cand: CandidateRelation, chunk: dict) -> str:
        return (f"[주장]\nsource: {cand.source_name} ({cand.source_type.value})\n"
                f"relationship: {cand.relationship.value}\ntarget: {cand.target_name} ({cand.target_type.value})\n"
                f"claim_type: {cand.claim_type}\nreasoning: {cand.reasoning}\n\n"
                f"[인용]\n{cand.quote}\n\n[Chunk 전문 - {chunk['corp_name']} / {chunk['section_path']} / "
                f"{chunk['published_at']}]\n{chunk['raw_text']}")

    def semantic(self, cand: CandidateRelation, chunk: dict) -> SemanticVerdict | None:
        if self.llm is None:
            return None
        return self.llm.parse(load_prompt("verification", "semantic"), self._claim_block(cand, chunk),
                              SemanticVerdict, purpose="verification")

    def semantic_batch(self, items: list[tuple[CandidateRelation, dict]]) -> dict[int, SemanticVerdict]:
        """여러 주장을 한 번의 LLM 호출로 검증한다 (주장마다 독립적으로 판정)."""
        if self.llm is None or not items:
            return {}
        blocks = [f"### 주장 {i}\n{self._claim_block(c, ch)}" for i, (c, ch) in enumerate(items)]
        prompt = (load_prompt("verification", "semantic")
                  + "\n\n여러 주장이 '### 주장 N' 형식으로 주어진다. 각 주장을 서로 독립적으로 판정하고, "
                    "모든 주장에 대해 index=N 인 결과를 하나씩 반환한다.")
        out = self.llm.parse(prompt, "\n\n".join(blocks), SemanticVerdictBatch, purpose="verification")
        return {v.index: SemanticVerdict(**v.model_dump(exclude={"index"})) for v in out.verdicts
                if 0 <= v.index < len(items)}

    # ---- 전체 ----
    def _prepare(self, cand: CandidateRelation | dict, run_semantic: bool
                 ) -> tuple[ValidationOutcome, dict | None, Evidence | None]:
        """Structural + Quote 검증. 의미 검증이 필요하면 (outcome, chunk, evidence)를, 아니면 chunk=None."""
        parsed, reasons, ctx = self.structural(cand)
        if parsed is None:
            dummy = CandidateRelation(source_name="?", source_type=NodeType.INDUSTRY, target_name="?",
                                      target_type=NodeType.INDUSTRY, relationship=RelationshipType.EXPOSED_TO,
                                      claim_type="HYPOTHESIS", chunk_id=None, quote=None, reasoning="", conditions=[])
            return ValidationOutcome(dummy, VS.INSUFFICIENT_EVIDENCE, False, None, None, reasons=reasons), None, None
        cand = parsed
        out = ValidationOutcome(cand, VS.HYPOTHESIS, not reasons, None, None, conditions=list(cand.conditions),
                                reasons=reasons, source_company=ctx.get("source_company"),
                                target_company=ctx.get("target_company"))
        if reasons:
            # 기업 식별 실패 / 근거 불량 → 사실로 확정하지 않음
            fatal = any(r.startswith(("존재하지 않", "평가 기준일", "자기 자신", "유효하지 않은")) for r in reasons)
            out.status = VS.INSUFFICIENT_EVIDENCE if fatal or cand.claim_type != "HYPOTHESIS" else VS.HYPOTHESIS
            return out, None, None
        if cand.claim_type == "HYPOTHESIS":
            out.status = VS.HYPOTHESIS
            return out, None, None
        chunk = ctx["chunk"]
        out.quote_ok = self.quote_check(cand, chunk)
        if not out.quote_ok:
            out.reasons.append("인용 구절이 원문 Chunk에 존재하지 않음")
            out.status = VS.INSUFFICIENT_EVIDENCE
            return out, None, None
        ev = Evidence(chunk_id=chunk["chunk_id"], document_id=chunk["document_id"], quote=cand.quote,
                      source_url=chunk.get("source_url") or "", published_at=chunk["published_at"])
        if not run_semantic or self.llm is None:
            # 의미 검증 없이 인용만 확인된 경우 사실로 확정하지 않는다
            out.status = VS.HYPOTHESIS
            out.evidence = [ev]
            out.reasons.append("의미 검증 미실행 - 인용 존재만 확인")
            return out, None, None
        return out, chunk, ev

    @staticmethod
    def _apply(out: ValidationOutcome, verdict: SemanticVerdict | None, ev: Evidence, missing_reason: str) -> None:
        cand = out.candidate
        out.evidence = [ev]
        if verdict is None:
            out.status = VS.HYPOTHESIS
            out.reasons.append(missing_reason)
            return
        out.semantic = verdict
        status = VS(verdict.verdict)
        if verdict.contradicting_points and status != VS.CONTRADICTED:
            out.conditions.extend(f"반대 근거: {p}" for p in verdict.contradicting_points)
        if not verdict.supports_claim:
            status = weaker(status, VS.HYPOTHESIS)
        if not verdict.entities_correct or not verdict.direction_correct:
            status = VS.CONTRADICTED if status == VS.CONTRADICTED else weaker(status, VS.INSUFFICIENT_EVIDENCE)
            out.reasons.append("기업/제품 연결 또는 관계 방향 오류")
        if cand.relationship == RelationshipType.SUPPLIES_TO and not verdict.direct_relation:
            status = weaker(status, VS.HYPOTHESIS)
            out.reasons.append("간접 관계를 직접 공급 관계로 확정하지 않음")
        out.status = weaker(status, CLAIM_CAP[cand.claim_type])
        out.conditions.extend(verdict.additional_conditions)

    def validate(self, cand: CandidateRelation | dict, run_semantic: bool = True) -> ValidationOutcome:
        out, chunk, ev = self._prepare(cand, run_semantic)
        if chunk is None:
            return out
        try:
            verdict = self.semantic(out.candidate, chunk)
        except LLMBudgetExceeded:
            self._apply(out, None, ev, "LLM 예산 소진으로 의미 검증 미실행")
            return out
        self._apply(out, verdict, ev, "의미 검증 결과 없음")
        return out

    def validate_many(self, cands: list[CandidateRelation | dict], run_semantic: bool = True
                      ) -> list[ValidationOutcome]:
        """Structural/Quote는 주장별로, Semantic은 통과한 주장들을 묶어 LLM 1회로 검증한다."""
        prepared = [self._prepare(c, run_semantic) for c in cands]
        pending = [(i, out, chunk, ev) for i, (out, chunk, ev) in enumerate(prepared) if chunk is not None]
        verdicts: dict[int, SemanticVerdict] = {}
        reason = "의미 검증 응답 누락"
        if pending:
            try:
                verdicts = self.semantic_batch([(out.candidate, chunk) for _, out, chunk, _ in pending])
            except LLMBudgetExceeded:
                reason = "LLM 예산 소진으로 의미 검증 미실행"
        for j, (_, out, _, ev) in enumerate(pending):
            self._apply(out, verdicts.get(j), ev, reason)
        return [p[0] for p in prepared]

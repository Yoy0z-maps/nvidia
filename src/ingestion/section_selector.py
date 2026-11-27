"""공시 문서 섹션 선택기.

규칙(config/sections.yaml)에 따라 사업 관련 섹션만 선택하고, 선택/제외/실패 내역과
전처리 전후 글자·토큰 수를 보고한다. 추출에 실패하면 문서 전체를 임베딩하지 않고 실패로 기록한다.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from config.settings import PROJECT_ROOT
from src.ingestion.xml_parser import ParsedDocument, Section

log = logging.getLogger(__name__)

_ROMAN = r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_PREFIX_RE = re.compile(
    rf"^\s*(?:(?:{_ROMAN})\s*[\.\)]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+\s*[\.\)]?|\d+(?:-\d+)*\s*[\.\)]|"
    r"[가-하]\s*[\.\)]|\(\s*[\d가-하]+\s*\)|[①-⑳]|【|\[)\s*", re.I)


def normalize_title(title: str) -> str:
    s = unicodedata.normalize("NFKC", title or "")
    prev = None
    while prev != s:  # 'II. 1.' 처럼 중첩된 번호 제거
        prev = s
        s = _PREFIX_RE.sub("", s, count=1)
    s = re.sub(r"[\s\.\,·ㆍ\-\(\)\[\]【】<>『』「」:：/&]", "", s)
    return s.lower()


@dataclass
class SelectedSection:
    section: Section
    section_type: str
    rule_id: str

    @property
    def path_str(self) -> str:
        return self.section.path_str


@dataclass
class SelectionReport:
    selected: list[dict] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    chars_before: int = 0
    chars_after: int = 0
    tokens_before: int | None = None
    tokens_after: int | None = None
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_rules(path: Path | None = None) -> dict:
    path = path or PROJECT_ROOT / "config" / "sections.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class SectionSelector:
    def __init__(self, rules: dict | None = None, token_counter: Callable[[str], int] | None = None,
                 title_classifier: Callable[[list[str]], list[str]] | None = None):
        self.rules = rules or load_rules()
        self.count_tokens = token_counter
        self.title_classifier = title_classifier

    @staticmethod
    def _matches(norm: str, patterns: list[str]) -> bool:
        return any(normalize_title(p) in norm for p in patterns)

    def select(self, doc: ParsedDocument, periodic: bool) -> tuple[list[SelectedSection], SelectionReport]:
        report = SelectionReport(chars_before=doc.raw_char_count)
        chosen: dict[int, SelectedSection] = {}
        excluded_ids: set[int] = set()

        def mark_excluded(sec: Section, reason: str) -> None:
            for s in sec.walk():
                excluded_ids.add(id(s))
            report.excluded.append({"path": sec.path_str, "reason": reason, "chars": sec.char_count()})

        def choose_subtree(sec: Section, stype: str, rule_id: str) -> None:
            for s in sec.walk():
                if id(s) in excluded_ids or id(s) in chosen:
                    continue
                if s.blocks:
                    chosen[id(s)] = SelectedSection(s, stype, rule_id)

        sections = doc.sections()
        cfg = self.rules["periodic" if periodic else "major"]
        for s in sections:
            norm = normalize_title(s.title)
            if self._matches(norm, cfg.get("exclude", [])):
                mark_excluded(s, "exclude 규칙")

        if periodic:
            for rule in cfg.get("include_subtree", []):
                hits = [s for s in sections if id(s) not in excluded_ids
                        and self._matches(normalize_title(s.title), rule["match"])]
                # 가장 상위 매칭만 사용 (하위 '사업의 개요' 등과 중복 방지)
                hits = [h for h in hits
                        if not any(h is not o and any(h is x for x in o.walk()) for o in hits)]
                if not hits:
                    report.failed_rules.append(rule["id"])
                for h in hits:
                    choose_subtree(h, rule["section_type"], rule["id"])

            for rule in cfg.get("include_only", []):
                parents = [s for s in sections if id(s) not in excluded_ids
                           and self._matches(normalize_title(s.title), rule["parent"])]
                found = False
                for p in parents:
                    subs = [c for c in p.walk() if c is not p]
                    targets = [c for c in subs if self._matches(normalize_title(c.title), rule["children"])]
                    if not subs and p.blocks:
                        targets = [p]  # 하위 섹션이 없는 경우 부모 자체
                    for t in targets:
                        choose_subtree(t, rule["section_type"], rule["id"])
                        found = True
                if not found:
                    report.failed_rules.append(rule["id"])

            for rule in cfg.get("include_standalone", []):
                for s in sections:
                    if id(s) not in excluded_ids and self._matches(normalize_title(s.title), rule["match"]):
                        choose_subtree(s, rule["section_type"], rule["id"])

            # 핵심 섹션(사업의 내용) 규칙 실패 시 제목 목록 기반 추가 분류
            if "business" in report.failed_rules and self.title_classifier is not None:
                titles = [s.path_str for s in sections if id(s) not in excluded_ids]
                picked = set(self.title_classifier(titles))
                for s in sections:
                    if s.path_str in picked and id(s) not in excluded_ids:
                        choose_subtree(s, "business", "llm_title_fallback")
                        report.used_fallback = True
        else:
            # 주요사항/거래소 공시: 표지/확인서 제외 본문 전체
            targets = sections or [doc.root]
            for s in targets:
                if id(s) not in excluded_ids and s.blocks and id(s) not in chosen:
                    chosen[id(s)] = SelectedSection(s, "major_disclosure", "major_body")
            if doc.root.blocks and id(doc.root) not in chosen:
                if not doc.root.path:
                    doc.root.path = [doc.document_name or "본문"]
                chosen[id(doc.root)] = SelectedSection(doc.root, "major_disclosure", "major_body")
            if not chosen:
                report.failed_rules.append("major_body")

        # 문서 순서 유지
        order = {id(s): i for i, s in enumerate([doc.root] + sections)}
        selected = sorted(chosen.values(), key=lambda x: order.get(id(x.section), 0))
        for sel in selected:
            text = sel.section.own_text()
            entry = {"path": sel.path_str, "section_type": sel.section_type, "rule": sel.rule_id,
                     "chars": len(text)}
            if self.count_tokens:
                entry["tokens"] = self.count_tokens(text)
            report.selected.append(entry)
        report.chars_after = sum(e["chars"] for e in report.selected)
        if self.count_tokens:
            report.tokens_after = sum(e.get("tokens", 0) for e in report.selected)
        log.info("섹션 선택: %d개 선택, %d개 제외, 실패 규칙=%s, 글자수 %d → %d",
                 len(report.selected), len(report.excluded), report.failed_rules,
                 report.chars_before, report.chars_after)
        return selected, report

"""DART 공시 XML 파서.

고정된 태그 경로를 가정하지 않고, 'TITLE 자식을 가진 컨테이너'를 섹션으로 인식한다.
DART 문서(dart4 형식)에서 확인된 구조:
    DOCUMENT > BODY > (COVER) > PART|LIBRARY > TITLE[ATOC=Y, AASSOCNOTE] > SECTION-1 > TITLE > SECTION-2 ...
    본문 블록: P, TABLE(THEAD/TBODY > TR > TH|TD|TE|TU), TABLE-GROUP(TITLE[ATOC=N] + TABLE)
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from lxml import etree

log = logging.getLogger(__name__)

CONTAINER_RE = re.compile(r"^(SECTION-\d+|PART|LIBRARY|CORRECTION|SECTION)$", re.I)
SKIP_TAGS = {"COVER", "TOC", "TOC-GROUP", "IMAGE", "IMG", "STYLE", "COLGROUP", "COL", "PGBRK", "EXTRACTION",
             "DOCUMENT-NAME", "FORMULA-VERSION", "COMPANY-NAME", "SUMMARY"}
CELL_TAGS = {"TH", "TD", "TE", "TU"}
UNIT_RE = re.compile(r"[\(\[]?\s*단\s*위\s*[:：]?\s*([^\)\]\n]{1,30})[\)\]]?")


def clean_text(s: str) -> str:
    s = s.replace(" ", " ").replace("&cr;", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def _text(el: etree._Element) -> str:
    return clean_text(" ".join(t for t in el.itertext()))


def _tag(el) -> str:
    return el.tag.upper() if isinstance(el.tag, str) else ""


@dataclass
class TableBlock:
    title: str
    unit: str
    header: list[list[str]]
    rows: list[list[str]]

    kind: str = "table"

    @property
    def n_cols(self) -> int:
        return max([len(r) for r in self.header + self.rows] or [0])

    def to_markdown(self, rows: list[list[str]] | None = None) -> str:
        rows = self.rows if rows is None else rows
        n = self.n_cols
        lines: list[str] = []
        if self.title:
            lines.append(f"[표] {self.title}")
        if self.unit:
            lines.append(f"(단위: {self.unit})")
        header = self.header or ([rows[0]] if rows else [])
        body = rows if self.header else rows[1:]

        def fmt(r: list[str]) -> str:
            r = [c.replace("|", "/").replace("\n", " ") for c in r] + [""] * (n - len(r))
            return "| " + " | ".join(r) + " |"

        for h in header:
            lines.append(fmt(h))
        if header:
            lines.append("|" + "---|" * n)
        lines.extend(fmt(r) for r in body)
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"title": self.title, "unit": self.unit, "header": self.header, "rows": self.rows}


@dataclass
class ParagraphBlock:
    text: str
    kind: str = "paragraph"


Block = TableBlock | ParagraphBlock


@dataclass(eq=False)
class Section:
    title: str
    level: int
    path: list[str]
    section_id: str = ""
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def own_text(self) -> str:
        parts = []
        for b in self.blocks:
            parts.append(b.text if isinstance(b, ParagraphBlock) else b.to_markdown())
        return "\n".join(parts)

    def char_count(self, recursive: bool = True) -> int:
        n = len(self.own_text())
        if recursive:
            n += sum(c.char_count() for c in self.children)
        return n


@dataclass
class ParsedDocument:
    document_name: str
    company_name: str
    root: Section
    raw_char_count: int
    content_hash: str

    def sections(self) -> list[Section]:
        return [s for s in self.root.walk() if s is not self.root]


def _parse_bytes(xml_bytes: bytes) -> etree._Element:
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, remove_comments=True)
    root = etree.fromstring(xml_bytes, parser=parser)
    if root is None or len(root) == 0:
        # 선언된 인코딩이 실제와 다를 때 (구형 공시의 EUC-KR 등)
        for enc in ("utf-8", "euc-kr", "cp949"):
            try:
                text = xml_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
            text = re.sub(r"^<\?xml[^>]*\?>", "", text.lstrip("﻿"))
            root = etree.fromstring(text.encode("utf-8"), parser=parser)
            if root is not None and len(root):
                break
    if root is None:
        raise ValueError("XML 파싱 실패: 루트 요소를 찾을 수 없습니다.")
    return root


def _is_container(el) -> bool:
    tag = _tag(el)
    if tag in ("TABLE-GROUP", "TABLE"):
        return False
    if CONTAINER_RE.match(tag):
        return True
    for ch in el:
        if _tag(ch) == "TITLE" and (ch.get("ATOC") or "").upper() == "Y":
            return True
    return False


def _row_cells(tr) -> list[str]:
    cells: list[str] = []
    for cell in tr:
        if _tag(cell) not in CELL_TAGS:
            continue
        txt = _text(cell)
        try:
            span = max(1, min(int(cell.get("COLSPAN", "1") or 1), 50))
        except ValueError:
            span = 1
        cells.extend([txt] * span)
    return cells


def _parse_table(table_el, title: str, unit: str) -> TableBlock | None:
    header: list[list[str]] = []
    rows: list[list[str]] = []
    for tr in table_el.iter("TR", "tr"):
        cells = _row_cells(tr)
        if not any(c.strip() for c in cells):
            continue
        parent_tag = _tag(tr.getparent())
        is_header = parent_tag == "THEAD" or all(_tag(c) == "TH" for c in tr if _tag(c) in CELL_TAGS)
        (header if is_header and not rows else rows).append(cells)
    if not header and not rows:
        return None
    # 표 내부에 단위 표기가 있는 경우 (예: TU 셀 또는 첫 행)
    if not unit:
        for r in (header + rows)[:2]:
            for c in r:
                m = UNIT_RE.search(c)
                if m:
                    unit = m.group(1).strip()
                    break
    return TableBlock(title=title, unit=unit, header=header, rows=rows)


def _collect_blocks(el, section: Section, level: int, ctx: dict) -> None:
    """el의 자식들을 순서대로 블록/하위 섹션으로 변환한다."""
    for child in el:
        tag = _tag(child)
        if not tag or tag in SKIP_TAGS:
            continue
        if tag == "TITLE":
            continue
        if _is_container(child):
            title_el = next((c for c in child if _tag(c) == "TITLE"), None)
            title = _text(title_el) if title_el is not None else tag
            sub = Section(title=title, level=level + 1, path=section.path + [title],
                          section_id=(title_el.get("AASSOCNOTE", "") if title_el is not None else ""))
            section.children.append(sub)
            _collect_blocks(child, sub, level + 1, {"last_para": ""})
            continue
        if tag == "TABLE-GROUP":
            title_el = next((c for c in child if _tag(c) == "TITLE"), None)
            g_title = _text(title_el) if title_el is not None else ""
            for t in child.iter("TABLE"):
                tb = _parse_table(t, g_title or ctx.get("pending_title", ""), ctx.get("pending_unit", ""))
                if tb:
                    section.blocks.append(tb)
            ctx["pending_title"] = ctx["pending_unit"] = ""
            continue
        if tag == "TABLE":
            tb = _parse_table(child, ctx.get("pending_title", ""), ctx.get("pending_unit", ""))
            if tb:
                section.blocks.append(tb)
            ctx["pending_title"] = ctx["pending_unit"] = ""
            continue
        # 문단 안에 표가 포함된 경우 재귀
        if any(_tag(d) == "TABLE" for d in child.iterdescendants()):
            _collect_blocks(child, section, level, ctx)
            continue
        txt = _text(child)
        if not txt:
            continue
        m = UNIT_RE.search(txt)
        if m and len(txt) <= 40:
            ctx["pending_unit"] = m.group(1).strip()
            # 단위 표기만 있는 문단은 다음 표에 귀속시킨다
            remaining = UNIT_RE.sub("", txt).strip()
            if remaining:
                ctx["pending_title"] = remaining
            continue
        if len(txt) <= 60 and not txt.endswith(("다.", "다", ".")):
            # 짧은 제목성 문단은 다음 표의 제목 후보로도 보관
            ctx["pending_title"] = txt
        else:
            ctx["pending_title"] = ""
        section.blocks.append(ParagraphBlock(text=txt))


def parse_dart_xml(xml_bytes: bytes) -> ParsedDocument:
    root_el = _parse_bytes(xml_bytes)
    doc_name = clean_text(root_el.findtext(".//DOCUMENT-NAME") or "")
    company = clean_text(root_el.findtext(".//COMPANY-NAME") or "")
    body = root_el.find(".//BODY")
    if body is None:
        body = root_el
    root = Section(title="(문서)", level=0, path=[])
    _collect_blocks(body, root, 0, {"last_para": ""})
    raw_chars = len(clean_text(" ".join(body.itertext())))
    return ParsedDocument(document_name=doc_name, company_name=company, root=root,
                          raw_char_count=raw_chars, content_hash=hashlib.sha256(xml_bytes).hexdigest())


def outline(doc: ParsedDocument, max_depth: int = 4) -> list[dict]:
    """XML 구조 조사용 목차 (scripts/inspect_xml.py 에서 사용)."""
    out = []
    for s in doc.sections():
        if s.level <= max_depth:
            out.append({"level": s.level, "title": s.title, "id": s.section_id,
                        "blocks": len(s.blocks), "tables": sum(isinstance(b, TableBlock) for b in s.blocks),
                        "chars": s.char_count(recursive=False)})
    return out

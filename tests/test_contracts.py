"""C0·C1 계약 검증 테스트. tests/fixtures/c1_sample.json은 가상 데이터다."""
from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from src.contracts import ANCHOR_COMPANY, SCHEMA_VERSION, SECTION_TITLES, C0Input, C1Bundle, CompanyDoc
from tests.conftest import FIXTURES


def load_sample() -> dict:
    data = json.loads((FIXTURES / "c1_sample.json").read_text(encoding="utf-8"))
    data.pop("_note")  # 가상 데이터 표시는 계약 필드가 아니다
    return data


def test_sample_is_valid_and_complete():
    c1 = C1Bundle.model_validate(load_sample())
    assert c1.schema_version == SCHEMA_VERSION
    assert len(c1.news) == 10
    assert c1.company_doc.missing_section_ids == []
    assert [s.section_id for s in c1.company_doc.sections] == list(SECTION_TITLES)


def test_json_round_trip():
    c1 = C1Bundle.model_validate(load_sample())
    assert C1Bundle.model_validate_json(c1.model_dump_json()) == c1


def test_c0_company_is_fixed():
    c0 = C0Input(field=" HBM ")
    assert c0.company == ANCHOR_COMPANY == "SK하이닉스"
    assert c0.field == "HBM"
    assert c0.request_id
    assert C0Input(company="SK하이닉스", field="DRAM").company == ANCHOR_COMPANY
    with pytest.raises(ValidationError):
        C0Input(company="삼성전자", field="HBM")
    with pytest.raises(ValidationError):
        C0Input(field="  ")


def test_c1_company_is_fixed():
    data = load_sample()
    data["company"] = "삼성전자"
    with pytest.raises(ValidationError):
        C1Bundle.model_validate(data)


def test_company_doc_must_be_anchor():
    data = load_sample()
    data["company_doc"]["stock_code"] = "005930"
    with pytest.raises(ValidationError, match="SK하이닉스"):
        C1Bundle.model_validate(data)


def test_unknown_field_rejected():
    data = load_sample()
    data["prompt"] = "오타 필드"
    with pytest.raises(ValidationError):
        C1Bundle.model_validate(data)


def test_section_chars_must_match_text():
    data = load_sample()
    data["company_doc"]["sections"][0]["chars"] += 1
    with pytest.raises(ValidationError, match="chars"):
        C1Bundle.model_validate(data)


def test_section_id_and_part_checked():
    data = load_sample()
    bad_id = copy.deepcopy(data)
    bad_id["company_doc"]["sections"][0]["section_id"] = "III-1"
    with pytest.raises(ValidationError):
        C1Bundle.model_validate(bad_id)
    bad_part = copy.deepcopy(data)
    bad_part["company_doc"]["sections"][0]["part"] = "II. 사업의 내용"
    with pytest.raises(ValidationError, match="part"):
        C1Bundle.model_validate(bad_part)


def test_highlight_must_point_to_existing_section():
    data = load_sample()
    data["company_doc"]["sections"] = [s for s in data["company_doc"]["sections"] if s["section_id"] != "II-2"]
    with pytest.raises(ValidationError, match="field_highlights"):
        C1Bundle.model_validate(data)


def test_news_rank_order_and_timezone():
    data = load_sample()
    bad_rank = copy.deepcopy(data)
    bad_rank["news"][0]["rank"] = 2
    with pytest.raises(ValidationError, match="rank"):
        C1Bundle.model_validate(bad_rank)
    no_tz = copy.deepcopy(data)
    no_tz["news"][0]["published_at"] = "2099-01-10T09:00:00"
    with pytest.raises(ValidationError, match="시간대"):
        C1Bundle.model_validate(no_tz)


def test_ok_bundle_requires_ok_company_doc():
    data = load_sample()
    data["company_doc"] = {"status": "not_found", "corp_name": "SK하이닉스", "error": "사업보고서 없음"}
    with pytest.raises(ValidationError, match="company_doc"):
        C1Bundle.model_validate(data)
    data["status"] = "partial"
    data["errors"] = ["company_doc: 사업보고서 없음"]
    c1 = C1Bundle.model_validate(data)
    assert c1.company_doc.sections == []


def test_failed_company_doc_needs_reason():
    with pytest.raises(ValidationError, match="error"):
        CompanyDoc(status="parse_error")


def test_edited_flag_must_match_prompts():
    data = load_sample()
    data["edited"] = True
    with pytest.raises(ValidationError, match="edited"):
        C1Bundle.model_validate(data)


def test_finalize():
    c1 = C1Bundle.model_validate(load_sample())
    same = c1.finalize(None)
    assert same.prompt_final == c1.prompt_draft and same.edited is False
    changed = c1.finalize("  수정한 프롬프트  ")
    assert changed.prompt_final == "수정한 프롬프트" and changed.edited is True
    assert c1.prompt_final == c1.prompt_draft  # 원본은 그대로
    with pytest.raises(ValueError):
        c1.finalize("   ")

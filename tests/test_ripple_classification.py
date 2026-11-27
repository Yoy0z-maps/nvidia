"""역할 2: C1 + C2 → C3 배정 테스트. LLM은 가짜이고 C1·배정 이유는 가상 데이터다.

C2 풀은 실제 `data/ripple_pool.json`을 그대로 읽는다 (구조 검증용).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_ripple_classification as rc
from config.settings import Settings
from src.contracts import C1Bundle
from tests.fake_llm import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = ROOT / "data" / "ripple_pool.json"
C1_PATH = ROOT / "tests" / "fixtures" / "c1_sample.json"
PURPOSE = "ripple_classification"


def load_pool() -> rc.Pool:
    return rc.Pool.model_validate_json(POOL_PATH.read_text(encoding="utf-8-sig"))


def load_c1(**changes) -> C1Bundle:
    data = json.loads(C1_PATH.read_text(encoding="utf-8"))
    data.pop("_note", None)
    data.update(changes)
    return C1Bundle.model_validate(data)


def decisions(pool: rc.Pool, **tiers: int) -> dict:
    """기본 차수를 그대로 따르는 가상 배정. tiers로 특정 item_id의 차수를 바꾼다."""
    return {"assignments": [
        {"item_id": item.item_id, "tier": tiers.get(item.item_id, item.default_tier),
         "impact": "positive", "reason": f"[가상] {item.name} 배정 이유", "confidence": 0.7}
        for item in pool.items
    ]}


def mutate_pool(change) -> rc.Pool:
    data = load_pool().model_dump()
    change(data["items"])
    return rc.Pool.model_validate(data)


def first(items, item_type=None, tier=None) -> dict:
    return next(i for i in items
                if (item_type is None or i["item_type"] == item_type)
                and (tier is None or i["default_tier"] == tier))


# ---------------------------------------------------------------------------
# C2 풀 검증
# ---------------------------------------------------------------------------
def test_real_pool_passes_check():
    pool = load_pool()
    assert rc.check_pool(pool) == {1: 5, 2: 15, 3: len(pool.items) - 20}
    assert pool.anchor.corp_code == "00164779"


def _set(key, value, **where):
    def change(items):
        first(items, **where)[key] = value
    return change


def _drop_tier3(items):
    items.remove(first(items, tier=3))


def _dup_id(items):
    items[1]["item_id"] = items[0]["item_id"]


@pytest.mark.parametrize("change, message", [
    (_dup_id, "중복 item_id"),
    (_drop_tier3, "35~40개"),
    (_set("default_tier", 2, tier=3), "기본 차수 개수"),
    (_set("corp_code", "0012638A", item_type="company"), "8자리 숫자"),
    (_set("item_id", "COM-99999999", item_type="company"), "항목 ID가 일치"),
    (_set("rep_companies", [], item_type="field"), "1~3개"),
    (_set("rep_companies", None, item_type="field"), "1~3개"),
    (_set("rep_companies", ["COM-00126380", "COM-00126380"], item_type="field"), "1~3개"),
    (_set("rep_companies", ["COM-00126380", "COM-00445054", "COM-00161383", "COM-00531014"],
          item_type="field"), "1~3개"),
    (_set("rep_companies", ["COM-99999999"], item_type="category"), "C2에 없는 대표 기업"),
])
def test_check_pool_rejects(change, message):
    with pytest.raises(ValueError, match=message):
        rc.check_pool(mutate_pool(change))


def test_pool_contract_forbids_unknown_fields():
    data = load_pool().model_dump()
    data["items"][0]["tier"] = 1  # default_tier 오타
    with pytest.raises(ValueError):
        rc.Pool.model_validate(data)


# ---------------------------------------------------------------------------
# LLM 배정 검증
# ---------------------------------------------------------------------------
def _decisions(pool, edit=None):
    out = rc.Decisions.model_validate(decisions(pool)).assignments
    if edit:
        edit(out)
    return out


@pytest.mark.parametrize("edit, message", [
    (lambda d: d.append(d[0]), "중복 항목 ID"),
    (lambda d: d.pop(), "누락 ID"),
    (lambda d: setattr(d[0], "item_id", "COM-99999999"), "풀 밖 ID"),
    (lambda d: setattr(next(x for x in d if x.tier == 3), "tier", 2), "차수별 개수"),
    (lambda d: setattr(d[0], "reason", "  "), "이유가 비어"),
    (lambda d: setattr(d[0], "confidence", 1.5), "confidence"),
    (lambda d: setattr(d[0], "confidence", -0.1), "confidence"),
])
def test_check_decisions_rejects(edit, message):
    pool = load_pool()
    counts = rc.check_pool(pool)
    with pytest.raises(ValueError, match=message):
        rc.check_decisions(_decisions(pool, edit), {i.item_id for i in pool.items}, counts)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------
def test_classify_ok_keeps_pool_order_and_sends_c1_payload():
    pool, c1 = load_pool(), load_c1()
    # LLM이 풀과 다른 순서로 돌려줘도 C3는 풀 순서를 따른다
    reply = decisions(pool)
    reply["assignments"].reverse()
    llm = FakeLLM({PURPOSE: reply})

    c3 = rc.classify(c1, pool, llm)

    assert c3.status == "ok" and c3.errors == []
    assert (c3.request_id, c3.field) == (c1.request_id, c1.field)
    assert [a.item_id for a in c3.assignments] == [i.item_id for i in pool.items]
    assert [(a.name, a.item_type) for a in c3.assignments] == [(i.name, i.item_type) for i in pool.items]
    assert len(llm.calls) == 1
    purpose, data = llm.calls[0]
    payload = json.loads(data)
    assert purpose == PURPOSE
    assert payload["prompt_final"] == c1.prompt_final
    assert payload["keywords"] == c1.keywords
    assert len(payload["pool"]["items"]) == len(pool.items)
    assert "previous_error" not in payload


def test_classify_partial_c1_gives_partial_c3():
    pool = load_pool()
    c1 = load_c1(status="partial", errors=["[가상] 뉴스 일부 누락"])
    c3 = rc.classify(c1, pool, FakeLLM({PURPOSE: decisions(pool)}))
    assert c3.status == "partial"
    assert c3.errors == ["C1이 partial 상태입니다."]
    assert len(c3.assignments) == len(pool.items)


def test_classify_retries_with_previous_error():
    pool = load_pool()
    tier3 = next(i.item_id for i in pool.items if i.default_tier == 3)
    llm = FakeLLM({PURPOSE: [decisions(pool, **{tier3: 2}), decisions(pool)]})

    c3 = rc.classify(load_c1(), pool, llm)

    assert c3.status == "ok"
    assert len(llm.calls) == 2
    assert "차수별 개수" in json.loads(llm.calls[1][1])["previous_error"]


def _raise(data):
    raise RuntimeError("[가상] API 장애")


@pytest.mark.parametrize("reply, message", [
    (_raise, "LLM 호출 실패 (RuntimeError)"),
    ({"assignments": [{"item_id": "x", "tier": 4}]}, "구조화 출력"),
])
def test_classify_falls_back_to_default_tiers_after_two_failures(reply, message):
    pool = load_pool()
    llm = FakeLLM({PURPOSE: reply})

    c3 = rc.classify(load_c1(), pool, llm)

    assert len(llm.calls) == 2
    assert c3.status == "partial"
    assert len(c3.errors) == 2 and all(message in e for e in c3.errors)
    assert [a.tier for a in c3.assignments] == [i.default_tier for i in pool.items]
    assert {a.impact for a in c3.assignments} == {"unclear"}
    assert {a.confidence for a in c3.assignments} == {0}


def test_classify_without_final_prompt_skips_llm():
    pool = load_pool()
    c1 = load_c1(status="error", errors=["[가상] 실패"], prompt_draft="", prompt_final="")
    llm = FakeLLM({PURPOSE: decisions(pool)})

    c3 = rc.classify(c1, pool, llm)

    assert c3.status == "error" and c3.assignments == []
    assert llm.calls == []


def test_classify_rejects_different_anchor():
    pool = load_pool()
    data = pool.model_dump()
    data["anchor"]["corp_code"] = "00126380"
    with pytest.raises(ValueError, match="기준 기업"):
        rc.classify(load_c1(), rc.Pool.model_validate(data), FakeLLM({PURPOSE: decisions(pool)}))


def test_c3_carries_pool_version():
    pool = load_pool()
    c3 = rc.classify(load_c1(), pool, FakeLLM({PURPOSE: decisions(pool)}))
    assert pool.pool_version and c3.pool_version == pool.pool_version != pool.schema_version


@pytest.mark.parametrize("change", [lambda d: d.pop("pool_version"), lambda d: d.update(pool_version="")])
def test_pool_requires_pool_version(change):
    data = load_pool().model_dump()
    change(data)
    with pytest.raises(ValueError, match="pool_version"):
        rc.Pool.model_validate(data)


# ---------------------------------------------------------------------------
# 출력과 CLI
# ---------------------------------------------------------------------------
def test_write_output_roundtrip_without_temp_files(tmp_path):
    pool = load_pool()
    c3 = rc.classify(load_c1(), pool, FakeLLM({PURPOSE: decisions(pool)}))
    path = tmp_path / "out" / "c3_x.json"

    rc.write_output(c3, path)

    assert rc.C3.model_validate_json(path.read_text(encoding="utf-8")) == c3
    assert list(path.parent.iterdir()) == [path]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """main()을 가짜 설정·LLM으로 실행한다."""
    settings = Settings(data_dir=tmp_path / "data", openai_api_key="TESTKEY")
    monkeypatch.setattr(rc, "Settings", SimpleNamespace(from_env=lambda: settings))
    llm = FakeLLM({PURPOSE: decisions(load_pool())})
    monkeypatch.setattr(rc, "OpenAILLM", lambda *a, **k: llm)
    c1_path = tmp_path / "c1_HBM_test.json"
    c1_path.write_text(load_c1().model_dump_json(), encoding="utf-8")

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["run_ripple_classification", *map(str, args)])
        return rc.main()

    return SimpleNamespace(run=run, settings=settings, c1_path=c1_path, llm=llm, tmp=tmp_path)


def test_main_writes_c3_to_runs_dir(cli, capsys):
    assert cli.run("--c1", cli.c1_path) == 0
    out = cli.settings.runs_dir / "c3" / "c3_HBM_test.json"
    c3 = rc.C3.model_validate_json(out.read_text(encoding="utf-8"))
    assert c3.status == "ok" and len(c3.assignments) == 35
    assert "HBM: ok · 35개" in capsys.readouterr().out


def test_main_rejects_colliding_outputs(cli, capsys):
    assert cli.run("--c1", cli.c1_path, cli.c1_path) == 2
    assert "출력 경로가 겹칩니다" in capsys.readouterr().err
    assert cli.llm.calls == []


def test_main_rejects_output_overwriting_input(cli, capsys):
    # c1_HBM_test.json의 출력은 <out-dir>/c3_HBM_test.json → 같은 이름의 C2 파일을 덮어쓰면 안 된다
    pool_copy = cli.tmp / "c3_HBM_test.json"
    pool_copy.write_text(POOL_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
    assert cli.run("--c1", cli.c1_path, "--pool", pool_copy, "--out-dir", cli.tmp) == 2
    assert "C1 또는 C2 파일과 같습니다" in capsys.readouterr().err
    assert rc.Pool.model_validate_json(pool_copy.read_text(encoding="utf-8"))


def test_main_requires_api_key(cli, capsys):
    cli.settings.openai_api_key = ""
    assert cli.run("--c1", cli.c1_path) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_reports_invalid_c1(cli, capsys):
    bad = cli.tmp / "c1_bad.json"
    bad.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    assert cli.run("--c1", bad) == 2
    assert "계약 검증 오류" in capsys.readouterr().err

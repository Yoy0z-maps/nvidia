"""역할 2의 C1 + C2 → C3 배정과 실행 명령."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from config.settings import PROJECT_ROOT, Settings
from src.contracts import C1Bundle, Status
from src.llm import LLM, OpenAILLM

Tier = Literal[1, 2, 3]
Impact = Literal["positive", "negative", "mixed", "unclear"]
ItemType = Literal["company", "field", "category"]

INSTRUCTIONS = """
C2 사전 풀의 모든 항목에 대해 C1의 분야, 핵심어, 확정 프롬프트에 담긴 이슈를 바탕으로
차수, 영향 방향, 이유를 배정한다. C2의 criteria를 차수 기준으로 사용한다.
항목 ID는 풀에 있는 값을 각각 정확히 한 번 쓴다. 1차 5개, 2차 15개,
3차는 전체 항목 수에서 20을 뺀 개수로 배정한다.
impact는 해당 항목에 미치는 방향이다. 근거가 부족하면 unclear를 사용한다.
reason은 한국어 한두 문장으로 작성한다. 확인되지 않은 거래나 수치는 단정하지 않는다.
confidence는 판단의 확실성을 0 이상 1 이하로 표시한다.
이전 응답의 검증 오류가 있으면 이를 고친 완전한 배정 목록을 반환한다.
""".strip()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Anchor(Contract):
    name: str
    corp_code: str


class Criteria(Contract):
    tier1: str
    tier2: str
    tier3: str


class PoolItem(Contract):
    item_id: str
    name: str
    item_type: ItemType
    default_tier: Tier
    relation: str
    keywords: list[str]
    corp_code: str | None = None
    stock_code: str | None = None
    rep_companies: list[str] | None = None


class Pool(Contract):
    schema_version: Literal["1.0"]
    pool_version: str | None = None
    anchor: Anchor
    criteria: Criteria
    items: list[PoolItem]


class Decision(Contract):
    item_id: str
    tier: Tier
    impact: Impact
    reason: str
    confidence: float


class Decisions(Contract):
    assignments: list[Decision]


class Assignment(Decision):
    name: str
    item_type: ItemType


class C3(Contract):
    schema_version: Literal["1.0"] = "1.0"
    status: Status
    errors: list[str]
    request_id: str
    field: str
    pool_version: str
    assignments: list[Assignment]


def check_pool(pool: Pool) -> dict[int, int]:
    items = {item.item_id: item for item in pool.items}
    size = len(pool.items)
    if size != len(items):
        raise ValueError("C2에 중복 item_id가 있습니다.")
    if not 35 <= size <= 40:
        raise ValueError("C2 항목이 35~40개여야 합니다.")

    counts = {1: 5, 2: 15, 3: size - 20}
    if Counter(item.default_tier for item in pool.items) != counts:
        raise ValueError(f"C2 기본 차수 개수가 {counts}와 다릅니다.")

    company_ids = {item.item_id for item in pool.items if item.item_type == "company"}
    for item in pool.items:
        if item.item_type == "company":
            code = item.corp_code or ""
            if len(code) != 8 or not code.isascii() or not code.isdecimal():
                raise ValueError(f"{item.item_id}: corp_code가 8자리 숫자가 아닙니다.")
            if item.item_id != f"COM-{code}":
                raise ValueError(f"{item.item_id}: corp_code와 항목 ID가 일치하지 않습니다.")
            continue
        representatives = item.rep_companies or []
        if not 1 <= len(set(representatives)) == len(representatives) <= 3:
            raise ValueError(f"{item.item_id}: 서로 다른 대표 기업이 1~3개여야 합니다.")
        if not set(representatives) <= company_ids:
            raise ValueError(f"{item.item_id}: C2에 없는 대표 기업이 연결돼 있습니다.")
    return counts


def check_decisions(
    decisions: list[Decision], pool_ids: set[str], counts: dict[int, int]
) -> None:
    ids = [decision.item_id for decision in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("중복 항목 ID가 있습니다.")
    if set(ids) != pool_ids:
        raise ValueError(
            f"풀 밖 ID: {sorted(set(ids) - pool_ids)}, 누락 ID: {sorted(pool_ids - set(ids))}"
        )
    if Counter(decision.tier for decision in decisions) != counts:
        raise ValueError(f"차수별 개수가 {counts}와 다릅니다.")
    if any(not decision.reason.strip() or not 0 <= decision.confidence <= 1 for decision in decisions):
        raise ValueError("이유가 비어 있거나 confidence가 0~1 범위를 벗어났습니다.")


def classify(c1: C1Bundle, pool: Pool, llm: LLM) -> C3:
    counts = check_pool(pool)
    if c1.company != pool.anchor.name or (
        c1.company_doc.corp_code and c1.company_doc.corp_code != pool.anchor.corp_code
    ):
        raise ValueError("C1의 기준 기업과 C2의 기준 기업이 다릅니다.")
    metadata = {
        "request_id": c1.request_id,
        "field": c1.field,
        "pool_version": pool.pool_version or pool.schema_version,
    }
    if c1.status == "error" or not c1.prompt_final.strip():
        return C3(
            **metadata, status="error", errors=["유효한 C1 확정 프롬프트가 없습니다."],
            assignments=[],
        )

    payload = {
        "field": c1.field,
        "keywords": c1.keywords,
        "prompt_final": c1.prompt_final,
        "pool": pool.model_dump(exclude_none=True),
    }
    errors = []
    pool_ids = {item.item_id for item in pool.items}
    for attempt in range(1, 3):
        try:
            result = llm.parse(
                INSTRUCTIONS, json.dumps(payload, ensure_ascii=False), Decisions,
                purpose="ripple_classification",
            )
            check_decisions(result.assignments, pool_ids, counts)
        except ValidationError:
            error = "C3 구조화 출력의 필드 또는 값이 잘못되었습니다."
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"LLM 호출 실패 ({type(exc).__name__})."
        else:
            decisions = {decision.item_id: decision for decision in result.assignments}
            return C3(
                **metadata,
                status="ok" if c1.status == "ok" else "partial",
                errors=[] if c1.status == "ok" else ["C1이 partial 상태입니다."],
                assignments=[
                    Assignment(
                        **decisions[item.item_id].model_dump(),
                        name=item.name, item_type=item.item_type,
                    )
                    for item in pool.items
                ],
            )
        errors.append(f"{attempt}차 시도: {error}")
        payload["previous_error"] = error

    return C3(
        **metadata, status="partial", errors=errors,
        assignments=[
            Assignment(
                item_id=item.item_id, name=item.name, item_type=item.item_type,
                tier=item.default_tier, impact="unclear", confidence=0,
                reason="자동 배정 결과를 확정하지 못해 사전 풀의 기본 차수를 사용했습니다.",
            )
            for item in pool.items
        ],
    )


def write_output(result: C3, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as temp:
        temp_path = Path(temp.name)
        temp.write(result.model_dump_json(indent=2) + "\n")
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="C1 + C2에서 분야별 C3 JSON을 생성합니다.")
    parser.add_argument("--c1", nargs="+", required=True, type=Path)
    parser.add_argument("--pool", type=Path, default=PROJECT_ROOT / "data" / "ripple_pool.json")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
        if not settings.openai_api_key:
            raise ValueError("프로젝트 .env에 OPENAI_API_KEY를 설정하세요.")
        pool = Pool.model_validate_json(args.pool.read_text(encoding="utf-8-sig"))
        check_pool(pool)
        bundle_paths = [
            (path, C1Bundle.model_validate_json(path.read_text(encoding="utf-8-sig")))
            for path in args.c1
        ]
        out_dir = args.out_dir or settings.runs_dir / "c3"
        output_paths = [
            out_dir / f"c3_{path.stem.removeprefix('c1_')}.json" for path, _ in bundle_paths
        ]
        if len({path.resolve() for path in output_paths}) != len(output_paths):
            raise ValueError("C1 파일 이름이 중복되어 C3 출력 경로가 겹칩니다.")
        input_paths = {args.pool.resolve(), *(path.resolve() for path, _ in bundle_paths)}
        if input_paths.intersection(path.resolve() for path in output_paths):
            raise ValueError("C3 출력 경로가 C1 또는 C2 파일과 같습니다.")

        llm = OpenAILLM(settings.openai_api_key, settings.llm_model, max_retries=1)
        exit_code = 0
        for (_, c1), path in zip(bundle_paths, output_paths, strict=True):
            result = classify(c1, pool, llm)
            write_output(result, path)
            print(f"{c1.field}: {result.status} · {len(result.assignments)}개 · {path}")
            if result.status != "ok":
                exit_code = 1
                for error in result.errors:
                    print(error, file=sys.stderr)
        return exit_code
    except ValidationError as exc:
        print(f"계약 검증 오류: {exc.error_count()}개", file=sys.stderr)
        for error in exc.errors(include_input=False):
            print(f"{'.'.join(map(str, error['loc']))}: {error['msg']}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

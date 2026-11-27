"""KRX Open API 인증·서비스 승인 상태 진단.

uv run python -m scripts.check_krx

KRX의 401 응답은 두 가지를 구분한다.
- "Unauthorized Key"      : 키 자체를 인식하지 못함 → 키 값(공백/줄바꿈 포함 여부)과 키 승인 상태 확인
- "Unauthorized API Call" : 키는 유효, 해당 서비스 이용 승인이 없음 → 서비스별 [API 이용신청] 후 승인 대기
키 값은 출력하지 않는다.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402

BASE = "https://data-dbg.krx.co.kr/svc/apis"
SERVICES = [
    ("sto/stk_bydd_trd", "유가증권 일별매매정보", "주가 (KOSPI 종목) - 필수"),
    ("sto/ksq_bydd_trd", "코스닥 일별매매정보", "주가 (KOSDAQ 종목) - 필수"),
    ("idx/kospi_dd_trd", "KOSPI 시리즈 일별시세정보", "시장 대비 수익률 벤치마크 - 선택"),
    ("idx/kosdaq_dd_trd", "KOSDAQ 시리즈 일별시세정보", "시장 대비 수익률 벤치마크 - 선택"),
]


def last_weekday() -> str:
    d = dt.date.today() - dt.timedelta(days=3)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d.strftime("%Y%m%d")


def explain(status: int, body: str) -> str:
    if status == 200:
        return "정상 (승인됨)"
    if "Unauthorized Key" in body:
        return "키를 인식하지 못함 → .env의 KRX_API_KEY 값과 KRX 마이페이지의 '인증키 승인' 상태 확인"
    if "Unauthorized API Call" in body:
        return "키는 유효, 이 서비스 이용 승인이 없음 → 서비스 목록에서 [API 이용신청] 후 승인 대기"
    if status == 401:
        return "401 (원인 메시지 없음) → 키 승인 및 서비스 이용 승인 모두 확인"
    if status == 429:
        return "호출 한도 초과"
    return f"HTTP {status}"


def main() -> int:
    s = Settings.from_env()
    key = s.krx_api_key
    if not key:
        print("KRX_API_KEY가 .env에 없습니다.")
        return 1
    raw = key
    key = key.strip().strip('"').strip("'")
    print(f"키 길이 {len(key)}자" + (" (앞뒤 공백/따옴표 제거됨 - .env 값을 정리하세요)" if raw != key else ""))
    bas_dd = last_weekday()
    print(f"기준일 basDd={bas_dd}\n")
    ok = 0
    with httpx.Client(timeout=20) as c:
        for path, name, usage in SERVICES:
            try:
                r = c.get(f"{BASE}/{path}", params={"basDd": bas_dd}, headers={"AUTH_KEY": key})
                body = r.text[:300]
                rows = len((r.json() or {}).get("OutBlock_1") or []) if r.status_code == 200 else 0
                msg = explain(r.status_code, body)
                ok += r.status_code == 200
                extra = f" · {rows}행" if r.status_code == 200 else f" · 응답: {body.strip()}"
                print(f"[{'OK' if r.status_code == 200 else 'NG'}] {name} ({path}) - {usage}\n     {msg}{extra}")
            except Exception as exc:
                print(f"[ERR] {name} ({path}): {type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(SERVICES)} 서비스 사용 가능")
    if ok < 2:
        print("승인 절차: openapi.krx.co.kr 로그인 → 서비스 목록 → 각 서비스에서 [API 이용신청] → "
              "마이페이지 → 이용현황에서 '승인' 확인 (인증키 승인과 서비스별 승인은 별개)")
    return 0 if ok >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())

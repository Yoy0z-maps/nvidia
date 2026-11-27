"""OpenDART API 클라이언트.

참고: https://opendart.fss.or.kr/guide/main.do
- corpCode.xml : 공시대상회사 고유번호 (ZIP)
- list.json    : 공시검색
- document.xml : 공시서류원본파일 (ZIP)
- company.json : 기업개황 (corp_cls: Y=유가, K=코스닥, N=코넥스, E=기타)
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path, PurePosixPath

import httpx
from lxml import etree

log = logging.getLogger(__name__)

BASE_URL = "https://opendart.fss.or.kr/api"
DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"

# OpenDART 응답 상태 코드
STATUS_MESSAGES = {
    "000": "정상",
    "010": "등록되지 않은 키",
    "011": "사용할 수 없는 키",
    "012": "접근할 수 없는 IP",
    "013": "조회된 데이터 없음",
    "014": "파일이 존재하지 않음",
    "020": "요청 제한 초과",
    "021": "조회 가능한 회사 개수 초과",
    "100": "필드의 부적절한 값",
    "101": "부적절한 접근",
    "800": "시스템 점검",
    "900": "정의되지 않은 오류",
    "901": "개인정보 보유기간 만료 계정",
}
AUTH_ERRORS = {"010", "011", "012", "901"}
RETRYABLE = {"020", "800", "900"}

MAX_ZIP_BYTES = 200 * 1024 * 1024          # 압축 해제 총량 상한
MAX_ZIP_ENTRY_BYTES = 100 * 1024 * 1024    # 엔트리 하나 상한
MAX_ZIP_ENTRIES = 200


class DartError(Exception):
    def __init__(self, status: str, message: str, url: str = ""):
        self.status = status
        self.message = message
        super().__init__(f"OpenDART 오류 [{status}] {message or STATUS_MESSAGES.get(status, '')} {url}".strip())


class DartAuthError(DartError):
    pass


class DartNoData(DartError):
    pass


class DartRateLimit(DartError):
    pass


class UnsafeZipError(Exception):
    pass


def safe_extract_zip(data: bytes) -> dict[str, bytes]:
    """ZIP을 메모리에서 안전하게 해제한다 (경로 조작, 압축 폭탄 방지)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UnsafeZipError("올바른 ZIP 파일이 아닙니다.") from exc
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise UnsafeZipError(f"ZIP 엔트리 수가 너무 많습니다: {len(infos)}")
    total = 0
    out: dict[str, bytes] = {}
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or not p.name:
            raise UnsafeZipError(f"안전하지 않은 ZIP 경로: {info.filename!r}")
        if info.file_size > MAX_ZIP_ENTRY_BYTES:
            raise UnsafeZipError(f"ZIP 엔트리가 너무 큽니다: {info.filename} ({info.file_size} bytes)")
        total += info.file_size
        if total > MAX_ZIP_BYTES:
            raise UnsafeZipError("ZIP 압축 해제 총량이 상한을 초과했습니다.")
        with zf.open(info) as fh:
            content = fh.read(MAX_ZIP_ENTRY_BYTES + 1)
        if len(content) > MAX_ZIP_ENTRY_BYTES:
            raise UnsafeZipError(f"ZIP 엔트리 실제 크기가 상한을 초과했습니다: {info.filename}")
        out[p.name] = content
    return out


def _parse_error_xml(content: bytes) -> tuple[str, str] | None:
    """ZIP 대신 반환된 오류 XML(<result><status/><message/></result>)을 해석한다."""
    try:
        root = etree.fromstring(content, parser=etree.XMLParser(recover=True))
    except etree.XMLSyntaxError:
        return None
    if root is None:
        return None
    status = root.findtext("status")
    if status is None:
        return None
    return status.strip(), (root.findtext("message") or "").strip()


class DartClient:
    def __init__(self, api_key: str, *, timeout: float = 60.0, max_retries: int = 3,
                 transport: httpx.BaseTransport | None = None, min_interval: float = 0.2):
        if not api_key:
            raise DartAuthError("NO_KEY", "DART_API_KEY 환경변수가 설정되지 않았습니다.")
        self._key = api_key
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call = 0.0
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout, transport=transport,
                                    headers={"User-Agent": "market-ripple/0.1"})
        self.call_count = 0

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    def _raise_for_status(self, status: str, message: str, endpoint: str) -> None:
        if status == "000":
            return
        if status in AUTH_ERRORS:
            raise DartAuthError(status, message, endpoint)
        if status in {"013", "014"}:
            raise DartNoData(status, message, endpoint)
        if status == "020":
            raise DartRateLimit(status, message, endpoint)
        raise DartError(status, message, endpoint)

    def _get(self, endpoint: str, params: dict) -> httpx.Response:
        params = {"crtfc_key": self._key, **params}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                self.call_count += 1
                resp = self._client.get(endpoint, params=params)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise DartError(str(exc.response.status_code), "HTTP 오류", endpoint) from exc
                backoff = 2 ** attempt
                log.warning("OpenDART %s 요청 실패 (%s) - %ss 후 재시도 [%d/%d]",
                            endpoint, type(exc).__name__, backoff, attempt + 1, self.max_retries)
                time.sleep(backoff)
        raise DartError("NETWORK", f"네트워크 오류: {last_exc}", endpoint)

    def _get_json(self, endpoint: str, params: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            resp = self._get(endpoint, params)
            try:
                data = resp.json()
            except ValueError as exc:
                raise DartError("BAD_JSON", "JSON 응답 해석 실패", endpoint) from exc
            status = str(data.get("status", ""))
            if status in RETRYABLE and attempt < self.max_retries:
                log.warning("OpenDART %s 상태 %s - 재시도", endpoint, status)
                time.sleep(2 ** (attempt + 1))
                continue
            self._raise_for_status(status, data.get("message", ""), endpoint)
            return data
        raise DartError("RETRY_EXHAUSTED", "재시도 초과", endpoint)

    def _get_zip(self, endpoint: str, params: dict) -> dict[str, bytes]:
        for attempt in range(self.max_retries + 1):
            resp = self._get(endpoint, params)
            content = resp.content
            if content[:2] == b"PK":
                return safe_extract_zip(content)
            parsed = _parse_error_xml(content)
            if parsed is None:
                raise DartError("BAD_RESPONSE", f"ZIP도 오류 XML도 아닌 응답 (앞부분: {content[:80]!r})", endpoint)
            status, message = parsed
            if status in RETRYABLE and attempt < self.max_retries:
                time.sleep(2 ** (attempt + 1))
                continue
            self._raise_for_status(status, message, endpoint)
            raise DartError(status, message or "ZIP이 아닌 정상 응답", endpoint)
        raise DartError("RETRY_EXHAUSTED", "재시도 초과", endpoint)

    # ------------------------------------------------------------------
    def download_corp_codes(self) -> bytes:
        files = self._get_zip("/corpCode.xml", {})
        for name, data in files.items():
            if name.lower().endswith(".xml"):
                return data
        raise DartError("NO_XML", "corpCode ZIP 안에 XML이 없습니다.", "/corpCode.xml")

    def list_disclosures(self, corp_code: str, bgn_de: str, end_de: str, pblntf_ty: str | None = None,
                         page_count: int = 100, max_pages: int = 20) -> list[dict]:
        """공시 목록 (페이지네이션). 데이터 없음(013)은 빈 목록으로 반환."""
        items: list[dict] = []
        page = 1
        while page <= max_pages:
            params = {"corp_code": corp_code, "bgn_de": bgn_de, "end_de": end_de,
                      "page_no": page, "page_count": page_count}
            if pblntf_ty:
                params["pblntf_ty"] = pblntf_ty
            try:
                data = self._get_json("/list.json", params)
            except DartNoData:
                break
            items.extend(data.get("list", []))
            total_page = int(data.get("total_page", 1) or 1)
            if page >= total_page:
                break
            page += 1
        return items

    def download_document(self, rcept_no: str) -> dict[str, bytes]:
        return self._get_zip("/document.xml", {"rcept_no": rcept_no})

    def company_info(self, corp_code: str) -> dict:
        return self._get_json("/company.json", {"corp_code": corp_code})

    @staticmethod
    def viewer_url(rcept_no: str) -> str:
        return DART_VIEWER_URL.format(rcept_no=rcept_no)


def save_document_files(files: dict[str, bytes], target_dir: Path, rcept_no: str) -> tuple[Path, list[Path]]:
    """원본 XML들을 저장하고 (주 문서, 첨부 문서들)을 반환한다.

    DART ZIP에는 보통 `{rcept_no}.xml`(본문)과 `{rcept_no}_XXXXX.xml`(감사보고서 등 첨부)이 함께 있다.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    main: Path | None = None
    attachments: list[Path] = []
    for name, data in sorted(files.items()):
        p = target_dir / name
        p.write_bytes(data)
        if not name.lower().endswith(".xml"):
            attachments.append(p)
            continue
        if Path(name).stem == rcept_no:
            main = p
        else:
            attachments.append(p)
    if main is None:
        xmls = [p for p in attachments if p.suffix.lower() == ".xml"]
        if not xmls:
            raise DartError("NO_XML", "공시 ZIP 안에 XML 문서가 없습니다.", rcept_no)
        main = min(xmls, key=lambda p: len(p.name))
        attachments.remove(main)
    return main, attachments

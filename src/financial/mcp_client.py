"""Korea Stock MCP 클라이언트 (Python MCP SDK, stdio).

Tool 이름과 파라미터는 실제 list_tools 결과로 확인한 뒤 사용한다.
korea-stock-mcp@1.4.1 에서 list_tools로 확인된 Tool (scripts/check_environment.py 출력 참조):
  get_corp_code, get_disclosure_list, get_disclosure, get_financial_statement, get_market_type,
  get_stock_base_info, get_stock_trade_info, get_today_date
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# 금융 분석에서 호출을 허용하는 Tool (Python에서 통제)
DEFAULT_ALLOWED_TOOLS = {"get_stock_trade_info", "get_stock_base_info", "get_market_type", "get_corp_code",
                         "get_financial_statement", "get_today_date"}


def _attr(obj, *names):
    """MCP SDK 1.x(camelCase) / 2.x(snake_case) 호환."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


class MCPUnavailable(RuntimeError):
    pass


class MCPCallBudgetExceeded(RuntimeError):
    pass


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description.strip(), "input_schema": self.input_schema}


@dataclass
class MCPCallRecord:
    tool: str
    arguments: dict
    ok: bool
    error: str | None = None


@dataclass
class KoreaStockMCP:
    command: str
    args: list[str]
    dart_api_key: str = field(repr=False, default="")
    krx_api_key: str = field(repr=False, default="")
    allowed_tools: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_TOOLS))
    max_calls: int = 100
    timeout_s: float = 120.0

    def __post_init__(self):
        self.calls: list[MCPCallRecord] = []
        self.tools: dict[str, ToolInfo] = {}
        self._stack: AsyncExitStack | None = None
        self._session = None

    # ---- 연결 ----
    def check_runtime(self) -> list[str]:
        problems = []
        if os.path.basename(self.command).startswith("node") and self.args and not os.path.exists(self.args[0]):
            problems.append(f"MCP 서버 미설치: {self.args[0]} (--install-mcp 옵션으로 설치)")
        if shutil.which(self.command) is None:
            problems.append(f"'{self.command}' 실행 파일을 찾을 수 없습니다 (Node.js/npx 설치 필요).")
        if shutil.which("node") is None:
            problems.append("Node.js(node)가 설치되어 있지 않습니다.")
        if not self.dart_api_key:
            problems.append("DART_API_KEY 미설정 - DART 관련 Tool이 실패합니다.")
        if not self.krx_api_key:
            problems.append("KRX_API_KEY 미설정 - 주가(KRX) Tool이 실패합니다.")
        return problems

    async def __aenter__(self) -> "KoreaStockMCP":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if shutil.which(self.command) is None:
            raise MCPUnavailable(f"'{self.command}'를 찾을 수 없습니다. Node.js를 설치하세요.")
        if os.path.basename(self.command).startswith("node") and self.args and not os.path.exists(self.args[0]):
            raise MCPUnavailable(f"MCP 서버 스크립트가 없습니다: {self.args[0]} - "
                                 "`uv run python -m scripts.check_environment --install-mcp` 로 설치하세요.")
        # 서드파티 프로세스에는 필요한 환경변수만 전달한다 (다른 비밀값 유출 방지)
        passthrough = {"PATH", "HOME", "USER", "LANG", "TMPDIR", "SystemRoot", "APPDATA", "NODE_EXTRA_CA_CERTS",
                       "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy"}
        env = {k: v for k, v in os.environ.items()
               if k in passthrough or k.lower().startswith("npm_config_") or k == "NPM_CONFIG_USERCONFIG"}
        if self.dart_api_key:
            env["DART_API_KEY"] = self.dart_api_key
        if self.krx_api_key:
            env["KRX_API_KEY"] = self.krx_api_key
        params = StdioServerParameters(command=self.command, args=self.args, env=env)
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self._session.initialize(), self.timeout_s)
            listed = await asyncio.wait_for(self._session.list_tools(), self.timeout_s)
        except Exception as exc:
            await self._stack.aclose()
            raise MCPUnavailable(f"MCP 서버 초기화 실패: {exc}") from exc
        self.tools = {t.name: ToolInfo(t.name, t.description or "", _attr(t, "input_schema", "inputSchema") or {})
                      for t in listed.tools}
        log.info("MCP Tool %d개 확인: %s", len(self.tools), sorted(self.tools))
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.aclose()

    # ---- 호출 ----
    def _validate_args(self, name: str, arguments: dict) -> None:
        schema = self.tools[name].input_schema
        props = schema.get("properties", {})
        unknown = set(arguments) - set(props)
        if unknown:
            raise ValueError(f"{name}: 스키마에 없는 파라미터 {sorted(unknown)}")
        missing = set(schema.get("required", [])) - set(arguments)
        if missing:
            raise ValueError(f"{name}: 필수 파라미터 누락 {sorted(missing)}")

    async def call(self, name: str, arguments: dict) -> Any:
        if name not in self.tools:
            raise ValueError(f"MCP 서버에 존재하지 않는 Tool: {name} (확인된 Tool: {sorted(self.tools)})")
        if name not in self.allowed_tools:
            raise PermissionError(f"허용되지 않은 Tool 호출: {name}")
        if len(self.calls) >= self.max_calls:
            raise MCPCallBudgetExceeded(f"MCP 호출 예산({self.max_calls}) 소진")
        self._validate_args(name, arguments)
        rec = MCPCallRecord(name, arguments, ok=False)
        self.calls.append(rec)
        try:
            res = await asyncio.wait_for(self._session.call_tool(name, arguments), self.timeout_s)
        except Exception as exc:
            rec.error = str(exc)
            raise
        texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
        if _attr(res, "is_error", "isError"):
            rec.error = " ".join(texts)[:500]
            raise RuntimeError(f"MCP Tool 오류 ({name}): {rec.error}")
        rec.ok = True
        joined = "\n".join(texts)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined

    # ---- LLM Function Calling 연결 ----
    def openai_tools(self, names: set[str] | None = None) -> list[dict]:
        """MCP Tool 스키마를 OpenAI Responses API function tool 형식으로 변환 (허용 목록만)."""
        out = []
        for name, t in sorted(self.tools.items()):
            if name not in self.allowed_tools or (names and name not in names):
                continue
            out.append({"type": "function", "name": name, "description": t.description.strip()[:1000],
                        "parameters": t.input_schema})
        return out


async def list_tools_once(command: str, args: list[str], dart_key: str = "", krx_key: str = "") -> list[dict]:
    async with KoreaStockMCP(command, args, dart_key, krx_key) as mcp:
        return [t.to_dict() for t in mcp.tools.values()]


async def run_tool_agent(mcp: KoreaStockMCP, openai_client, model: str, question: str,
                         date_range: tuple[str, str], max_tool_calls: int = 5) -> dict:
    """LLM이 필요한 Tool을 선택하도록 하되, 허용 Tool·날짜 범위·호출 횟수는 Python에서 통제한다."""
    tools = mcp.openai_tools()
    instructions = ("너는 한국 주식 데이터 조회 도우미다. 제공된 Tool만 사용하고, 조회 결과에 없는 수치를 만들지 않는다. "
                    f"조회 날짜는 {date_range[0]}~{date_range[1]} 범위(YYYYMMDD)만 허용된다.")
    inputs: list = [{"role": "user", "content": question}]
    log_calls = []
    for _ in range(max_tool_calls + 1):
        resp = openai_client.responses.create(model=model, instructions=instructions, input=inputs, tools=tools)
        calls = [o for o in resp.output if getattr(o, "type", "") == "function_call"]
        if not calls:
            return {"answer": resp.output_text, "tool_calls": log_calls}
        inputs += [o.model_dump() if hasattr(o, "model_dump") else o for o in resp.output]
        for c in calls:
            args = json.loads(c.arguments or "{}")
            for key in ("basDdList",):
                if key in args:
                    args[key] = [d for d in args[key] if date_range[0] <= d <= date_range[1]]
            try:
                if len(log_calls) >= max_tool_calls:
                    raise MCPCallBudgetExceeded("Tool 호출 횟수 제한 초과")
                result = await mcp.call(c.name, args)
                out = json.dumps(result, ensure_ascii=False)[:20000]
            except Exception as exc:
                out = json.dumps({"error": str(exc)}, ensure_ascii=False)
            log_calls.append({"tool": c.name, "arguments": args})
            inputs.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
    return {"answer": "(Tool 호출 제한으로 중단)", "tool_calls": log_calls}


MCP_PACKAGE = "korea-stock-mcp@1.4.1"


def make_mcp(settings, max_calls: int = 100) -> KoreaStockMCP:
    return KoreaStockMCP(settings.mcp_command, settings.mcp_args, settings.dart_api_key, settings.krx_api_key,
                         max_calls=max_calls)

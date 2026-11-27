"""테스트 전용 스크립트형 LLM."""
from __future__ import annotations

from src.llm import LLMBudgetExceeded, Usage


class FakeLLM:
    def __init__(self, responses: dict, max_calls: int | None = None):
        """responses: purpose -> list[obj | callable(data)->obj] 또는 단일 callable"""
        self.model = "fake-llm"
        self.responses = responses
        self.max_calls = max_calls
        self.usage = Usage()
        self.calls: list[tuple[str, str]] = []

    def reset_budget(self, max_calls):
        self.max_calls = max_calls
        self.usage = Usage()

    def _next(self, purpose: str, data: str):
        if self.max_calls is not None and self.usage.calls >= self.max_calls:
            raise LLMBudgetExceeded("fake budget")
        self.usage.calls += 1
        self.calls.append((purpose, data))
        r = self.responses.get(purpose)
        if r is None:
            raise AssertionError(f"FakeLLM: 정의되지 않은 purpose {purpose}")
        if callable(r):
            return r(data)
        if isinstance(r, list):
            if not r:
                raise AssertionError(f"FakeLLM: {purpose} 응답 소진")
            item = r.pop(0) if len(r) > 1 else r[0]
            return item(data) if callable(item) else item
        return r

    def parse(self, instructions, data, schema, purpose=""):
        out = self._next(purpose, data)
        return schema.model_validate(out) if isinstance(out, dict) else out

    def text(self, instructions, data, purpose=""):
        return self._next(purpose, data)

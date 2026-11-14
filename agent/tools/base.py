"""Tool registry: schema validation, safety policy, human confirmation,
timeouts and structured logging for every tool call."""

from __future__ import annotations

import concurrent.futures as cf
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.config import settings
from agent.logging_utils import get_logger, log_event
from agent.schemas import ToolResult, ToolStatus

logger = get_logger("mlops_agent.tools")

# A shared pool so a hung call cannot block the agent forever.
_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


class ServiceUnavailable(Exception):
    """The external system (MLflow, Airflow, cluster, git/dvc) could not be reached."""


class ServiceError(Exception):
    """The external system answered, but with an error."""


class PolicyDenied(Exception):
    """The request violates a safety policy (namespace, DAG allow-list, command allow-list)."""


ConfirmFn = Callable[[str, dict[str, Any]], bool]


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[Any], Any]
    module: str
    mutating: bool = False
    policy: Callable[[Any], None] | None = None

    def anthropic_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("properties", {})
        desc = self.description + (" [CHANGES STATE: requires explicit human confirmation]" if self.mutating else "")
        return {"name": self.name, "description": desc, "input_schema": schema}


class ToolRegistry:
    def __init__(self, timeout_s: float | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.timeout_s = timeout_s if timeout_s is not None else settings.tool_timeout_s

    # registration ---------------------------------------------------------
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.anthropic_schema() for t in self._tools.values()]

    # execution ------------------------------------------------------------
    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        request_id: str,
        confirm: ConfirmFn | None = None,
    ) -> ToolResult:
        start = time.perf_counter()
        arguments = arguments or {}
        tool = self._tools.get(name)

        def finish(status: ToolStatus, data: Any = None, error: str | None = None) -> ToolResult:
            duration = round((time.perf_counter() - start) * 1000, 1)
            result = ToolResult(
                tool=name, status=status, data=data, error=error, duration_ms=duration, request_id=request_id
            )
            log_event(
                logger,
                "tool_call",
                request_id=request_id,
                tool=name,
                module=tool.module if tool else None,
                status=status.value,
                duration_ms=duration,
                arguments=arguments,
                error=error,
            )
            return result

        if tool is None:
            return finish(ToolStatus.INVALID_REQUEST, error=f"Unknown tool '{name}'")

        try:
            parsed = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            return finish(ToolStatus.INVALID_REQUEST, error=f"Invalid arguments: {exc.errors(include_url=False)}")

        try:
            if tool.policy:
                tool.policy(parsed)
        except PolicyDenied as exc:
            return finish(ToolStatus.DENIED, error=str(exc))

        if tool.mutating and not (confirm and confirm(name, parsed.model_dump())):
            return finish(
                ToolStatus.NEEDS_CONFIRMATION,
                error="State-changing operation was NOT executed: the human did not confirm it.",
            )

        future = _POOL.submit(tool.func, parsed)
        try:
            data = future.result(timeout=self.timeout_s)
        except cf.TimeoutError:
            future.cancel()
            return finish(ToolStatus.TIMEOUT, error=f"Tool exceeded {self.timeout_s}s timeout")
        except PolicyDenied as exc:
            return finish(ToolStatus.DENIED, error=str(exc))
        except ServiceUnavailable as exc:
            return finish(ToolStatus.UNAVAILABLE, error=str(exc))
        except ServiceError as exc:
            return finish(ToolStatus.ERROR, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - every failure must become a structured result
            return finish(ToolStatus.ERROR, error=f"{type(exc).__name__}: {exc}")
        return finish(ToolStatus.OK, data=data)

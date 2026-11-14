"""Structured JSON logging with secret redaction.

Every tool call is logged with: request_id, tool, status, duration_ms.
Values whose key looks sensitive (password, token, key, secret, authorization)
are replaced with "***" before anything is written.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from typing import Any

SENSITIVE_KEY = re.compile(r"pass(word)?|token|secret|api[_-]?key|authorization|credential", re.I)
# Bearer tokens / Anthropic-style keys that might appear inside free text
SENSITIVE_VALUE = re.compile(r"(Bearer\s+[A-Za-z0-9\-_.=]+|sk-ant-[A-Za-z0-9\-_]+)")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if SENSITIVE_KEY.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SENSITIVE_VALUE.sub("***", value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(redact(extra))
        return json.dumps(payload, default=str)


def get_logger(name: str = "mlops_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"fields": fields})

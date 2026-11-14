"""Tool execution, validation, timeouts, failures and safety controls."""

import json
import logging
import time

from pydantic import BaseModel

from agent.schemas import Empty, ToolStatus
from agent.tools import build_registry
from agent.tools.base import Tool, ToolRegistry


def test_tool_selection_and_execution_success(fake_mlflow):
    result = build_registry().execute("mlflow_find_best_run", {"metric": "f1"}, "req1")
    assert result.status == ToolStatus.OK
    assert result.data["best"]["run_id"] == "b" * 32
    assert result.duration_ms >= 0 and result.request_id == "req1"


def test_unknown_tool_is_invalid_request():
    result = build_registry().execute("drop_database", {}, "req2")
    assert result.status == ToolStatus.INVALID_REQUEST


def test_invalid_arguments_are_rejected_before_execution(fake_mlflow):
    result = build_registry().execute("mlflow_find_best_run", {"lookback": 9999}, "req3")
    assert result.status == ToolStatus.INVALID_REQUEST
    assert "lookback" in result.error


def test_tool_timeout_is_reported():
    reg = ToolRegistry(timeout_s=0.2)
    reg.register(Tool("slow", "sleeps", Empty, lambda _: time.sleep(2), "test"))
    result = reg.execute("slow", {}, "req4")
    assert result.status == ToolStatus.TIMEOUT


def test_service_failure_becomes_unavailable(mlflow_down):
    result = build_registry().execute("mlflow_list_runs", {}, "req5")
    assert result.status == ToolStatus.UNAVAILABLE
    assert result.data is None


def test_unauthorized_k8s_modification_is_rejected(fake_k8s):
    _, apps = fake_k8s
    reg = build_registry()
    # 1) no human confirmation -> not executed
    r1 = reg.execute("k8s_restart_deployment", {"deployment": "inference-api"}, "req6", confirm=None)
    assert r1.status == ToolStatus.NEEDS_CONFIRMATION
    # 2) human says no -> not executed
    r2 = reg.execute("k8s_restart_deployment", {"deployment": "inference-api"}, "req6", confirm=lambda *_: False)
    assert r2.status == ToolStatus.NEEDS_CONFIRMATION
    # 3) other namespace -> denied even if confirmed
    r3 = reg.execute("k8s_restart_deployment", {"deployment": "x", "namespace": "kube-system"}, "req6",
                     confirm=lambda *_: True)
    assert r3.status == ToolStatus.DENIED
    assert apps.patched == []
    # 4) approved namespace + confirmation -> executed
    r4 = reg.execute("k8s_restart_deployment", {"deployment": "inference-api"}, "req6", confirm=lambda *_: True)
    assert r4.status == ToolStatus.OK and len(apps.patched) == 1


def test_unapproved_dag_trigger_denied(fake_airflow):
    r = build_registry().execute("airflow_trigger_dag", {"dag_id": "delete_everything"}, "req7",
                                 confirm=lambda *_: True)
    assert r.status == ToolStatus.DENIED
    assert not any(req.method == "POST" and "dagRuns" in req.url.path for req in fake_airflow)


def test_structured_logs_have_fields_and_no_secrets(caplog):
    class In(BaseModel):
        password: str
        note: str

    reg = ToolRegistry()
    reg.register(Tool("echo", "echo", In, lambda i: "ok", "test"))
    with caplog.at_level(logging.INFO, logger="mlops_agent.tools"):
        reg.execute("echo", {"password": "hunter2", "note": "Bearer abc.def"}, "req8")
    from agent.logging_utils import JsonFormatter

    line = JsonFormatter().format(caplog.records[-1])
    payload = json.loads(line)
    assert payload["tool"] == "echo" and payload["status"] == "ok" and payload["request_id"] == "req8"
    assert "duration_ms" in payload
    assert "hunter2" not in line and "abc.def" not in line

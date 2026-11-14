"""Task 3 - Apache Airflow 3 workflow tools (REST API v2, JWT-authenticated)."""

from __future__ import annotations

from typing import Any

import httpx

from agent.config import settings
from agent.schemas import DagIdInput, DagRunInput, DagRunsInput, Empty, TaskLogInput, TriggerDagInput
from agent.tools.base import PolicyDenied, ServiceError, ServiceUnavailable, Tool


class AirflowClient:
    """Minimal Airflow 3 client: POST /auth/token for a JWT, then /api/v2/* with a Bearer header."""

    def __init__(self, base_url: str, username: str, password: str, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self._username, self._password = username, password
        self._token: str | None = None
        self._http = httpx.Client(base_url=self.base_url, timeout=15, transport=transport)

    def _login(self) -> str:
        try:
            r = self._http.post("/auth/token", json={"username": self._username, "password": self._password})
        except httpx.TransportError as exc:
            raise ServiceUnavailable(f"Airflow API at {self.base_url} is unreachable") from exc
        if r.status_code not in (200, 201):
            raise ServiceError(f"Airflow authentication failed (HTTP {r.status_code})")
        return r.json()["access_token"]

    def request(self, method: str, path: str, **kwargs) -> Any:
        if self._token is None:
            self._token = self._login()
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
            try:
                r = self._http.request(method, f"/api/v2{path}", headers=headers, **kwargs)
            except httpx.TransportError as exc:
                raise ServiceUnavailable(f"Airflow API at {self.base_url} is unreachable") from exc
            if r.status_code == 401 and attempt == 1:  # token expired -> re-login once
                self._token = self._login()
                continue
            if r.status_code >= 400:
                raise ServiceError(f"Airflow {method} {path} -> HTTP {r.status_code}: {r.text[:300]}")
            return r.json() if r.content else {}
        raise ServiceError("Airflow authentication failed")


_client: AirflowClient | None = None


def get_client() -> AirflowClient:
    global _client
    if _client is None:
        _client = AirflowClient(settings.airflow_base_url, settings.airflow_username, settings.airflow_password)
    return _client


def _runs(dag_id: str, limit: int) -> list[dict[str, Any]]:
    data = get_client().request("GET", f"/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-run_after"})
    return data.get("dag_runs", [])


def _latest_run_id(dag_id: str) -> str:
    runs = _runs(dag_id, 1)
    if not runs:
        raise ServiceError(f"DAG '{dag_id}' has no runs yet")
    return runs[0]["dag_run_id"]


# ----------------------------------------------------------------------------- tools


def list_dags(_: Empty) -> list[dict[str, Any]]:
    data = get_client().request("GET", "/dags", params={"limit": 100})
    return [
        {"dag_id": d["dag_id"], "is_paused": d.get("is_paused"), "tags": [t.get("name") for t in d.get("tags", [])],
         "approved_for_trigger": d["dag_id"] in settings.approved_dags}
        for d in data.get("dags", [])
    ]


def get_dag_runs(inp: DagRunsInput) -> list[dict[str, Any]]:
    return [
        {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "run_after", "start_date", "end_date", "conf")}
        for r in _runs(inp.dag_id, inp.limit)
    ]


def _trigger_policy(inp: TriggerDagInput) -> None:
    if inp.dag_id not in settings.approved_dags:
        raise PolicyDenied(f"DAG '{inp.dag_id}' is not in the approved list {settings.approved_dags}")


def trigger_dag(inp: TriggerDagInput) -> dict[str, Any]:
    client = get_client()
    dag = client.request("GET", f"/dags/{inp.dag_id}")
    if dag.get("is_paused"):
        client.request("PATCH", f"/dags/{inp.dag_id}", json={"is_paused": False})
    run = client.request("POST", f"/dags/{inp.dag_id}/dagRuns", json={"logical_date": None, "conf": inp.conf})
    return {"dag_run_id": run.get("dag_run_id"), "state": run.get("state"), "unpaused": bool(dag.get("is_paused"))}


def get_task_instances(inp: DagRunInput) -> dict[str, Any]:
    run_id = inp.dag_run_id or _latest_run_id(inp.dag_id)
    data = get_client().request("GET", f"/dags/{inp.dag_id}/dagRuns/{run_id}/taskInstances")
    tis = [
        {k: t.get(k) for k in ("task_id", "state", "try_number", "start_date", "end_date", "duration", "operator")}
        for t in data.get("task_instances", [])
    ]
    return {"dag_run_id": run_id, "task_instances": tis}


def get_failed_tasks(inp: DagRunInput) -> dict[str, Any]:
    result = get_task_instances(inp)
    failed = [t for t in result["task_instances"] if t["state"] in ("failed", "upstream_failed", "up_for_retry")]
    return {"dag_run_id": result["dag_run_id"], "failed_tasks": failed, "any_failed": bool(failed)}


def get_task_logs(inp: TaskLogInput) -> dict[str, Any]:
    data = get_client().request(
        "GET",
        f"/dags/{inp.dag_id}/dagRuns/{inp.dag_run_id}/taskInstances/{inp.task_id}/logs/{inp.try_number}",
        params={"full_content": "true"},
    )
    content = data.get("content", "")
    if isinstance(content, list):  # Airflow 3 returns structured log events
        lines = [c.get("event", str(c)) if isinstance(c, dict) else str(c) for c in content]
    else:
        lines = str(content).splitlines()
    errors = [ln for ln in lines if any(w in ln for w in ("ERROR", "Error", "Exception", "Traceback", "FAILED"))]
    return {"task_id": inp.task_id, "try_number": inp.try_number, "error_lines": errors[-15:],
            "tail": lines[-inp.tail_lines:]}


def get_dag_details(inp: DagIdInput) -> dict[str, Any]:
    client = get_client()
    dag = client.request("GET", f"/dags/{inp.dag_id}")
    tasks = client.request("GET", f"/dags/{inp.dag_id}/tasks").get("tasks", [])
    return {
        "dag_id": inp.dag_id, "is_paused": dag.get("is_paused"),
        "tasks": [{"task_id": t["task_id"], "downstream": t.get("downstream_task_ids", []),
                   "retries": t.get("retries")} for t in tasks],
    }


TOOLS = [
    Tool("airflow_list_dags", "List Airflow DAGs and whether each is approved for triggering.",
         Empty, list_dags, "airflow"),
    Tool("airflow_get_dag_details", "Tasks, dependencies and retry settings of a DAG.",
         DagIdInput, get_dag_details, "airflow"),
    Tool("airflow_get_dag_runs", "Most recent runs of a DAG with their state.", DagRunsInput, get_dag_runs, "airflow"),
    Tool("airflow_get_task_instances", "Task statuses for a DAG run (latest run if none given).",
         DagRunInput, get_task_instances, "airflow"),
    Tool("airflow_get_failed_tasks", "Failed / upstream-failed tasks of a DAG run.",
         DagRunInput, get_failed_tasks, "airflow"),
    Tool("airflow_get_task_logs", "Log tail and error lines of one task try, for failure diagnosis.",
         TaskLogInput, get_task_logs, "airflow"),
    Tool("airflow_trigger_dag", "Trigger an APPROVED training DAG.", TriggerDagInput, trigger_dag, "airflow",
         mutating=True, policy=_trigger_policy),
]

"""Shared fakes: no test talks to a real MLflow server, Airflow, cluster or LLM."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

os.environ["LLM_PROVIDER"] = "offline"
os.environ.pop("ANTHROPIC_API_KEY", None)

from agent.config import settings  # noqa: E402
from agent.tools import airflow_tools, kubernetes_tools, mlflow_tools  # noqa: E402

NOW_MS = int(time.time() * 1000)


def make_run(run_id, f1, md5="abc123", gate="passed", age_days=1):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status="FINISHED", start_time=NOW_MS - age_days * 86_400_000,
                             run_name=f"run-{run_id[:4]}"),
        data=SimpleNamespace(
            metrics={"f1": f1, "accuracy": f1 - 0.01},
            params={"train.model": "random_forest"},
            tags={"data.dvc_md5": md5, "git.commit": "c0ffee" * 6 + "abcd", "quality_gate": gate,
                  "data.version": "v1", "mlflow.runName": f"run-{run_id[:4]}"},
        ),
    )


class FakeMlflow:
    def __init__(self, runs):
        self.runs = runs

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1", name=name)

    def search_experiments(self):
        return [SimpleNamespace(experiment_id="1", name="a3-training", lifecycle_stage="active", creation_time=NOW_MS)]

    def search_runs(self, exp_ids, filter_string="", order_by=None, max_results=10):
        return self.runs[:max_results]

    def get_run(self, run_id):
        return next(r for r in self.runs if r.info.run_id == run_id)

    def list_artifacts(self, run_id):
        return [SimpleNamespace(path="model")]


class DownMlflow:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("Failed to establish a new connection: Connection refused")
        return fail


@pytest.fixture
def fake_mlflow(monkeypatch):
    runs = [make_run("a" * 32, 0.95), make_run("b" * 32, 0.97, gate="failed"), make_run("c" * 32, 0.93)]
    client = FakeMlflow(runs)
    monkeypatch.setattr(mlflow_tools, "get_client", lambda: client)
    return client


@pytest.fixture
def mlflow_down(monkeypatch):
    monkeypatch.setattr(mlflow_tools, "get_client", lambda: DownMlflow())


# ---------------------------------------------------------------- Kubernetes fakes
def _pod(name, ready=True, restarts=0, waiting=None):
    cs = SimpleNamespace(
        name="api", ready=ready, restart_count=restarts, image="mlops-a3-inference:1.0.0",
        state=SimpleNamespace(waiting=SimpleNamespace(reason=waiting, message="back-off") if waiting else None),
        last_state=SimpleNamespace(terminated=None),
    )
    return SimpleNamespace(metadata=SimpleNamespace(name=name, creation_timestamp=None),
                           status=SimpleNamespace(phase="Running", container_statuses=[cs], conditions=[]))


class FakeCore:
    def __init__(self, pods):
        self.pods = pods

    def list_namespaced_pod(self, ns, label_selector=None):
        return SimpleNamespace(items=self.pods)

    def list_namespaced_event(self, ns):
        ev = SimpleNamespace(type="Warning", reason="Unhealthy", message="Readiness probe failed: HTTP 503", count=7,
                             involved_object=SimpleNamespace(kind="Pod", name=self.pods[0].metadata.name),
                             last_timestamp=None, event_time=None, metadata=SimpleNamespace(creation_timestamp=1))
        return SimpleNamespace(items=[ev])

    def read_namespaced_config_map(self, name, ns):
        return SimpleNamespace(data={"MODEL_VERSION": "3"})


class FakeApps:
    def __init__(self):
        self.patched = []

    def list_namespaced_deployment(self, ns):
        d = SimpleNamespace(
            metadata=SimpleNamespace(name="inference-api", labels={"app.kubernetes.io/version": "1.0.0"},
                                     annotations={}),
            spec=SimpleNamespace(replicas=2, template=SimpleNamespace(spec=SimpleNamespace(
                containers=[SimpleNamespace(image="mlops-a3-inference:1.0.0")]))),
            status=SimpleNamespace(ready_replicas=1, available_replicas=1),
        )
        return SimpleNamespace(items=[d])

    def patch_namespaced_deployment(self, name, ns, body):
        self.patched.append((name, ns, body))


@pytest.fixture
def fake_k8s(monkeypatch):
    core = FakeCore([_pod("inference-api-1"), _pod("inference-api-2", ready=False, restarts=4,
                                                   waiting="CrashLoopBackOff")])
    apps = FakeApps()
    monkeypatch.setattr(kubernetes_tools, "get_apis", lambda: (core, apps))
    return core, apps


# ---------------------------------------------------------------- Airflow fake (httpx MockTransport)
@pytest.fixture
def fake_airflow(monkeypatch):
    import httpx

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt-123"})
        assert request.headers["Authorization"] == "Bearer jwt-123"
        if path.endswith("/dagRuns") and request.method == "GET":
            return httpx.Response(200, json={"dag_runs": [
                {"dag_run_id": "manual__2", "state": "failed"}, {"dag_run_id": "manual__1", "state": "success"}]})
        if path.endswith("/dagRuns") and request.method == "POST":
            return httpx.Response(200, json={"dag_run_id": "manual__3", "state": "queued"})
        if path.endswith("/taskInstances"):
            return httpx.Response(200, json={"task_instances": [
                {"task_id": "validate_data", "state": "success"},
                {"task_id": "quality_gate", "state": "failed", "try_number": 1},
                {"task_id": "promote_model", "state": "upstream_failed"}]})
        if "/logs/" in path:
            return httpx.Response(200, json={"content": [
                {"event": "starting"}, {"event": "ERROR Quality gate FAILED: f1=0.85 is below 0.9"}]})
        if path.endswith("/dags/ml_training_pipeline"):
            return httpx.Response(200, json={"dag_id": "ml_training_pipeline", "is_paused": False})
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    client = airflow_tools.AirflowClient("http://airflow.test", "u", "secret-pw", transport=transport)
    monkeypatch.setattr(airflow_tools, "get_client", lambda: client)
    return calls


@pytest.fixture
def approved_namespace():
    return settings.k8s_namespace

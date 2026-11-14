"""Module-level behaviour of the DVC, MLflow, Airflow and Kubernetes tools."""

import shutil
import subprocess

import pytest

from agent.config import settings
from agent.schemas import ToolStatus
from agent.tools import build_registry


def test_mlflow_best_run_explains_caveats(fake_mlflow):
    data = build_registry().execute("mlflow_find_best_run", {"metric": "f1"}, "r").data
    assert [r["f1"] for r in data["ranking"]] == [0.97, 0.95, 0.93]
    assert any("production-ready" in c for c in data["caveats"])
    assert any("failed" in c for c in data["caveats"])  # best run failed its quality gate


def test_airflow_failed_tasks_and_logs(fake_airflow):
    reg = build_registry()
    failed = reg.execute("airflow_get_failed_tasks", {"dag_id": "ml_training_pipeline"}, "r").data
    assert failed["dag_run_id"] == "manual__2"
    assert {t["task_id"] for t in failed["failed_tasks"]} == {"quality_gate", "promote_model"}
    logs = reg.execute("airflow_get_task_logs", {"dag_id": "ml_training_pipeline", "dag_run_id": "manual__2",
                                                 "task_id": "quality_gate"}, "r").data
    assert "below 0.9" in logs["error_lines"][0]


def test_airflow_trigger_with_confirmation(fake_airflow):
    r = build_registry().execute("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"}, "r",
                                 confirm=lambda *_: True)
    assert r.status == ToolStatus.OK and r.data["dag_run_id"] == "manual__3"
    post = [c for c in fake_airflow if c.method == "POST" and c.url.path.endswith("/dagRuns")]
    assert len(post) == 1


def test_k8s_diagnosis_finds_crashlooping_pod(fake_k8s):
    data = build_registry().execute("k8s_diagnose_unhealthy", {}, "r").data
    assert [p["name"] for p in data["unhealthy_pods"]] == ["inference-api-2"]
    assert data["unhealthy_pods"][0]["waiting"][0]["reason"] == "CrashLoopBackOff"
    assert "Readiness probe failed" in data["warning_events"][0]["message"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_dvc_provenance_from_git_history(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "data/raw").mkdir(parents=True)
    dvc = tmp_path / "data/raw/dataset.csv.dvc"
    for version, md5 in (("v1", "1" * 32), ("v2", "2" * 32)):
        dvc.write_text(f"outs:\n- md5: {md5}\n  size: 100\n  path: dataset.csv\n")
        git("add", ".")
        git("commit", "-qm", f"data {version}")
        git("tag", f"data-{version}")
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    reg = build_registry()

    resolved = reg.execute("dvc_resolve_dataset_version", {"md5": "1" * 32}, "r")
    assert resolved.status == ToolStatus.OK and resolved.data["tags"] == ["data-v1"]
    cmp_ = reg.execute("dvc_compare_dataset_versions", {"rev_a": "data-v1", "rev_b": "data-v2"}, "r").data
    assert cmp_["identical"] is False
    bad = reg.execute("dvc_compare_dataset_versions", {"rev_a": "--output=/etc/x"}, "r")
    assert bad.status == ToolStatus.DENIED

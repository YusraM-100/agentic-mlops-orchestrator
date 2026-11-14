"""Task 2 - MLflow experiment intelligence tools (MlflowClient only, no web scraping)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from agent.config import settings
from agent.schemas import (
    BestRunInput,
    CompareExperimentsInput,
    CompareRunsInput,
    Empty,
    ListRunsInput,
    RegistryInput,
    RunIdInput,
)
from agent.tools.base import ServiceError, ServiceUnavailable, Tool

# Fail fast instead of MLflow's default multi-minute retry loop when the server is down.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

PROVENANCE_TAGS = ("data.version", "data.dvc_md5", "data.dvc_file", "git.commit", "quality_gate", "pipeline")


def get_client():  # pragma: no cover - replaced by a fake in tests
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _call(fn, *args, **kwargs):
    """Translate MLflow / network errors into the agent's error taxonomy."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        if any(s in text for s in ("ConnectionError", "Connection refused", "Max retries", "Failed to establish",
                                   "NewConnectionError", "timed out", "ConnectTimeout")):
            raise ServiceUnavailable(f"MLflow at {settings.mlflow_tracking_uri} is unreachable") from exc
        raise ServiceError(text[:500]) from exc


def _ts(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds") if ms else None


def _summ(run) -> dict[str, Any]:
    tags = run.data.tags or {}
    return {
        "run_id": run.info.run_id,
        "run_name": tags.get("mlflow.runName") or getattr(run.info, "run_name", None),
        "status": run.info.status,
        "start_time": _ts(run.info.start_time),
        "metrics": {k: round(v, 4) for k, v in (run.data.metrics or {}).items()},
        "params": dict(run.data.params or {}),
        "provenance": {k: tags[k] for k in PROVENANCE_TAGS if k in tags},
    }


def _experiment_id(client, name: str | None) -> str:
    name = name or settings.mlflow_experiment
    exp = _call(client.get_experiment_by_name, name)
    if exp is None:
        raise ServiceError(f"Experiment '{name}' does not exist")
    return exp.experiment_id


def _recent_runs(client, experiment_name: str | None, n: int) -> list:
    exp_id = _experiment_id(client, experiment_name)
    return list(
        _call(
            client.search_runs,
            [exp_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=n,
        )
    )


# ----------------------------------------------------------------------------- tools


def list_experiments(_: Empty) -> list[dict[str, Any]]:
    client = get_client()
    exps = _call(client.search_experiments)
    return [
        {"experiment_id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage,
         "created": _ts(getattr(e, "creation_time", None))}
        for e in exps
    ]


def list_runs(inp: ListRunsInput) -> list[dict[str, Any]]:
    return [_summ(r) for r in _recent_runs(get_client(), inp.experiment_name, inp.max_results)]


def get_run_details(inp: RunIdInput) -> dict[str, Any]:
    client = get_client()
    run = _call(client.get_run, inp.run_id)
    out = _summ(run)
    out["tags"] = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
    try:
        out["artifacts"] = [a.path for a in _call(client.list_artifacts, inp.run_id)]
    except ServiceError as exc:
        out["artifacts"] = f"unavailable: {exc}"
    return out


def find_best_run(inp: BestRunInput) -> dict[str, Any]:
    runs = _recent_runs(get_client(), inp.experiment_name, inp.lookback)
    scored = [r for r in runs if inp.metric in (r.data.metrics or {})]
    if not scored:
        return {"best": None, "considered": len(runs),
                "note": f"No finished run in the last {inp.lookback} recorded metric '{inp.metric}'"}
    scored.sort(key=lambda r: r.data.metrics[inp.metric], reverse=inp.higher_is_better)
    best = _summ(scored[0])
    caveats = ["A higher offline metric alone does not make a model production-ready."]
    gate = best["provenance"].get("quality_gate")
    if gate != "passed":
        caveats.append(f"Quality gate status for this run is '{gate or 'not recorded'}'.")
    if "data.dvc_md5" not in best["provenance"]:
        caveats.append("Run has no recorded dataset provenance (data.dvc_md5).")
    if len(scored) > 1:
        gap = abs(scored[0].data.metrics[inp.metric] - scored[1].data.metrics[inp.metric])
        if gap < 0.01:
            caveats.append(f"Lead over the runner-up is only {gap:.4f} - within typical noise.")
    return {
        "metric": inp.metric,
        "best": best,
        "ranking": [{"run_id": r.info.run_id, inp.metric: round(r.data.metrics[inp.metric], 4)} for r in scored],
        "considered": len(runs),
        "caveats": caveats,
    }


def compare_runs(inp: CompareRunsInput) -> dict[str, Any]:
    client = get_client()
    runs = [_call(client.get_run, rid) for rid in inp.run_ids] if inp.run_ids else _recent_runs(
        client, inp.experiment_name, inp.last_n
    )
    rows = [_summ(r) for r in runs]
    with_metric = [r for r in rows if inp.metric in r["metrics"]]
    best = max(with_metric, key=lambda r: r["metrics"][inp.metric]) if with_metric else None
    return {"metric": inp.metric, "runs": rows, "best_by_metric": best["run_id"] if best else None}


def compare_experiments(inp: CompareExperimentsInput) -> list[dict[str, Any]]:
    client = get_client()
    out = []
    for name in inp.experiment_names:
        runs = _recent_runs(client, name, 50)
        vals = [r.data.metrics[inp.metric] for r in runs if inp.metric in r.data.metrics]
        out.append({
            "experiment": name, "finished_runs": len(runs),
            f"best_{inp.metric}": round(max(vals), 4) if vals else None,
            f"mean_{inp.metric}": round(sum(vals) / len(vals), 4) if vals else None,
        })
    return out


def get_model_registry_info(inp: RegistryInput) -> dict[str, Any]:
    client = get_client()
    name = inp.model_name or settings.registered_model
    versions = _call(client.search_model_versions, f"name='{name}'")
    model = _call(client.get_registered_model, name)
    return {
        "model_name": name,
        "aliases": dict(getattr(model, "aliases", {}) or {}),
        "versions": [
            {"version": v.version, "run_id": v.run_id, "status": v.status,
             "created": _ts(v.creation_timestamp), "tags": dict(v.tags or {})}
            for v in sorted(versions, key=lambda v: int(v.version), reverse=True)[:10]
        ],
    }


TOOLS = [
    Tool("mlflow_list_experiments", "List MLflow experiments.", Empty, list_experiments, "mlflow"),
    Tool("mlflow_list_runs", "List the most recent finished runs with metrics, params and data provenance tags.",
         ListRunsInput, list_runs, "mlflow"),
    Tool("mlflow_get_run_details", "Metrics, params, tags (incl. data.dvc_md5, git.commit) and artifacts of a run.",
         RunIdInput, get_run_details, "mlflow"),
    Tool("mlflow_find_best_run", "Best recent run by a metric, with full ranking and caveats.",
         BestRunInput, find_best_run, "mlflow"),
    Tool("mlflow_compare_runs", "Side-by-side comparison of runs (explicit ids or the last N).",
         CompareRunsInput, compare_runs, "mlflow"),
    Tool("mlflow_compare_experiments", "Compare two or more experiments on a metric.",
         CompareExperimentsInput, compare_experiments, "mlflow"),
    Tool("mlflow_get_model_registry_info", "Registered model versions, aliases (e.g. champion) and source runs.",
         RegistryInput, get_model_registry_info, "mlflow"),
]

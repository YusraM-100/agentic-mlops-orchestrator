"""Structured inputs and outputs shared by every tool and by the orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # tool ran but the service returned an error
    UNAVAILABLE = "unavailable"  # service could not be reached
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"  # input failed schema validation / unknown tool
    DENIED = "denied"  # safety policy rejected the call
    NEEDS_CONFIRMATION = "needs_confirmation"  # mutating call, human said no / not asked


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK

    def for_llm(self) -> str:
        """Compact JSON given back to the model as the tool_result content."""
        return self.model_dump_json(exclude={"duration_ms"})


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: ToolResult


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    mode: Literal["anthropic", "offline"] = "offline"


# ---------- tool input models (one per tool; JSON schema is generated from them) ----------


class Empty(BaseModel):
    pass


# DVC
class DatasetPathInput(BaseModel):
    dvc_file: str = Field("data/raw/dataset.csv.dvc", description="Path of the .dvc file, relative to repo root")


class DatasetHistoryInput(DatasetPathInput):
    max_entries: int = Field(10, ge=1, le=50)


class CompareDatasetInput(DatasetPathInput):
    rev_a: str = Field(..., description="Git revision (commit, tag or branch) of the first version")
    rev_b: str = Field("HEAD", description="Git revision of the second version")


class ResolveDatasetInput(DatasetPathInput):
    md5: str | None = Field(None, description="DVC md5 recorded in an MLflow run tag (data.dvc_md5)")
    git_commit: str | None = Field(None, description="Git commit recorded in an MLflow run tag (git.commit)")


class ReproduceInput(DatasetPathInput):
    rev: str = Field(..., description="Git revision or tag of the dataset version to reproduce")


# MLflow
class ListRunsInput(BaseModel):
    experiment_name: str | None = Field(None, description="Defaults to the configured training experiment")
    max_results: int = Field(5, ge=1, le=50)


class RunIdInput(BaseModel):
    run_id: str


class BestRunInput(BaseModel):
    metric: str = Field("f1", description="Metric name, e.g. f1, accuracy, roc_auc")
    experiment_name: str | None = None
    lookback: int = Field(10, ge=1, le=100, description="Only consider the most recent N finished runs")
    higher_is_better: bool = True


class CompareRunsInput(BaseModel):
    run_ids: list[str] | None = Field(None, description="Explicit run ids; if omitted the last N runs are used")
    last_n: int = Field(5, ge=2, le=20)
    metric: str = "f1"
    experiment_name: str | None = None


class CompareExperimentsInput(BaseModel):
    experiment_names: list[str] = Field(..., min_length=2)
    metric: str = "f1"


class RegistryInput(BaseModel):
    model_name: str | None = Field(None, description="Defaults to the configured registered model")


# Airflow
class DagIdInput(BaseModel):
    dag_id: str


class DagRunsInput(DagIdInput):
    limit: int = Field(5, ge=1, le=50)


class TriggerDagInput(DagIdInput):
    conf: dict[str, Any] = Field(default_factory=dict, description="Optional DAG run parameters")


class DagRunInput(DagIdInput):
    dag_run_id: str | None = Field(None, description="If omitted, the most recent run is used")


class TaskLogInput(DagIdInput):
    dag_run_id: str
    task_id: str
    try_number: int = Field(1, ge=1)
    tail_lines: int = Field(60, ge=5, le=500)


# Kubernetes
class NamespaceInput(BaseModel):
    namespace: str | None = Field(None, description="Must be the approved namespace; defaults to it")


class PodInput(NamespaceInput):
    pod_name: str


class PodLogInput(PodInput):
    tail_lines: int = Field(80, ge=5, le=1000)
    previous: bool = Field(False, description="Logs of the previous (crashed) container")


class RestartInput(NamespaceInput):
    deployment: str


class ScaleInput(RestartInput):
    replicas: int = Field(..., ge=0, le=5)

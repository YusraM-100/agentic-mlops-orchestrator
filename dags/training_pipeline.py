"""Task 3 - required training DAG (Airflow 3).

validate_data -> preprocess -> train_model -> evaluate_and_log (MLflow) -> quality_gate -> promote_model

* retries + retry_delay on every task (transient failures),
* AirflowFailException for deterministic failures (bad data, failed gate) -> no pointless retries,
* promote_model only runs when quality_gate succeeds (default all_success trigger rule),
* everything is logged through the task logger (visible in the UI and via the REST API).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # pragma: no cover
    from airflow.exceptions import AirflowFailException

PROJECT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT))
log = logging.getLogger(__name__)


def _work_dir() -> Path:
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", get_current_context()["run_id"])
    return PROJECT / "artifacts" / "airflow" / run_id


@dag(
    dag_id="ml_training_pipeline",
    description="Validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> promote",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlops-a3", "training"],
    default_args={"retries": 2, "retry_delay": timedelta(seconds=20), "owner": "mlops"},
    params={
        "min_f1": Param(0.90, type="number", minimum=0, maximum=1, description="Quality gate threshold"),
        "model": Param("random_forest", enum=["random_forest", "logistic_regression"]),
    },
)
def ml_training_pipeline():
    @task
    def validate_data() -> dict:
        from training.pipeline import DataValidationError, load_params, validate

        p = load_params(PROJECT)
        try:
            return validate(PROJECT / p["data"]["path"], p["validate"]["min_rows"])
        except DataValidationError as exc:
            log.error("Data validation failed: %s", exc)
            raise AirflowFailException(f"Data validation failed: {exc}") from exc

    @task
    def preprocess(_report: dict) -> str:
        from training.pipeline import load_params
        from training.pipeline import preprocess as prep

        p = load_params(PROJECT)
        out = _work_dir() / "processed"
        prep(PROJECT / p["data"]["path"], out, p["train"]["test_size"], p["train"]["seed"])
        return str(out)

    @task
    def train_model(processed_dir: str) -> str:
        from training.pipeline import load_params, train

        p = load_params(PROJECT)
        params = {**p["train"], "model": get_current_context()["params"]["model"]}
        return train(Path(processed_dir), _work_dir() / "model.joblib", params)

    @task
    def evaluate_and_log(processed_dir: str, model_path: str) -> dict:
        from training.pipeline import evaluate, load_params, log_to_mlflow

        p = load_params(PROJECT)
        p["train"]["model"] = get_current_context()["params"]["model"]
        metrics = evaluate(Path(processed_dir), Path(model_path))
        run_id = log_to_mlflow(metrics, p, Path(model_path), pipeline="airflow", root=PROJECT)
        return {"metrics": metrics, "mlflow_run_id": run_id}

    @task
    def quality_gate(result: dict) -> dict:
        from training.pipeline import QualityGateError, set_gate_tag
        from training.pipeline import quality_gate as gate

        threshold = float(get_current_context()["params"]["min_f1"])
        try:
            gate(result["metrics"], threshold)
        except QualityGateError as exc:
            set_gate_tag(result["mlflow_run_id"], "failed")
            log.error("Quality gate FAILED: %s", exc)
            raise AirflowFailException(str(exc)) from exc
        set_gate_tag(result["mlflow_run_id"], "passed")
        return result

    @task
    def promote_model(result: dict) -> str:
        from training.pipeline import promote

        version = promote(result["mlflow_run_id"])
        log.info("Model version %s promoted to 'champion'", version)
        return version

    report = validate_data()
    processed = preprocess(report)
    model = train_model(processed)
    result = evaluate_and_log(processed, model)
    promote_model(quality_gate(result))


ml_training_pipeline()

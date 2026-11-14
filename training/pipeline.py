"""Training stages shared by the DVC pipeline (dvc repro) and the Airflow DAG.

validate -> preprocess -> train -> evaluate (+ MLflow logging) -> quality gate -> promote
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.provenance import dvc_md5, git_commit

log = logging.getLogger("training")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
LABEL = "target"


class DataValidationError(Exception):
    pass


class QualityGateError(Exception):
    pass


def load_params(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((root / "params.yaml").read_text())


# ----------------------------------------------------------------------------- stages


def validate(data_path: Path, min_rows: int = 100) -> dict[str, Any]:
    if not data_path.exists():
        raise DataValidationError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    problems = []
    if LABEL not in df.columns:
        problems.append(f"missing label column '{LABEL}'")
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (< {min_rows})")
    nulls = int(df.isna().sum().sum())
    if nulls:
        problems.append(f"{nulls} null values")
    if LABEL in df.columns and set(df[LABEL].unique()) - {0, 1}:
        problems.append("label is not binary 0/1")
    report = {"rows": len(df), "columns": len(df.columns), "nulls": nulls, "problems": problems}
    log.info("validation report: %s", report)
    if problems:
        raise DataValidationError("; ".join(problems))
    return report


def preprocess(data_path: Path, out_dir: Path, test_size: float, seed: int) -> dict[str, str]:
    df = pd.read_csv(data_path)
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[LABEL])
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    log.info("preprocess: train=%d test=%d -> %s", len(train), len(test), out_dir)
    return {"train": str(out_dir / "train.csv"), "test": str(out_dir / "test.csv")}


def train(processed_dir: Path, model_path: Path, params: dict[str, Any]) -> str:
    df = pd.read_csv(processed_dir / "train.csv")
    x, y = df.drop(columns=[LABEL]), df[LABEL]
    if params["model"] == "logistic_regression":
        est = LogisticRegression(C=params.get("C", 1.0), max_iter=2000)
    else:
        est = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                     random_state=params["seed"])
    model = make_pipeline(StandardScaler(), est).fit(x, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("trained %s -> %s", params["model"], model_path)
    return str(model_path)


def evaluate(processed_dir: Path, model_path: Path) -> dict[str, float]:
    df = pd.read_csv(processed_dir / "test.csv")
    x, y = df.drop(columns=[LABEL]), df[LABEL]
    model = joblib.load(model_path)
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "roc_auc": roc_auc_score(y, proba),
    }
    log.info("metrics: %s", metrics)
    return {k: round(float(v), 4) for k, v in metrics.items()}


def log_to_mlflow(metrics: dict[str, float], params: dict[str, Any], model_path: Path, pipeline: str,
                  root: Path = ROOT) -> str:
    """Log the run with full data provenance tags. Returns the MLflow run id."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "a3-training"))
    dvc_file = "data/raw/dataset.csv.dvc"
    tags = {
        "data.version": params["data"]["version"],
        "data.dvc_file": dvc_file,
        "data.dvc_md5": dvc_md5(root / dvc_file) or "untracked",
        "git.commit": git_commit(root) or "unknown",
        "pipeline": pipeline,
        "quality_gate": "pending",
    }
    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params({f"train.{k}": v for k, v in params["train"].items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            joblib.load(model_path),
            name="model",
            registered_model_name=os.getenv("REGISTERED_MODEL", "a3-classifier"),
            serialization_format="cloudpickle",  # our own model, trusted; avoids skops type allow-listing
            pip_requirements=["scikit-learn==1.8.0", "pandas==2.3.3"],  # skip slow requirement inference
        )
        log.info("MLflow run %s logged with tags %s", run.info.run_id, tags)
        return run.info.run_id


def quality_gate(metrics: dict[str, float], min_f1: float) -> None:
    if metrics["f1"] < min_f1:
        raise QualityGateError(f"f1={metrics['f1']} is below the quality gate threshold {min_f1}")
    log.info("quality gate passed: f1=%s >= %s", metrics["f1"], min_f1)


def set_gate_tag(run_id: str, status: str) -> None:
    from mlflow.tracking import MlflowClient

    MlflowClient(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")).set_tag(run_id, "quality_gate", status)


def promote(run_id: str, alias: str = "champion") -> str:
    """Point the registry alias at the model version created by this run."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    name = os.getenv("REGISTERED_MODEL", "a3-classifier")
    version = next(v.version for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id)
    client.set_registered_model_alias(name, alias, version)
    log.info("promoted %s version %s to alias '%s'", name, version, alias)
    return version


# ----------------------------------------------------------------------------- CLI for dvc.yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validate", "preprocess", "train", "evaluate"])
    stage = ap.parse_args().stage
    p = load_params()
    raw, processed, model = ROOT / p["data"]["path"], ROOT / "data/processed", ROOT / "models/model.joblib"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if stage == "validate":
        (reports / "validation.json").write_text(json.dumps(validate(raw, p["validate"]["min_rows"]), indent=2))
    elif stage == "preprocess":
        preprocess(raw, processed, p["train"]["test_size"], p["train"]["seed"])
    elif stage == "train":
        train(processed, model, p["train"])
    elif stage == "evaluate":
        metrics = evaluate(processed, model)
        (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if os.getenv("MLFLOW_LOG", "1") == "1":
            run_id = log_to_mlflow(metrics, p, model, pipeline="dvc")
            try:
                quality_gate(metrics, p["quality_gate"]["min_f1"])
                set_gate_tag(run_id, "passed")
            except QualityGateError:
                set_gate_tag(run_id, "failed")
                raise


if __name__ == "__main__":
    main()

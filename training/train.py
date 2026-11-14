"""Run the whole training pipeline locally (same code the Airflow DAG uses).

    python -m training.train
"""

from training.pipeline import (
    ROOT,
    evaluate,
    load_params,
    log_to_mlflow,
    preprocess,
    quality_gate,
    set_gate_tag,
    validate,
)
from training.pipeline import train as fit

if __name__ == "__main__":
    p = load_params()
    raw, processed, model = ROOT / p["data"]["path"], ROOT / "data/processed", ROOT / "models/model.joblib"
    validate(raw, p["validate"]["min_rows"])
    preprocess(raw, processed, p["train"]["test_size"], p["train"]["seed"])
    fit(processed, model, p["train"])
    metrics = evaluate(processed, model)
    run_id = log_to_mlflow(metrics, p, model, pipeline="local")
    quality_gate(metrics, p["quality_gate"]["min_f1"])
    set_gate_tag(run_id, "passed")
    print(f"done: run_id={run_id} metrics={metrics}")

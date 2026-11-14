"""FastAPI inference service deployed on Kubernetes.

/health  - liveness probe (process is up)
/ready   - readiness probe (model file loaded) -> 503 until a model is available
/version - app + model version (read from the ConfigMap env vars)
/predict - prediction for one or more feature rows
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference-api")

APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

_state: dict[str, Any] = {"model": None, "error": None}


def load_model() -> None:
    try:
        _state["model"] = joblib.load(MODEL_PATH)
        _state["error"] = None
        log.info("model loaded from %s (model_version=%s)", MODEL_PATH, MODEL_VERSION)
    except Exception as exc:  # noqa: BLE001
        _state["model"] = None
        _state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("ERROR loading model from %s: %s", MODEL_PATH, _state["error"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="A3 Inference API", version=APP_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    rows: list[dict[str, float]] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_state['error']}")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"app_version": APP_VERSION, "model_version": MODEL_VERSION, "model_path": str(MODEL_PATH)}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import pandas as pd

    start = time.perf_counter()
    try:
        frame = pd.DataFrame(req.rows)[list(model.feature_names_in_)]
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"missing features: {exc}") from exc
    preds = model.predict(frame).tolist()
    proba = model.predict_proba(frame)[:, 1].round(4).tolist()
    log.info(json.dumps({"event": "predict", "rows": len(preds), "ms": round((time.perf_counter() - start) * 1000, 1)}))
    return {"predictions": preds, "probabilities": proba, "model_version": MODEL_VERSION}

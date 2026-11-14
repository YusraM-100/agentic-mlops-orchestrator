"""Central configuration. Every value comes from environment variables (.env),
so no secret is ever hard-coded or committed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime (CI sets env vars directly)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "12")))

    # Tool execution
    tool_timeout_s: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "30")))

    # DVC / Git
    repo_root: Path = field(default_factory=lambda: Path(os.getenv("REPO_ROOT", str(REPO_ROOT))))

    # MLflow
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow_experiment: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "a3-training"))
    registered_model: str = field(default_factory=lambda: os.getenv("REGISTERED_MODEL", "a3-classifier"))

    # Airflow 3
    airflow_base_url: str = field(default_factory=lambda: os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080"))
    airflow_username: str = field(default_factory=lambda: os.getenv("AIRFLOW_USERNAME", "airflow"))
    airflow_password: str = field(default_factory=lambda: os.getenv("AIRFLOW_PASSWORD", ""))
    approved_dags: list[str] = field(
        default_factory=lambda: _split(os.getenv("APPROVED_DAGS", "ml_training_pipeline"))
    )

    # Kubernetes
    k8s_namespace: str = field(default_factory=lambda: os.getenv("K8S_NAMESPACE", "mlops-a3"))
    k8s_app_label: str = field(default_factory=lambda: os.getenv("K8S_APP_LABEL", "app=inference-api"))
    k8s_context: str | None = field(default_factory=lambda: os.getenv("K8S_CONTEXT") or None)


settings = Settings()

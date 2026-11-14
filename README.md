# Assignment 03 – Production-Ready MLOps AI Agent

A conversational agent that answers operational questions about an ML project by calling real tools:

| Module | Technology | Tools (prefix) |
|---|---|---|
| Data management | DVC + Git | `dvc_*` (7 tools) |
| Experiment intelligence | MLflow 3 (`MlflowClient`) | `mlflow_*` (7 tools) |
| Workflow operations | Apache Airflow 3 REST API v2 (JWT auth) | `airflow_*` (7 tools) |
| Deployment operations | Kubernetes Python client | `k8s_*` (8 tools) |

The agent runs on Claude (Messages API tool use) when `ANTHROPIC_API_KEY` is set, or in a
deterministic **offline mode** (keyword planner) otherwise. In both modes answers are built only
from tool results; if no tool returns data the agent says so instead of inventing values.

See [architecture.md](architecture.md) for the design and [report/TECHNICAL_REPORT.md](report/TECHNICAL_REPORT.md) for the report.

## 1. Prerequisites (Windows 10/11)

- Python 3.12 (python.org installer, tick "Add to PATH")
- Git for Windows
- Docker Desktop (WSL 2 backend) with **Settings → Kubernetes → Enable Kubernetes**
- VS Code (+ Python extension)

## 2. Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env      # then edit .env
```

## 3. Environment configuration (`.env`, never committed)

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the Claude planner (empty = offline mode) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |
| `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow 3 API credentials |
| `APPROVED_DAGS` | DAGs the agent may trigger |
| `K8S_NAMESPACE`, `K8S_CONTEXT` | The only namespace the agent may touch |

## 4. Launching each service

```powershell
# MLflow + Airflow 3 (postgres, api-server, scheduler, dag-processor)
docker compose up -d --build
docker compose ps                       # wait until airflow-apiserver is healthy
# MLflow UI: http://localhost:5000   Airflow UI: http://localhost:8080

# Data + local training pipeline (DVC)
python -m training.make_dataset --version v1
dvc add data/raw/dataset.csv
dvc repro                               # validate -> preprocess -> train -> evaluate (+MLflow)

# Inference service on Kubernetes (Docker Desktop)
docker build -t mlops-a3-inference:1.0.0 .
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl -n mlops-a3 get pods,svc
kubectl -n mlops-a3 port-forward svc/inference-api 8000:80   # http://localhost:8000/docs
```

## 5. Tests

```powershell
ruff check .
pytest -v
```

All external systems are mocked in tests (fake MLflow client, `httpx.MockTransport` for Airflow,
fake Kubernetes APIs, scripted LLM), so CI needs no services.

## 6. Talking to the agent

```powershell
python -m agent.main --tools                     # list the tools
python -m agent.main                             # interactive chat (Claude if key set)
python -m agent.main --offline --trace -q "Which dataset version is in use and has it changed?"
```

Demo questions (the five evaluated scenarios):

1. *Dataset provenance* – "Which dataset version was used for the best run? How do I reproduce it?"
2. *Model selection* – "Compare the last five training runs and recommend the best candidate based on F1 score."
3. *Pipeline execution* – "Trigger the model training pipeline and report whether execution succeeded."
4. *Deployment diagnosis* – "Why is my prediction API unavailable?"
5. *Multi-tool* – "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

State-changing tools (`airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`) always
ask for a typed `yes` in the terminal and are limited to the approved DAG list / namespace.

## 7. Security

No secrets in the repo: credentials come from `.env` (git-ignored); logs redact keys such as
`password`, `token`, `api_key`, `authorization` and Bearer tokens.

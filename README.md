# Agentic MLOps Orchestrator – Assignment 03

**Student:** Yusra Masood (23i-0565) · **Course:** MLOps & Production AI Engineering (Fall 2026)
**Repository:** https://github.com/YusraM-100/agentic-mlops-orchestrator

A conversational agent that answers operational questions about an ML project by calling real tools across four systems:

| Module | Technology | Tools |
|---|---|---|
| Data management | DVC + Git | `dvc_*` (7) |
| Experiment intelligence | MLflow 3 (`MlflowClient`) | `mlflow_*` (7) |
| Workflow operations | Apache Airflow 3 REST API v2 (JWT) | `airflow_*` (7) |
| Deployment operations | Kubernetes Python client | `k8s_*` (8) |

The agent uses Claude (Messages API tool use) when `ANTHROPIC_API_KEY` is set, otherwise a deterministic **offline mode** (keyword planner). All evidence in this repo was produced in offline mode. In both modes answers are built only from tool results; if no tool returns data, the agent says so instead of inventing values.

- Design: [architecture.md](architecture.md)
- Report with evidence: [report/TECHNICAL_REPORT.md](report/TECHNICAL_REPORT.md)
- Screenshots: [screenshots/](screenshots/)

## 1. Prerequisites (Windows 10/11)
- Python 3.12
- Git for Windows
- Docker Desktop (WSL 2) with **Settings → Kubernetes → Enable Kubernetes** (tested with v4.99, Kubernetes v1.36.1 in kind mode, context `docker-desktop`)
- VS Code with the Python extension

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
| `AIRFLOW_JWT_SECRET` | Any long random string (`python -c "import secrets; print(secrets.token_hex(32))"`) |
| `APPROVED_DAGS` | DAGs the agent may trigger |
| `K8S_NAMESPACE`, `K8S_CONTEXT` | The only namespace the agent may touch |

## 4. Launching each service
```powershell
# MLflow + Airflow 3 (postgres, api-server, scheduler, dag-processor)
docker compose up -d --build
docker compose ps                  # wait until airflow-apiserver is healthy
# MLflow UI: http://localhost:5000   Airflow UI: http://localhost:8080 (airflow / airflow)

# Data versioning + local training (DVC)
dvc pull                           # or: python -m training.make_dataset --version v2
dvc repro                          # validate -> preprocess -> train -> evaluate (+ MLflow)

# Inference API on Kubernetes
kubectl config use-context docker-desktop
docker build -t mlops-a3-inference:1.0.0 .
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl -n mlops-a3 get pods,svc
kubectl -n mlops-a3 port-forward svc/inference-api 8000:80   # http://localhost:8000/docs
```
Dataset versions are tagged `data-v1.0` (569 rows) and `data-v2.0` (869 rows). To reproduce v1:
`git checkout data-v1.0 -- data/raw/dataset.csv.dvc; dvc checkout`

## 5. Tests
```powershell
ruff check .
pytest -v                          # 19 tests
```
All external systems are mocked (fake MLflow client, `httpx.MockTransport` for Airflow, fake Kubernetes APIs, scripted LLM), so CI needs no running services. GitHub Actions runs ruff + pytest + a Docker build on every pull request.

## 6. Talking to the agent
```powershell
python -m agent.main --tools                 # list the 29 tools
python -m agent.main --offline --trace       # interactive chat, prints every tool call
python -m agent.main --offline --trace -q "Which dataset version is in use and has it changed?"
```
The five evaluated scenarios:
1. **Dataset provenance:** "Which dataset version was used for the best run? How do I reproduce it?"
2. **Model selection:** "Compare the last five training runs and recommend the best candidate based on F1 score."
3. **Pipeline execution:** "Trigger the model training pipeline and report whether execution succeeded."
4. **Deployment diagnosis:** "Why is my prediction API unavailable?"
5. **Multi-tool:** "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

State-changing tools (`airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`) always require a typed `yes` and are restricted to the approved DAG list and the `mlops-a3` namespace. Requests outside it (e.g. `kube-system`) are denied.

## 7. Failure injection (demonstrated in the report)
- **Corrupt data:** `python -m training.make_dataset --version corrupt`, then trigger the DAG. `validate_data` fails and nothing downstream runs. Fix: `dvc checkout --force data/raw/dataset.csv.dvc`
- **API unavailable:** set `MODEL_PATH` to a missing file in `kubernetes/configmap.yaml`, apply, and restart. Pods are alive but never Ready (503). Fix: restore the path, apply, and restart.

## 8. Security
No secrets in the repo: credentials come from `.env` (git-ignored), and logs redact `password`, `token`, `api_key`, `authorization` and Bearer tokens. The agent can only change state in one namespace, after explicit human confirmation.
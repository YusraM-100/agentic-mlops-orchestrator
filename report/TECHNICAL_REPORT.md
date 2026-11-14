# Technical Report – Assignment 03: Agentic MLOps Orchestrator

Repository: https://github.com/YusraM-100/agentic-mlops-orchestrator
Environment: Windows 11, Python 3.12.10, Docker Desktop 4.99 (Kubernetes v1.36.1, kind), MLflow 3.15.2, Airflow 3.3.2, DVC 3.67.1

## 1. System architecture
See `architecture.md` for the full diagram.

The user talks to `MLOpsAgent` through a CLI. The agent plans tool calls (Claude tool-use loop when an API key is set, otherwise a deterministic keyword `OfflinePlanner`; all results below are from offline mode). Every call goes through `ToolRegistry`, which does pydantic input validation → safety policy → human confirmation for state-changing tools → execution with a timeout → structured `ToolResult` + JSON log line with `request_id` and duration.

Four tool modules:
- **DVC** (git/dvc via an allow-listed subprocess): dataset versions, history, provenance resolution.
- **MLflow** (MlflowClient): experiments, runs, comparison, model registry.
- **Airflow** (REST API v2 + JWT): DAGs, runs, task states, task logs, guarded trigger.
- **Kubernetes** (python client): workloads, pod status/logs, events, services; guarded restart/scale.

**Provenance chain:** every training run is tagged in MLflow with `data.version`, `data.dvc_md5`, `data.dvc_file`, `git.commit` and `quality_gate`. The agent reads these tags and passes them to `dvc_resolve_dataset_version`, which looks up the matching commit/tag in git history.

**Why these tools:** DVC versions data alongside git without storing large files in git; MLflow gives experiment tracking plus a registry with aliases (`champion`); Airflow orchestrates the same stage functions as `dvc.yaml`, so local and scheduled training are identical; Kubernetes provides probes, resource limits, rolling updates and replicas for the inference API.

## 2. Tool definitions and permissions
29 tools (`python -m agent.main --tools`, screenshot in `screenshots/`): 7 DVC, 7 MLflow, 7 Airflow, 8 Kubernetes. Three are state-changing and are marked `[CHANGES STATE: requires explicit human confirmation]`: `airflow_trigger_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`.

| Tool group | Reads | Writes | Guard |
|---|---|---|---|
| `dvc_*` | git log/show/tag, dvc status/diff | none | command allow-list, revision/path regex |
| `mlflow_*` | experiments, runs, registry | none | – |
| `airflow_*` (read) | DAGs, runs, task instances, logs | none | JWT auth |
| `airflow_trigger_dag` | – | new DAG run | approved DAG list + human confirmation |
| `k8s_*` (read) | pods, deployments, services, events, logs | none | approved namespace `mlops-a3` |
| `k8s_restart_deployment`, `k8s_scale_deployment` | – | patch Deployment | approved namespace + human confirmation |

**Safety demonstrated:**
- `k8s_restart_deployment` with the human answering "no" → `needs_confirmation`: "State-changing operation was NOT executed: the human did not confirm it."
- `k8s_restart_deployment` in `kube-system` → `denied`: "Namespace 'kube-system' is not approved; this agent may only operate in 'mlops-a3'" (rejected before any prompt).

## 3. Evidence from each system (all in `screenshots/`)
| # | Evidence |
|---|---|
| 1 | DVC: `git log --oneline`, `git tag` (data-v1.0, data-v2.0), `dvc status`, `dataset.csv.dvc`, `dvc dag`, `dvc metrics show` |
| 2 | Reproducible checkout: v1 → 570 lines, back to v2 → 870 lines |
| 3 | MLflow: `a3-training` run list, run comparison, run tags (data.version, data.dvc_md5, git.commit, pipeline), registry `a3-classifier` v1–v3 with `champion` on v3 |
| 4 | Airflow: DAG graph (6 tasks), successful run, `evaluate_and_log` log, MLflow run tagged `pipeline=airflow` |
| 5 | Kubernetes: `get pods,svc,configmap` (2/2 Running), `describe deployment` (probes, limits), API `/version`, `/ready`, `/predict` |
| 6 | Agent: tool list, five scenarios, safety demo, two failure investigations |

**Training results**
| Run | Data | Pipeline | F1 | Accuracy | Registry |
|---|---|---|---|---|---|
| receptive-skunk-935 | v1 (569 rows) | dvc | 0.9583 | 0.9474 | v1 |
| thundering-skunk-953 | v2 (869 rows) | dvc | 0.9774 | 0.9713 | v2 |
| placid-stork-807 | v2 | airflow | 0.9774 | 0.9713 | v3 (`champion`) |

## 4. Five tested agent conversations (offline mode, `--trace`)
**1. Dataset provenance** – "Which dataset version was used for the best run? How do I reproduce it?"
Tools: `dvc_get_current_dataset_version` ok, `mlflow_find_best_run` ok.
Answer: best run `placid-stork-807` (F1 0.9774) trained on data v2, md5 `8dbaba79…`, tag `data-v2.0`. Caveat: lead over runner-up is 0.0000 (within noise). Reproduce: `git checkout data-v2.0 -- data/raw/dataset.csv.dvc; dvc checkout` (demonstrated in §3).

**2. Model selection** – "Compare the last five training runs and recommend the best candidate based on F1 score."
Tools: `mlflow_compare_runs(last_n=5)` ok, `mlflow_find_best_run` ok.
Answer: 3 runs existed; recommended `placid-stork-807` (v2, airflow, F1 0.9774), tied with the v2 dvc run; v1 run lower (0.9583). Caveat: an offline metric alone does not make a model production-ready.

**3. Pipeline execution** – "Trigger the model training pipeline and report whether execution succeeded."
Tools: confirmation prompt → user typed `yes` → `airflow_trigger_dag` ok (state `queued`). Follow-up `airflow_get_dag_runs` showed the run `success` in 38 s, `airflow_get_failed_tasks` → none.

**4. Deployment diagnosis** – "Why is my prediction API unavailable?"
Tools: `k8s_list_workloads`, `k8s_diagnose_unhealthy`, `k8s_list_services`, all ok.
Answer (healthy state): deployment 2/2 ready, 0 unhealthy pods, service has 2 ready endpoints; two old "Readiness probe failed: connection refused" events from startup only.

**5. Multi-tool (mandatory)** – "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
Tools (6, all ok): `mlflow_find_best_run`, `dvc_resolve_dataset_version`, `dvc_get_current_dataset_version`, `k8s_diagnose_unhealthy`, `k8s_get_deployed_version`, `airflow_get_dag_runs`.
Answer:
- Facts: best run F1 0.9774, quality_gate passed; training data md5 `8dbaba79…` equals the current dataset; 2 pods, 0 unhealthy; deployed image `mlops-a3-inference:1.0.0`, model_version 1; last 2 DAG runs success.
- Uncertainties: none.
- Recommended actions: no evidence that retraining is necessary. **No DAG was triggered.**

## 5. Failure investigations
### Failure 1 – Corrupt data blocks the pipeline
Injection: `python -m training.make_dataset --version corrupt` (drops `target`, adds nulls), then triggered the DAG via the agent (confirmed `yes`).
Observed: `validate_data` failed on try 1 (no pointless retries); preprocess → promote_model all `upstream_failed`; no model trained or promoted.
Agent diagnosis: `airflow_get_dag_runs` + `airflow_get_failed_tasks` identified the failed run and task; `airflow_get_task_logs` returned:
`Data validation failed: missing label column 'target'; 125 null values` (rows 869, columns 30).
Resolution: `dvc checkout --force data/raw/dataset.csv.dvc` restored v2 (870 lines); re-triggered the DAG → success.

### Failure 2 – Prediction API pods never become Ready
Injection: `MODEL_PATH: "/app/models/missing.joblib"` in `configmap.yaml`, `kubectl apply`, `kubectl rollout restart`.
Observed: new pod `Running` but `0/1` Ready; because the rolling update uses maxUnavailable=0, the 2 old pods kept serving.
Agent diagnosis: `k8s_diagnose_unhealthy` listed the unready pod and 21× "Readiness probe failed: HTTP probe failed with statuscode: 503"; `k8s_get_pod_logs` showed
`ERROR loading model from /app/models/missing.joblib: FileNotFoundError` while `/health` returned 200 (alive but not ready).
Resolution: restored the ConfigMap, `kubectl apply`, then `k8s_restart_deployment` with human confirmation `yes` → 2 new pods 1/1, 0 unhealthy, 2 ready endpoints.

## 6. Known limitations and proposed improvements
- The offline planner is keyword-based: it did not map "restart" to the restart tool, and in Failure 1 it fetched failed tasks but not the task log automatically (the log tool had to be called directly). An LLM planner (Claude) handles free-form questions and chains tools better.
- The model is baked into the image: Kubernetes serves model_version 1 while the registry `champion` is v3, and the agent reports both facts without flagging the mismatch. Improvement: load the `champion` alias from MLflow at start-up and add a deployed-vs-champion check.
- Single-node local cluster and local DVC remote; improvement: RBAC service account for the agent, remote DVC storage (S3/GDrive).
- Kubernetes warning events persist after recovery and can confuse diagnosis; the agent should filter events by current pods/age.
- No drift monitoring; the retraining policy is evidence-based (quality gate / freshness) only.
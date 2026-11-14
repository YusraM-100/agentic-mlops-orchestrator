# Architecture

```
            ┌─────────────────────── User (CLI: python -m agent.main) ───────────────────────┐
            │                                                                                 │
            ▼                                                                                 │
   MLOpsAgent (agent/orchestrator.py)                                                         │
   ├─ AnthropicPlanner: Claude Messages API tool-use loop  (or OfflinePlanner: keyword plan)  │
   ├─ grounding check: no successful tool result ⇒ "No operational data could be retrieved"   │
   └─ human confirmation callback for state-changing tools ◄──────────────────────────────────┘
            │ structured tool calls (pydantic-validated JSON)
            ▼
   ToolRegistry (agent/tools/base.py)
   validate input → safety policy → confirmation → run with timeout → ToolResult + JSON log
            │
   ┌────────┼──────────────────┬──────────────────────┬──────────────────────────┐
   ▼        ▼                  ▼                      ▼
 dvc_tools  mlflow_tools       airflow_tools          kubernetes_tools
 git/dvc    MlflowClient       REST /api/v2 + JWT     kubernetes client
 (allow-    (tracking +        (/auth/token)          (approved namespace only)
  listed)    registry)
   │        │                  │                      │
   ▼        ▼                  ▼                      ▼
 Git repo + MLflow server     Airflow 3 (Docker)     Docker Desktop Kubernetes
 DVC remote (docker, :5000)   training DAG           FastAPI inference (2 replicas,
                                                      probes, limits, ConfigMap)
```

## Provenance chain

`training.pipeline.log_to_mlflow` tags every run with `data.dvc_md5`, `data.dvc_file`, `data.version`,
`git.commit` and `quality_gate`. The agent links a model to its data by reading those tags
(`mlflow_get_run_details` / `mlflow_find_best_run`) and passing them to `dvc_resolve_dataset_version`,
which searches the git history of `data/raw/dataset.csv.dvc` for the matching md5/commit and tag.

## Training DAG (`dags/training_pipeline.py`)

`validate_data → preprocess → train_model → evaluate_and_log (MLflow) → quality_gate → promote_model`

- `retries=2`, `retry_delay=20s` for transient errors; `AirflowFailException` for bad data and failed gates (no retry).
- `promote_model` (sets registry alias `champion`) only runs if `quality_gate` succeeds.
- The same stage functions back `dvc.yaml`, so local `dvc repro` and Airflow runs are identical.

## Tool permissions

| Tool group | Reads | Writes | Guard |
|---|---|---|---|
| `dvc_*` | git log/show/tag, dvc status/diff | none | command allow-list, revision/path regex |
| `mlflow_*` | experiments, runs, registry | none | – |
| `airflow_*` (read) | DAGs, runs, task instances, logs | none | JWT auth |
| `airflow_trigger_dag` | – | new DAG run | approved DAG list + human confirmation |
| `k8s_*` (read) | pods, deployments, services, events, logs | none | approved namespace |
| `k8s_restart_deployment`, `k8s_scale_deployment` | – | patch Deployment | approved namespace + human confirmation |

## Error taxonomy (`ToolStatus`)

`ok`, `error`, `unavailable`, `timeout`, `invalid_request`, `denied`, `needs_confirmation`. Every value is
returned to the LLM as a `tool_result` (`is_error=true` when not ok) and logged with request id and duration.

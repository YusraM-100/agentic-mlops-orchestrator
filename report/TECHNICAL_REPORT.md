# Technical Report – Assignment 03 (fill in with your own evidence)

## 1. System architecture
Summarise architecture.md; paste the diagram. Explain why each tool was chosen.

## 2. Tool definitions and permissions
Output of `python -m agent.main --tools` + the permission table from architecture.md.

## 3. Evidence from each system (screenshots in ../screenshots/)
| # | Evidence | File |
|---|---|---|
| 1 | `dvc dag`, `dvc status`, `git tag` showing data-v1.0 / data-v2.0 | screenshots/dvc_*.png |
| 2 | MLflow experiment list, run comparison, registry with `champion` alias | screenshots/mlflow_*.png |
| 3 | Airflow DAG graph, successful run, failed quality-gate run | screenshots/airflow_*.png |
| 4 | `kubectl get pods,svc -n mlops-a3`, `kubectl describe pod`, /docs page | screenshots/k8s_*.png |
| 5 | GitHub Actions green run on a pull request | screenshots/ci_*.png |

## 4. Five tested agent conversations (paste question, tool trace, answer)
Run each with `python -m agent.main --trace -q "..."`.
1. Dataset provenance
2. Model selection
3. Pipeline execution
4. Deployment diagnosis
5. Multi-tool reasoning (mandatory scenario)

## 5. Failure investigations
### Failure 1 – Bad data blocks the pipeline
Injection: `python -m training.make_dataset --version corrupt`, then trigger the DAG.
Agent diagnosis: (paste) – expected: `validate_data` failed, error "missing label column", downstream not run.
Root cause / resolution: restore the dataset with `git checkout data-v2.0 -- data/raw/dataset.csv.dvc; dvc checkout`.

### Failure 2 – Inference pods never become Ready
Injection: set `MODEL_PATH: "/app/models/missing.joblib"` in configmap.yaml, `kubectl apply`, `kubectl rollout restart`.
Agent diagnosis: (paste) – expected: pods Running but not Ready, Warning "Readiness probe failed: HTTP 503", log line "ERROR loading model".
Root cause / resolution: fix the ConfigMap path, re-apply, restart the deployment (agent asks for confirmation).

(Optional third) Quality gate failure: trigger with conf `{"min_f1": 0.999}` → `quality_gate` failed, `promote_model` upstream_failed.

## 6. Known limitations and proposed improvements
- Offline planner is keyword-based; the LLM planner is needed for free-form questions.
- Model is baked into the image; improvement: pull the `champion` alias from MLflow at start-up.
- Local single-node cluster; improvement: RBAC service account for the agent, remote DVC storage (S3/GDrive).
- No drift monitoring; freshness policy is age-based only.

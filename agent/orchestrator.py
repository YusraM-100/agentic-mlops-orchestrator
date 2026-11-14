"""Integration (Section 5): one agent that selects tools from all four modules.

Two planners share the same ToolRegistry (validation, safety, timeouts, logging):

* AnthropicPlanner - Claude chooses tools through the Messages API tool-use loop.
* OfflinePlanner   - deterministic keyword planner, used when no API key is set,
                     in CI, and as a reproducible baseline for the demo.

In both modes the final answer is built ONLY from tool results; if no tool
returned data the answer says so instead of inventing operational facts.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

from agent.config import settings
from agent.logging_utils import get_logger, log_event, new_request_id
from agent.schemas import AgentResponse, ToolCallRecord, ToolResult, ToolStatus
from agent.tools import build_registry
from agent.tools.base import ConfirmFn, ToolRegistry

logger = get_logger("mlops_agent.orchestrator")

SYSTEM_PROMPT = """You are an MLOps operations agent for one ML project. You can call tools that query
DVC (dataset versions), MLflow (experiments, runs, model registry), Apache Airflow (training DAG) and
Kubernetes (the FastAPI inference service in namespace '{namespace}').

Rules:
1. Every operational claim (versions, metrics, statuses, commits, pod health) must come from a tool
   result in this conversation. Never use prior knowledge or guess values.
2. If a tool fails (status unavailable/timeout/error/denied), say exactly which system could not be
   checked and treat that part as UNKNOWN. Do not fill the gap with plausible numbers.
3. To link a model to its data: read the MLflow run tags data.dvc_md5 / git.commit, then call
   dvc_resolve_dataset_version.
4. The best metric alone does not make a model production-ready: mention quality gate status,
   data provenance and deployment health.
5. Never trigger training or change the cluster unless the user explicitly asked for that action.
   Recommend retraining only with evidence: a failed quality gate, data changed since the model's
   training data, or the model being older than the freshness policy of {freshness} days.
6. Finish with three sections: "Facts (with source tool)", "Uncertainties", "Recommended actions".
"""

NUMBER = re.compile(r"\d")


def _fallback_kwargs() -> dict[str, Any]:
    if os.getenv("ANTHROPIC_FALLBACKS", "default") == "off":
        return {}
    return {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class MLOpsAgent:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_client: Any | None = None,
        confirm: ConfirmFn | None = None,
        mode: str | None = None,
    ) -> None:
        self.registry = registry or build_registry()
        self.confirm = confirm
        mode = mode or settings.llm_provider
        if mode == "anthropic" and llm_client is None and not (
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        ):
            mode = "offline"
        self.mode = mode
        self.llm = llm_client
        if self.mode == "anthropic" and self.llm is None:
            import anthropic

            self.llm = anthropic.Anthropic()

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> AgentResponse:
        request_id = new_request_id()
        log_event(logger, "request_start", request_id=request_id, mode=self.mode, question=question[:200])
        if self.mode == "anthropic":
            answer, calls = self._run_anthropic(question, request_id)
        else:
            answer, calls = OfflinePlanner(self).run(question, request_id)
        answer = self._ground_check(answer, calls)
        log_event(logger, "request_end", request_id=request_id, tool_calls=len(calls),
                  ok_calls=sum(c.result.ok for c in calls))
        return AgentResponse(request_id=request_id, answer=answer, tool_calls=calls, mode=self.mode)

    def call(self, name: str, args: dict[str, Any], request_id: str, calls: list[ToolCallRecord]) -> ToolResult:
        result = self.registry.execute(name, args, request_id, confirm=self.confirm)
        calls.append(ToolCallRecord(tool=name, arguments=args, result=result))
        return result

    # ------------------------------------------------------------------ safety net
    @staticmethod
    def _ground_check(answer: str, calls: list[ToolCallRecord]) -> str:
        if calls and not any(c.result.ok for c in calls):
            failed = ", ".join(sorted({f"{c.tool} ({c.result.status.value})" for c in calls}))
            return (
                "No operational data could be retrieved, so no results are reported. "
                f"Failed checks: {failed}.\n\n" + answer
            )
        return answer

    # ------------------------------------------------------------------ Claude tool-use loop
    def _run_anthropic(self, question: str, request_id: str) -> tuple[str, list[ToolCallRecord]]:
        calls: list[ToolCallRecord] = []
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        system = SYSTEM_PROMPT.format(namespace=settings.k8s_namespace, freshness=_freshness_days())
        tools = self.registry.anthropic_tools()
        for _ in range(settings.max_agent_steps):
            response = self.llm.beta.messages.create(
                model=settings.anthropic_model,
                max_tokens=16000,
                system=system,
                tools=tools,
                messages=messages,
                output_config={"effort": "medium"},
                **_fallback_kwargs(),
            )
            if response.stop_reason == "refusal":
                return "The model declined this request.", calls
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                return text or "(no answer)", calls
            results = []
            for block in tool_uses:
                result = self.call(block.name, dict(block.input or {}), request_id, calls)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result.for_llm(),
                                "is_error": not result.ok})
            messages.append({"role": "user", "content": results})  # all results in ONE user message
        return "Stopped: reached the maximum number of agent steps.", calls


def _freshness_days() -> int:
    return int(os.getenv("RETRAIN_MAX_MODEL_AGE_DAYS", "30"))


class OfflinePlanner:
    """Deterministic tool selection + evidence-only report (no LLM needed)."""

    def __init__(self, agent: MLOpsAgent) -> None:
        self.agent = agent

    def run(self, question: str, request_id: str) -> tuple[str, list[ToolCallRecord]]:
        q = question.lower()
        calls: list[ToolCallRecord] = []
        call = lambda n, a=None: self.agent.call(n, a or {}, request_id, calls)  # noqa: E731
        dag = settings.approved_dags[0] if settings.approved_dags else "ml_training_pipeline"

        wants_all = "retrain" in q or ("best" in q and ("deploy" in q or "kubernetes" in q))
        if wants_all:
            return self._full_scenario(call, dag), calls

        sections: list[str] = []
        if re.search(r"\btrigger\b|\brun the (training )?pipeline", q):
            r = call("airflow_trigger_dag", {"dag_id": dag})
            sections.append(_fmt("Trigger training DAG", r))
        if re.search(r"dataset|dvc|data version|provenance", q):
            m = re.search(r"run[_ ]?(id)?[:= ]+([0-9a-f]{32})", q)
            if m:
                run = call("mlflow_get_run_details", {"run_id": m.group(2)})
                if run.ok:
                    prov = run.data.get("provenance", {})
                    sections.append(_fmt("MLflow run provenance", run))
                    sections.append(_fmt("Dataset version", call("dvc_resolve_dataset_version", {
                        "md5": prov.get("data.dvc_md5"), "git_commit": prov.get("git.commit")})))
                else:
                    sections.append(_fmt("MLflow run", run))
            sections.append(_fmt("Current dataset version", call("dvc_get_current_dataset_version")))
            if "change" in q or "status" in q:
                sections.append(_fmt("DVC status", call("dvc_check_data_status")))
            if "history" in q or "compare" in q:
                sections.append(_fmt("Dataset history", call("dvc_dataset_history")))
        if re.search(r"best|compare|runs?\b|experiment|f1|accuracy|metric", q) and "dag" not in q:
            metric = next((m for m in ("f1", "accuracy", "roc_auc", "precision", "recall") if m in q), "f1")
            if "compare" in q:
                sections.append(_fmt("Run comparison", call("mlflow_compare_runs", {"metric": metric, "last_n": 5})))
            sections.append(_fmt("Best run", call("mlflow_find_best_run", {"metric": metric})))
        if re.search(r"dag|airflow|pipeline|workflow|fail", q) and "trigger" not in q:
            sections.append(_fmt("DAG runs", call("airflow_get_dag_runs", {"dag_id": dag})))
            sections.append(_fmt("Failed tasks", call("airflow_get_failed_tasks", {"dag_id": dag})))
        if re.search(r"pod|kubernetes|k8s|deploy|api|unavailable|service|health", q):
            sections.append(_fmt("Workloads", call("k8s_list_workloads")))
            sections.append(_fmt("Unhealthy pods & events", call("k8s_diagnose_unhealthy")))
            if "unavailable" in q or "why" in q:
                sections.append(_fmt("Services", call("k8s_list_services")))
        if not sections:
            return ("I could not map this request to a supported operation. Ask about datasets (DVC), "
                    "runs/models (MLflow), the training DAG (Airflow) or the deployment (Kubernetes)."), calls
        return "\n\n".join(sections), calls

    def _full_scenario(self, call, dag: str) -> str:
        facts, unknowns, actions = [], [], []
        best = call("mlflow_find_best_run", {"metric": "f1"})
        run_prov: dict[str, Any] = {}
        if best.ok and best.data.get("best"):
            b = best.data["best"]
            run_prov = b["provenance"]
            facts.append(f"[mlflow] Best recent run {b['run_id']} f1={b['metrics'].get('f1')} "
                         f"started {b['start_time']}; quality_gate={run_prov.get('quality_gate', 'not recorded')}")
            facts.extend(f"[mlflow] caveat: {c}" for c in best.data.get("caveats", []))
        else:
            unknowns.append(f"MLflow: {best.error or 'no runs with f1 found'}")

        if run_prov.get("data.dvc_md5") or run_prov.get("git.commit"):
            ds = call("dvc_resolve_dataset_version", {"md5": run_prov.get("data.dvc_md5"),
                                                      "git_commit": run_prov.get("git.commit")})
            if ds.ok:
                facts.append(f"[dvc] Training data md5={ds.data.get('md5')} commit={ds.data.get('commit')} "
                             f"tags={ds.data.get('tags')}")
            else:
                unknowns.append(f"DVC provenance: {ds.error}")
        else:
            unknowns.append("Dataset version of the best run (no provenance tags or MLflow unavailable)")

        cur = call("dvc_get_current_dataset_version")
        data_changed = None
        if cur.ok:
            facts.append(f"[dvc] Current dataset md5={cur.data['md5']} "
                         f"(last changed {cur.data['dataset_last_changed_at']})")
            if run_prov.get("data.dvc_md5"):
                data_changed = cur.data["md5"] != run_prov["data.dvc_md5"]
        else:
            unknowns.append(f"Current dataset version: {cur.error}")

        k8s = call("k8s_diagnose_unhealthy")
        ver = call("k8s_get_deployed_version")
        if k8s.ok:
            facts.append(f"[kubernetes] {k8s.data['total_pods']} pods, {len(k8s.data['unhealthy_pods'])} unhealthy, "
                         f"{len(k8s.data['warning_events'])} warning events")
            if k8s.data["unhealthy_pods"]:
                actions.append("Investigate unhealthy pods (k8s_get_pod_logs) before any redeploy.")
        else:
            unknowns.append(f"Kubernetes health: {k8s.error}")
        if ver.ok:
            facts.extend(f"[kubernetes] {d['deployment']} image={d['image']} model_version={d['model_version_config']}"
                         for d in ver.data)

        runs = call("airflow_get_dag_runs", {"dag_id": dag, "limit": 5})
        gate_failed = False
        if runs.ok:
            states = [r["state"] for r in runs.data]
            facts.append(f"[airflow] Last {len(states)} runs of {dag}: {states}")
            if states and states[0] == "failed":
                failed = call("airflow_get_failed_tasks", {"dag_id": dag})
                if failed.ok:
                    ids = [t["task_id"] for t in failed.data["failed_tasks"]]
                    facts.append(f"[airflow] Latest run failed tasks: {ids}")
                    gate_failed = "quality_gate" in ids
        else:
            unknowns.append(f"Airflow: {runs.error}")

        # retraining decision: only explicit evidence
        reasons = []
        if gate_failed:
            reasons.append("the latest pipeline run failed its quality gate")
        if data_changed:
            reasons.append("the current DVC dataset differs from the data the best model was trained on")
        if best.ok and best.data.get("best") and best.data["best"]["start_time"]:
            age = (datetime.now(UTC) - datetime.fromisoformat(best.data["best"]["start_time"])).days
            if age > _freshness_days():
                reasons.append(f"the best model is {age} days old (policy: {_freshness_days()} days)")
        if reasons:
            actions.append("Retraining is recommended because " + "; ".join(reasons)
                           + ". It was NOT started - trigger the approved DAG explicitly if you agree.")
        elif data_changed is None and unknowns:
            actions.append("Cannot decide on retraining until the unknown items above can be checked.")
        else:
            note = " (based only on the systems that could be checked - see Uncertainties)" if unknowns else ""
            actions.append("No evidence that retraining is necessary right now" + note + ".")

        def block(title, items):
            return f"{title}:\n" + ("\n".join(f"- {i}" for i in items) if items else "- none")

        return "\n\n".join([block("Facts (with source tool)", facts), block("Uncertainties", unknowns),
                            block("Recommended actions", actions)])


def _fmt(title: str, r: ToolResult) -> str:
    if r.status != ToolStatus.OK:
        return f"{title}: NOT AVAILABLE ({r.status.value}) - {r.error}"
    return f"{title} [{r.tool}]:\n{json.dumps(r.data, indent=2, default=str)[:3000]}"

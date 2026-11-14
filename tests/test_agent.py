"""End-to-end agent behaviour: multi-tool reasoning and grounding."""

from types import SimpleNamespace

from agent.orchestrator import MLOpsAgent
from agent.schemas import ToolStatus


def test_agent_does_not_invent_results_when_tool_unavailable(mlflow_down):
    resp = MLOpsAgent(mode="offline").ask("Compare the last five training runs and recommend the best by F1")
    assert resp.tool_calls and all(c.result.status == ToolStatus.UNAVAILABLE for c in resp.tool_calls)
    assert resp.answer.startswith("No operational data could be retrieved")
    assert "0.9" not in resp.answer  # no fabricated metric values


def test_mandatory_integration_scenario_uses_all_four_systems(fake_mlflow, fake_k8s, fake_airflow, monkeypatch):
    from agent.tools import dvc_tools

    monkeypatch.setattr(dvc_tools, "run_command", lambda args, timeout=20: "")
    q = ("Find the best recent model, identify the dataset version used to train it, check the deployment "
         "status, and explain whether retraining is necessary.")
    resp = MLOpsAgent(mode="offline").ask(q)
    modules = {c.tool.split("_")[0] for c in resp.tool_calls}
    assert {"mlflow", "dvc", "k8s", "airflow"} <= modules
    assert "airflow_trigger_dag" not in [c.tool for c in resp.tool_calls]  # never retrains automatically
    for section in ("Facts (with source tool)", "Uncertainties", "Recommended actions"):
        assert section in resp.answer


class FakeLLM:
    """Scripted Messages API: first asks for a tool, then answers using the tool_result it received."""

    def __init__(self):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot
        if len(self.requests) == 1:
            block = SimpleNamespace(type="tool_use", id="tu_1", name="mlflow_find_best_run", input={"metric": "f1"})
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="MLflow could not be reached.")])


def test_llm_loop_returns_errors_to_model_and_grounds_answer(mlflow_down):
    llm = FakeLLM()
    resp = MLOpsAgent(mode="anthropic", llm_client=llm).ask("best model?")
    tool_result = llm.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["is_error"] is True
    assert '"unavailable"' in tool_result["content"]
    assert resp.answer.startswith("No operational data could be retrieved")
    assert all(t["name"] for t in llm.requests[0]["tools"]) and len(llm.requests[0]["tools"]) >= 12

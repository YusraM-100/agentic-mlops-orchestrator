"""Task 4 - Kubernetes operations tools.

Read tools inspect the approved namespace only. State-changing tools
(restart / scale) additionally require explicit human confirmation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent.config import settings
from agent.schemas import NamespaceInput, PodInput, PodLogInput, RestartInput, ScaleInput
from agent.tools.base import PolicyDenied, ServiceError, ServiceUnavailable, Tool

_apis: tuple[Any, Any] | None = None


def get_apis():  # pragma: no cover - replaced by fakes in tests
    """Return (CoreV1Api, AppsV1Api) using the local kubeconfig (Docker Desktop / minikube)."""
    global _apis
    if _apis is None:
        from kubernetes import client, config

        try:
            config.load_kube_config(context=settings.k8s_context)
        except Exception:  # noqa: BLE001
            try:
                config.load_incluster_config()
            except Exception as exc:  # noqa: BLE001
                raise ServiceUnavailable("No Kubernetes configuration found (kubeconfig / in-cluster)") from exc
        _apis = (client.CoreV1Api(), client.AppsV1Api())
    return _apis


def _ns(inp: NamespaceInput) -> str:
    return inp.namespace or settings.k8s_namespace


def namespace_policy(inp: NamespaceInput) -> None:
    if _ns(inp) != settings.k8s_namespace:
        raise PolicyDenied(
            f"Namespace '{_ns(inp)}' is not approved; this agent may only operate in '{settings.k8s_namespace}'"
        )


def _k8s(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ServiceUnavailable, PolicyDenied):
        raise
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None)
        if status:
            raise ServiceError(f"Kubernetes API error {status}: {getattr(exc, 'reason', exc)}") from exc
        raise ServiceUnavailable(f"Kubernetes API unreachable: {type(exc).__name__}") from exc


def _age(ts) -> str | None:
    if not ts:
        return None
    return f"{int((datetime.now(UTC) - ts).total_seconds() // 60)}m"


def _pod_summary(p) -> dict[str, Any]:
    statuses = p.status.container_statuses or []
    waiting = [
        {"container": c.name, "reason": c.state.waiting.reason, "message": (c.state.waiting.message or "")[:200]}
        for c in statuses
        if c.state and c.state.waiting
    ]
    last_term = [
        {"container": c.name, "reason": c.last_state.terminated.reason, "exit_code": c.last_state.terminated.exit_code}
        for c in statuses
        if c.last_state and c.last_state.terminated
    ]
    return {
        "name": p.metadata.name,
        "phase": p.status.phase,
        "ready": bool(statuses) and all(c.ready for c in statuses),
        "restarts": sum(c.restart_count for c in statuses),
        "image": statuses[0].image if statuses else None,
        "waiting": waiting,
        "last_terminated": last_term,
        "age": _age(p.metadata.creation_timestamp),
    }


# ----------------------------------------------------------------------------- tools


def list_workloads(inp: NamespaceInput) -> dict[str, Any]:
    core, apps = get_apis()
    ns = _ns(inp)
    deps = _k8s(apps.list_namespaced_deployment, ns).items
    pods = _k8s(core.list_namespaced_pod, ns, label_selector=settings.k8s_app_label).items
    return {
        "namespace": ns,
        "deployments": [
            {"name": d.metadata.name, "desired": d.spec.replicas, "ready": d.status.ready_replicas or 0,
             "available": d.status.available_replicas or 0, "image": d.spec.template.spec.containers[0].image}
            for d in deps
        ],
        "pods": [_pod_summary(p) for p in pods],
    }


def get_pod_status(inp: PodInput) -> dict[str, Any]:
    core, _ = get_apis()
    pod = _k8s(core.read_namespaced_pod, inp.pod_name, _ns(inp))
    summary = _pod_summary(pod)
    summary["conditions"] = [
        {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
        for c in (pod.status.conditions or [])
    ]
    return summary


def get_pod_logs(inp: PodLogInput) -> dict[str, Any]:
    core, _ = get_apis()
    text = _k8s(core.read_namespaced_pod_log, inp.pod_name, _ns(inp), tail_lines=inp.tail_lines,
                previous=inp.previous)
    lines = (text or "").splitlines()
    return {"pod": inp.pod_name, "previous": inp.previous,
            "error_lines": [ln for ln in lines if "ERROR" in ln or "Traceback" in ln][-15:], "tail": lines}


def list_services(inp: NamespaceInput) -> list[dict[str, Any]]:
    core, _ = get_apis()
    ns = _ns(inp)
    out = []
    for s in _k8s(core.list_namespaced_service, ns).items:
        ep = _k8s(core.read_namespaced_endpoints, s.metadata.name, ns)
        ready = sum(len(sub.addresses or []) for sub in (ep.subsets or []))
        out.append({"name": s.metadata.name, "type": s.spec.type, "cluster_ip": s.spec.cluster_ip,
                    "ports": [f"{p.port}->{p.target_port}" for p in s.spec.ports or []],
                    "selector": s.spec.selector, "ready_endpoints": ready})
    return out


def diagnose_unhealthy(inp: NamespaceInput) -> dict[str, Any]:
    core, _ = get_apis()
    ns = _ns(inp)
    pods = [_pod_summary(p) for p in _k8s(core.list_namespaced_pod, ns, label_selector=settings.k8s_app_label).items]
    unhealthy = [p for p in pods if not p["ready"] or p["restarts"] > 0 or p["waiting"]]
    events = _k8s(core.list_namespaced_event, ns).items
    warnings = sorted(
        (e for e in events if e.type == "Warning"),
        key=lambda e: e.last_timestamp or e.event_time or e.metadata.creation_timestamp,
        reverse=True,
    )[:20]
    return {
        "namespace": ns,
        "total_pods": len(pods),
        "unhealthy_pods": unhealthy,
        "warning_events": [
            {"object": f"{e.involved_object.kind}/{e.involved_object.name}", "reason": e.reason,
             "message": (e.message or "")[:300], "count": e.count}
            for e in warnings
        ],
    }


def get_deployed_version(inp: NamespaceInput) -> list[dict[str, Any]]:
    core, apps = get_apis()
    ns = _ns(inp)
    out = []
    for d in _k8s(apps.list_namespaced_deployment, ns).items:
        labels = d.metadata.labels or {}
        cm_version = None
        try:
            cm = _k8s(core.read_namespaced_config_map, "inference-config", ns)
            cm_version = (cm.data or {}).get("MODEL_VERSION")
        except ServiceError:
            pass
        out.append({"deployment": d.metadata.name, "image": d.spec.template.spec.containers[0].image,
                    "app_version_label": labels.get("app.kubernetes.io/version"),
                    "model_version_config": cm_version, "revision": (d.metadata.annotations or {}).get(
                        "deployment.kubernetes.io/revision")})
    return out


def restart_deployment(inp: RestartInput) -> dict[str, Any]:
    _, apps = get_apis()
    now = datetime.now(UTC).isoformat()
    body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}}
    _k8s(apps.patch_namespaced_deployment, inp.deployment, _ns(inp), body)
    return {"deployment": inp.deployment, "namespace": _ns(inp), "restarted_at": now}


def scale_deployment(inp: ScaleInput) -> dict[str, Any]:
    _, apps = get_apis()
    _k8s(apps.patch_namespaced_deployment_scale, inp.deployment, _ns(inp), {"spec": {"replicas": inp.replicas}})
    return {"deployment": inp.deployment, "namespace": _ns(inp), "replicas": inp.replicas}


TOOLS = [
    Tool("k8s_list_workloads", "List Deployments and application Pods (readiness, restarts, image).",
         NamespaceInput, list_workloads, "kubernetes", policy=namespace_policy),
    Tool("k8s_get_pod_status", "Detailed status and conditions of a Pod.", PodInput, get_pod_status,
         "kubernetes", policy=namespace_policy),
    Tool("k8s_get_pod_logs", "Recent logs of a Pod (optionally the previous crashed container).",
         PodLogInput, get_pod_logs, "kubernetes", policy=namespace_policy),
    Tool("k8s_list_services", "Services, ports and how many ready endpoints back them.",
         NamespaceInput, list_services, "kubernetes", policy=namespace_policy),
    Tool("k8s_diagnose_unhealthy", "Unhealthy Pods, restart counts, waiting reasons and Warning events.",
         NamespaceInput, diagnose_unhealthy, "kubernetes", policy=namespace_policy),
    Tool("k8s_get_deployed_version", "Currently deployed image tag, version label and model version.",
         NamespaceInput, get_deployed_version, "kubernetes", policy=namespace_policy),
    Tool("k8s_restart_deployment", "Rolling restart of a Deployment.", RestartInput, restart_deployment,
         "kubernetes", mutating=True, policy=namespace_policy),
    Tool("k8s_scale_deployment", "Scale a Deployment (0-5 replicas).", ScaleInput, scale_deployment,
         "kubernetes", mutating=True, policy=namespace_policy),
]

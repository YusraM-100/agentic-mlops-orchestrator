"""Task 1 - DVC data management tools.

Only read-only, allow-listed git/dvc sub-commands are executed (no shell,
arguments passed as a list). Dataset provenance is read from the .dvc files
in git history, so every answer is backed by recorded evidence.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agent.config import settings
from agent.schemas import (
    CompareDatasetInput,
    DatasetHistoryInput,
    DatasetPathInput,
    Empty,
    ReproduceInput,
    ResolveDatasetInput,
)
from agent.tools.base import PolicyDenied, ServiceError, ServiceUnavailable, Tool

ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("git", "log"),
    ("git", "show"),
    ("git", "rev-parse"),
    ("git", "describe"),
    ("git", "tag"),
    ("dvc", "status"),
    ("dvc", "diff"),
    ("dvc", "remote", "list"),
)
SAFE_REV = re.compile(r"^[A-Za-z0-9_./~^@{}-]{1,100}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]{1,200}\.dvc$")


def _check_rev(rev: str) -> str:
    if rev.startswith("-") or not SAFE_REV.match(rev):
        raise PolicyDenied(f"Rejected unsafe git revision: {rev!r}")
    return rev


def _check_dvc_file(path: str) -> str:
    if path.startswith(("/", "-")) or ".." in path or not SAFE_PATH.match(path):
        raise PolicyDenied(f"Rejected unsafe .dvc path: {path!r}")
    return path


def run_command(args: list[str], timeout: float = 20) -> str:
    """Run an allow-listed git/dvc command inside the repository."""
    if not any(tuple(args[: len(p)]) == p for p in ALLOWED_COMMANDS):
        raise PolicyDenied(f"Command not allowed: {' '.join(args[:3])}")
    try:
        proc = subprocess.run(  # noqa: S603 - args are a validated list, no shell
            args,
            cwd=settings.repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ServiceUnavailable(f"'{args[0]}' is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ServiceUnavailable(f"'{' '.join(args[:2])}' timed out") from exc
    if proc.returncode != 0:
        raise ServiceError(f"{' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def _parse_dvc(text: str) -> dict[str, Any]:
    doc = yaml.safe_load(text) or {}
    out = (doc.get("outs") or [{}])[0]
    md5 = out.get("md5")
    # str(): an all-digit md5 would otherwise be parsed by YAML as an integer
    return {"path": out.get("path"), "md5": str(md5) if md5 is not None else None, "size": out.get("size"),
            "nfiles": out.get("nfiles")}


def _dvc_at(rev: str, dvc_file: str) -> dict[str, Any]:
    return _parse_dvc(run_command(["git", "show", f"{_check_rev(rev)}:{dvc_file}"]))


def _tags_at(commit: str) -> list[str]:
    return [t for t in run_command(["git", "tag", "--points-at", commit]).split() if t]


def _data_version_param() -> str | None:
    params = settings.repo_root / "params.yaml"
    if params.exists():
        return (yaml.safe_load(params.read_text()) or {}).get("data", {}).get("version")
    return None


# ----------------------------------------------------------------------------- tools


def list_tracked_datasets(_: Empty) -> dict[str, Any]:
    root = settings.repo_root
    files = [
        p
        for p in root.rglob("*.dvc")
        if p.is_file() and not any(part in {".dvc", ".venv", "venv", "node_modules"} for part in p.parts)
    ]
    datasets = []
    for f in files:
        info = _parse_dvc(f.read_text())
        info["dvc_file"] = f.relative_to(root).as_posix()
        datasets.append(info)
    try:
        remotes = run_command(["dvc", "remote", "list"]).strip().splitlines()
    except (ServiceError, ServiceUnavailable) as exc:
        remotes = [f"unavailable: {exc}"]
    return {"datasets": datasets, "dvc_remotes": remotes}


def get_current_dataset_version(inp: DatasetPathInput) -> dict[str, Any]:
    dvc_file = _check_dvc_file(inp.dvc_file)
    path = settings.repo_root / dvc_file
    if not path.exists():
        raise ServiceError(f"{dvc_file} not found - is the dataset tracked with 'dvc add'?")
    current = _parse_dvc(path.read_text())
    head = run_command(["git", "rev-parse", "HEAD"]).strip()
    last = run_command(["git", "log", "-1", "--format=%H|%ad|%s", "--date=iso", "--", dvc_file]).strip()
    last_commit, last_date, last_msg = (last.split("|", 2) + ["", "", ""])[:3] if last else ("", "", "")
    return {
        "dvc_file": dvc_file,
        "md5": current["md5"],
        "size_bytes": current["size"],
        "data_version_param": _data_version_param(),
        "repo_head_commit": head,
        "dataset_last_changed_in_commit": last_commit,
        "dataset_last_changed_at": last_date,
        "dataset_commit_message": last_msg,
        "tags_on_dataset_commit": _tags_at(last_commit) if last_commit else [],
    }


def dataset_history(inp: DatasetHistoryInput) -> list[dict[str, Any]]:
    dvc_file = _check_dvc_file(inp.dvc_file)
    log = run_command(
        ["git", "log", f"-n{inp.max_entries}", "--format=%H|%h|%ad|%s", "--date=iso", "--", dvc_file]
    )
    history = []
    for line in log.strip().splitlines():
        full, short, date, msg = line.split("|", 3)
        entry = {"commit": full, "short": short, "date": date, "message": msg, "tags": _tags_at(full)}
        entry.update({k: v for k, v in _dvc_at(full, dvc_file).items() if k in {"md5", "size"}})
        history.append(entry)
    return history


def compare_dataset_versions(inp: CompareDatasetInput) -> dict[str, Any]:
    dvc_file = _check_dvc_file(inp.dvc_file)
    a, b = _dvc_at(inp.rev_a, dvc_file), _dvc_at(inp.rev_b, dvc_file)
    try:
        diff = json.loads(run_command(["dvc", "diff", "--json", _check_rev(inp.rev_a), _check_rev(inp.rev_b)]) or "{}")
    except (ServiceError, ServiceUnavailable, json.JSONDecodeError) as exc:
        diff = {"unavailable": str(exc)}
    return {
        "rev_a": {"rev": inp.rev_a, **a},
        "rev_b": {"rev": inp.rev_b, **b},
        "identical": a["md5"] == b["md5"],
        "size_change_bytes": (b["size"] or 0) - (a["size"] or 0),
        "dvc_diff": diff,
    }


def check_data_status(_: Empty) -> dict[str, Any]:
    raw = run_command(["dvc", "status", "--json"]).strip() or "{}"
    status = json.loads(raw)
    return {
        "in_sync": status == {},
        "changes": status,
        "explanation": "Workspace data matches the .dvc/dvc.lock records"
        if status == {}
        else "Tracked data or pipeline outputs differ from what is recorded",
    }


def resolve_dataset_version(inp: ResolveDatasetInput) -> dict[str, Any]:
    dvc_file = _check_dvc_file(inp.dvc_file)
    if not inp.md5 and not inp.git_commit:
        raise PolicyDenied("Provide md5 or git_commit (e.g. from MLflow run tags data.dvc_md5 / git.commit)")
    if inp.git_commit:
        info = _dvc_at(inp.git_commit, dvc_file)
        return {"matched_by": "git_commit", "commit": inp.git_commit, "tags": _tags_at(inp.git_commit), **info}
    for entry in dataset_history(DatasetHistoryInput(dvc_file=dvc_file, max_entries=50)):
        if entry.get("md5") == inp.md5:
            return {"matched_by": "md5", **entry}
    return {"matched_by": None, "md5": inp.md5, "note": "No commit in the last 50 dataset changes has this md5"}


def get_reproduce_instructions(inp: ReproduceInput) -> dict[str, Any]:
    dvc_file = _check_dvc_file(inp.dvc_file)
    commit = run_command(["git", "rev-parse", "--verify", f"{_check_rev(inp.rev)}^{{commit}}"]).strip()
    info = _dvc_at(commit, dvc_file)
    return {
        "rev": inp.rev,
        "commit": commit,
        "md5": info["md5"],
        "steps": [
            f"git checkout {commit} -- {dvc_file}",
            f"dvc checkout {dvc_file}   (or: dvc pull {dvc_file} if the cache is empty)",
            "dvc status   # should report the data is up to date",
            "git checkout HEAD -- " + dvc_file + " && dvc checkout   # return to the current version",
        ],
        "why_it_is_reproducible": "The .dvc file at that commit pins the content hash (md5); "
        "DVC restores exactly those bytes from the cache/remote.",
    }


TOOLS = [
    Tool("dvc_list_tracked_datasets", "List datasets tracked by DVC and the configured DVC remotes.",
         Empty, list_tracked_datasets, "dvc"),
    Tool("dvc_get_current_dataset_version",
         "Current dataset version: DVC md5, git commit that last changed it, tags, data.version param.",
         DatasetPathInput, get_current_dataset_version, "dvc"),
    Tool("dvc_dataset_history", "History of dataset versions (commit, date, tags, md5) from git log.",
         DatasetHistoryInput, dataset_history, "dvc"),
    Tool("dvc_compare_dataset_versions", "Compare two dataset versions (git revisions or tags).",
         CompareDatasetInput, compare_dataset_versions, "dvc"),
    Tool("dvc_check_data_status", "Detect whether tracked data changed vs. what DVC recorded (dvc status).",
         Empty, check_data_status, "dvc"),
    Tool("dvc_resolve_dataset_version",
         "Map an MLflow run's recorded data.dvc_md5 or git.commit tag to a dataset version/commit/tag.",
         ResolveDatasetInput, resolve_dataset_version, "dvc"),
    Tool("dvc_get_reproduce_instructions", "Exact commands to reproduce (checkout) a previous dataset version.",
         ReproduceInput, get_reproduce_instructions, "dvc"),
]

__all__ = ["TOOLS", "run_command", "Path"]

"""Read dataset/code provenance without needing the git or dvc binaries
(works inside the Airflow container where only the project folder is mounted)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def dvc_md5(dvc_file: Path) -> str | None:
    if not dvc_file.exists():
        return None
    doc = yaml.safe_load(dvc_file.read_text()) or {}
    md5 = (doc.get("outs") or [{}])[0].get("md5")
    return str(md5) if md5 is not None else None


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    git_dir = repo / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        return None
    head = head_file.read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1]
    ref_file = git_dir / ref
    if ref_file.exists():
        return ref_file.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    return None

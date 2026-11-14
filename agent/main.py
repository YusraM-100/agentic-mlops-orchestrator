"""Command-line chat interface.

    python -m agent.main                 # interactive chat
    python -m agent.main -q "question"   # single question
    python -m agent.main --offline       # no LLM, deterministic planner
    python -m agent.main --tools         # list tools
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.orchestrator import MLOpsAgent
from agent.tools import build_registry


def ask_human(tool: str, args: dict) -> bool:
    print(f"\n⚠  The agent wants to run a STATE-CHANGING operation:\n   {tool} {json.dumps(args)}")
    return input("   Type 'yes' to allow: ").strip().lower() == "yes"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MLOps AI Agent")
    p.add_argument("-q", "--question")
    p.add_argument("--offline", action="store_true", help="Use the deterministic planner (no LLM)")
    p.add_argument("--tools", action="store_true", help="List available tools and exit")
    p.add_argument("--trace", action="store_true", help="Print every tool call and its status")
    args = p.parse_args(argv)

    if args.tools:
        for t in build_registry().anthropic_tools():
            print(f"- {t['name']}: {t['description']}")
        return 0

    agent = MLOpsAgent(confirm=ask_human, mode="offline" if args.offline else None)
    print(f"MLOps agent ready (mode: {agent.mode}). Type 'exit' to quit.", file=sys.stderr)

    def handle(q: str) -> None:
        resp = agent.ask(q)
        if args.trace:
            for c in resp.tool_calls:
                print(f"  -> {c.tool} {c.arguments} = {c.result.status.value} ({c.result.duration_ms} ms)")
        print(f"\n{resp.answer}\n\n[request_id={resp.request_id}]")

    if args.question:
        handle(args.question)
        return 0
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            handle(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

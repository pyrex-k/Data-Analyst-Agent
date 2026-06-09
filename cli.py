#!/usr/bin/env python3
"""
Command-line entry point for the data-analyst agent.

Usage:
    python cli.py "Which region had the highest total revenue?"
    python cli.py --show-steps "What is the correlation between revenue and profit?"
    python cli.py --interactive
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Make sure src/ is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import AgentResult, TextStep, ThinkingStep, ToolCallStep, run_agent


RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def _print_step(step: TextStep | ThinkingStep | ToolCallStep, show_thinking: bool) -> None:
    if isinstance(step, ThinkingStep):
        if show_thinking:
            excerpt = textwrap.shorten(step.thinking, width=200, placeholder=" …")
            print(f"{DIM}[thinking] {excerpt}{RESET}")
    elif isinstance(step, TextStep):
        print(f"{CYAN}[agent]{RESET} {step.text}")
    elif isinstance(step, ToolCallStep):
        print(f"{YELLOW}[tool]{RESET} {BOLD}{step.tool_name}{RESET}  "
              f"input={list(step.tool_input.keys())}")
        if "error" in step.tool_result:
            print(f"  {RESET}  ↳ ERROR: {step.tool_result['error']}")
        elif "path" in step.tool_result or "chart_saved" in step.tool_result:
            path = step.tool_result.get("path") or step.tool_result.get("chart_saved")
            print(f"  {GREEN}  ↳ saved: {path}{RESET}")


def run_query_cli(query: str, show_steps: bool, show_thinking: bool) -> None:
    print(f"\n{BOLD}Query:{RESET} {query}\n{'─' * 60}")

    def on_step(step: TextStep | ThinkingStep | ToolCallStep) -> None:
        if show_steps:
            _print_step(step, show_thinking)

    result = run_agent(query, on_step=on_step)

    print(f"\n{'─' * 60}")
    print(f"{BOLD}{GREEN}Final answer:{RESET}\n{result.final_answer}")
    print(f"\n{DIM}[iterations: {result.iterations}  stop: {result.stop_reason}]{RESET}\n")


def interactive_mode(show_steps: bool, show_thinking: bool) -> None:
    print(f"{BOLD}Data Analyst Agent — interactive mode{RESET}")
    print("Type your question and press Enter.  Type 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input(f"{CYAN}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        run_query_cli(query, show_steps=show_steps, show_thinking=show_thinking)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Analyst Agent powered by Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python cli.py "Which product category generates the most profit?"
              python cli.py --show-steps "Show monthly revenue trends"
              python cli.py --interactive
        """),
    )
    parser.add_argument("query", nargs="?", help="Question to ask the agent")
    parser.add_argument("--show-steps",   action="store_true", help="Print each agent step")
    parser.add_argument("--show-thinking", action="store_true", help="Also print thinking excerpts")
    parser.add_argument("--interactive",  action="store_true", help="Start interactive REPL")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.show_steps, args.show_thinking)
    elif args.query:
        run_query_cli(args.query, show_steps=args.show_steps, show_thinking=args.show_thinking)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

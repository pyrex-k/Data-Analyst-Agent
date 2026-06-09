"""
Core agentic loop for the data-analyst agent.

Design rationale
----------------
We implement a *manual* ReAct-style loop rather than using the beta tool
runner so that we have full visibility into every intermediate step — useful
for displaying progress in both the CLI and the Streamlit UI, and for
collecting the step trace that feeds the evaluation section.

The loop:
  1. Send messages (+ tool schemas) to Claude.
  2. If stop_reason == "tool_use", execute the requested tools and append
     their results as tool_result blocks.
  3. Repeat until stop_reason == "end_turn" or the iteration cap is reached.

Streaming is used to avoid HTTP timeouts on long analytical tasks; we use
.get_final_message() to obtain the complete response after the stream finishes.
Adaptive thinking is enabled so Claude decides how much reasoning each step
warrants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import anthropic

from .config import MODEL, MAX_ITERATIONS, get_api_key
from .tools import TOOL_SCHEMAS, dispatch

# ─── step types ───────────────────────────────────────────────────────────────

@dataclass
class ThinkingStep:
    thinking: str

@dataclass
class TextStep:
    text: str

@dataclass
class ToolCallStep:
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any

@dataclass
class AgentResult:
    final_answer: str
    steps: list[ThinkingStep | TextStep | ToolCallStep] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "end_turn"

# ─── main agent function ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data analyst with deep knowledge of pandas, \
statistics, and data visualisation.  When given a question about data, you:

1. First load and explore the dataset to understand its structure.
2. Break the analysis into clear steps, using the available tools one at a time.
3. Produce charts and summary tables where they add insight.
4. Conclude with a concise, plain-English interpretation of findings.

Always explain *why* you chose each analytical step so the reasoning is transparent."""


def run_agent(
    user_query: str,
    on_step: Callable[[ThinkingStep | TextStep | ToolCallStep], None] | None = None,
) -> AgentResult:
    """
    Run the agentic loop for a single user query and return an AgentResult.

    `on_step` is an optional callback invoked after each step, which lets the
    CLI and Streamlit UI stream progress without coupling to this module.
    """
    client = anthropic.Anthropic(api_key=get_api_key())

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]
    steps: list[Any] = []
    iterations = 0
    stop_reason = "max_iterations"
    final_text = ""

    while iterations < MAX_ITERATIONS:
        iterations += 1

        # Stream the response to avoid long-poll timeouts.
        with client.messages.stream(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            thinking={"type": "adaptive"},
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        stop_reason = response.stop_reason  # "end_turn" | "tool_use" | "max_tokens"

        # ── collect content blocks ────────────────────────────────────────────
        assistant_content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "thinking":
                step = ThinkingStep(thinking=block.thinking)
                steps.append(step)
                if on_step:
                    on_step(step)
                assistant_content.append({"type": "thinking", "thinking": block.thinking})

            elif block.type == "text":
                final_text = block.text          # updated each iteration
                step = TextStep(text=block.text)
                steps.append(step)
                if on_step:
                    on_step(step)
                assistant_content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })

        # Append the full assistant turn (preserves thinking blocks for compaction).
        messages.append({"role": "assistant", "content": assistant_content})

        if stop_reason != "tool_use":
            break

        # ── execute tool calls and collect results ────────────────────────────
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            raw_result = dispatch(block.name, block.input)
            result_obj = json.loads(raw_result)

            step = ToolCallStep(
                tool_name=block.name,
                tool_input=block.input,
                tool_result=result_obj,
            )
            steps.append(step)
            if on_step:
                on_step(step)

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     raw_result,
            })

        messages.append({"role": "user", "content": tool_results})

    return AgentResult(
        final_answer=final_text,
        steps=steps,
        iterations=iterations,
        stop_reason=stop_reason,
    )

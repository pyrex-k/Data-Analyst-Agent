"""
Core agentic loop — powered by Google Gemini (free tier, google-genai SDK).

Manual ReAct loop: send message → execute function calls → repeat until
the model returns only text (no more tool calls) or iteration cap is hit.
Every step is surfaced via the optional `on_step` callback so the CLI and
Streamlit UI can display live progress.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai
from google.genai import types

from .config import MODEL, MAX_ITERATIONS, get_api_key
from .tools import dispatch

# ─── step types ───────────────────────────────────────────────────────────────

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
    steps: list[TextStep | ToolCallStep] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "end_turn"

# ─── tool declarations ────────────────────────────────────────────────────────

def _str(desc: str = "") -> types.Schema:
    return types.Schema(type="STRING", description=desc)

def _bool(desc: str = "") -> types.Schema:
    return types.Schema(type="BOOLEAN", description=desc)

def _arr(items: types.Schema, desc: str = "") -> types.Schema:
    return types.Schema(type="ARRAY", items=items, description=desc)

def _obj(props: dict[str, types.Schema], required: list[str], desc: str = "") -> types.Schema:
    return types.Schema(type="OBJECT", properties=props, required=required, description=desc)

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="load_dataset",
        description="Load a CSV or JSON file into memory and return a preview. The dataset is stored under `dataset_name` for subsequent tool calls.",
        parameters=_obj(
            {"file_path": _str("Path to the CSV or JSON file."),
             "dataset_name": _str("Short alias for this dataset.")},
            ["file_path", "dataset_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="describe_dataset",
        description="Return summary statistics, dtypes, shape, missing-value counts, and first rows of a loaded dataset.",
        parameters=_obj({"dataset_name": _str()}, ["dataset_name"]),
    ),
    types.FunctionDeclaration(
        name="run_query",
        description="Execute a pandas expression against a loaded dataset using `df` as the DataFrame variable. Returns the result as JSON.",
        parameters=_obj(
            {"dataset_name": _str(),
             "query": _str("Pandas expression using `df` as the DataFrame variable.")},
            ["dataset_name", "query"],
        ),
    ),
    types.FunctionDeclaration(
        name="filter_data",
        description="Filter a dataset by column conditions (==, !=, >, <, >=, <=, contains, isin) and optionally select columns.",
        parameters=_obj(
            {"dataset_name": _str(),
             "conditions": _arr(
                 _obj({"column": _str(), "operator": _str(), "value": _str()}, []),
                 "List of {column, operator, value} filter conditions.",
             ),
             "output_dataset_name": _str("Alias for the resulting filtered dataset."),
             "select_columns": _arr(_str(), "Columns to keep.")},
            ["dataset_name", "conditions"],
        ),
    ),
    types.FunctionDeclaration(
        name="aggregate_data",
        description="Group a dataset by columns and compute aggregations (sum, mean, count, min, max, std, median).",
        parameters=_obj(
            {"dataset_name": _str(),
             "group_by": _arr(_str(), "Columns to group by."),
             "aggregations": types.Schema(
                 type="OBJECT",
                 description='Map of column→list of functions e.g. {"revenue":["sum","mean"]}',
             ),
             "sort_by": _str("Column to sort result by (descending).")},
            ["dataset_name", "group_by", "aggregations"],
        ),
    ),
    types.FunctionDeclaration(
        name="correlation_analysis",
        description="Compute Pearson correlation matrix for numeric columns and optionally save a heatmap PNG.",
        parameters=_obj(
            {"dataset_name": _str(),
             "columns": _arr(_str(), "Subset of columns (default: all numeric)."),
             "plot": _bool("Save a heatmap PNG if true.")},
            ["dataset_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="create_visualization",
        description="Create and save a chart (bar, line, scatter, histogram, box, pie) for a loaded dataset.",
        parameters=_obj(
            {"dataset_name": _str(),
             "chart_type": types.Schema(type="STRING", enum=["bar","line","scatter","histogram","box","pie"]),
             "output_filename": _str("Filename without extension for the saved PNG."),
             "x_column": _str("Column for x-axis."),
             "y_column": _str("Column for y-axis."),
             "title": _str("Chart title."),
             "hue": _str("Column for colour grouping."),
             "aggregate": types.Schema(type="STRING", enum=["sum","mean","count","none"],
                                       description="Pre-aggregate y by x before plotting.")},
            ["dataset_name", "chart_type", "output_filename"],
        ),
    ),
    types.FunctionDeclaration(
        name="save_results",
        description="Save a text report or analysis summary to the outputs folder.",
        parameters=_obj(
            {"content": _str("Text content to save."),
             "filename": _str("Filename without extension.")},
            ["content", "filename"],
        ),
    ),
]

GEMINI_TOOLS = [types.Tool(function_declarations=TOOL_DECLARATIONS)]

# ─── system prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert data analyst with deep knowledge of pandas, \
statistics, and data visualisation. When given a question about data, you:

1. First load and explore the dataset to understand its structure.
2. Break the analysis into clear steps, using the available tools one at a time.
3. Produce charts and summary tables where they add insight.
4. Conclude with a concise, plain-English interpretation of findings.

Always explain *why* you chose each analytical step so the reasoning is transparent."""

# ─── agentic loop ─────────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    on_step: Callable[[TextStep | ToolCallStep], None] | None = None,
) -> AgentResult:
    client = genai.Client(api_key=get_api_key())

    # conversation history
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]

    steps: list[Any] = []
    iterations = 0
    stop_reason = "max_iterations"
    final_text = ""

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=GEMINI_TOOLS,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
            ),
        )

        candidate = response.candidates[0]
        assistant_parts: list[types.Part] = []
        fn_calls = []
        text_parts = []

        for part in candidate.content.parts:
            assistant_parts.append(part)
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                fn_calls.append(part.function_call)

        # Append assistant turn to history
        contents.append(types.Content(role="model", parts=assistant_parts))

        if text_parts:
            final_text = "\n".join(text_parts)
            step = TextStep(text=final_text)
            steps.append(step)
            if on_step:
                on_step(step)

        if not fn_calls:
            stop_reason = "end_turn"
            break

        # Execute tools and collect responses
        fn_response_parts: list[types.Part] = []
        for fc in fn_calls:
            tool_input = dict(fc.args) if fc.args else {}
            raw = dispatch(fc.name, tool_input)
            result_obj = json.loads(raw)

            step = ToolCallStep(tool_name=fc.name, tool_input=tool_input, tool_result=result_obj)
            steps.append(step)
            if on_step:
                on_step(step)

            fn_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"output": raw},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=fn_response_parts))

    return AgentResult(
        final_answer=final_text,
        steps=steps,
        iterations=iterations,
        stop_reason=stop_reason,
    )

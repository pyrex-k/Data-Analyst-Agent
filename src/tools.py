"""
Tool implementations for the data-analyst agent.

Each function is the Python side of a Claude tool.  The JSON schemas below
are what we register with the Anthropic API; the matching callables are what
we invoke when Claude calls a tool.
"""
from __future__ import annotations

import io
import json
import os
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # headless — no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ─── shared in-memory state ───────────────────────────────────────────────────
_datasets: dict[str, pd.DataFrame] = {}
OUTPUT_DIR = Path("outputs")

def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── tool schemas (JSON definitions sent to the API) ──────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "load_dataset",
        "description": (
            "Load a CSV or JSON file into memory and return a preview.  "
            "The dataset is stored under `dataset_name` for subsequent tool calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV or JSON file to load.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Short alias used to reference this dataset in later calls.",
                },
            },
            "required": ["file_path", "dataset_name"],
        },
    },
    {
        "name": "describe_dataset",
        "description": (
            "Return summary statistics, dtypes, shape, missing-value counts, and "
            "the first few rows of a loaded dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "run_query",
        "description": (
            "Execute a pandas expression against a loaded dataset and return the result "
            "as a JSON string.  Use `df` as the variable name for the DataFrame.\n"
            "Examples:\n"
            "  `df.groupby('region')['revenue'].sum().sort_values(ascending=False)`\n"
            "  `df[df['category'] == 'Electronics'][['product','profit']].head(10)`"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "query": {
                    "type": "string",
                    "description": "Pandas expression using `df` as the DataFrame variable.",
                },
            },
            "required": ["dataset_name", "query"],
        },
    },
    {
        "name": "filter_data",
        "description": (
            "Filter a dataset by applying one or more column conditions and optionally "
            "selecting a subset of columns.  Returns a new in-memory dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "conditions": {
                    "type": "array",
                    "description": (
                        "List of filter conditions, each an object with keys: "
                        "`column`, `operator` (one of ==, !=, >, <, >=, <=, contains, isin), "
                        "and `value`."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "column":   {"type": "string"},
                            "operator": {"type": "string"},
                            "value":    {},
                        },
                        "required": ["column", "operator", "value"],
                    },
                },
                "output_dataset_name": {
                    "type": "string",
                    "description": "Alias for the resulting filtered dataset (optional).",
                },
                "select_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If provided, keep only these columns.",
                },
            },
            "required": ["dataset_name", "conditions"],
        },
    },
    {
        "name": "aggregate_data",
        "description": (
            "Group a dataset by one or more columns and compute aggregation functions "
            "(sum, mean, count, min, max, std, median) on selected numeric columns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by.",
                },
                "aggregations": {
                    "type": "object",
                    "description": (
                        "Mapping of column → list of aggregation functions.  "
                        'Example: {"revenue": ["sum", "mean"], "units_sold": ["sum"]}'
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "description": "Optional column name to sort the result by (descending).",
                },
            },
            "required": ["dataset_name", "group_by", "aggregations"],
        },
    },
    {
        "name": "correlation_analysis",
        "description": (
            "Compute the Pearson correlation matrix for numeric columns in a dataset "
            "and optionally plot a heatmap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of columns to include (defaults to all numeric).",
                },
                "plot": {
                    "type": "boolean",
                    "description": "If true, save a heatmap image to the outputs folder.",
                    "default": True,
                },
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "create_visualization",
        "description": (
            "Create and save a chart for a loaded dataset.  "
            "Supported chart types: bar, line, scatter, histogram, box, pie."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "histogram", "box", "pie"],
                },
                "x_column": {"type": "string", "description": "Column for the x-axis."},
                "y_column": {"type": "string", "description": "Column for the y-axis (ignored for histogram)."},
                "title":    {"type": "string", "description": "Chart title."},
                "hue":      {"type": "string", "description": "Optional column for colour grouping."},
                "output_filename": {
                    "type": "string",
                    "description": "Filename (without extension) for the saved PNG.",
                },
                "aggregate": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "none"],
                    "description": "Pre-aggregate y by x before plotting (bar/line).  Default 'none'.",
                    "default": "none",
                },
            },
            "required": ["dataset_name", "chart_type", "output_filename"],
        },
    },
    {
        "name": "save_results",
        "description": "Save a text report or analysis summary to a file in the outputs folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content":  {"type": "string", "description": "Text content to save."},
                "filename": {"type": "string", "description": "Filename (without extension)."},
            },
            "required": ["content", "filename"],
        },
    },
]


# ─── implementations ───────────────────────────────────────────────────────────

def load_dataset(file_path: str, dataset_name: str) -> dict[str, Any]:
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        if path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)
        _datasets[dataset_name] = df
        return {
            "status": "loaded",
            "dataset_name": dataset_name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def describe_dataset(dataset_name: str) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    desc = df.describe(include="all").fillna("").astype(str)
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": desc.to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }


def run_query(dataset_name: str, query: str) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    try:
        result = eval(query, {"df": df, "pd": pd})  # noqa: S307
        if isinstance(result, pd.DataFrame):
            return {"result": result.to_dict(orient="records"), "shape": list(result.shape)}
        if isinstance(result, pd.Series):
            return {"result": result.to_dict()}
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def filter_data(
    dataset_name: str,
    conditions: list[dict],
    output_dataset_name: str | None = None,
    select_columns: list[str] | None = None,
) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    try:
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in conditions:
            col, op, val = cond["column"], cond["operator"], cond["value"]
            if op == "==":   mask &= df[col] == val
            elif op == "!=": mask &= df[col] != val
            elif op == ">":  mask &= df[col] > val
            elif op == "<":  mask &= df[col] < val
            elif op == ">=": mask &= df[col] >= val
            elif op == "<=": mask &= df[col] <= val
            elif op == "contains": mask &= df[col].astype(str).str.contains(str(val), na=False)
            elif op == "isin":     mask &= df[col].isin(val if isinstance(val, list) else [val])
            else: return {"error": f"Unknown operator: {op}"}
        filtered = df[mask]
        if select_columns:
            filtered = filtered[select_columns]
        out_name = output_dataset_name or f"{dataset_name}_filtered"
        _datasets[out_name] = filtered
        return {
            "output_dataset_name": out_name,
            "original_rows": len(df),
            "filtered_rows": len(filtered),
            "preview": filtered.head(10).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def aggregate_data(
    dataset_name: str,
    group_by: list[str],
    aggregations: dict[str, list[str]],
    sort_by: str | None = None,
) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    try:
        result = df.groupby(group_by).agg(aggregations)
        result.columns = ["_".join(c) if isinstance(c, tuple) else c for c in result.columns]
        result = result.reset_index()
        if sort_by and sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=False)
        return {
            "result": result.to_dict(orient="records"),
            "shape": list(result.shape),
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def correlation_analysis(
    dataset_name: str,
    columns: list[str] | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    try:
        numeric = df.select_dtypes(include="number")
        if columns:
            numeric = numeric[[c for c in columns if c in numeric.columns]]
        corr = numeric.corr()
        result: dict[str, Any] = {"correlation_matrix": corr.round(4).to_dict()}
        if plot:
            _ensure_output_dir()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            out_path = OUTPUT_DIR / f"{dataset_name}_correlation.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            result["chart_saved"] = str(out_path)
        return result
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def create_visualization(
    dataset_name: str,
    chart_type: str,
    output_filename: str,
    x_column: str | None = None,
    y_column: str | None = None,
    title: str | None = None,
    hue: str | None = None,
    aggregate: str = "none",
) -> dict[str, Any]:
    df = _datasets.get(dataset_name)
    if df is None:
        return {"error": f"Dataset '{dataset_name}' not loaded."}
    _ensure_output_dir()
    try:
        plot_df = df.copy()

        # optional pre-aggregation for bar / line
        if aggregate != "none" and x_column and y_column and chart_type in ("bar", "line"):
            agg_map = {"sum": "sum", "mean": "mean", "count": "count"}
            func = agg_map.get(aggregate, "sum")
            plot_df = df.groupby(x_column)[y_column].agg(func).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        chart_title = title or f"{chart_type.title()} chart"

        if chart_type == "bar":
            if hue and hue in plot_df.columns:
                pivot = plot_df.pivot_table(index=x_column, columns=hue, values=y_column, aggfunc="sum")
                pivot.plot(kind="bar", ax=ax)
            else:
                ax.bar(plot_df[x_column].astype(str), plot_df[y_column])
                ax.set_xlabel(x_column); ax.set_ylabel(y_column)
                ax.tick_params(axis="x", rotation=45)

        elif chart_type == "line":
            if hue and hue in plot_df.columns:
                for label, grp in plot_df.groupby(hue):
                    ax.plot(grp[x_column], grp[y_column], marker="o", label=str(label))
                ax.legend()
            else:
                ax.plot(plot_df[x_column], plot_df[y_column], marker="o")
            ax.set_xlabel(x_column); ax.set_ylabel(y_column)

        elif chart_type == "scatter":
            c = plot_df[hue].astype("category").cat.codes if hue else None
            sc = ax.scatter(plot_df[x_column], plot_df[y_column], c=c, alpha=0.6, cmap="tab10")
            ax.set_xlabel(x_column); ax.set_ylabel(y_column)
            if hue:
                ax.legend(*sc.legend_elements(), title=hue)

        elif chart_type == "histogram":
            col = x_column or y_column
            ax.hist(plot_df[col].dropna(), bins=30, edgecolor="black")
            ax.set_xlabel(col); ax.set_ylabel("Frequency")

        elif chart_type == "box":
            if x_column and y_column:
                sns.boxplot(data=plot_df, x=x_column, y=y_column, ax=ax)
            else:
                col = y_column or x_column
                ax.boxplot(plot_df[col].dropna())
                ax.set_ylabel(col)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "pie":
            sizes = plot_df.set_index(x_column)[y_column]
            ax.pie(sizes, labels=sizes.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")

        ax.set_title(chart_title)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"{output_filename}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return {"status": "saved", "path": str(out_path), "title": chart_title}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def save_results(content: str, filename: str) -> dict[str, Any]:
    _ensure_output_dir()
    out_path = OUTPUT_DIR / f"{filename}.txt"
    out_path.write_text(content, encoding="utf-8")
    return {"status": "saved", "path": str(out_path)}


# ─── dispatcher ───────────────────────────────────────────────────────────────

TOOL_HANDLERS: dict[str, Any] = {
    "load_dataset":         load_dataset,
    "describe_dataset":     describe_dataset,
    "run_query":            run_query,
    "filter_data":          filter_data,
    "aggregate_data":       aggregate_data,
    "correlation_analysis": correlation_analysis,
    "create_visualization": create_visualization,
    "save_results":         save_results,
}


def dispatch(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Call the appropriate tool and return its result as a JSON string."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    result = handler(**tool_input)
    return json.dumps(result, default=str, ensure_ascii=False)

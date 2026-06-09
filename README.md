# Data Analyst Agent

An end-to-end AI agent that answers natural-language questions about tabular data using Claude claude-opus-4-8 with tool use.

---

## Demo

```
python cli.py "Which region had the highest total revenue?"
python cli.py --show-steps "Show monthly revenue trends with a line chart"
python cli.py --interactive
streamlit run app.py           # web UI
jupyter notebook notebooks/demo.ipynb
```

---

## Project structure

```
data-analyst-agent/
├── src/
│   ├── agent.py      # agentic ReAct loop
│   ├── tools.py      # 8 tool implementations + JSON schemas
│   └── config.py     # model name, iteration cap, helpers
├── cli.py            # CLI entry point
├── app.py            # Streamlit web interface
├── notebooks/
│   └── demo.ipynb    # reproducible walkthrough
├── data/
│   └── sales_data.csv
├── outputs/          # charts + reports written by the agent
└── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd data-analyst-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-..."
```

---

## Problem framing

Business analysts spend a large fraction of their time translating analytical questions into pandas code, debugging it, and formatting the results. The bottleneck is the translation step — not the computation. An LLM that can interpret a question, decide which operations to perform, call them via structured tools, and synthesise the results eliminates that bottleneck.

The goal is an agent that behaves like a senior analyst sitting next to you: it explores before committing, explains its reasoning, produces visual artefacts, and delivers a plain-English conclusion.

---

## Why a tool-use agent, not a code-interpreter approach?

Two main alternatives exist:

| Approach | Description | Rejected because |
|---|---|---|
| **Direct code generation** | Ask the LLM to write a pandas script, execute it | No recovery path when code errors; model has to get everything right in one shot |
| **Managed Agents / sandbox** | Anthropic runs the execution environment | Adds infrastructure dependency; harder to inspect intermediate steps locally |
| **Tool-use agentic loop** *(chosen)* | Structured tool schemas + manual ReAct loop | Full visibility, custom error handling, works locally without extra infra |

Tool use lets Claude call deterministic Python functions (load_dataset, run_query, create_visualization, etc.) instead of generating arbitrary code. This is safer and more controllable: the LLM selects *which* operation to perform and *with which parameters*, while the execution stays in our code.

---

## Retrieval / analysis approach

There is no retrieval-augmented generation step here — the dataset fits in memory and the agent operates directly on pandas DataFrames. The "retrieval" analogy is the `run_query` tool, which evaluates a pandas expression that Claude constructs. This is appropriate for structured tabular data; for unstructured documents a vector-search layer would be added.

---

## Model and inference design

| Choice | Decision | Rationale |
|---|---|---|
| Model | `claude-opus-4-8` | Best reasoning on multi-step analytical tasks; handles ambiguous questions well |
| Thinking mode | `adaptive` | Claude decides how much reasoning each step warrants; avoids paying for full extended thinking on simple tool calls |
| Streaming | Yes (`client.messages.stream`) | Prevents HTTP timeouts on long analysis chains; `.get_final_message()` gives the full response after streaming |
| Max iterations | 20 | Enough for complex multi-chart analyses; prevents runaway loops |
| System prompt | Analyst persona with reasoning instructions | Encourages step-by-step decomposition and plain-English conclusions |

The manual agentic loop (in `src/agent.py`) was chosen over the beta tool runner because:
- We need to capture every step (thinking, text, tool calls) for the Streamlit UI and evaluation
- We want to pass the full `assistant` content block (including thinking blocks) back on each turn — required for correct compaction behaviour
- Fine-grained control lets us add approval gates or logging in future

---

## Tool design

| Tool | Purpose | Design note |
|---|---|---|
| `load_dataset` | Load CSV/JSON into a named in-memory store | Named store (`_datasets` dict) decouples loading from querying across turns |
| `describe_dataset` | Shape, dtypes, nulls, summary stats | Returns structured dict so Claude can plan next steps |
| `run_query` | Execute a pandas expression | Power-escape-hatch for arbitrary computation; restricted to a trusted local context |
| `filter_data` | Declarative row/column filtering | Safer than raw eval for simple slicing; avoids injecting large DataFrames into the prompt |
| `aggregate_data` | GroupBy + multi-aggregation | Maps cleanly to common business questions ("total revenue by region") |
| `correlation_analysis` | Pearson correlation matrix + heatmap | Seaborn heatmap saved to disk; path returned so agent can cite it |
| `create_visualization` | Bar, line, scatter, histogram, box, pie | Single tool for all chart types; `aggregate` param avoids a separate query+chart call pattern |
| `save_results` | Write text report to file | Lets Claude produce artefacts that persist after the session |

All tools return a JSON-serialisable dict. Errors are returned as `{"error": "…"}` rather than raised — this keeps the agentic loop running and lets Claude decide whether to retry or explain the failure.

---

## Alternatives considered

**LangChain / LlamaIndex agents** — both provide ready-made pandas agents. Rejected in favour of the bare Anthropic SDK to have full control over the loop, avoid framework abstractions that obscure what the model actually does, and keep the dependency surface small.

**GPT-4 / Gemini** — no technical barrier, but the assignment context calls for Claude, and claude-opus-4-8's adaptive thinking is well-suited to multi-step reasoning.

**Function calling via JSON mode** — superseded by the native tool-use API which handles schema validation, parallel tool calls, and streaming tool responses natively.

---

## Evaluation

The agent is evaluated on three dimensions:

### 1. Functional correctness
After each demo notebook cell we inspect `steps_log` for any `ToolCallStep` where `tool_result` contains `"error"`. A clean run should have zero errors.

### 2. Analytical accuracy
We compare agent-computed aggregates against ground-truth pandas calculations run independently in the notebook's cells 1–2. For the sample dataset the correct values are deterministic and easy to verify.

### 3. Answer quality (rubric)
| Dimension | 1 – Poor | 3 – Acceptable | 5 – Excellent |
|---|---|---|---|
| Correctness | Wrong numbers | Correct but incomplete | Fully correct |
| Explanation | None | Mentions steps | Clear, cites chart |
| Efficiency | >15 iterations | 8–14 iterations | ≤7 iterations |
| Chart quality | No chart | Chart saved, not cited | Chart cited + described |

The notebooks/demo.ipynb section 5 ("Manual evaluation") shows how to extract the step trace and count tool-call errors programmatically.

### Limitations
- `run_query` uses `eval()` — safe in a local trusted context, but not appropriate for a multi-tenant service.
- No persistent memory across Streamlit sessions (datasets are in-process memory; restarting the server clears them).
- Large datasets (>1 M rows) will be slow because all data is held in RAM.

---

## AI usage disclosure

This project was built with the assistance of **Claude** (via the Anthropic Claude Code CLI):

- **Architecture design**: the agentic loop pattern, tool schema design, and system prompt were co-designed in conversation with Claude.
- **Code generation**: `src/tools.py`, `src/agent.py`, `cli.py`, `app.py`, and `notebooks/demo.ipynb` were generated by Claude and reviewed/edited by the author.
- **README**: drafted by Claude; structure, rationale sections, and evaluation rubric were guided by the author's requirements.
- **Sample data**: the `sales_data.csv` file was synthetically generated by Claude.

The author is responsible for the overall architecture decisions, the choice of tool set, the evaluation rubric, and the final quality of all outputs.

Tool: **Anthropic Claude Code (claude-sonnet-4-6)** — used as the primary coding assistant throughout this session.

---

## License

MIT

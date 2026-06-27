# Customer Support Multi-Agent System

> A production-grade, AI-powered customer support pipeline built with **LangGraph**, **Groq (llama-3.1-8b-instant)**, **Gemini Embeddings**, **FAISS**, and **SQLite** — featuring intelligent routing, Retrieval-Augmented Generation (RAG), and a terminal-based Human-in-the-Loop (HITL) approval workflow.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Environment Variables](#environment-variables)
6. [How to Run](#how-to-run)
7. [Demonstration Walkthrough](#demonstration-walkthrough)
8. [Human-in-the-Loop (HITL) Workflow](#human-in-the-loop-hitl-workflow)
9. [SQLite Memory Persistence](#sqlite-memory-persistence)
10. [Verifying the Database](#verifying-the-database)

---

## Architecture Overview

The system processes incoming support tickets through a stateful, multi-node LangGraph pipeline. Each query enters via an **Intent Classifier**, is routed to the appropriate **Department Agent** (which performs RAG retrieval), and is reviewed by a **Supervisor Node** — with high-risk actions triggering a mandatory human review step.

```
                    ┌─────────────────────────────────────────────────┐
                    │           LANGGRAPH STATE MACHINE                │
                    │                                                   │
  User Query  ───►  │  [START] ──► [Classifier] ──► Conditional Router│
                    │                                      │           │
                    │              ┌───────────────────────┤           │
                    │              │                       │           │
                    │         [Sales Agent]         [Tech Agent]       │
                    │         [Billing Agent]        [Account Agent]   │
                    │         [Memory Agent]                           │
                    │              │                                   │
                    │         RAG Retrieval (FAISS + Gemini)           │
                    │              │                                   │
                    │         High-Risk Detection                      │
                    │              │                                   │
                    │         YES ─┤─ NO                              │
                    │              │   └──────────────────────┐        │
                    │    [HITL Interrupt]                      │        │
                    │    Terminal input()                      │        │
                    │    app.update_state()                    │        │
                    │              │                           │        │
                    │              └──────► [Supervisor Node] ◄┘       │
                    │                           │                      │
                    │                    [SQLite Save]                 │
                    │                           │                      │
                    │                         [END]                    │
                    └─────────────────────────────────────────────────┘
```

### State Persistence

Every node transition is serialised to `memory.db` via `SqliteSaver`. Using the same `thread_id` across multiple separate terminal invocations enables **full cross-session conversational memory** — the system remembers every past interaction with a customer.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) `>=1.2.6` | Stateful, cyclic multi-agent workflow engine |
| **LLM (Reasoning)** | [Groq](https://groq.com/) — `llama-3.1-8b-instant` | Ultra-fast inference for classification, RAG responses, supervision |
| **Embeddings** | [Gemini](https://ai.google.dev/) — `models/gemini-embedding-001` | High-quality text-to-vector conversion for knowledge base indexing |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) | Local, serverless similarity search over knowledge base chunks |
| **Memory / State** | SQLite + `SqliteSaver` | Persistent cross-session conversation history via checkpoint protocol |
| **Env Management** | `python-dotenv` | Secure API key loading from `.env` |
| **Python Runtime** | `uv` + Python 3.13 | Fast, reproducible environment and package management |

### Why These Choices?

- **Groq**: Sub-100ms LLM inference makes the interactive HITL terminal feel real-time.
- **Gemini Embeddings**: `gemini-embedding-001` produces dense, high-quality vectors, outperforming many open-source alternatives on semantic retrieval benchmarks.
- **FAISS (local)**: No external vector DB server required — the index lives in RAM and is rebuilt from the knowledge base on each startup. Ideal for demos and edge deployments.
- **LangGraph + SQLite**: The `SqliteSaver` checkpointer stores the *entire* `SupportState` (messages, intent, draft, approval flags) as a JSON blob after every node, making memory recall trivially simple.

---

## Project Structure

```
customer-support/
├── main.py              # Core multi-agent application (all logic lives here)
├── verify_db.py         # Helper script to inspect SQLite checkpoint schema
├── memory.db            # Auto-generated SQLite database (persistent memory)
├── .env                 # API keys (NOT committed to version control)
├── .gitignore           # Excludes .env and memory.db from git
├── pyproject.toml       # Project metadata and pinned dependencies (uv)
├── requirements.txt     # Pip-compatible dependency list
└── README.md            # This file
```

> **Knowledge Base Note:** In this demonstration, the four knowledge base files (`pricing_guide.txt`, `company_policy.txt`, `technical_manual.txt`, `faq.txt`) are embedded directly as `Document` objects inside `setup_knowledge_base()`. To use physical files, replace that function with `TextLoader` or `PyPDFLoader` as shown in the inline comments.

---

## Setup & Installation

This project uses **`uv`** for fast, reproducible Python environment and package management.

### Step 1: Install `uv`

If you don't have `uv` installed:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```powershell
uv --version
```

### Step 2: Clone and Navigate to the Project

```powershell
git clone <your-repo-url>
cd customer-support
```

### Step 3: Create the Virtual Environment

`uv` will automatically read `pyproject.toml` and create an isolated environment:

```powershell
uv venv
```

This creates a `.venv/` folder in the project directory.

### Step 4: Install All Dependencies

```powershell
uv sync
```

`uv sync` resolves the full dependency graph from `uv.lock` and installs all packages in a single, reproducible step. This is faster and more reliable than `pip install -r requirements.txt`.

### Step 5: Verify Installation

```powershell
uv run python -c "import langgraph, langchain_groq, langchain_google_genai, faiss; print('All dependencies OK')"
```

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# Groq API Key — used for llama-3.1-8b-instant inference
# Get yours free at: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

# Google / Gemini API Key — used for models/gemini-embedding-001
# Get yours at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIzaSy_your_google_api_key_here
```

> **Security Warning:** Never commit your `.env` file to version control. The provided `.gitignore` already excludes it.

> **Note on Duplicate Keys:** If both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are set, `langchain-google-genai` will prefer `GOOGLE_API_KEY` and print a deprecation notice. Use only `GOOGLE_API_KEY` to avoid the warning.

---

## How to Run

Execute the interactive demonstration directly with `uv`:

```powershell
uv run main.py
```

The script will:
1. Build the FAISS knowledge base (calls Gemini Embeddings API once at startup)
2. Compile the LangGraph state machine
3. Process all 5 demonstration queries sequentially
4. **Pause and wait for your terminal input** on Query 4 (the refund request)
5. Save the full session to `memory.db`

### Running Interactively

The system pauses on high-risk queries and waits for your input:

```
[HITL INTERRUPT] -- Manager Approval Required
----------------------------------------------------------
Draft Response Under Review:

  """Based on our policy, annual plan refunds are available within 14 days..."""
----------------------------------------------------------

[MANAGER ACTION REQUIRED] Approve this action? (yes/no): _
```

Type `yes` to approve the refund or `no` to reject it. The system continues automatically after your input.

---

## Demonstration Walkthrough

The system processes 5 pre-defined queries that exercise every component of the architecture.

---

### Query 1 — Sales Routing + RAG Pricing Retrieval

**Input:** `"What are the pricing plans available for your software?"`

**Pipeline:**
1. **Classifier Node** → Groq classifies query as `Sales`
2. **Conditional Router** → Routes to `sales_node`
3. **RAG Retrieval** → Gemini Embeddings converts query to vector; FAISS returns top-2 chunks from `pricing_guide.txt`
4. **Draft Generation** → Groq generates a response grounded in retrieved pricing data
5. **HITL Check** → `requires_approval = False` (no high-risk keywords)
6. **Supervisor Node** → Refines draft; saves `AIMessage` to SQLite
7. **Output** → Professional pricing summary with monthly/annual plan details

**Key Demonstration:** Shows the full RAG pipeline — embedding → retrieval → grounded generation.

---

### Query 2 — Account Routing + FAQ Retrieval

**Input:** `"I forgot my account password."`

**Pipeline:**
1. **Classifier** → Classified as `Account`
2. **Router** → Routes to `account_node`
3. **RAG Retrieval** → FAISS returns FAQ chunk about password reset procedure
4. **Draft Generation** → Step-by-step reset instructions grounded in `faq.txt`
5. **HITL Check** → `requires_approval = False`
6. **Supervisor** → Polishes draft; appends to conversation history in SQLite
7. **Output** → Clear password reset instructions

**Key Demonstration:** Shows department-specific routing and FAQ-based knowledge retrieval.

---

### Query 3 — Technical Routing + Manual Retrieval

**Input:** `"My application crashes whenever I upload a file."`

**Pipeline:**
1. **Classifier** → Classified as `Technical`
2. **Router** → Routes to `tech_node`
3. **RAG Retrieval** → FAISS returns the file-upload crash troubleshooting chunk from `technical_manual.txt`
4. **Draft Generation** → Groq generates cache-clear and version-update instructions
5. **HITL Check** → `requires_approval = False` (no financial/legal risk)
6. **Supervisor** → Refines and saves to SQLite
7. **Output** → Actionable troubleshooting steps (clear cache, update to v2.1)

**Key Demonstration:** Shows technical RAG retrieval and grounded troubleshooting guidance.

---

### Query 4 — Billing Routing + HITL Interrupt ⚠️

**Input:** `"I need a refund for my annual subscription."`

**Pipeline:**
1. **Classifier** → Classified as `Billing`
2. **Router** → Routes to `billing_node`
3. **RAG Retrieval** → FAISS returns `company_policy.txt` refund policy chunk
4. **Draft Generation** → Groq drafts a refund response citing the 14-day policy
5. **High-Risk Detection** → `"refund"` keyword detected → `requires_approval = True`
6. **LangGraph Interrupt** → Graph **pauses** at `interrupt_before=["supervisor"]`; state saved to `memory.db`
7. **Terminal HITL Block** → System prints the draft and prompts: `[MANAGER ACTION REQUIRED] Approve this action? (yes/no):`
8. **State Injection** → `app.update_state()` writes `approval_status = "approved"` or `"rejected"` into the SQLite checkpoint
9. **Graph Resume** → `app.stream(None, ...)` resumes execution from the supervisor node
10. **Supervisor Node** → Either refines the approval draft OR generates a policy refusal based on the injected status
11. **Output** → Either a refund confirmation or a polite rejection message

**Key Demonstration:** This is the primary HITL demonstration. The terminal becomes a live approval console, and the SQLite checkpoint freezes the complete agent state mid-execution.

---

### Query 5 — Memory Routing + History Recall

**Input:** `"What was my previous support issue?"`

**Pipeline:**
1. **Classifier** → Classified as `Memory` (keyword: "previous")
2. **Router** → Routes to `memory_agent_node`
3. **History Access** → Node reads `state["messages"]` — which now contains **all 8 prior messages** (4 HumanMessages + 4 AIMessages from Queries 1–4) loaded from the SQLite checkpoint
4. **Draft Generation** → Groq identifies and summarises the most recent prior issue (Query 4: the refund request)
5. **HITL Check** → `requires_approval = False`
6. **Supervisor** → Finalises and saves the memory recall response
7. **Output** → "Your previous support issue was a request for a refund on your annual subscription."

**Key Demonstration:** Proves that SQLite-backed state persistence enables true multi-turn memory within a single `thread_id`. The Memory agent has no dedicated database — it reads directly from the accumulated `messages` list in the checkpoint.

---

## Human-in-the-Loop (HITL) Workflow

The HITL mechanism is implemented using three LangGraph primitives:

### 1. Compile-Time Interrupt Declaration

```python
app = builder.compile(
    checkpointer=memory,
    interrupt_before=["supervisor"]  # Graph pauses BEFORE this node
)
```

### 2. Detecting the Pause

```python
current_state = app.get_state(thread_config)
if current_state.next and "supervisor" in current_state.next:
    # Graph is paused — collect human input
```

### 3. Injecting Human Decision into Frozen State

```python
app.update_state(
    thread_config,
    {"approval_status": "approved"},  # or "rejected"
    as_node=current_state.next[0]     # Apply from supervisor's perspective
)
```

### 4. Resuming Execution

```python
for event in app.stream(None, config=thread_config):
    pass  # Resume from the frozen checkpoint
```

---

## SQLite Memory Persistence

LangGraph's `SqliteSaver` implements the `BaseCheckpointSaver` protocol and creates three tables in `memory.db`:

| Table | Purpose |
|---|---|
| `checkpoints` | Full serialised `SupportState` snapshots keyed by `(thread_id, checkpoint_ns, checkpoint_id)`. Each row is a msgpack-encoded blob of the complete graph state at one node transition. |
| `writes` | Pending write operations recording each channel value update during node execution, keyed by `(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`. |

Each table row represents the **complete agent state** at a specific moment in time, enabling:
- **Cross-session recall**: Restart the script with the same `thread_id` and the state is restored
- **Time-travel debugging**: Roll back to any prior checkpoint
- **HITL mid-execution state freeze**: Pause the graph with full state integrity

---

## Verifying the Database

Run the included helper script to inspect the SQLite schema and confirm memory is working:

```powershell
uv run verify_db.py
```

This prints the full schema of all LangGraph checkpoint tables and a sample of stored rows from `memory.db`.

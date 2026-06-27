# LangGraph Workflow Diagram — Customer Support Multi-Agent System

Paste the code block below into [Mermaid Live Editor](https://mermaid.live) to render the interactive diagram.

```mermaid
flowchart TD
    START([🚀 START]) --> Classifier

    %% ── Classifier Node ──────────────────────────────────────────
    Classifier["🔀 Intent Classifier Node\n──────────────────────\nGroq llama-3.1-8b-instant\nCategorises query into:\nSales / Tech / Billing /\nAccount / Memory"]

    %% ── Conditional Routing ──────────────────────────────────────
    Classifier -->|intent = Sales| SalesAgent
    Classifier -->|intent = Technical| TechAgent
    Classifier -->|intent = Billing| BillingAgent
    Classifier -->|intent = Account| AccountAgent
    Classifier -->|intent = Memory| MemoryAgent

    %% ── Department Agent Nodes ───────────────────────────────────
    SalesAgent["💰 Sales Agent\n──────────────────────\nRAG: FAISS retrieval\nprice_guide.txt\nGroq: Draft response"]
    TechAgent["🔧 Technical Support Agent\n──────────────────────\nRAG: FAISS retrieval\ntechnical_manual.txt\nGroq: Draft response"]
    BillingAgent["🧾 Billing Agent\n──────────────────────\nRAG: FAISS retrieval\ncompany_policy.txt\nGroq: Draft response\n⚠️ Refund keywords → flag"]
    AccountAgent["👤 Account Mgmt Agent\n──────────────────────\nRAG: FAISS retrieval\nfaq.txt\nGroq: Draft response"]
    MemoryAgent["🧠 Memory Agent\n──────────────────────\nReads full message\nhistory from SQLite\nGroq: Summarise past\ninteraction"]

    %% ── High-Risk Detection ──────────────────────────────────────
    SalesAgent --> RiskCheck{"⚠️ High-Risk\nDetection\n──────────────\nrequires_approval\n= True / False?"}
    TechAgent --> RiskCheck
    BillingAgent --> RiskCheck
    AccountAgent --> RiskCheck
    MemoryAgent --> RiskCheck

    %% ── HITL Branch ──────────────────────────────────────────────
    RiskCheck -->|requires_approval = TRUE\nGraph pauses here| HITL
    RiskCheck -->|requires_approval = FALSE\nGraph continues| Supervisor

    HITL["🚨 HITL INTERRUPT\n─────────────────────────────\ninterrupt_before=['supervisor']\n─────────────────────────────\n1. State frozen in memory.db\n2. Draft response displayed\n3. Terminal: input() prompt\n   'Approve? yes/no'\n4. app.update_state() injects\n   approval_status into SQLite\n5. app.stream(None,...) resumes"]

    HITL -->|Approved ✅| Supervisor
    HITL -->|Rejected ❌| Supervisor

    %% ── Supervisor Node ──────────────────────────────────────────
    Supervisor["🧑‍💼 Supervisor Node\n──────────────────────\nIf REJECTED:\n  → Policy refusal message\nIf APPROVED / No flag:\n  → Groq refines draft\n  → Returns AIMessage"]

    %% ── SQLite Save ──────────────────────────────────────────────
    Supervisor --> SQLiteSave["💾 SQLite Checkpointer\n──────────────────────\nSqliteSaver writes full\nSupportState to memory.db\n\nTables:\n• checkpoints\n• checkpoint_blobs\n• checkpoint_writes\n\nKeyed by thread_id +\ncheckpoint_id"]

    SQLiteSave --> END([✅ END])

    %% ── Memory Feedback Loop ─────────────────────────────────────
    SQLiteSave -.->|"Next query:\nadd_messages reducer\nloads full history\nfrom same thread_id"| MemoryAgent

    %% ── Styling ──────────────────────────────────────────────────
    style START fill:#1a1a2e,color:#e0e0e0,stroke:#4a4a8a
    style END fill:#1a2e1a,color:#e0e0e0,stroke:#4a8a4a
    style Classifier fill:#16213e,color:#a8d8ea,stroke:#4a90d9
    style SalesAgent fill:#0f3460,color:#e0e0e0,stroke:#4a90d9
    style TechAgent fill:#0f3460,color:#e0e0e0,stroke:#4a90d9
    style BillingAgent fill:#0f3460,color:#e0e0e0,stroke:#4a90d9
    style AccountAgent fill:#0f3460,color:#e0e0e0,stroke:#4a90d9
    style MemoryAgent fill:#1a0f3c,color:#d4a8ff,stroke:#9b59b6
    style RiskCheck fill:#3c1a0f,color:#ffd700,stroke:#e67e22
    style HITL fill:#3c0f0f,color:#ff9999,stroke:#e74c3c
    style Supervisor fill:#0f2c0f,color:#90ee90,stroke:#27ae60
    style SQLiteSave fill:#1a1a0f,color:#fffacd,stroke:#f39c12
```

## Node Reference Table

| Node | Role | Model/Tech |
|---|---|---|
| **Intent Classifier** | Routes query to correct department | Groq `llama-3.1-8b-instant` |
| **Sales Agent** | Answers pricing & plan questions | FAISS + Gemini Embeddings + Groq |
| **Technical Agent** | Resolves application errors & bugs | FAISS + Gemini Embeddings + Groq |
| **Billing Agent** | Handles invoices, refunds, subscriptions | FAISS + Gemini Embeddings + Groq |
| **Account Agent** | Manages passwords, profiles, login | FAISS + Gemini Embeddings + Groq |
| **Memory Agent** | Recalls prior interactions from history | SQLite message history + Groq |
| **High-Risk Check** | Detects refund/cancel/escalate keywords | Python keyword scan (no LLM) |
| **HITL Interrupt** | Pauses graph; collects human decision | `interrupt_before` + `input()` + `update_state()` |
| **Supervisor Node** | Finalises or refuses based on approval | Groq `llama-3.1-8b-instant` |
| **SQLite Checkpointer** | Persists full state after every node | `SqliteSaver` → `memory.db` |

## Key LangGraph Primitives Used

```python
# 1. add_messages reducer — appends instead of overwrites
messages: Annotated[list, add_messages]

# 2. Conditional routing — maps intent to node names
builder.add_conditional_edges("classifier", route_intent)

# 3. HITL interrupt — pauses graph before supervisor
app = builder.compile(checkpointer=memory, interrupt_before=["supervisor"])

# 4. Read paused state
current_state = app.get_state(thread_config)

# 5. Inject human decision into frozen checkpoint
app.update_state(thread_config, {"approval_status": "approved"}, as_node="supervisor")

# 6. Resume from frozen checkpoint
for event in app.stream(None, config=thread_config):
    pass
```

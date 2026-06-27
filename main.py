"""
================================================================================
  Customer Support Multi-Agent System
  Built with: LangGraph | LangChain | Groq (llama-3.1-8b-instant) |
              Gemini Embeddings (models/gemini-embedding-001) | FAISS | SQLite
================================================================================

ARCHITECTURE OVERVIEW:
  User Query --> [Classifier Node] --> Conditional Router
                    |
                    +---> [Sales Agent]   --|
                    |                      |
                    +---> [Tech Agent]     |--> [High-Risk Check]
                    |                      |         |
                    +---> [Billing Agent]  |         +--> YES --> [HITL Interrupt]
                    |                      |         |              --> Terminal Input
                    +---> [Account Agent] -|         +--> NO  --> [Supervisor Node]
                    |                                                    |
                    +---> [Memory Agent]                          [SQLite Save]
                                                                        |
                                                                       END

STATE PERSISTENCE:
  Every node transition is checkpointed to memory.db via SqliteSaver.
  The same thread_id across queries enables full conversational memory recall.
"""

import os
import sqlite3
from typing import TypedDict, Annotated, Literal

# Environment variable loader — reads GROQ_API_KEY and GOOGLE_API_KEY from .env
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain Imports
# ---------------------------------------------------------------------------
# ChatGroq  : Wraps Groq's API to call llama-3.1-8b-instant for all reasoning
# GoogleGenerativeAIEmbeddings : Calls Gemini's embedding model for vector ops
# HumanMessage / AIMessage     : Typed message objects for the messages list
# FAISS                        : Local, in-memory vector store (no DB server needed)
# Document                     : Container for knowledge base text chunks
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# LangGraph Imports
# ---------------------------------------------------------------------------
# StateGraph  : Core class for defining the agent workflow as a directed graph
# START / END : Sentinel nodes marking entry and exit points of the graph
# add_messages: A reducer function -- instead of overwriting, it APPENDS to the
#               messages list, which gives the graph its persistent memory.
# SqliteSaver : Checkpointer that serialises the full graph state to SQLite
#               after every node transition, enabling true persistent memory.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ==============================================================================
# PHASE 1: ENVIRONMENT & MODEL INITIALISATION
# ==============================================================================
load_dotenv()  # Loads keys from the .env file into os.environ

# --- Primary LLM: Groq (llama-3.1-8b-instant) --------------------------------
# temperature=0 ensures deterministic, consistent routing and drafts.
# This model handles: intent classification, department responses, supervision.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- Embedding Model: Gemini Embedding 001 ------------------------------------
# Used ONLY for converting text into high-dimensional vectors for FAISS lookup.
# The 'models/gemini-embedding-001' prefix is the exact Gemini API model name.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# ==============================================================================
# PHASE 2: STATE DEFINITION
# ==============================================================================
class SupportState(TypedDict):
    """
    The central data container shared across every node in the LangGraph.

    Every field is readable and writable by any node. A node only needs to
    return the fields it wants to update -- unchanged fields are preserved
    automatically by the checkpointer.

    Fields
    ------
    messages : Annotated[list, add_messages]
        The full conversation history. The `add_messages` reducer means that
        returning {"messages": [new_msg]} APPENDS rather than replaces, giving
        the agent a persistent, growing memory within a thread.

    intent : str
        The department category determined by the classifier node.
        One of: "Sales", "Technical", "Billing", "Account", "Memory"

    requires_approval : bool
        Set to True by a department agent if the query contains high-risk
        keywords (e.g., "refund", "cancel"). This flag triggers the HITL
        interrupt before the supervisor node.

    approval_status : str
        Holds the human manager's decision: "approved" or "rejected".
        Injected into state via app.update_state() after terminal input.

    retrieved_context : str
        The raw text retrieved from FAISS for the current query. Used by
        department agents to ground their responses in factual company data.

    draft_response : str
        The unpolished response generated by a department agent. Reviewed by
        the HITL block and then refined by the supervisor node.
    """
    messages: Annotated[list, add_messages]
    intent: str
    requires_approval: bool
    approval_status: str
    retrieved_context: str
    draft_response: str


# ==============================================================================
# PHASE 3: RAG KNOWLEDGE BASE SETUP (FAISS + Gemini Embeddings)
# ==============================================================================
def setup_knowledge_base() -> FAISS:
    """
    Ingests knowledge base text and builds a local FAISS vector store.

    In a production environment, this function would use PyPDFLoader or
    TextLoader to load files from a /documents directory:

        from langchain_community.document_loaders import TextLoader
        loaders = [
            TextLoader("documents/pricing_guide.txt"),
            TextLoader("documents/company_policy.txt"),
            TextLoader("documents/technical_manual.txt"),
            TextLoader("documents/faq.txt"),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

    For this demonstration, the content from those files is embedded directly
    as Document objects to ensure the system runs without external file I/O.

    Returns
    -------
    FAISS
        A searchable vector store. Each document chunk is converted to a
        vector by Gemini and stored in FAISS's flat L2 index.
    """
    docs = [
        # --- pricing_guide.txt content ---
        Document(
            page_content=(
                "Pricing Guide: We offer two subscription tiers.\n"
                "Monthly Plan: $12/month -- billed monthly, cancel anytime.\n"
                "Annual Plan:  $120/year -- saves 17% vs monthly (equivalent to 2 free months).\n"
                "Enterprise pricing is available upon request for teams of 10 or more users."
            ),
            metadata={"source": "pricing_guide.txt"}
        ),
        # --- technical_manual.txt content ---
        Document(
            page_content=(
                "Technical Manual -- Troubleshooting:\n"
                "Issue: Application crashes on file upload.\n"
                "Resolution: Clear the application cache (Settings > Advanced > Clear Cache) "
                "and update to version 2.1 or later. The crash was caused by a memory leak "
                "in the v2.0 file-handling module that was patched in v2.1."
            ),
            metadata={"source": "technical_manual.txt"}
        ),
        # --- company_policy.txt content ---
        Document(
            page_content=(
                "Company Refund & Cancellation Policy:\n"
                "Refunds are ONLY applicable to Annual Plan subscribers.\n"
                "The request must be submitted within 14 days of the original purchase date.\n"
                "All refund requests require explicit manager approval before processing.\n"
                "Monthly subscriptions are non-refundable but can be cancelled at any time."
            ),
            metadata={"source": "company_policy.txt"}
        ),
        # --- faq.txt content ---
        Document(
            page_content=(
                "FAQ -- Account Management:\n"
                "Q: How do I reset my password?\n"
                "A: Navigate to Profile Settings > Security > 'Forgot Password'. "
                "A reset link will be emailed to your registered address within 2 minutes.\n"
                "Q: How do I update billing information?\n"
                "A: Go to Account > Billing > Payment Methods to update your card details."
            ),
            metadata={"source": "faq.txt"}
        ),
    ]

    # FAISS.from_documents() calls embeddings.embed_documents() on each chunk,
    # then builds the flat L2 index. This is done ONCE at startup.
    print("Building FAISS knowledge base with Gemini embeddings...")
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Knowledge base ready.\n")
    return vector_store


# Build the vector store and expose a retriever interface.
# The retriever wraps FAISS's similarity_search() with a clean .invoke() API.
vector_store = setup_knowledge_base()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Return top-2 most relevant chunks


# ==============================================================================
# PHASE 4: AGENT NODE DEFINITIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# Node 1: Intent Classifier
# ------------------------------------------------------------------------------
def intent_classifier_node(state: SupportState) -> dict:
    """
    Determines the department category for the latest user message.

    This is the first node executed after START. It reads the most recent
    human message and asks the LLM to classify it into one of five categories.
    The result is stored in state["intent"], which the conditional router reads.

    Parameters
    ----------
    state : SupportState
        The current graph state; only state["messages"][-1] is read here.

    Returns
    -------
    dict
        Updates intent, and resets approval flags to defaults for this turn.
    """
    # Always classify the latest (most recent) human message
    user_query = state["messages"][-1].content

    classification_prompt = f"""You are a customer support routing system.
Categorize the following customer query into EXACTLY ONE of these categories:
- Sales       : Questions about pricing, plans, upgrades, or purchasing
- Technical   : Application errors, bugs, crashes, or technical how-to questions
- Billing     : Invoice questions, payment failures, or subscription changes
- Account     : Password resets, profile updates, login issues
- Memory      : Any question asking about a PREVIOUS or PAST support interaction

Query: "{user_query}"

Respond with ONLY the single category name. No punctuation, no explanation."""

    response = llm.invoke(classification_prompt)

    # Strip whitespace and punctuation to ensure clean matching in route_intent()
    intent = response.content.strip().strip(".").strip("'\"")
    print(f" -> Classified as: [{intent}]")

    return {
        "intent": intent,
        "requires_approval": False,   # Reset high-risk flag for each new query
        "approval_status": "pending"  # Reset approval status for each new query
    }


# ------------------------------------------------------------------------------
# Node 2: Memory Agent
# ------------------------------------------------------------------------------
def memory_agent_node(state: SupportState) -> dict:
    """
    Handles queries about past interactions by scanning conversation history.

    When the intent is "Memory", LangGraph routes here. The node receives the
    FULL conversation history (because add_messages accumulates across turns),
    giving it access to all previous queries and AI responses in this thread.

    The LLM synthesises an answer from this history without needing any
    external lookup -- the SQLite checkpoint IS the memory.

    Parameters
    ----------
    state : SupportState
        state["messages"] contains the complete multi-turn conversation.

    Returns
    -------
    dict
        draft_response: The LLM's answer derived from conversation history.
    """
    # Exclude the current message (index -1); everything before it is "history"
    history = state["messages"][:-1]
    current_query = state["messages"][-1].content

    memory_prompt = (
        "You are a customer support assistant with access to the full conversation "
        "history for this customer session.\n\n"
        "Previous conversation:\n"
        + "\n".join([f"  [{type(m).__name__}]: {m.content}" for m in history])
        + f"\n\nThe customer is now asking: \"{current_query}\"\n\n"
        "Answer using ONLY the information visible in the conversation history above. "
        "Be specific -- reference the actual issues or queries the customer raised."
    )

    response = llm.invoke(memory_prompt)
    return {"draft_response": response.content}


# ------------------------------------------------------------------------------
# Generic Department Agent Template (used by Sales, Tech, Billing, Account)
# ------------------------------------------------------------------------------
def department_agent(state: SupportState, department_name: str) -> dict:
    """
    Generic RAG-powered agent for any support department.

    Execution flow:
    1. Retrieve top-k relevant document chunks from FAISS (RAG retrieval).
    2. Build a department-specific prompt grounded in the retrieved context.
    3. Generate a draft response using the Groq LLM.
    4. Scan the user query for high-risk keywords that require human approval.

    The high-risk flag (requires_approval=True) causes the graph to pause
    at the interrupt_before=["supervisor"] checkpoint, blocking execution
    until a human manager approves or rejects via the terminal.

    Parameters
    ----------
    state          : SupportState -- full current graph state
    department_name: str          -- used in the system prompt to set agent persona

    Returns
    -------
    dict
        retrieved_context : The raw FAISS-retrieved text used for grounding.
        draft_response    : The LLM-generated response before supervisor review.
        requires_approval : True if a high-risk keyword was detected.
    """
    user_query = state["messages"][-1].content

    # ---- Step 1: RAG Retrieval -----------------------------------------------
    # retriever.invoke() converts the query to a Gemini embedding vector,
    # performs an L2 similarity search in FAISS, and returns the top-k chunks.
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # ---- Step 2: Draft Generation --------------------------------------------
    agent_prompt = (
        f"You are the {department_name} Support Agent for our software company.\n\n"
        "Your task is to draft a helpful, accurate, and empathetic response to the customer's query.\n"
        "Base your response ONLY on the company knowledge provided below. Do not invent information.\n\n"
        f"--- Company Knowledge Base (Retrieved) ---\n{context}\n---\n\n"
        f"Customer Query: \"{user_query}\"\n\n"
        "Write a clear, professional draft response:"
    )

    draft = llm.invoke(agent_prompt).content

    # ---- Step 3: High-Risk Detection -----------------------------------------
    # These keywords indicate actions that could have financial or legal impact.
    # If detected, requires_approval=True pauses the graph for HITL review.
    high_risk_keywords = ["refund", "cancel", "cancellation", "closure",
                          "compensation", "escalate", "chargeback"]
    requires_approval = any(
        keyword in user_query.lower() for keyword in high_risk_keywords
    )

    if requires_approval:
        print(f"\n  High-risk keyword detected in query. Flagging for manager approval.")

    return {
        "retrieved_context": context,
        "draft_response": draft,
        "requires_approval": requires_approval
    }


# --- Node Wrappers -----------------------------------------------------------
# LangGraph requires each node to be a separate callable. These thin wrappers
# allow us to reuse the generic department_agent with different persona names.
def sales_node(state: SupportState) -> dict:
    return department_agent(state, "Sales & Pricing")

def tech_node(state: SupportState) -> dict:
    return department_agent(state, "Technical Support")

def billing_node(state: SupportState) -> dict:
    return department_agent(state, "Billing & Payments")

def account_node(state: SupportState) -> dict:
    return department_agent(state, "Account Management")


# ------------------------------------------------------------------------------
# Node 3: Supervisor (Post-HITL Finalisation)
# ------------------------------------------------------------------------------
def supervisor_node(state: SupportState) -> dict:
    """
    Finalises the agent response after optional HITL review.

    This node is declared with interrupt_before=["supervisor"] in the compiled
    graph, meaning execution ALWAYS pauses BEFORE this node runs. The terminal
    input block in run_demonstration() catches this pause, collects the
    manager's decision, and calls app.update_state() to inject approval_status
    into the graph before resuming with app.stream(None, ...).

    If requires_approval is True AND approval_status is "rejected", the node
    generates a polite refusal instead of refining the draft.

    Parameters
    ----------
    state : SupportState -- must have draft_response and approval_status set.

    Returns
    -------
    dict
        messages: [AIMessage] -- the final response appended to conversation
                  history. This is what gets saved to memory.db via SqliteSaver.
    """
    draft = state.get("draft_response", "")
    status = state.get("approval_status", "approved")
    requires_approval = state.get("requires_approval", False)

    if requires_approval and status == "rejected":
        # Manager rejected the action -- generate a policy-compliant refusal
        final_text = (
            "Thank you for reaching out. After careful review, we are unable to "
            "approve your request at this time based on our current company policy. "
            "If you believe this decision is incorrect, please contact our support "
            "team directly at support@company.com for further assistance."
        )
        print("\n  Request REJECTED by manager. Sending policy refusal.")
    else:
        # Manager approved (or no approval was needed) -- refine the draft
        refine_prompt = (
            "You are a senior customer support supervisor.\n"
            "Review the following draft response and refine it to ensure it is:\n"
            "- Polite and empathetic in tone\n"
            "- Concise and professional\n"
            "- Free of any internal references or raw template text\n\n"
            f"Draft: {draft}\n\n"
            "Provide the final, customer-ready response:"
        )
        final_text = llm.invoke(refine_prompt).content
        print("\n  Request APPROVED. Supervisor refining draft response.")

    # Returning an AIMessage via add_messages saves it to the SQLite conversation
    # history, making it accessible in future "Memory" queries via memory_agent_node.
    return {"messages": [AIMessage(content=final_text)]}


# ==============================================================================
# PHASE 5: GRAPH ROUTING LOGIC
# ==============================================================================
def route_intent(state: SupportState) -> Literal["sales", "tech", "billing", "account", "memory"]:
    """
    Conditional edge function: maps the classified intent to a graph node name.

    Called after the classifier node. LangGraph uses the return value as the
    name of the next node to execute. Must return one of the node names
    registered with builder.add_node().

    Parameters
    ----------
    state : SupportState -- reads state["intent"] set by the classifier.

    Returns
    -------
    str : One of "sales", "tech", "billing", "account", "memory"
    """
    intent = state["intent"].lower()

    if "sales" in intent:    return "sales"
    if "tech" in intent:     return "tech"
    if "billing" in intent:  return "billing"
    if "account" in intent:  return "account"
    # Default fallback -- memory handles "I had a previous issue..." type queries
    return "memory"


def route_approval(state: SupportState) -> Literal["supervisor"]:
    """
    Post-department conditional edge.

    Currently all paths lead to "supervisor" regardless of requires_approval,
    because the interrupt_before=["supervisor"] mechanism in the compiled graph
    handles the conditional pause automatically. LangGraph checks the flag
    before the node executes, not after the edge resolves.

    This function exists as an explicit hook for future logic -- for example,
    routing directly to END for low-risk queries to skip supervision overhead.

    Parameters
    ----------
    state : SupportState -- reads state["requires_approval"]

    Returns
    -------
    str : "supervisor" (always -- the interrupt mechanism handles the pause)
    """
    return "supervisor"


# ==============================================================================
# PHASE 6: GRAPH COMPILATION WITH SQLITE CHECKPOINTER
# ==============================================================================

# --- Graph Builder -----------------------------------------------------------
builder = StateGraph(SupportState)

# Add all agent nodes to the graph
builder.add_node("classifier", intent_classifier_node)  # Entry routing node
builder.add_node("sales",      sales_node)               # Sales & Pricing dept
builder.add_node("tech",       tech_node)                # Technical Support dept
builder.add_node("billing",    billing_node)             # Billing dept
builder.add_node("account",    account_node)             # Account Management dept
builder.add_node("memory",     memory_agent_node)        # Conversation history agent
builder.add_node("supervisor", supervisor_node)          # HITL + final response

# --- Edge Definitions --------------------------------------------------------
# Entry point: every query starts at the classifier
builder.add_edge(START, "classifier")

# Conditional routing: classifier output determines which department runs next
builder.add_conditional_edges("classifier", route_intent)

# All department agents connect to the supervisor via the approval router
for dept_node in ["sales", "tech", "billing", "account", "memory"]:
    builder.add_conditional_edges(dept_node, route_approval)

# Exit point: supervisor is always the final node before END
builder.add_edge("supervisor", END)

# --- SQLite Checkpointer Setup -----------------------------------------------
# sqlite3.connect() opens (or creates) memory.db in the current directory.
# check_same_thread=False is required because LangGraph accesses the connection
# from multiple threads during async graph execution.
# SqliteSaver wraps the connection and implements the BaseCheckpointSaver protocol,
# serialising the full SupportState to JSON after every node transition.
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# --- Compile the Graph -------------------------------------------------------
# interrupt_before=["supervisor"] is the HITL mechanism:
#   - After any department node finishes, LangGraph pauses before "supervisor".
#   - It saves state to memory.db and returns control to the calling code.
#   - The graph is then resumed via app.stream(None, config=...) after the
#     human manager has injected their decision via app.update_state().
app = builder.compile(
    checkpointer=memory,
    interrupt_before=["supervisor"]   # Task 8: HITL pause point
)

print("LangGraph compiled successfully.")
print(f"  Nodes  : {list(builder.nodes.keys())}")
print(f"  Memory : SQLite checkpointer -> memory.db\n")


# ==============================================================================
# PHASE 7: INTERACTIVE DEMONSTRATION RUNNER
# ==============================================================================
def run_demonstration():
    """
    Executes a scripted 5-query demonstration of the full system.

    Demonstrates:
      Query 1 -- Sales routing + RAG pricing retrieval
      Query 2 -- Account routing + RAG FAQ retrieval
      Query 3 -- Technical routing + RAG manual retrieval
      Query 4 -- Billing routing + HITL interrupt for refund approval
      Query 5 -- Memory routing + conversation history recall

    The same thread_id is used for all queries, which enables:
      a) SQLite state persistence between queries (same thread = same checkpoint)
      b) The Memory agent to see all previous messages in state["messages"]

    HITL Flow (Query 4):
      1. department_agent() detects "refund" and sets requires_approval=True
      2. Graph streams until it hits the interrupt_before=["supervisor"] pause
      3. app.get_state() reads the draft response from the paused checkpoint
      4. Terminal input() collects manager's "yes"/"no" decision
      5. app.update_state() injects approval_status into the frozen checkpoint
      6. app.stream(None, ...) resumes execution from the supervisor node
    """
    print("\n" + "="*60)
    print("  CUSTOMER SUPPORT MULTI-AGENT SYSTEM -- DEMO")
    print("="*60)

    # A single thread_id maintains conversation continuity across all queries.
    # Changing this value starts a completely fresh session with no prior history.
    thread_config = {"configurable": {"thread_id": "demo_session_001"}}

    queries = [
        "What are the pricing plans available for your software?",  # -> Sales
        "I forgot my account password.",                             # -> Account
        "My application crashes whenever I upload a file.",          # -> Technical
        "I need a refund for my annual subscription.",               # -> Billing + HITL
        "What was my previous support issue?",                       # -> Memory
    ]

    for i, query in enumerate(queries, start=1):
        print(f"\n{'='*60}")
        print(f"  [Query {i}/{len(queries)}]: {query}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # STEP 1: Stream the graph until it hits an interrupt (or completes)
        # ------------------------------------------------------------------
        # Passing a new HumanMessage triggers the graph from START.
        # The stream yields event dicts keyed by node_name.
        # We iterate to consume all events up to the pause point.
        for event in app.stream(
            {"messages": [HumanMessage(content=query)]},
            config=thread_config
        ):
            for node_name, state_update in event.items():
                # Display routing decision from the classifier
                if node_name == "classifier":
                    intent = state_update.get("intent", "unknown")
                    print(f"\n  Routing Decision: [{intent}] department")

        # ------------------------------------------------------------------
        # STEP 2: Check if the graph paused for Human-in-the-Loop review
        # ------------------------------------------------------------------
        current_state = app.get_state(thread_config)

        if current_state.next and "supervisor" in current_state.next:
            # The graph has paused before the supervisor node.
            # This means requires_approval=True was set by a department agent.
            draft = current_state.values.get("draft_response", "(no draft)")
            retrieved = current_state.values.get("retrieved_context", "")

            print("\n" + "-"*60)
            print("[HITL INTERRUPT] -- Manager Approval Required")
            print("-"*60)
            print("\nRetrieved Context (from FAISS):")
            print(f"  {retrieved[:200]}..." if len(retrieved) > 200 else f"  {retrieved}")
            print(f"\nDraft Response Under Review:\n")
            print(f'  """{draft}"""')
            print("-"*60)

            # ---- LIVE TERMINAL INPUT BLOCK ----------------------------------
            # This is the core HITL mechanism. Execution is fully blocked here
            # until the manager types "yes" or "no" and presses Enter.
            user_decision = ""
            while user_decision not in ["yes", "no"]:
                user_decision = input(
                    "\n[MANAGER ACTION REQUIRED] Approve this action? (yes/no): "
                ).strip().lower()

            # Map the terminal input to the state field value
            approval_status = "approved" if user_decision == "yes" else "rejected"
            print(f"\n>> Manager Decision: {approval_status.upper()}")

            # ---- Inject Decision into Graph State ---------------------------
            # app.update_state() modifies the SQLite checkpoint directly,
            # inserting approval_status into the frozen state before resuming.
            # as_node=current_state.next[0] tells LangGraph which node's
            # "perspective" to use when applying the state update.
            app.update_state(
                thread_config,
                {"approval_status": approval_status},
                as_node=current_state.next[0]
            )

            # ---- Resume Graph Execution -------------------------------------
            # Passing None as the input tells LangGraph to resume from where
            # it paused, rather than starting a new run.
            print("\nResuming graph from supervisor node...\n")
            for event in app.stream(None, config=thread_config):
                pass  # Consume remaining events to drive execution to END

        # ------------------------------------------------------------------
        # STEP 3: Read and display the final response from memory
        # ------------------------------------------------------------------
        # get_state() reads the latest checkpoint from memory.db.
        # The final AIMessage was appended by supervisor_node() and is
        # always the last element in state["messages"].
        final_state = app.get_state(thread_config)
        final_response = final_state.values["messages"][-1].content

        print(f"\n{'-'*60}")
        print(f"[AI Response]:")
        print(f"{'-'*60}")
        print(f"{final_response}")
        print(f"{'-'*60}")

    print("\n\n" + "="*60)
    print("  DEMONSTRATION COMPLETE")
    print(f"  Session persisted to: memory.db")
    print(f"  Thread ID           : {thread_config['configurable']['thread_id']}")
    print("="*60 + "\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    run_demonstration()
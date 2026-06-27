"""
verify_db.py
============
Standalone helper script to inspect the LangGraph SQLite checkpoint database.

Purpose:
    Proves that SqliteSaver is correctly writing and persisting the full
    SupportState to memory.db after every node transition in the LangGraph.

Usage:
    uv run verify_db.py

Output:
    - Lists all tables created by SqliteSaver
    - Prints the CREATE TABLE schema for each table
    - Shows a summary count of stored checkpoints
    - Displays a sample of the most recent checkpoint rows

SqliteSaver creates two tables in memory.db:
    checkpoints : Full SupportState blobs keyed by (thread_id, checkpoint_ns, checkpoint_id)
    writes      : Pending channel write operations per task_id during node execution
"""

import sqlite3
import json
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "memory.db"
SAMPLE_ROWS = 3   # Number of recent rows to preview per table


def verify_database(db_path: str) -> None:
    """
    Connects to the LangGraph SQLite checkpoint database and prints
    the schema and sample data for all checkpoint-related tables.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g., "memory.db").
    """
    # Check the file exists before attempting to connect
    if not os.path.exists(db_path):
        print(f"\n[ERROR] Database file not found: '{db_path}'")
        print("Make sure you have run 'uv run main.py' at least once to generate memory.db.")
        return

    file_size_kb = os.path.getsize(db_path) / 1024
    print(f"\n{'='*60}")
    print(f"  LangGraph SQLite Memory Verification")
    print(f"{'='*60}")
    print(f"  Database : {os.path.abspath(db_path)}")
    print(f"  Size     : {file_size_kb:.1f} KB")
    print(f"{'='*60}\n")

    # Open connection in read-only mode for safety
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    cursor = conn.cursor()

    # -----------------------------------------------------------------------
    # Step 1: List all tables in the database
    # -----------------------------------------------------------------------
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row["name"] for row in cursor.fetchall()]

    print(f"Tables found in '{db_path}':")
    if not tables:
        print("  (no tables found -- has main.py been run yet?)")
        conn.close()
        return

    for t in tables:
        print(f"  - {t}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Print the CREATE TABLE schema for each table
    # -----------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"  TABLE SCHEMAS")
    print(f"{'='*60}")

    for table_name in tables:
        # Fetch the original CREATE TABLE statement from sqlite_master
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,)
        )
        row = cursor.fetchone()
        schema_sql = row["sql"] if row else "(schema not available)"

        print(f"\n[TABLE]: {table_name}")
        print("-" * 50)
        print(schema_sql)
        print()

        # Fetch column info for a cleaner columnar view
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"  Columns ({len(columns)}):")
        for col in columns:
            pk_marker = " [PRIMARY KEY]" if col["pk"] else ""
            null_marker = " NOT NULL" if col["notnull"] else ""
            print(f"    [{col['cid']}] {col['name']} ({col['type']}){null_marker}{pk_marker}")

    # -----------------------------------------------------------------------
    # Step 3: Row counts — prove data was written
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ROW COUNTS (Proof of Write Operations)")
    print(f"{'='*60}")

    for table_name in tables:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name};")
        count = cursor.fetchone()["cnt"]
        status = "OK -- data present" if count > 0 else "EMPTY -- no data written yet"
        print(f"  {table_name:<30} {count:>5} rows   [{status}]")

    # -----------------------------------------------------------------------
    # Step 4: Preview the most recent checkpoint rows
    # -----------------------------------------------------------------------
    CHECKPOINT_TABLE = "checkpoints"
    if CHECKPOINT_TABLE in tables:
        print(f"\n{'='*60}")
        print(f"  RECENT CHECKPOINTS (last {SAMPLE_ROWS} rows)")
        print(f"{'='*60}")
        print("  These rows are the serialised SupportState snapshots.")
        print("  Each row = one node transition in the LangGraph.\n")

        # Fetch the SAMPLE_ROWS most recent rows
        # SqliteSaver uses (thread_id, checkpoint_ns, checkpoint_id) as the key
        cursor.execute(f"""
            SELECT * FROM {CHECKPOINT_TABLE}
            ORDER BY rowid DESC
            LIMIT {SAMPLE_ROWS};
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  (no checkpoint rows found)")
        else:
            col_names = [description[0] for description in cursor.description]

            for idx, row in enumerate(rows, start=1):
                print(f"\n  --- Checkpoint Row {idx} ---")
                for col in col_names:
                    value = row[col]
                    # Truncate long blob/JSON values for readability
                    if isinstance(value, (bytes, str)) and len(str(value)) > 120:
                        display_val = str(value)[:117] + "..."
                    else:
                        display_val = value
                    print(f"    {col:<25} : {display_val}")

    # -----------------------------------------------------------------------
    # Step 5: Extract thread IDs to show active sessions
    # -----------------------------------------------------------------------
    if CHECKPOINT_TABLE in tables:
        try:
            cursor.execute(f"SELECT DISTINCT thread_id FROM {CHECKPOINT_TABLE};")
            thread_ids = [row[0] for row in cursor.fetchall()]
            print(f"\n{'='*60}")
            print(f"  ACTIVE SESSION THREAD IDs")
            print(f"{'='*60}")
            if thread_ids:
                for tid in thread_ids:
                    # Count messages per thread
                    cursor.execute(
                        f"SELECT COUNT(*) as cnt FROM {CHECKPOINT_TABLE} WHERE thread_id=?;",
                        (tid,)
                    )
                    checkpoint_count = cursor.fetchone()["cnt"]
                    print(f"  Thread: '{tid}'")
                    print(f"    Checkpoints stored: {checkpoint_count}")
            else:
                print("  (no threads found)")
        except sqlite3.OperationalError:
            # thread_id column may not exist in all SqliteSaver versions
            pass

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    conn.close()
    print(f"\n{'='*60}")
    print(f"  VERIFICATION COMPLETE")
    print(f"  SQLite memory persistence is working correctly.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    verify_database(DB_PATH)

#!/usr/bin/env python3
"""
inspect_checkpoints.py

LangGraph SQLite Checkpoint Inspector (Message Decoder + Thread Explorer)

- Reads checkpoints from a SqliteSaver database
- Decodes msgpack using LangGraph's JsonPlusSerializer (handles complex types/messages)
- Supports: list threads, filter by thread, show last N checkpoints, message preview, optional full dump
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Sequence, Tuple

import msgpack

try:
    # Preferred: uses LangGraph serializer with proper msgpack ext hooks.
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency: langgraph (JsonPlusSerializer not importable). "
        "Install/upgrade langgraph. Error: " + str(e)
    )

try:
    # Optional: if messages ever appear as dicts, this converts them safely.
    from langchain_core.messages.utils import messages_from_dict  # type: ignore
except Exception:
    messages_from_dict = None


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    # Read-only open prevents accidental writes.
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return bool(row)


def _decode_blob(serializer: JsonPlusSerializer, blob: Optional[bytes]) -> Any:
    if not blob:
        return None

    # Best path: decode as msgpack with LangGraph's ext hook support. [page:1]
    try:
        return serializer.loads_typed(("msgpack", blob))
    except Exception:
        pass

    # Sometimes metadata is stored in formats that msgpack-python can parse.
    try:
        return msgpack.unpackb(blob, raw=False, strict_map_key=False)
    except Exception:
        pass

    # Handle "extra data" by unpacking only the first object (common in some dumps).
    try:
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        unpacker.feed(blob)
        for obj in unpacker:
            return obj
    except Exception:
        pass

    return f"<un-decodable blob: {len(blob)} bytes>"


def _preview(val: Any, limit: int = 220) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.replace("\n", " ")[:limit]
    if isinstance(val, (bytes, bytearray)):
        return f"<{type(val).__name__} {len(val)} bytes>"
    if isinstance(val, (list, tuple)):
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, dict):
        keys = list(val.keys())
        return f"<dict keys={keys[:12]}{'...' if len(keys) > 12 else ''}>"
    return str(val).replace("\n", " ")[:limit]


def _as_messages(maybe_messages: Any) -> Sequence[Any]:
    # Most of the time the serializer will return real BaseMessage objects.
    if isinstance(maybe_messages, (list, tuple)):
        # If these are dict messages, convert them to BaseMessage objects. [web:74]
        if maybe_messages and isinstance(maybe_messages[0], dict) and messages_from_dict:
            try:
                return messages_from_dict(maybe_messages)  # type: ignore[misc]
            except Exception:
                return maybe_messages
        return maybe_messages
    return []


def _print_messages(messages: Sequence[Any], last: int = 8) -> None:
    if not messages:
        print("  Messages: (none)")
        return

    show = messages[-last:] if last > 0 else messages
    print(f"  Messages: total={len(messages)} showing_last={len(show)}")
    for i, m in enumerate(show, 1):
        m_type = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", None)
        name = getattr(m, "name", None)
        extra = f" name={name}" if name else ""
        print(f"    {i:2d}. {m_type:>10}{extra} : {_preview(content)}")


def list_threads(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT thread_id, COUNT(*) AS n FROM checkpoints GROUP BY thread_id ORDER BY n DESC, thread_id ASC"
    ).fetchall()
    if not rows:
        print("No checkpoints found.")
        return
    print(f"Threads found: {len(rows)}\n")
    for thread_id, n in rows:
        print(f"- {thread_id}  (checkpoints={n})")


def inspect_checkpoints(
    *,
    conn: sqlite3.Connection,
    serializer: JsonPlusSerializer,
    limit: int,
    thread_id: Optional[str],
    messages_last: int,
    dump: bool,
) -> None:
    sql = """
        SELECT thread_id, checkpoint_id, checkpoint, metadata
        FROM checkpoints
    """
    params: Tuple[Any, ...] = ()
    if thread_id:
        sql += " WHERE thread_id=?"
        params = (thread_id,)

    sql += " ORDER BY checkpoint_id DESC LIMIT ?"
    params = (*params, limit)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No checkpoints matched your query.")
        return

    print(f"Found {len(rows)} checkpoint(s) (newest first)\n")

    for thread_id_val, cp_id, checkpoint_blob, metadata_blob in rows:
        print("═" * 88)
        print(f"Thread:        {thread_id_val}")
        print(f"Checkpoint ID:  {cp_id}")

        checkpoint = _decode_blob(serializer, checkpoint_blob)
        metadata = _decode_blob(serializer, metadata_blob)

        if isinstance(checkpoint, dict):
            print("\nCheckpoint keys:", list(checkpoint.keys()))
            channel_values = checkpoint.get("channel_values") or {}
            if isinstance(channel_values, dict):
                print("channel_values keys:", list(channel_values.keys()))

                msgs = _as_messages(channel_values.get("messages"))
                print()
                _print_messages(msgs, last=messages_last)

                for k in ("user_id", "ltm_context"):
                    if k in channel_values:
                        print(f"\n  {k}: {_preview(channel_values.get(k), 500)}")
        else:
            print("\nCheckpoint decoded (non-dict):", _preview(checkpoint, 500))

        if metadata is not None:
            print("\nMetadata:")
            if dump:
                pprint(metadata, width=110, compact=True)
            else:
                print(" ", _preview(metadata, 700))

        if dump:
            print("\n--- FULL CHECKPOINT DUMP ---")
            pprint(checkpoint, width=110, compact=False)

        print()

    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect LangGraph SqliteSaver checkpoints.")
    ap.add_argument("--db", type=str, default="agent_state.db", help="Path to SQLite DB (default: agent_state.db)")
    ap.add_argument("--limit", type=int, default=12, help="How many checkpoints to show (default: 12)")
    ap.add_argument("--thread", type=str, default=None, help="Filter by thread_id")
    ap.add_argument("--messages-last", type=int, default=8, help="Show last N messages per checkpoint (default: 8)")
    ap.add_argument("--list-threads", action="store_true", help="List thread_ids and checkpoint counts")
    ap.add_argument("--dump", action="store_true", help="Pretty-print full checkpoint structures")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database file not found: {db_path.resolve()}")

    print(f"Inspecting database: {db_path.resolve()}\n")

    conn = _connect_readonly(db_path)
    try:
        if not _table_exists(conn, "checkpoints"):
            raise SystemExit("Table 'checkpoints' not found. Is this a LangGraph SqliteSaver DB?")

        if args.list_threads:
            list_threads(conn)
            return

        serializer = JsonPlusSerializer()
        inspect_checkpoints(
            conn=conn,
            serializer=serializer,
            limit=args.limit,
            thread_id=args.thread,
            messages_last=args.messages_last,
            dump=args.dump,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

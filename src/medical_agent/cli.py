"""Terminal entry point for the medical AI agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from medical_agent.graph import build_graph
from medical_agent.session import checkpoint_thread_id

DEFAULT_QUESTION = "请查询患者 P10086 的最新肾功能"
DEFAULT_CHECKPOINT_DB = Path(".medical-agent-data/checkpoints.sqlite")


def configure_terminal_encoding() -> None:
    """Use UTF-8 for Chinese text and medical units on Windows terminals."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical AI agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question sent to the agent",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep one process open for a checkpoint-backed multi-turn chat",
    )
    parser.add_argument(
        "--actor-id",
        default="doctor-chen",
        help="Authenticated actor identity for local CLI testing",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation identity used by the checkpointer",
    )
    return parser.parse_args()


def run_turn(
    graph,
    question: str,
    thread_id: str,
    actor_id: str = "doctor-chen",
) -> None:
    """Run one user turn while keeping prior state under the same thread ID."""

    request_id = str(uuid4())
    print(f"\nrequest_id: {request_id}")
    print(f"user: {question}\n")

    for update in graph.stream(
        {
            "messages": [HumanMessage(content=question)],
            "request_id": request_id,
            "actor_id": actor_id,
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 10,
        },
        stream_mode="updates",
    ):
        node_name, node_update = next(iter(update.items()))
        if node_name == "query_plan":
            print(
                f"[{node_name}] intent={node_update['intent']} "
                f"scope={node_update.get('query_scope')} "
                f"patient={node_update.get('patient_id')} "
                f"indicators={node_update.get('requested_indicator_codes', [])}"
            )
            continue
        if node_name == "authorize_patient":
            decision = "allowed" if node_update["authorization_allowed"] else "denied"
            print(f"[{node_name}] authorization -> {decision}")
            continue
        if node_name == "choose_result_source":
            if node_update["use_cached_result"]:
                source = "checkpoint cache"
            elif node_update["force_tool_call"]:
                source = "forced tool"
            else:
                source = "agent decision"
            print(f"[{node_name}] source -> {source}")
            continue

        message = node_update["messages"][-1]

        if isinstance(message, AIMessage) and message.tool_calls:
            call = message.tool_calls[0]
            print(f"[{node_name}] tool_call -> {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[{node_name}] observation -> {message.content}")
        elif isinstance(message, AIMessage):
            print(f"[{node_name}] final -> {message.content}")


def main() -> None:
    configure_terminal_encoding()
    load_dotenv()
    args = parse_args()
    thread_id = args.thread_id or str(uuid4())
    try:
        storage_thread_id = checkpoint_thread_id(args.actor_id, thread_id)
    except ValueError as exc:
        raise SystemExit(f"invalid session identity: {exc}") from exc

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    DEFAULT_CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(DEFAULT_CHECKPOINT_DB)) as checkpointer:
        try:
            graph = build_graph(checkpointer=checkpointer)
        except RuntimeError as exc:
            raise SystemExit(f"configuration error: {exc}") from exc

        print(f"model: {os.getenv('QWEN_MODEL', 'qwen-plus')}")
        print(f"actor_id: {args.actor_id}")
        print(f"thread_id: {thread_id}")
        print(f"checkpoint_db: {DEFAULT_CHECKPOINT_DB.resolve()}")

        if not args.interactive:
            run_turn(graph, args.question or DEFAULT_QUESTION, storage_thread_id, args.actor_id)
            return

        print("多轮模式已启动。输入 exit 结束；Checkpoint 会保存到本地 SQLite。")
        if args.question:
            run_turn(graph, args.question, storage_thread_id, args.actor_id)

        while True:
            try:
                question = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会话结束。")
                break

            if question.lower() in {"exit", "quit"}:
                print("会话结束。")
                break
            if question:
                run_turn(graph, question, storage_thread_id, args.actor_id)


if __name__ == "__main__":
    main()

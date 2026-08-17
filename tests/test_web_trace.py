from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from medical_agent import web


class _CheckpointerContext:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *args: object) -> None:
        return None


class _GraphWithNoOpUpdate:
    def stream(self, *args: Any, **kwargs: Any):
        yield {
            "query_plan": {
                "intent": "patient_data",
                "query_scope": "specific_indicators",
                "requested_indicator_codes": ["UREA"],
            }
        }
        yield {"patient_request": None}
        yield {
            "patient_id_required": {
                "messages": [AIMessage(content="请提供患者 ID，例如 P10086。")]
            }
        }


def test_web_trace_skips_noop_graph_updates(monkeypatch) -> None:
    monkeypatch.setattr(
        web.SqliteSaver,
        "from_conn_string",
        lambda *args, **kwargs: _CheckpointerContext(),
    )
    monkeypatch.setattr(
        web,
        "build_graph",
        lambda checkpointer: _GraphWithNoOpUpdate(),
    )

    answer, trace = web._run_agent(
        web.ChatRequest(thread_id="session-001", message="查询尿素"),
        "doctor-chen",
        "request-001",
    )

    assert answer == "请提供患者 ID，例如 P10086。"
    assert [step["node"] for step in trace] == ["query_plan", "patient_id_required"]

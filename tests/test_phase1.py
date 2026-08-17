import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from medical_agent.cli import configure_terminal_encoding
from medical_agent.models import create_model
from medical_agent.osteoporosis_db import (
    actor_can_access_patient,
    get_indicator_history,
    get_latest_indicators,
    initialize_osteoporosis_database,
    list_authorized_patients,
    list_synthetic_patients,
)
from medical_agent.session import checkpoint_thread_id
from medical_agent.tools import query_indicator_trend, query_patient_indicators
from tests.fakes import SemanticTestPlanner, build_test_graph


def test_tool_returns_known_synthetic_patient() -> None:
    result = query_patient_indicators.invoke(
        {"patient_id": "p10086", "indicator_codes": ["BMD_QCT"]}
    )

    assert result["status"] == "success"
    assert result["source"] == "synthetic_osteoporosis_sqlite"
    assert result["results"][0]["indicator_code"] == "BMD_QCT"


def test_latest_tool_returns_only_requested_indicator() -> None:
    result = query_patient_indicators.invoke({"patient_id": "P10086", "indicator_codes": ["CREA"]})

    assert result["indicator_codes"] == ["CREA"]
    assert [item["indicator_code"] for item in result["results"]] == ["CREA"]


def test_graph_renders_only_requested_indicator() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询 P10086 的肌酐")],
            "request_id": "creatinine-only-request",
            "actor_id": "doctor-chen",
        }
    )

    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert result["requested_indicator_codes"] == ["CREA"]
    assert '"indicator_code": "CREA"' in tool_message.content
    assert '"indicator_code": "EGFR"' not in tool_message.content
    assert "肌酐：" in result["messages"][-1].content
    assert "肾小球滤过率" not in result["messages"][-1].content


def test_graph_queries_osteoporosis_specific_bone_density() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询 P10086 的最新骨密度")],
            "request_id": "bone-density-request",
            "actor_id": "doctor-chen",
        }
    )

    assert result["requested_indicator_codes"] == ["BMD_QCT"]
    assert result["messages"][-3].tool_calls[0]["name"] == "query_patient_indicators"
    assert "骨密度：" in result["messages"][-1].content
    assert "mg/cm3" in result["messages"][-1].content


def test_graph_renders_only_egfr_in_trend_request() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查看 P30005 的 eGFR 历史趋势")],
            "request_id": "egfr-trend-only-request",
            "actor_id": "doctor-chen",
        }
    )

    tool_message = next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )
    assert result["requested_indicator_codes"] == ["EGFR"]
    assert '"indicator_code": "EGFR"' in tool_message.content
    assert '"indicator_code": "CREA"' not in tool_message.content
    assert "肾小球滤过率" in result["messages"][-1].content
    assert "肌酐" not in result["messages"][-1].content


def test_medical_database_contains_patients_and_history(tmp_path) -> None:
    database = tmp_path / "osteoporosis.sqlite"
    initialize_osteoporosis_database(database)

    patients = list_synthetic_patients(database)
    latest = get_latest_indicators("P30005", ["BMD_QCT"], database)

    assert len(patients) == 12
    assert latest is not None
    assert latest["results"][0]["indicator_code"] == "BMD_QCT"
    assert latest["results"][0]["visited_at"] == "2026-04-05"

    history = get_indicator_history("P30005", ["BMD_QCT"], database=database)
    assert len(history) == 4
    assert history[0]["visited_at"] < history[-1]["visited_at"]


def test_authorized_patient_lists_are_filtered(tmp_path) -> None:
    database = tmp_path / "authorization.sqlite"
    initialize_osteoporosis_database(database)
    chen_patients = list_authorized_patients("doctor-chen", database)
    lin_patients = list_authorized_patients("doctor-lin", database)

    assert {patient["patient_id"] for patient in chen_patients} == {
        "P10086",
        "P30002",
        "P30005",
        "P30009",
    }
    assert "P10086" not in {patient["patient_id"] for patient in lin_patients}
    assert actor_can_access_patient("doctor-chen", "P10086", database) is True
    assert actor_can_access_patient("doctor-lin", "P10086", database) is False


def test_graph_denies_unauthorized_patient_before_tool_call() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询患者 P10086 的最新肾功能")],
            "request_id": "unauthorized-request",
            "actor_id": "doctor-lin",
            "patient_id": None,
        }
    )

    assert result["authorization_allowed"] is False
    assert "无权访问" in result["messages"][-1].content
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_trend_tool_returns_chronological_history() -> None:
    result = query_indicator_trend.invoke(
        {"patient_id": "P30005", "indicator_codes": ["BMD_QCT"], "limit": 4}
    )

    assert result["query_type"] == "trend"
    assert len(result["results"]) == 4
    assert result["results"][-1]["results"][0]["indicator_code"] == "BMD_QCT"


def test_graph_routes_trend_request_to_trend_tool() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询 P30005 最近几个月的肾功能变化")],
            "request_id": "test-trend-request",
            "actor_id": "doctor-chen",
            "patient_id": None,
        }
    )

    assert result["query_kind"] == "trend"
    assert result["messages"][-3].tool_calls[0]["name"] == "query_indicator_trend"
    assert "2026-01-05" in result["messages"][-1].content
    assert "2026-04-05" in result["messages"][-1].content
    assert "不判断升降原因" in result["messages"][-1].content


def test_planner_test_double_reads_patient_from_structured_context() -> None:
    planner = SemanticTestPlanner()
    plan = planner.invoke(
        [
            HumanMessage(
                content=(
                    '{"latest_message":"查询这位患者的肾功能",'
                    '"current_patient_id":"P10086","has_previous_result":false}'
                )
            )
        ]
    )

    assert plan.patient_reference == "current"
    assert plan.indicator_codes == ["CREA", "UREA", "UA", "CO2CP", "EGFR"]


def test_graph_completes_protected_tool_loop() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询 P10086 的肾功能")],
            "request_id": "test-request-001",
            "actor_id": "doctor-chen",
            "patient_id": None,
        }
    )

    assert [type(message) for message in result["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert result["messages"][1].tool_calls[0]["name"] == "query_patient_indicators"
    assert "肌酐" in result["messages"][-1].content
    assert "肾小球滤过率" in result["messages"][-1].content
    assert result["patient_id"] == "P10086"


def test_safe_render_does_not_add_clinical_interpretation() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="查询 P10086 的肾功能")],
            "request_id": "test-request-safe-render",
            "actor_id": "doctor-chen",
            "patient_id": None,
        }
    )

    answer = result["messages"][-1].content
    assert "肌酐" in answer
    assert "肾小球滤过率" in answer
    assert "CKD" not in answer
    assert "3a" not in answer
    assert "参考范围通常" not in answer
    assert "诊断或治疗建议" in answer


def test_graph_stops_when_patient_id_is_missing() -> None:
    result = build_test_graph().invoke(
        {
            "messages": [HumanMessage(content="帮我查一下肾功能")],
            "request_id": "test-request-002",
            "actor_id": "doctor-chen",
            "patient_id": None,
        }
    )

    assert len(result["messages"]) == 2
    assert "患者 ID" in result["messages"][-1].content
    assert result["patient_id"] is None


def test_checkpoint_restores_patient_id_in_the_same_thread() -> None:
    graph = build_test_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "patient-session-001"}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="查询患者 P10086 的肾功能")],
            "request_id": "turn-001",
            "actor_id": "doctor-chen",
        },
        config=config,
    )
    second_turn = graph.invoke(
        {
            "messages": [HumanMessage(content="再查一下他的结果")],
            "request_id": "turn-002",
            "actor_id": "doctor-chen",
        },
        config=config,
    )

    assert second_turn["patient_id"] == "P10086"
    assert "P10086" in second_turn["messages"][-1].content
    assert sum(isinstance(message, HumanMessage) for message in second_turn["messages"]) == 2
    assert sum(isinstance(message, ToolMessage) for message in second_turn["messages"]) == 2
    assert second_turn["force_tool_call"] is True


def test_checkpoint_replays_cached_result_without_calling_tool() -> None:
    graph = build_test_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "patient-session-cache"}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="查询患者 P10086 的肾功能")],
            "request_id": "turn-001",
            "actor_id": "doctor-chen",
        },
        config=config,
    )
    replay = graph.invoke(
        {
            "messages": [HumanMessage(content="刚才的结果是什么？")],
            "request_id": "turn-002",
            "actor_id": "doctor-chen",
        },
        config=config,
    )

    assert replay["use_cached_result"] is True
    assert "上一次缓存的" in replay["messages"][-1].content
    assert sum(isinstance(message, ToolMessage) for message in replay["messages"]) == 1


def test_latest_request_forces_tool_call_without_model_discretion() -> None:
    graph = build_test_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "patient-session-refresh"}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="查询患者 P10086 的肾功能")],
            "request_id": "turn-001",
            "actor_id": "doctor-chen",
        },
        config=config,
    )
    refreshed = graph.invoke(
        {
            "messages": [HumanMessage(content="最新的呢？")],
            "request_id": "turn-002",
            "actor_id": "doctor-chen",
        },
        config=config,
    )

    assert refreshed["force_tool_call"] is True
    assert refreshed["messages"][-3].tool_calls[0]["name"] == "query_patient_indicators"
    assert isinstance(refreshed["messages"][-2], ToolMessage)
    assert "最新模拟" in refreshed["messages"][-1].content


def test_sqlite_checkpoint_survives_graph_restart(tmp_path) -> None:
    database = tmp_path / "checkpoints.sqlite"
    config = {"configurable": {"thread_id": "persistent-patient-session"}}

    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        first_graph = build_test_graph(checkpointer=checkpointer)
        first_graph.invoke(
            {
                "messages": [HumanMessage(content="查询患者 P10086 的肾功能")],
                "request_id": "turn-001",
                "actor_id": "doctor-chen",
            },
            config=config,
        )

    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        restarted_graph = build_test_graph(checkpointer=checkpointer)
        resumed = restarted_graph.invoke(
            {
                "messages": [HumanMessage(content="刚才的结果是什么？")],
                "request_id": "turn-002",
                "actor_id": "doctor-chen",
            },
            config=config,
        )

    assert resumed["patient_id"] == "P10086"
    assert resumed["use_cached_result"] is True
    assert "上一次缓存的" in resumed["messages"][-1].content


def test_actor_identity_isolates_the_same_public_thread_id() -> None:
    graph = build_test_graph(checkpointer=InMemorySaver())
    doctor_a = {"configurable": {"thread_id": checkpoint_thread_id("doctor-chen", "shared-name")}}
    doctor_b = {"configurable": {"thread_id": checkpoint_thread_id("doctor-lin", "shared-name")}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="查询患者 P10086 的肾功能")],
            "request_id": "doctor-a-turn",
            "actor_id": "doctor-chen",
        },
        config=doctor_a,
    )
    isolated = graph.invoke(
        {
            "messages": [HumanMessage(content="刚才的结果是什么？")],
            "request_id": "doctor-b-turn",
            "actor_id": "doctor-lin",
        },
        config=doctor_b,
    )

    assert isolated.get("patient_id") is None
    assert isolated.get("lab_result") is None
    assert isolated["intent"] == "unclear"
    assert isolated["messages"][-1].content == "我不明白您的意思，请明确一下。"


def test_session_identity_rejects_ambiguous_ids() -> None:
    with pytest.raises(ValueError, match="actor_id"):
        checkpoint_thread_id("doctor::admin", "session-001")


def test_terminal_encoding_is_utf8() -> None:
    configure_terminal_encoding()

    assert __import__("sys").stdout.encoding.lower() == "utf-8"


def test_qwen_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        create_model()

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from medical_agent.state import QueryPlan
from tests.fakes import StaticPlanner, build_test_graph


def test_query_plan_rejects_medical_scope_for_general_chat() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="general_chat",
            scope="specific_indicators",
            indicator_codes=["RBC"],
        )


def test_query_plan_allows_clarification_without_field_scope() -> None:
    plan = QueryPlan(
        intent="patient_data",
        scope="none",
        needs_clarification=True,
    )

    assert plan.needs_clarification is True


@pytest.mark.parametrize(
    ("text", "plan", "expected_text"),
    [
        ("你好", QueryPlan(intent="general_chat"), "你好"),
        ("好的知道了", QueryPlan(intent="general_chat"), "测试用通用对话回答"),
        (
            "有什么问题吗",
            QueryPlan(intent="clinical_interpretation"),
            "不能仅根据当前展示的单项数据判断",
        ),
        (
            "我能查询哪些病人",
            QueryPlan(intent="patient_access_list"),
            "P10086",
        ),
        (
            "你能做什么",
            QueryPlan(intent="capability_question"),
            "需要我帮你查询吗",
        ),
        ("帮我处理一下", QueryPlan(intent="unclear"), "请明确一下"),
    ],
)
def test_non_patient_intents_route_without_medical_tools(
    text: str,
    plan: QueryPlan,
    expected_text: str,
) -> None:
    result = build_test_graph(planner=StaticPlanner(plan)).invoke(
        {
            "messages": [HumanMessage(content=text)],
            "request_id": "non-patient-plan",
            "actor_id": "doctor-chen",
        }
    )

    assert expected_text in result["messages"][-1].content
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_invalid_planner_output_fails_closed() -> None:
    result = build_test_graph(
        planner=StaticPlanner({"intent": "patient_data", "scope": "made_up"})
    ).invoke(
        {
            "messages": [HumanMessage(content="查询一些东西")],
            "request_id": "invalid-plan",
            "actor_id": "doctor-chen",
        }
    )

    assert result["intent"] == "unclear"
    assert result["messages"][-1].content == "我不明白您的意思，请明确一下。"


def test_planner_receives_current_patient_and_prior_query_context() -> None:
    planner = StaticPlanner(
        QueryPlan(
            intent="patient_data",
            patient_reference="explicit",
            scope="specific_indicators",
            indicator_codes=["UREA"],
        ),
        QueryPlan(
            intent="patient_data",
            patient_reference="current",
            scope="specific_indicators",
            indicator_codes=["RBC"],
        ),
    )
    graph = build_test_graph(checkpointer=InMemorySaver(), planner=planner)
    config = {"configurable": {"thread_id": "context-plan"}}
    graph.invoke(
        {
            "messages": [HumanMessage(content="查患者P10086关于尿素的")],
            "request_id": "context-first",
            "actor_id": "doctor-chen",
        },
        config=config,
    )
    second = graph.invoke(
        {
            "messages": [HumanMessage(content="红细胞的呢")],
            "request_id": "context-second",
            "actor_id": "doctor-chen",
        },
        config=config,
    )

    assert planner.contexts[1]["current_patient_id"] == "P10086"
    assert planner.contexts[1]["previous_indicator_codes"] == ["UREA"]
    assert planner.contexts[1]["recent_user_messages"] == [
        "查患者P10086关于尿素的",
        "红细胞的呢",
    ]
    assert second["requested_indicator_codes"] == ["RBC"]
    assert "红细胞：" in second["messages"][-1].content


def test_patient_query_without_patient_id_requests_it_deterministically() -> None:
    plan = QueryPlan(
        intent="patient_data",
        patient_reference="none",
        scope="specific_indicators",
        indicator_codes=["UREA"],
    )

    result = build_test_graph(planner=StaticPlanner(plan)).invoke(
        {
            "messages": [HumanMessage(content="查关于尿素的")],
            "request_id": "patient-required",
            "actor_id": "doctor-chen",
        }
    )

    assert "请提供患者 ID" in result["messages"][-1].content
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from medical_agent.indicator_catalog import INDICATOR_CATALOG
from medical_agent.planner import make_query_planner_node
from medical_agent.state import QueryPlan
from tests.fakes import StaticPlanner, build_test_graph


@pytest.mark.parametrize(
    ("plan", "expected_codes", "expected_kind"),
    [
        (
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="specific_indicators",
                indicator_codes=["RBC"],
            ),
            ["RBC"],
            "latest",
        ),
        (
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="category",
                category_codes=["RENAL"],
                time_scope="trend",
            ),
            ["CREA", "UREA", "UA", "CO2CP", "EGFR"],
            "trend",
        ),
        (
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="all_indicators",
            ),
            [item.code for item in INDICATOR_CATALOG.definitions],
            "latest",
        ),
        (
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="full_record",
                time_scope="all_visits",
            ),
            [item.code for item in INDICATOR_CATALOG.definitions],
            "full_record",
        ),
    ],
)
def test_query_plan_resolves_only_allowlisted_scope(
    plan: QueryPlan,
    expected_codes: list[str],
    expected_kind: str,
) -> None:
    node = make_query_planner_node(StaticPlanner(plan))

    result = node(
        {
            "messages": [HumanMessage(content="给我患者p10086的请求")],
            "request_id": "scope-resolution",
            "actor_id": "doctor-chen",
        }
    )

    assert result["patient_id"] == "P10086"
    assert result["requested_indicator_codes"] == expected_codes
    assert result["query_kind"] == expected_kind


def test_hallucinated_indicator_code_fails_closed() -> None:
    plan = QueryPlan(
        intent="patient_data",
        patient_reference="explicit",
        scope="specific_indicators",
        indicator_codes=["NOT_A_REAL_FIELD"],
    )

    result = make_query_planner_node(StaticPlanner(plan))(
        {"messages": [HumanMessage(content="查询 P10086 的未知字段")]}
    )

    assert result["intent"] == "unclear"
    assert result["requested_indicator_codes"] == []


@pytest.mark.parametrize(
    ("text", "plan", "expected_fragment"),
    [
        (
            "查患者P10086关于尿素的",
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="specific_indicators",
                indicator_codes=["UREA"],
            ),
            "尿素：",
        ),
        (
            "查询P10086所有肾功能指标",
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="category",
                category_codes=["RENAL"],
            ),
            "肾小球滤过率",
        ),
        (
            "显示P10086的全部指标",
            QueryPlan(
                intent="patient_data",
                patient_reference="explicit",
                scope="all_indicators",
            ),
            "雌二醇：",
        ),
    ],
)
def test_graph_executes_semantic_plan(
    text: str,
    plan: QueryPlan,
    expected_fragment: str,
) -> None:
    result = build_test_graph(planner=StaticPlanner(plan)).invoke(
        {
            "messages": [HumanMessage(content=text)],
            "request_id": "semantic-plan-execution",
            "actor_id": "doctor-chen",
        }
    )

    assert expected_fragment in result["messages"][-1].content
    assert any(isinstance(message, ToolMessage) for message in result["messages"])


def test_full_record_plan_returns_all_visits_without_direct_identifiers() -> None:
    plan = QueryPlan(
        intent="patient_data",
        patient_reference="explicit",
        scope="full_record",
        time_scope="all_visits",
    )
    result = build_test_graph(planner=StaticPlanner(plan)).invoke(
        {
            "messages": [HumanMessage(content="给我患者p10086的全部信息")],
            "request_id": "full-record-plan",
            "actor_id": "doctor-chen",
        }
    )

    payload = json.loads(
        next(message for message in result["messages"] if isinstance(message, ToolMessage)).content
    )
    assert len(payload["visits"]) == 4
    assert all(len(visit["indicators"]) == 54 for visit in payload["visits"])
    assert "contact" not in payload["patient"]
    assert "identity_card" not in payload["patient"]
    assert "既往史：骨质疏松症（模拟）" in result["messages"][-1].content

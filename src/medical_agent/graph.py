"""Context-aware semantic planning followed by protected deterministic tools."""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from medical_agent.models import create_model, create_query_planner_model
from medical_agent.nodes import (
    authorize_patient_access,
    choose_result_source,
    force_lab_tool_call,
    make_general_chat_node,
    render_access_denied,
    render_cached_lab_result,
    render_capability_response,
    render_clinical_interpretation_boundary,
    render_indicator_clarification,
    render_patient_access_list,
    render_patient_id_required,
    render_unclear_response,
    safe_render_lab_result,
)
from medical_agent.planner import make_query_planner_node
from medical_agent.state import MedicalAgentState
from medical_agent.tools import MEDICAL_TOOLS


def route_by_intent(
    state: MedicalAgentState,
) -> Literal[
    "patient_request",
    "patient_access_response",
    "capability_response",
    "interpretation_response",
    "unclear_response",
    "general_chat",
]:
    routes = {
        "patient_data": "patient_request",
        "patient_access_list": "patient_access_response",
        "capability_question": "capability_response",
        "clinical_interpretation": "interpretation_response",
        "unclear": "unclear_response",
        "general_chat": "general_chat",
    }
    return routes[state["intent"]]


def route_patient_request(
    state: MedicalAgentState,
) -> Literal["patient_id_required", "indicator_clarification", "authorize_patient"]:
    if not state.get("patient_id"):
        return "patient_id_required"
    if not state.get("requested_indicator_codes"):
        return "indicator_clarification"
    return "authorize_patient"


def route_after_authorization(
    state: MedicalAgentState,
) -> Literal["choose_result_source", "access_denied"]:
    return "choose_result_source" if state.get("authorization_allowed") else "access_denied"


def route_by_result_source(
    state: MedicalAgentState,
) -> Literal["cached_render", "force_tool_call"]:
    return "cached_render" if state.get("use_cached_result") else "force_tool_call"


def build_graph(
    checkpointer: Any | None = None,
    *,
    planner_model: Any | None = None,
    general_model: Any | None = None,
):
    """Build the production Qwen graph, with explicit dependency seams for tests."""

    planner = planner_model or create_query_planner_model()
    chat_model = general_model or create_model()

    builder = StateGraph(MedicalAgentState)
    builder.add_node("query_plan", make_query_planner_node(planner))
    builder.add_node("patient_access_response", render_patient_access_list)
    builder.add_node("capability_response", render_capability_response)
    builder.add_node("interpretation_response", render_clinical_interpretation_boundary)
    builder.add_node("unclear_response", render_unclear_response)
    builder.add_node("general_chat", make_general_chat_node(chat_model))
    builder.add_node("patient_request", lambda state: {})
    builder.add_node("patient_id_required", render_patient_id_required)
    builder.add_node("indicator_clarification", render_indicator_clarification)
    builder.add_node("authorize_patient", authorize_patient_access)
    builder.add_node("access_denied", render_access_denied)
    builder.add_node("choose_result_source", choose_result_source)
    builder.add_node("force_tool_call", force_lab_tool_call)
    builder.add_node("tools", ToolNode(MEDICAL_TOOLS))
    builder.add_node("safe_render", safe_render_lab_result)
    builder.add_node("cached_render", render_cached_lab_result)

    builder.add_edge(START, "query_plan")
    builder.add_conditional_edges("query_plan", route_by_intent)
    builder.add_conditional_edges("patient_request", route_patient_request)
    builder.add_edge("patient_access_response", END)
    builder.add_edge("capability_response", END)
    builder.add_edge("interpretation_response", END)
    builder.add_edge("unclear_response", END)
    builder.add_edge("general_chat", END)
    builder.add_edge("patient_id_required", END)
    builder.add_edge("indicator_clarification", END)
    builder.add_conditional_edges("authorize_patient", route_after_authorization)
    builder.add_edge("access_denied", END)
    builder.add_conditional_edges("choose_result_source", route_by_result_source)
    builder.add_edge("force_tool_call", "tools")
    builder.add_edge("tools", "safe_render")
    builder.add_edge("safe_render", END)
    builder.add_edge("cached_render", END)

    return builder.compile(checkpointer=checkpointer)

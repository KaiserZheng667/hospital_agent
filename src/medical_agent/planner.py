"""Context-aware LLM query planning with deterministic safety validation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from medical_agent.indicator_catalog import INDICATOR_CATALOG
from medical_agent.state import MedicalAgentState, QueryPlan

PATIENT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])P\d+(?![A-Za-z0-9])", re.IGNORECASE)
_CATALOG_ROWS = "\n".join(
    f"- {item.code}: {item.name}; category={item.category}; aliases={','.join(item.aliases)}"
    for item in INDICATOR_CATALOG.definitions
)

QUERY_PLANNER_PROMPT = f"""You are the semantic query planner for a Chinese medical-data assistant.
Return exactly one JSON object matching the supplied QueryPlan schema. Do not answer the user.

Use the supplied conversation context, especially current_patient_id and previous query metadata.
Resolve natural paraphrases and short follow-ups such as “红细胞的呢” against that context.

Intent values:
- patient_data: retrieve one patient's authorized clinical data.
- patient_access_list: ask which patients the current signed-in account is authorized to access.
- capability_question: ask what this system can do, without requesting execution.
- clinical_interpretation: ask whether a patient result is normal, abnormal, serious, problematic,
  diagnostic, or what treatment should follow. This is not a retrieval request.
- general_chat: greetings, medical-term explanations, ID-format explanations, or ordinary chat.
- unclear: genuinely ambiguous after considering conversation context.

Patient reference:
- explicit when the latest message contains a patient number.
- current when it refers to the patient already in context.
- none when no patient is selected. Never invent or output a patient number.
An otherwise clear retrieval request does not become unclear merely because the patient number is
missing. Return patient_data with patient_reference=none; the graph will ask for the patient ID.

Scope values:
- specific_indicators: one or more particular indicators; return allowlisted indicator_codes.
- category: a complete category; return category_codes.
- all_indicators: every indicator, but not demographics or visit narrative fields.
- full_record: patient profile plus all visits and all indicators. Use time_scope=all_visits.
- none: required for every non-patient intent.

Time scope:
- latest for the newest result.
- trend for history/change across visits. Wording such as “历次”, “历史”, “变化”, “趋势”,
  or a comparison across dates requires trend, even when the word “趋势” is absent.
- all_visits only for full_record.

Distinguish retrieval from explanation: “红细胞的呢” with a current patient is retrieval, while
“红细胞是什么意思” is general_chat. “给我患者P10086的全部信息” is full_record.
Pure acknowledgements, thanks, or conversation-closing messages such as “ok”, “好的”, “知道了”,
“好的知道了”, and “谢谢” are general_chat. Do not reuse the previous patient query merely because
patient and result context exist. Reuse prior query metadata only when the latest message actually
asks to replay, refresh, continue, compare, or retrieve a field.
Questions such as “有什么问题吗”, “正常吗”, “有异常吗”, “严重吗”, or “该怎么治疗” after
a patient result are clinical_interpretation. Do not repeat or refresh the previous query for them.
Distinguish system capability from account scope: “你能查询哪些内容” is capability_question,
while “我能查询哪些病人/患者” or “我的患者名单” is patient_access_list.
For “刚才的结果/复述一下”, reuse previous_query_scope, previous_query_kind and
previous_indicator_codes so the checkpoint cache can be replayed. For “最新的呢/刷新”, reuse the
previous field scope but use time_scope=latest.

Examples:
- latest_message=“查患者P10086关于尿素的” -> patient_data, explicit,
  specific_indicators, [UREA], latest.
- latest_message=“帮我查一下尿素” with no current patient -> patient_data, none,
  specific_indicators, [UREA], latest.
- current_patient_id=P10086 and latest_message=“红细胞的呢” -> patient_data, current,
  specific_indicators, [RBC], latest.
- latest_message=“给我患者p10086的全部信息” -> patient_data, explicit, full_record,
  [], [], all_visits.
- latest_message=“查看P30005历次骨密度” -> patient_data, explicit,
  specific_indicators, [BMD_QCT], trend.
- latest_message=“所有肾功能指标” with a current patient -> patient_data, current, category,
  [], [RENAL], latest.
- latest_message=“红细胞是什么意思” -> general_chat, none, none, [], [], latest.
- latest_message=“好的知道了” after a patient result -> general_chat, none, none, [], [], latest.
- latest_message=“有什么问题吗” after a patient result -> clinical_interpretation, none, none,
  [], [], latest.
- latest_message=“我能查询哪些病人” -> patient_access_list, none, none, [], [], latest.
Use only these categories: BONE, CBC, URINE, LIVER, RENAL, CYTOKINE, HORMONE.
Use only these indicator codes:
{_CATALOG_ROWS}
"""


def _latest_user_text(state: MedicalAgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def _planner_context(state: MedicalAgentState) -> dict[str, Any]:
    recent_user_messages = [
        str(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage)
    ][-4:]
    return {
        "latest_message": recent_user_messages[-1] if recent_user_messages else "",
        "recent_user_messages": recent_user_messages,
        "current_patient_id": state.get("patient_id"),
        "previous_query_scope": state.get("query_scope"),
        "previous_query_kind": state.get("query_kind"),
        "previous_indicator_codes": state.get("requested_indicator_codes", []),
        "has_previous_result": bool(state.get("lab_result")),
    }


def _unwrap_plan(result: Any) -> QueryPlan:
    if isinstance(result, dict) and "parsed" in result:
        if result.get("parsing_error") is not None:
            raise ValueError("query plan parsing failed")
        result = result.get("parsed")
    return result if isinstance(result, QueryPlan) else QueryPlan.model_validate(result)


def make_query_planner_node(planner_model: Any) -> Callable[[MedicalAgentState], dict[str, Any]]:
    """Create one semantic planner that replaces separate intent and indicator routers."""

    def query_planner_node(state: MedicalAgentState) -> dict[str, Any]:
        context = _planner_context(state)
        try:
            result = planner_model.invoke(
                [
                    SystemMessage(content=QUERY_PLANNER_PROMPT),
                    HumanMessage(content=json.dumps(context, ensure_ascii=False)),
                ]
            )
            plan = _unwrap_plan(result)
        except (TypeError, ValidationError, ValueError):
            return {
                "intent": "unclear",
                "query_scope": "none",
                "requested_indicator_codes": [],
            }

        output: dict[str, Any] = {
            "intent": plan.intent,
            "query_scope": plan.scope,
            "requested_indicator_codes": [],
        }
        if plan.intent != "patient_data":
            return output

        latest_text = _latest_user_text(state)
        explicit_match = PATIENT_ID_PATTERN.search(latest_text)
        if explicit_match:
            output["patient_id"] = explicit_match.group(0).upper()
        elif plan.patient_reference == "current" and state.get("patient_id"):
            output["patient_id"] = state["patient_id"]

        try:
            if plan.scope in {"all_indicators", "full_record"}:
                selected = INDICATOR_CATALOG.definitions
            else:
                selected = INDICATOR_CATALOG.expand(
                    indicator_codes=plan.indicator_codes,
                    category_codes=plan.category_codes,
                )
        except ValueError:
            return {
                **output,
                "intent": "unclear",
                "query_scope": "none",
                "requested_indicator_codes": [],
            }
        output["requested_indicator_codes"] = [item.code for item in selected]
        output["query_kind"] = (
            "full_record"
            if plan.scope == "full_record"
            else "trend"
            if plan.time_scope == "trend"
            else "latest"
        )
        return output

    return query_planner_node

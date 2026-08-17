"""Deterministic execution, authorization, and safe-rendering nodes."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from medical_agent.repository import get_medical_repository
from medical_agent.state import MedicalAgentState

GENERAL_CHAT_PROMPT = """You are the conversational front desk of a medical data teaching system.
Answer in Chinese. Handle greetings, explanations, and ordinary conversation.
Do not claim that you queried patient data, and do not invent patient records.
Do not diagnose, prescribe, or claim that the output replaces clinical judgment.
"""
CACHE_REPLAY_PHRASES = ("刚才", "刚刚", "上次", "之前", "重复", "再说一遍", "复述")
REFRESH_PHRASES = ("最新", "重新查", "再查", "刷新", "现在")


def render_capability_response(state: MedicalAgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "我目前可以查询有权限访问的合成骨质疏松患者基本资料、访视记录、"
                    "单项或分类指标、全部指标以及历史趋势。数据仅用于教学演示。"
                    "需要我帮你查询吗？"
                )
            )
        ]
    }


def render_patient_access_list(state: MedicalAgentState) -> dict:
    actor_id = state.get("actor_id")
    patients = (
        get_medical_repository().list_authorized_patients(actor_id) if actor_id else []
    )
    if not patients:
        return {
            "messages": [
                AIMessage(content="当前账号没有已分配的可查询患者，请联系管理员配置患者权限。")
            ]
        }
    lines = [f"当前账号可以查询以下 {len(patients)} 名患者："]
    lines.extend(
        f"- {patient['patient_id']} · {patient.get('display_name') or '姓名未提供'}"
        for patient in patients
    )
    lines.append("患者访问范围由管理员分配。")
    return {"messages": [AIMessage(content="\n".join(lines))]}


def render_clinical_interpretation_boundary(state: MedicalAgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "我不能仅根据当前展示的单项数据判断是否存在问题，也不能据此作出诊断、"
                    "疾病分期或治疗建议。当前系统可以继续为你调取有权限的历史趋势或相关指标；"
                    "临床判断需要结合参考范围、症状、病史、检查条件和其他检查结果。"
                )
            )
        ]
    }


def render_unclear_response(state: MedicalAgentState) -> dict:
    return {"messages": [AIMessage(content="我不明白您的意思，请明确一下。")]}


def make_general_chat_node(model: Any) -> Callable[[MedicalAgentState], dict]:
    def general_chat_node(state: MedicalAgentState) -> dict:
        response = model.invoke([SystemMessage(content=GENERAL_CHAT_PROMPT), *state["messages"]])
        return {"messages": [response]}

    return general_chat_node


def authorize_patient_access(state: MedicalAgentState) -> dict:
    patient_id = state.get("patient_id")
    actor_id = state.get("actor_id")
    allowed = bool(
        patient_id
        and actor_id
        and get_medical_repository().actor_can_access_patient(actor_id, patient_id)
    )
    return {
        "authorization_allowed": allowed,
        "authorization_reason": "care_team_assignment" if allowed else "access_denied",
    }


def render_indicator_clarification(state: MedicalAgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "我没有识别出需要查询的具体指标。请明确指标名称、类别、全部指标或完整资料。"
                )
            )
        ]
    }


def render_patient_id_required(state: MedicalAgentState) -> dict:
    return {"messages": [AIMessage(content="请提供患者 ID，例如 P10086。")]}


def render_access_denied(state: MedicalAgentState) -> dict:
    return {
        "messages": [
            AIMessage(content="当前身份无权访问该患者数据。此次请求未调用模型生成医疗结果，也未查询患者记录。")
        ]
    }


def _latest_user_text(state: MedicalAgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def choose_result_source(state: MedicalAgentState) -> dict:
    latest_user_text = _latest_user_text(state)
    cached_result = state.get("lab_result")
    requested_codes = state.get("requested_indicator_codes", [])
    cached_codes = cached_result.get("indicator_codes", []) if cached_result else []
    query_kind = state.get("query_kind", "latest")
    same_patient = bool(
        cached_result and cached_result.get("patient_id") == state.get("patient_id")
    )
    same_scope = bool(
        cached_result
        and cached_codes == requested_codes
        and cached_result.get("query_type") == query_kind
    )
    asks_for_refresh = any(phrase in latest_user_text for phrase in REFRESH_PHRASES)
    asks_for_replay = any(phrase in latest_user_text for phrase in CACHE_REPLAY_PHRASES)
    use_cached_result = bool(
        same_patient and same_scope and asks_for_replay and not asks_for_refresh
    )
    return {
        "query_kind": query_kind,
        "use_cached_result": use_cached_result,
        "force_tool_call": not use_cached_result,
    }


def force_lab_tool_call(state: MedicalAgentState) -> dict:
    patient_id = state["patient_id"]
    query_kind = state.get("query_kind", "latest")
    indicator_codes = state.get("requested_indicator_codes", [])
    if query_kind == "full_record":
        tool_name = "query_patient_full_record"
        tool_args = {"patient_id": patient_id}
    elif query_kind == "trend":
        tool_name = "query_indicator_trend"
        tool_args = {
            "patient_id": patient_id,
            "indicator_codes": indicator_codes,
            "limit": 4,
        }
    else:
        tool_name = "query_patient_indicators"
        tool_args = {"patient_id": patient_id, "indicator_codes": indicator_codes}
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"fresh-medical-{state['request_id']}",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def _render_full_record(payload: dict, *, from_cache: bool) -> AIMessage:
    patient_id = payload["patient_id"]
    patient = payload.get("patient", {})
    heading = "上一次缓存的完整临床资料" if from_cache else "完整模拟临床资料"
    lines = [
        f"患者 {patient_id} {heading}：",
        "基本资料：",
        f"- 姓名：{patient.get('name', '未提供')}",
        f"- 性别：{patient.get('sex', '未提供')}",
        f"- 入组序号：{patient.get('source_sequence', '未提供')}",
        f"- 知情同意日期：{patient.get('consented_at', '未提供')}",
        f"- 合成数据：{'是' if patient.get('is_synthetic') else '否'}",
    ]
    for visit in payload.get("visits", []):
        blood_pressure = (
            f"{visit.get('systolic_bp') or '未提供'}/"
            f"{visit.get('diastolic_bp') or '未提供'} mmHg"
        )
        lines.extend(
            [
                f"访视：{visit.get('visited_at', '时间未知')} {visit.get('visit_label', '')}",
                f"- 研究者：{visit.get('researcher') or '未提供'}",
                f"- 访视者：{visit.get('visitor') or '未提供'}",
                f"- 年龄：{visit.get('age_years') or '未提供'} 岁",
                f"- 体重：{visit.get('weight_kg') or '未提供'} kg",
                f"- 身高：{visit.get('height_m') or '未提供'} m",
                f"- 血压：{blood_pressure}",
                f"- 脉搏：{visit.get('pulse') or '未提供'} 次/分",
                f"- 中医证候：{visit.get('tcm_syndrome') or '未提供'}",
                f"- 中医症状：{visit.get('tcm_symptoms') or '未提供'}",
                f"- 治疗记录：{visit.get('treatment') or '未提供'}",
                f"- 既往史：{visit.get('past_history') or '未提供'}",
                f"- 家族史：{visit.get('family_history') or '未提供'}",
                f"- 抗生素使用史：{visit.get('antibiotic_history') or '未提供'}",
                f"- 研究分组：{visit.get('study_group') or '未提供'}",
                "- 指标：",
            ]
        )
        for result in visit.get("indicators", []):
            unit = f" {result['standard_unit']}" if result.get("standard_unit") else ""
            lines.append(f"  - {result.get('name', '未知指标')}：{result.get('value_text', '未提供')}{unit}")
    lines.extend(
        [
            f"- 数据源标记：{payload.get('source', '数据源未提供')}",
            "联系方式和身份证号未包含在普通临床读取权限中。",
            "以上为教学用合成资料，只复述数据源字段，不提供诊断或治疗建议。",
        ]
    )
    return AIMessage(content="\n".join(lines))


def _render_payload(payload: dict, *, from_cache: bool) -> AIMessage:
    patient_id = str(payload.get("patient_id", "未知"))
    if payload.get("status") != "success":
        return AIMessage(content=f"未查到患者 {patient_id} 的模拟记录。")
    if payload.get("query_type") == "full_record":
        return _render_full_record(payload, from_cache=from_cache)
    if payload.get("query_type") == "trend":
        heading = "上一次缓存的指标历史" if from_cache else "模拟指标历史"
        lines = [f"患者 {patient_id} {heading}（按访视时间顺序）："]
        for visit in payload.get("results", []):
            lines.append(f"- {visit.get('visited_at', '时间未知')} {visit.get('visit_label', '')}：")
            for result in visit.get("results", []):
                unit = f" {result['standard_unit']}" if result.get("standard_unit") else ""
                lines.append(f"  - {result.get('name', '未知指标')}：{result.get('value_text', '未提供')}{unit}")
        lines.append("以上仅按时间列出模拟数据源字段，不判断升降原因，不提供疾病分期、诊断或治疗建议。")
        return AIMessage(content="\n".join(lines))
    result_kind = "上一次缓存的" if from_cache else "最新模拟"
    lines = [f"患者 {patient_id} {result_kind}指标结果："]
    for result in payload.get("results", []):
        unit = f" {result['standard_unit']}" if result.get("standard_unit") else ""
        lines.append(
            f"- {result.get('name', '未知指标')}：{result.get('value_text', '未提供')}{unit}"
            f"（{result.get('visited_at', '时间未知')}，{result.get('visit_label', '访视点未知')}）"
        )
    lines.extend(
        [
            f"- 数据源标记：{payload.get('source', '数据源未提供')}",
            "以上内容仅复述模拟数据源提供的字段，不提供参考范围、疾病分期、诊断或治疗建议。",
        ]
    )
    return AIMessage(content="\n".join(lines))


def safe_render_lab_result(state: MedicalAgentState) -> dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        return {"messages": [AIMessage(content="无法生成结果：未收到有效的医疗工具返回。")]}
    try:
        payload = json.loads(str(last_message.content))
    except (TypeError, json.JSONDecodeError):
        return {"messages": [AIMessage(content="无法生成结果：医疗工具返回格式无效。")]}
    return {"lab_result": payload, "messages": [_render_payload(payload, from_cache=False)]}


def render_cached_lab_result(state: MedicalAgentState) -> dict:
    payload = state.get("lab_result")
    if not payload:
        return {"messages": [AIMessage(content="当前会话没有可复述的医疗结果。")]}
    return {"messages": [_render_payload(payload, from_cache=True)]}

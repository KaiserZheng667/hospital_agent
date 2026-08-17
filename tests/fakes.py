from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from medical_agent.graph import build_graph
from medical_agent.indicator_catalog import INDICATOR_CATALOG
from medical_agent.planner import PATIENT_ID_PATTERN
from medical_agent.state import QueryPlan


class FakeGeneralModel:
    def invoke(self, messages: list[Any]) -> AIMessage:
        latest = str(messages[-1].content)
        if "你好" in latest or "您好" in latest:
            return AIMessage(content="你好。我是医疗数据助手，请问有什么可以帮助你？")
        if "格式" in latest or "是什么意思" in latest:
            return AIMessage(content="这是普通解释请求，没有读取患者数据。")
        return AIMessage(content="测试用通用对话回答。")


class SemanticTestPlanner:
    """Offline dependency-injected test double; it is not a runtime mode."""

    def invoke(self, messages: list[Any]) -> QueryPlan:
        context = json.loads(str(messages[-1].content))
        latest = context["latest_message"]
        normalized = latest.casefold()
        current_patient = context.get("current_patient_id")
        patient_reference = (
            "explicit"
            if PATIENT_ID_PATTERN.search(latest)
            else "current"
            if current_patient
            else "none"
        )
        if normalized.strip() in {"ok", "好的", "知道了", "好的知道了", "谢谢", "明白了"}:
            return QueryPlan(intent="general_chat")
        if any(word in normalized for word in ("有什么问题", "正常吗", "有异常", "严重吗", "怎么治疗")):
            return QueryPlan(intent="clinical_interpretation")
        if any(word in normalized for word in ("哪些病人", "哪些患者", "患者名单", "病人名单")):
            return QueryPlan(intent="patient_access_list")
        if any(word in normalized for word in ("你好", "您好", "是什么意思", "什么格式")):
            return QueryPlan(intent="general_chat", patient_reference=patient_reference)
        if any(word in normalized for word in ("能做什么", "有什么功能", "可以查询什么")):
            return QueryPlan(intent="capability_question", patient_reference=patient_reference)

        previous_codes = context.get("previous_indicator_codes", [])
        is_replay = any(word in normalized for word in ("刚才", "刚刚", "上次", "之前"))
        is_refresh = any(word in normalized for word in ("再查", "最新的呢", "刷新"))
        if (is_replay or is_refresh) and not current_patient:
            return QueryPlan(intent="unclear")
        if is_replay or is_refresh:
            codes = previous_codes
        else:
            selected_categories = {
                category
                for category, aliases in {
                    "BONE": ("骨代谢", "骨指标"),
                    "CBC": ("血常规",),
                    "URINE": ("尿常规", "尿检"),
                    "LIVER": ("肝功能",),
                    "RENAL": ("肾功能",),
                    "CYTOKINE": ("细胞因子",),
                    "HORMONE": ("激素", "性激素"),
                }.items()
                if any(alias in normalized for alias in aliases)
            }
            selected = [
                item.code
                for item in INDICATOR_CATALOG.definitions
                if any(
                    alias.casefold() in normalized
                    for alias in (item.code, item.name, *item.aliases)
                )
                or item.category in selected_categories
            ]
            codes = selected

        if any(word in normalized for word in ("全部信息", "全部资料", "完整病历")):
            return QueryPlan(
                intent="patient_data",
                patient_reference=patient_reference,
                scope="full_record",
                time_scope="all_visits",
            )
        if any(word in normalized for word in ("全部指标", "所有指标", "全部结果")):
            return QueryPlan(
                intent="patient_data",
                patient_reference=patient_reference,
                scope="all_indicators",
            )
        if not codes:
            return QueryPlan(
                intent="patient_data",
                patient_reference=patient_reference,
                scope="none",
                needs_clarification=True,
            )
        return QueryPlan(
            intent="patient_data",
            patient_reference=patient_reference,
            scope="specific_indicators",
            indicator_codes=codes,
            time_scope=(
                "trend"
                if any(word in normalized for word in ("趋势", "变化", "历史", "最近几个月"))
                else "latest"
            ),
        )


class StaticPlanner:
    def __init__(self, *plans: QueryPlan | dict[str, Any]) -> None:
        self.plans = list(plans)
        self.contexts: list[dict[str, Any]] = []

    def invoke(self, messages: list[Any]) -> QueryPlan | dict[str, Any]:
        self.contexts.append(json.loads(str(messages[-1].content)))
        if not self.plans:
            raise AssertionError("StaticPlanner has no remaining plan")
        return self.plans.pop(0)


def build_test_graph(checkpointer: Any | None = None, planner: Any | None = None):
    return build_graph(
        checkpointer=checkpointer,
        planner_model=planner or SemanticTestPlanner(),
        general_model=FakeGeneralModel(),
    )

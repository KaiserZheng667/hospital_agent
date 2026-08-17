"""Shared state schema for the LangGraph workflow."""

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

Intent = Literal[
    "patient_data",
    "patient_access_list",
    "capability_question",
    "clinical_interpretation",
    "general_chat",
    "unclear",
]


QueryScope = Literal[
    "none",
    "specific_indicators",
    "category",
    "all_indicators",
    "full_record",
]
TimeScope = Literal["latest", "trend", "all_visits"]
PatientReference = Literal["explicit", "current", "none"]


class QueryPlan(BaseModel):
    """One validated semantic plan for routing and patient-data retrieval."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    patient_reference: PatientReference = "none"
    scope: QueryScope = "none"
    indicator_codes: list[str] = Field(default_factory=list, max_length=54)
    category_codes: list[str] = Field(default_factory=list, max_length=7)
    time_scope: TimeScope = "latest"
    needs_clarification: bool = False

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> "QueryPlan":
        if self.intent != "patient_data":
            if self.scope != "none" or self.indicator_codes or self.category_codes:
                raise ValueError("non-patient intent cannot request medical fields")
            return self
        if self.scope == "none":
            if self.needs_clarification:
                return self
            raise ValueError("patient_data without a scope must require clarification")
        if self.scope == "specific_indicators" and not self.indicator_codes:
            raise ValueError("specific_indicators requires indicator_codes")
        if self.scope == "category" and not self.category_codes:
            raise ValueError("category requires category_codes")
        if self.scope == "full_record" and self.time_scope != "all_visits":
            raise ValueError("full_record requires all_visits")
        return self


class MedicalAgentState(TypedDict):
    """Data shared by every node during one graph run.

    ``add_messages`` is a reducer: node outputs are appended to the existing
    conversation instead of replacing it.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    request_id: str
    actor_id: str
    patient_id: str | None
    lab_result: dict[str, Any] | None
    use_cached_result: bool
    force_tool_call: bool
    query_kind: str
    query_scope: QueryScope
    requested_indicator_codes: list[str]
    authorization_allowed: bool
    authorization_reason: str
    intent: Intent

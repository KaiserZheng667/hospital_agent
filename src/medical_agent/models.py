"""Qwen model adapters used by the production graph."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from medical_agent.state import QueryPlan


def create_model(*, temperature: float | None = None) -> ChatOpenAI:
    """Create the configured Qwen chat model."""

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env "
            "and set a valid Alibaba Cloud Model Studio API key."
        )
    model_options = {} if temperature is None else {"temperature": temperature}
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        **model_options,
    )


def create_query_planner_model():
    """Create the single Structured Output planner used before protected tools."""

    return create_model(temperature=0).with_structured_output(
        QueryPlan,
        method="json_mode",
        include_raw=True,
    )

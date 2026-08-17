from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from medical_agent.models import create_query_planner_model
from medical_agent.planner import make_query_planner_node

CASES_FILE = Path(__file__).with_name("query_plan_cases.json")


def _matches(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("intent", "query_scope", "patient_id", "query_kind"):
        if key in expected and actual.get(key) != expected[key]:
            errors.append(f"{key}: expected {expected[key]!r}, got {actual.get(key)!r}")
    actual_codes = actual.get("requested_indicator_codes", [])
    if "codes" in expected and actual_codes != expected["codes"]:
        errors.append(f"codes: expected {expected['codes']!r}, got {actual_codes!r}")
    if "code_count" in expected and len(actual_codes) != expected["code_count"]:
        errors.append(f"code_count: expected {expected['code_count']}, got {len(actual_codes)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen QueryPlan semantics.")
    parser.add_argument("--case", help="Run only one named evaluation case")
    args = parser.parse_args()
    load_dotenv()
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            raise SystemExit(f"Unknown case: {args.case}")
    node = make_query_planner_node(create_query_planner_model())
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        state = {
            "messages": [HumanMessage(content=case["message"])],
            "request_id": f"eval-{index}",
            "actor_id": "doctor-chen",
            "patient_id": case.get("current_patient_id"),
            "query_scope": case.get("previous_query_scope"),
            "query_kind": case.get("previous_query_kind"),
            "requested_indicator_codes": case.get("previous_indicator_codes", []),
            "lab_result": {} if case.get("has_previous_result") else None,
        }
        actual = node(state)
        errors = _matches(actual, case["expected"])
        if errors:
            failures.append({"name": case["name"], "errors": errors, "actual": actual})
        print(f"[{index:02}/{len(cases)}] {'PASS' if not errors else 'FAIL'} {case['name']}")
    passed = len(cases) - len(failures)
    print(f"\nQueryPlan evaluation: {passed}/{len(cases)} passed")
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

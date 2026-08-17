"""Medical tools backed by the configured repository."""

from langchain_core.tools import tool

from medical_agent.indicator_catalog import INDICATOR_CATALOG
from medical_agent.repository import get_medical_repository


@tool
def query_patient_indicators(patient_id: str, indicator_codes: list[str]) -> dict:
    """Query the latest requested osteoporosis-study indicators for a patient.

    Indicator codes must come from the allowlisted osteoporosis-study catalog.
    All records in the current SQLite environment are synthetic.
    """

    normalized_id = patient_id.strip().upper()
    result = get_medical_repository().get_latest_indicators(
        normalized_id,
        indicator_codes=indicator_codes,
    )
    if result is None:
        return {
            "status": "not_found",
            "patient_id": normalized_id,
            "message": "No synthetic osteoporosis record exists for this patient ID.",
        }

    source = result.pop("source")
    result.pop("patient_id")
    return {
        "status": "success",
        "query_type": "latest",
        "indicator_codes": indicator_codes,
        "patient_id": normalized_id,
        "source": source,
        "results": result["results"],
    }


@tool
def query_indicator_trend(
    patient_id: str,
    indicator_codes: list[str],
    limit: int = 4,
) -> dict:
    """Query recent values for requested osteoporosis-study indicators.

    Returns up to 12 synthetic records in chronological order. This tool
    provides source values only and does not clinically interpret the trend.
    """

    normalized_id = patient_id.strip().upper()
    results = get_medical_repository().get_indicator_history(
        normalized_id,
        indicator_codes=indicator_codes,
        limit=limit,
    )
    if not results:
        return {
            "status": "not_found",
            "query_type": "trend",
            "patient_id": normalized_id,
            "message": "No synthetic indicator history exists for this patient ID.",
        }

    return {
        "status": "success",
        "query_type": "trend",
        "indicator_codes": indicator_codes,
        "patient_id": normalized_id,
        "source": "synthetic_osteoporosis_sqlite",
        "results": results,
    }


@tool
def query_patient_full_record(patient_id: str) -> dict:
    """Query the complete allowlisted clinical record for one authorized patient.

    The repository excludes direct contact and identity-card fields from this
    ordinary clinical-read capability.
    """

    normalized_id = patient_id.strip().upper()
    result = get_medical_repository().get_patient_full_record(normalized_id)
    if result is None:
        return {
            "status": "not_found",
            "query_type": "full_record",
            "patient_id": normalized_id,
            "message": "No synthetic osteoporosis record exists for this patient ID.",
        }
    return {
        "status": "success",
        "query_type": "full_record",
        "indicator_codes": [item.code for item in INDICATOR_CATALOG.definitions],
        **result,
    }


MEDICAL_TOOLS = [query_patient_indicators, query_indicator_trend, query_patient_full_record]

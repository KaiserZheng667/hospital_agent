import pytest

from medical_agent.indicator_catalog import INDICATOR_CATALOG


def test_catalog_contains_exactly_the_54_supplied_fields() -> None:
    assert len(INDICATOR_CATALOG.definitions) == 54
    assert INDICATOR_CATALOG.category_counts() == {
        "BONE": 5,
        "CBC": 24,
        "URINE": 12,
        "LIVER": 5,
        "RENAL": 5,
        "CYTOKINE": 1,
        "HORMONE": 2,
    }


def test_renal_fields_from_osteoporosis_schema_are_retained() -> None:
    renal_codes = {
        definition.code
        for definition in INDICATOR_CATALOG.definitions
        if definition.category == "RENAL"
    }

    assert renal_codes == {"CREA", "UREA", "UA", "CO2CP", "EGFR"}
    assert "creatinine_umol_l" not in renal_codes
    assert "egfr_ml_min_1_73m2" not in renal_codes


def test_single_indicator_request_does_not_expand_to_other_fields() -> None:
    selected = INDICATOR_CATALOG.expand(
        indicator_codes=["crea"],
        category_codes=[],
    )
    assert [definition.code for definition in selected] == ["CREA"]


def test_category_request_expands_using_catalog_order() -> None:
    selected = INDICATOR_CATALOG.expand(
        indicator_codes=[],
        category_codes=["renal"],
    )

    assert [definition.code for definition in selected] == [
        "CREA",
        "UREA",
        "UA",
        "CO2CP",
        "EGFR",
    ]


def test_alias_resolution_is_catalog_data_not_field_specific_branching() -> None:
    assert INDICATOR_CATALOG.resolve_alias("Cr").code == "CREA"
    assert INDICATOR_CATALOG.resolve_alias("谷丙转氨酶").code == "ALT"
    assert INDICATOR_CATALOG.resolve_alias("IL-6").code == "IL6"
    assert INDICATOR_CATALOG.resolve_alias("不存在的指标") is None


def test_unknown_codes_are_rejected_before_database_access() -> None:
    with pytest.raises(ValueError, match="unknown indicator code"):
        INDICATOR_CATALOG.expand(
            indicator_codes=["NOT_A_FIELD"],
            category_codes=[],
        )

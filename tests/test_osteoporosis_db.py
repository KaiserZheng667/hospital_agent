import sqlite3
from pathlib import Path

from medical_agent.osteoporosis_db import initialize_osteoporosis_database
from medical_agent.study_fields import STUDY_FIELDS


def test_all_78_excel_fields_have_one_continuous_mapping() -> None:
    assert len(STUDY_FIELDS) == 78
    assert [field.source_column for field in STUDY_FIELDS] == list(range(1, 79))
    assert len({field.code for field in STUDY_FIELDS}) == 78
    assert len([field for field in STUDY_FIELDS if field.storage_table == "observations"]) == 54
    assert STUDY_FIELDS[0].storage_table == "patients"
    assert STUDY_FIELDS[6].storage_key == "visited_at"


def test_renal_source_columns_match_the_workbook() -> None:
    renal_fields = STUDY_FIELDS[70:75]

    assert [field.source_column for field in renal_fields] == [71, 72, 73, 74, 75]
    assert [field.code for field in renal_fields] == ["CREA", "UREA", "UA", "CO2CP", "EGFR"]
    assert renal_fields[-1].excel_header == "肾小球滤过率mL/min"
    assert renal_fields[-1].unit == "mL/min"


def test_normalized_sqlite_schema_and_synthetic_seed(tmp_path: Path) -> None:
    database = tmp_path / "osteoporosis.sqlite"
    initialize_osteoporosis_database(database)
    initialize_osteoporosis_database(database)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        counts = {
            "patients": connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
            "visits": connection.execute("SELECT COUNT(*) FROM visits").fetchone()[0],
            "definitions": connection.execute(
                "SELECT COUNT(*) FROM indicator_definitions"
            ).fetchone()[0],
            "observations": connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
        }
        egfr_unit = connection.execute(
            """
            SELECT standard_unit FROM indicator_definitions
            WHERE indicator_code = 'EGFR'
            """
        ).fetchone()[0]

    assert {"patients", "visits", "indicator_definitions", "observations"} <= tables
    assert counts == {
        "patients": 12,
        "visits": 48,
        "definitions": 54,
        "observations": 2592,
    }
    assert egfr_unit == "mL/min"

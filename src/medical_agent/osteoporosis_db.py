"""Standalone SQLite test database for the osteoporosis visit model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_agent.indicator_catalog import INDICATOR_CATALOG

DEFAULT_OSTEOPOROSIS_DB = Path(".medical-agent-data/osteoporosis.sqlite")

_SYNTHETIC_PATIENTS = (
    ("P10086", 1, "模拟患者甲", "SYNTHETIC-CONTACT-001", "SYNTHETIC-ID-001", "2026-01-05", "女"),
    ("P20001", 2, "模拟患者乙", "SYNTHETIC-CONTACT-002", "SYNTHETIC-ID-002", "2026-01-05", "男"),
    ("P30002", 3, "模拟患者丙", "SYNTHETIC-CONTACT-003", "SYNTHETIC-ID-003", "2026-01-05", "女"),
    ("P30003", 4, "模拟患者丁", "SYNTHETIC-CONTACT-004", "SYNTHETIC-ID-004", "2026-01-05", "男"),
    ("P30004", 5, "模拟患者戊", "SYNTHETIC-CONTACT-005", "SYNTHETIC-ID-005", "2026-01-05", "女"),
    ("P30005", 6, "模拟患者己", "SYNTHETIC-CONTACT-006", "SYNTHETIC-ID-006", "2026-01-05", "男"),
    ("P30006", 7, "模拟患者庚", "SYNTHETIC-CONTACT-007", "SYNTHETIC-ID-007", "2026-01-05", "女"),
    ("P30007", 8, "模拟患者辛", "SYNTHETIC-CONTACT-008", "SYNTHETIC-ID-008", "2026-01-05", "男"),
    ("P30008", 9, "模拟患者壬", "SYNTHETIC-CONTACT-009", "SYNTHETIC-ID-009", "2026-01-05", "女"),
    ("P30009", 10, "模拟患者癸", "SYNTHETIC-CONTACT-010", "SYNTHETIC-ID-010", "2026-01-05", "男"),
    ("P30010", 11, "模拟患者子", "SYNTHETIC-CONTACT-011", "SYNTHETIC-ID-011", "2026-01-05", "女"),
    ("P30011", 12, "模拟患者丑", "SYNTHETIC-CONTACT-012", "SYNTHETIC-ID-012", "2026-01-05", "男"),
)

_DOCTOR_PATIENT_ACCESS = (
    ("doctor-chen", "P10086"),
    ("doctor-chen", "P30002"),
    ("doctor-chen", "P30005"),
    ("doctor-chen", "P30009"),
    ("doctor-lin", "P20001"),
    ("doctor-lin", "P30003"),
    ("doctor-lin", "P30004"),
    ("doctor-lin", "P30006"),
    ("doctor-lin", "P30007"),
    ("doctor-lin", "P30008"),
    ("doctor-lin", "P30010"),
    ("doctor-lin", "P30011"),
)

_BASE_NUMERIC_VALUES = {
    "BMD_QCT": 78.0,
    "OSTEOCALCIN": 20.0,
    "25OHD": 24.0,
    "PINP": 48.0,
    "BETA_CTX": 0.45,
    "WBC": 5.8,
    "RBC": 4.5,
    "HGB": 135.0,
    "HCT": 40.5,
    "MCV": 90.0,
    "MCH": 30.0,
    "MCHC": 333.0,
    "PLT": 220.0,
    "PCT": 0.2,
    "MPV": 10.0,
    "LYMPH_PCT": 30.0,
    "NEUT_PCT": 60.0,
    "MONO_PCT": 6.0,
    "EOS_PCT": 3.0,
    "BASO_PCT": 1.0,
    "LYMPH_ABS": 1.74,
    "NEUT_ABS": 3.48,
    "MONO_ABS": 0.35,
    "EOS_ABS": 0.17,
    "BASO_ABS": 0.06,
    "RDW_SD": 42.0,
    "RDW_CV": 12.5,
    "PDW": 15.0,
    "P_LCR": 28.0,
    "URINE_PH": 6.0,
    "URINE_SG": 1.015,
    "ALT": 22.0,
    "AST": 24.0,
    "GGT": 18.0,
    "TBIL": 12.0,
    "DBIL": 4.0,
    "CREA": 70.0,
    "UREA": 5.0,
    "UA": 300.0,
    "CO2CP": 25.0,
    "EGFR": 90.0,
    "IL6": 2.0,
    "TESTOSTERONE": 0.3,
    "ESTRADIOL": 40.0,
}


def _connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_osteoporosis_database(database: Path = DEFAULT_OSTEOPOROSIS_DB) -> None:
    """Create the normalized schema and deterministic teaching-only data."""

    with _connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_no TEXT PRIMARY KEY,
                source_sequence INTEGER UNIQUE,
                name TEXT NOT NULL,
                contact TEXT,
                identity_card TEXT,
                consented_at TEXT,
                sex TEXT NOT NULL CHECK (sex IN ('男', '女', '未知')),
                is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1)
            );

            CREATE TABLE IF NOT EXISTS visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_no TEXT NOT NULL,
                researcher TEXT,
                visitor TEXT,
                visited_at TEXT NOT NULL,
                visit_label TEXT NOT NULL,
                age_years REAL,
                weight_kg REAL,
                height_m REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                pulse REAL,
                tcm_syndrome TEXT,
                tcm_symptoms TEXT,
                treatment TEXT,
                past_history TEXT,
                family_history TEXT,
                antibiotic_history TEXT,
                study_group TEXT,
                source TEXT NOT NULL DEFAULT 'synthetic_seed',
                FOREIGN KEY (patient_no) REFERENCES patients(patient_no),
                UNIQUE (patient_no, visited_at, visit_label)
            );

            CREATE TABLE IF NOT EXISTS indicator_definitions (
                indicator_code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                value_type TEXT NOT NULL,
                standard_unit TEXT
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_id INTEGER NOT NULL,
                indicator_code TEXT NOT NULL,
                value_text TEXT NOT NULL,
                value_numeric REAL,
                result_flag TEXT NOT NULL DEFAULT 'unknown'
                    CHECK (result_flag IN ('low', 'high', 'normal', 'unknown')),
                source TEXT NOT NULL DEFAULT 'synthetic_seed',
                FOREIGN KEY (visit_id) REFERENCES visits(visit_id),
                FOREIGN KEY (indicator_code) REFERENCES indicator_definitions(indicator_code),
                UNIQUE (visit_id, indicator_code)
            );

            CREATE TABLE IF NOT EXISTS doctor_patient_access (
                actor_id TEXT NOT NULL,
                patient_no TEXT NOT NULL,
                access_reason TEXT NOT NULL DEFAULT 'study_assignment',
                PRIMARY KEY (actor_id, patient_no),
                FOREIGN KEY (patient_no) REFERENCES patients(patient_no)
            );

            CREATE INDEX IF NOT EXISTS idx_visits_patient_time
            ON visits(patient_no, visited_at DESC);

            CREATE INDEX IF NOT EXISTS idx_observations_visit_indicator
            ON observations(visit_id, indicator_code);
            """
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO indicator_definitions(
                indicator_code, name, category, value_type, standard_unit
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    item.code,
                    item.name,
                    item.category,
                    item.value_type,
                    item.standard_unit,
                )
                for item in INDICATOR_CATALOG.definitions
            ),
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO patients(
                patient_no, source_sequence, name, contact, identity_card, consented_at, sex
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            _SYNTHETIC_PATIENTS,
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO doctor_patient_access(actor_id, patient_no)
            VALUES (?, ?)
            """,
            _DOCTOR_PATIENT_ACCESS,
        )

        visit_dates = (
            ("2026-01-05", "治疗前"),
            ("2026-02-04", "30天访视点"),
            ("2026-03-06", "60天访视点"),
            ("2026-04-05", "90天访视点"),
        )
        for patient_index, patient in enumerate(_SYNTHETIC_PATIENTS):
            patient_no = patient[0]
            for visit_index, (visited_at, visit_label) in enumerate(visit_dates, start=1):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO visits(
                        patient_no, researcher, visitor, visited_at,
                        visit_label, age_years, weight_kg, height_m, systolic_bp,
                        diastolic_bp, pulse, tcm_syndrome, tcm_symptoms, treatment,
                        past_history, family_history, antibiotic_history, study_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_no,
                        "模拟研究者",
                        "模拟访视者",
                        visited_at,
                        visit_label,
                        65 + patient_index,
                        60 + patient_index * 5,
                        1.60 + patient_index * 0.05,
                        120 - visit_index,
                        75 - visit_index,
                        72 - visit_index,
                        "模拟证候",
                        "教学用模拟症状",
                        "教学用模拟治疗方案",
                        "骨质疏松症（模拟）",
                        "无（模拟）",
                        "无（模拟）",
                        "骨质疏松研究模拟组",
                    ),
                )
                visit_row = connection.execute(
                    """
                    SELECT visit_id FROM visits
                    WHERE patient_no = ? AND visited_at = ? AND visit_label = ?
                    """,
                    (patient_no, visited_at, visit_label),
                ).fetchone()
                if visit_row is None:
                    raise RuntimeError("failed to create synthetic visit")
                _seed_observations(connection, visit_row["visit_id"], visit_index)
        connection.execute("PRAGMA optimize")


def _validated_indicator_codes(indicator_codes: list[str]) -> list[str]:
    normalized: list[str] = []
    for code in indicator_codes:
        canonical = INDICATOR_CATALOG.get(code).code
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("at least one indicator code is required")
    return normalized


def get_latest_indicators(
    patient_id: str,
    indicator_codes: list[str],
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> dict[str, Any] | None:
    """Return the latest available observation for each requested indicator."""

    normalized_id = patient_id.strip().upper()
    codes = _validated_indicator_codes(indicator_codes)
    initialize_osteoporosis_database(database)
    placeholders = ", ".join("?" for _ in codes)
    with _connect(database) as connection:
        patient = connection.execute(
            "SELECT 1 FROM patients WHERE patient_no = ?",
            (normalized_id,),
        ).fetchone()
        if patient is None:
            return None
        rows = connection.execute(
            f"""
            WITH ranked AS (
                SELECT
                    v.patient_no,
                    v.visited_at,
                    v.visit_label,
                    o.indicator_code,
                    d.name,
                    d.standard_unit,
                    o.value_text,
                    o.value_numeric,
                    o.result_flag,
                    o.source,
                    ROW_NUMBER() OVER (
                        PARTITION BY o.indicator_code
                        ORDER BY v.visited_at DESC, v.visit_id DESC
                    ) AS row_number
                FROM observations AS o
                JOIN visits AS v ON v.visit_id = o.visit_id
                JOIN indicator_definitions AS d
                    ON d.indicator_code = o.indicator_code
                WHERE v.patient_no = ?
                  AND o.indicator_code IN ({placeholders})
            )
            SELECT * FROM ranked
            WHERE row_number = 1
            ORDER BY indicator_code
            """,
            (normalized_id, *codes),
        ).fetchall()
    return {
        "patient_id": normalized_id,
        "source": "synthetic_osteoporosis_sqlite",
        "results": [dict(row) for row in rows],
    }


def get_indicator_history(
    patient_id: str,
    indicator_codes: list[str],
    limit: int = 4,
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> list[dict[str, Any]]:
    """Return requested observations grouped by recent visits in time order."""

    normalized_id = patient_id.strip().upper()
    codes = _validated_indicator_codes(indicator_codes)
    safe_limit = min(max(limit, 1), 12)
    initialize_osteoporosis_database(database)
    placeholders = ", ".join("?" for _ in codes)
    with _connect(database) as connection:
        visits = connection.execute(
            """
            SELECT visit_id, visited_at, visit_label
            FROM visits
            WHERE patient_no = ?
            ORDER BY visited_at DESC, visit_id DESC
            LIMIT ?
            """,
            (normalized_id, safe_limit),
        ).fetchall()
        grouped: list[dict[str, Any]] = []
        for visit in reversed(visits):
            rows = connection.execute(
                f"""
                SELECT
                    o.indicator_code,
                    d.name,
                    d.standard_unit,
                    o.value_text,
                    o.value_numeric,
                    o.result_flag
                FROM observations AS o
                JOIN indicator_definitions AS d
                    ON d.indicator_code = o.indicator_code
                WHERE o.visit_id = ?
                  AND o.indicator_code IN ({placeholders})
                ORDER BY o.indicator_code
                """,
                (visit["visit_id"], *codes),
            ).fetchall()
            grouped.append(
                {
                    "visited_at": visit["visited_at"],
                    "visit_label": visit["visit_label"],
                    "results": [dict(row) for row in rows],
                }
            )
    return grouped


def get_patient_full_record(
    patient_id: str,
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> dict[str, Any] | None:
    """Return the complete allowlisted synthetic clinical record for one patient.

    Direct identifiers such as identity-card number and contact details are
    deliberately excluded from the ordinary patient-read permission.
    """

    normalized_id = patient_id.strip().upper()
    initialize_osteoporosis_database(database)
    with _connect(database) as connection:
        patient = connection.execute(
            """
            SELECT patient_no AS patient_id, source_sequence, name, consented_at,
                   sex, is_synthetic
            FROM patients WHERE patient_no = ?
            """,
            (normalized_id,),
        ).fetchone()
        if patient is None:
            return None
        visit_rows = connection.execute(
            """
            SELECT visit_id, researcher, visitor, visited_at, visit_label,
                   age_years, weight_kg, height_m, systolic_bp, diastolic_bp,
                   pulse, tcm_syndrome, tcm_symptoms, treatment, past_history,
                   family_history, antibiotic_history, study_group, source
            FROM visits WHERE patient_no = ?
            ORDER BY visited_at, visit_id
            """,
            (normalized_id,),
        ).fetchall()
        visits: list[dict[str, Any]] = []
        for visit_row in visit_rows:
            visit = dict(visit_row)
            visit_id = visit.pop("visit_id")
            observations = connection.execute(
                """
                SELECT o.indicator_code, d.name, d.category, d.standard_unit,
                       o.value_text, o.value_numeric, o.result_flag
                FROM observations AS o
                JOIN indicator_definitions AS d
                    ON d.indicator_code = o.indicator_code
                WHERE o.visit_id = ?
                ORDER BY o.indicator_code
                """,
                (visit_id,),
            ).fetchall()
            visit["indicators"] = [dict(row) for row in observations]
            visits.append(visit)
    return {
        "patient_id": normalized_id,
        "source": "synthetic_osteoporosis_sqlite",
        "patient": dict(patient),
        "visits": visits,
    }


def list_synthetic_patients(
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> list[dict[str, Any]]:
    """List the synthetic osteoporosis-study patients for the teaching UI."""

    initialize_osteoporosis_database(database)
    with _connect(database) as connection:
        rows = connection.execute(
            """
            SELECT
                p.patient_no AS patient_id,
                p.name AS display_name,
                p.sex,
                COALESCE(v.study_group, '骨质疏松研究') AS department
            FROM patients AS p
            LEFT JOIN visits AS v ON v.visit_id = (
                SELECT latest.visit_id FROM visits AS latest
                WHERE latest.patient_no = p.patient_no
                ORDER BY latest.visited_at DESC, latest.visit_id DESC
                LIMIT 1
            )
            WHERE p.patient_no LIKE 'P%'
            ORDER BY p.patient_no
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_authorized_patients(
    actor_id: str,
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> list[dict[str, Any]]:
    """List only osteoporosis-study patients assigned to one actor."""

    initialize_osteoporosis_database(database)
    with _connect(database) as connection:
        rows = connection.execute(
            """
            SELECT
                p.patient_no AS patient_id,
                p.name AS display_name,
                p.sex,
                COALESCE(v.study_group, '骨质疏松研究') AS department
            FROM patients AS p
            JOIN doctor_patient_access AS a ON a.patient_no = p.patient_no
            LEFT JOIN visits AS v ON v.visit_id = (
                SELECT latest.visit_id FROM visits AS latest
                WHERE latest.patient_no = p.patient_no
                ORDER BY latest.visited_at DESC, latest.visit_id DESC
                LIMIT 1
            )
            WHERE a.actor_id = ?
            ORDER BY p.patient_no
            """,
            (actor_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def actor_can_access_patient(
    actor_id: str,
    patient_id: str,
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> bool:
    """Check study assignment without exposing whether another patient exists."""

    initialize_osteoporosis_database(database)
    with _connect(database) as connection:
        row = connection.execute(
            """
            SELECT 1 FROM doctor_patient_access
            WHERE actor_id = ? AND patient_no = ?
            """,
            (actor_id, patient_id.strip().upper()),
        ).fetchone()
    return row is not None


def replace_patient_access(
    actor_id: str,
    patient_ids: list[str],
    database: Path = DEFAULT_OSTEOPOROSIS_DB,
) -> None:
    """Atomically replace one user's assigned synthetic-patient scope."""

    initialize_osteoporosis_database(database)
    normalized = sorted({patient_id.strip().upper() for patient_id in patient_ids})
    with _connect(database) as connection:
        if normalized:
            placeholders = ",".join("?" for _ in normalized)
            existing = {
                row["patient_no"]
                for row in connection.execute(
                    f"SELECT patient_no FROM patients WHERE patient_no IN ({placeholders})",
                    normalized,
                ).fetchall()
            }
            unknown = set(normalized) - existing
            if unknown:
                raise ValueError(f"未知患者编号：{', '.join(sorted(unknown))}")
        connection.execute("DELETE FROM doctor_patient_access WHERE actor_id = ?", (actor_id,))
        connection.executemany(
            "INSERT INTO doctor_patient_access(actor_id, patient_no) VALUES (?, ?)",
            ((actor_id, patient_id) for patient_id in normalized),
        )


@dataclass(frozen=True)
class SQLiteOsteoporosisRepository:
    """SQLite implementation of the storage-neutral medical repository."""

    database: Path = DEFAULT_OSTEOPOROSIS_DB

    def get_latest_indicators(
        self,
        patient_id: str,
        indicator_codes: list[str],
    ) -> dict[str, Any] | None:
        return get_latest_indicators(patient_id, indicator_codes, self.database)

    def get_indicator_history(
        self,
        patient_id: str,
        indicator_codes: list[str],
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return get_indicator_history(patient_id, indicator_codes, limit, self.database)

    def get_patient_full_record(self, patient_id: str) -> dict[str, Any] | None:
        return get_patient_full_record(patient_id, self.database)

    def list_patients(self) -> list[dict[str, Any]]:
        return list_synthetic_patients(self.database)

    def list_authorized_patients(self, actor_id: str) -> list[dict[str, Any]]:
        return list_authorized_patients(actor_id, self.database)

    def actor_can_access_patient(self, actor_id: str, patient_id: str) -> bool:
        return actor_can_access_patient(actor_id, patient_id, self.database)

    def replace_patient_access(self, actor_id: str, patient_ids: list[str]) -> None:
        replace_patient_access(actor_id, patient_ids, self.database)


def _seed_observations(
    connection: sqlite3.Connection,
    visit_id: int,
    visit_index: int,
) -> None:
    rows: list[tuple[object, ...]] = []
    for definition in INDICATOR_CATALOG.definitions:
        numeric_value = _BASE_NUMERIC_VALUES.get(definition.code)
        if numeric_value is not None:
            adjusted_value = round(numeric_value * (1 + (visit_index - 1) * 0.005), 3)
            value_text = str(adjusted_value)
        elif definition.code == "URINE_OTHER":
            adjusted_value = None
            value_text = "无（模拟）"
        else:
            adjusted_value = None
            value_text = "阴性（模拟）"
        rows.append(
            (
                visit_id,
                definition.code,
                value_text,
                adjusted_value,
                "normal",
                "synthetic_seed",
            )
        )
    connection.executemany(
        """
        INSERT OR IGNORE INTO observations(
            visit_id, indicator_code, value_text, value_numeric, result_flag, source
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

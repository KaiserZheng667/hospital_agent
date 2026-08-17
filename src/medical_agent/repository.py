"""Repository boundary between Agent logic and medical data storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from medical_agent.osteoporosis_db import (
    DEFAULT_OSTEOPOROSIS_DB,
    SQLiteOsteoporosisRepository,
)


@runtime_checkable
class MedicalRepository(Protocol):
    """Storage-neutral data capabilities currently required by the Agent."""

    def get_latest_indicators(
        self,
        patient_id: str,
        indicator_codes: list[str],
    ) -> dict[str, Any] | None: ...

    def get_indicator_history(
        self,
        patient_id: str,
        indicator_codes: list[str],
        limit: int = 4,
    ) -> list[dict[str, Any]]: ...

    def get_patient_full_record(self, patient_id: str) -> dict[str, Any] | None: ...

    def list_patients(self) -> list[dict[str, Any]]: ...

    def list_authorized_patients(self, actor_id: str) -> list[dict[str, Any]]: ...

    def actor_can_access_patient(self, actor_id: str, patient_id: str) -> bool: ...

    def replace_patient_access(self, actor_id: str, patient_ids: list[str]) -> None: ...


def get_medical_repository() -> MedicalRepository:
    """Build the configured repository at the application composition boundary."""

    backend = os.getenv("MEDICAL_DB_BACKEND", "sqlite").strip().lower()
    if backend == "sqlite":
        database = Path(os.getenv("SQLITE_MEDICAL_DB", str(DEFAULT_OSTEOPOROSIS_DB)))
        return SQLiteOsteoporosisRepository(database)
    if backend == "oracle":
        raise RuntimeError(
            "MEDICAL_DB_BACKEND=oracle is configured, but the Oracle repository "
            "will be enabled only after an Oracle server connection is available."
        )
    raise RuntimeError(f"Unsupported MEDICAL_DB_BACKEND: {backend!r}")

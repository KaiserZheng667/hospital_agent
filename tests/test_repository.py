from pathlib import Path
from typing import Any

import pytest

from medical_agent.osteoporosis_db import SQLiteOsteoporosisRepository
from medical_agent.repository import MedicalRepository, get_medical_repository
from medical_agent.tools import query_patient_indicators


class FakeMedicalRepository:
    def get_latest_indicators(
        self,
        patient_id: str,
        indicator_codes: list[str],
    ) -> dict[str, Any] | None:
        return {
            "patient_id": patient_id,
            "source": "fake_repository",
            "results": [
                {
                    "indicator_code": indicator_codes[0],
                    "name": "骨密度",
                    "standard_unit": "mg/cm3",
                    "value_text": "88.0",
                    "visited_at": "2030-01-01",
                    "visit_label": "模拟访视",
                }
            ],
        }

    def get_indicator_history(
        self,
        patient_id: str,
        indicator_codes: list[str],
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return []

    def get_patient_full_record(self, patient_id: str) -> dict[str, Any] | None:
        return None

    def list_patients(self) -> list[dict[str, Any]]:
        return []

    def list_authorized_patients(self, actor_id: str) -> list[dict[str, Any]]:
        return []

    def actor_can_access_patient(self, actor_id: str, patient_id: str) -> bool:
        return True

    def replace_patient_access(self, actor_id: str, patient_ids: list[str]) -> None:
        return None


def test_sqlite_adapter_satisfies_repository_contract(tmp_path: Path) -> None:
    repository = SQLiteOsteoporosisRepository(tmp_path / "osteoporosis.sqlite")

    assert isinstance(repository, MedicalRepository)
    assert len(repository.list_patients()) == 12


def test_tool_uses_repository_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repository = FakeMedicalRepository()
    monkeypatch.setattr(
        "medical_agent.tools.get_medical_repository",
        lambda: fake_repository,
    )

    result = query_patient_indicators.invoke(
        {"patient_id": "P99999", "indicator_codes": ["BMD_QCT"]}
    )

    assert result["source"] == "fake_repository"
    assert result["results"][0]["indicator_code"] == "BMD_QCT"


def test_repository_factory_rejects_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_DB_BACKEND", "unknown")

    with pytest.raises(RuntimeError, match="Unsupported MEDICAL_DB_BACKEND"):
        get_medical_repository()

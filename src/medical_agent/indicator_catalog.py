"""Canonical osteoporosis-study indicator catalog and query schema."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

IndicatorCategory = Literal[
    "BONE",
    "CBC",
    "URINE",
    "LIVER",
    "RENAL",
    "CYTOKINE",
    "HORMONE",
]
IndicatorValueType = Literal["NUMERIC", "TEXT", "SEMI_QUANT"]


@dataclass(frozen=True)
class IndicatorDefinition:
    """One canonical field from the supplied osteoporosis-study schema."""

    code: str
    name: str
    category: IndicatorCategory
    value_type: IndicatorValueType
    standard_unit: str | None
    aliases: tuple[str, ...] = ()


class IndicatorCatalog:
    """Allowlisted field registry used after LLM semantic extraction."""

    def __init__(
        self,
        definitions: tuple[IndicatorDefinition, ...],
        *,
        max_query_indicators: int = 54,
    ) -> None:
        self._definitions = definitions
        self._by_code = {definition.code: definition for definition in definitions}
        if len(self._by_code) != len(definitions):
            raise ValueError("indicator codes must be unique")

        self._by_alias: dict[str, IndicatorDefinition] = {}
        for definition in definitions:
            for alias in (definition.code, definition.name, *definition.aliases):
                normalized = _normalize_alias(alias)
                existing = self._by_alias.get(normalized)
                if existing is not None and existing.code != definition.code:
                    raise ValueError(f"indicator alias {alias!r} conflicts with {existing.code}")
                self._by_alias[normalized] = definition
        self.max_query_indicators = max_query_indicators

    @property
    def definitions(self) -> tuple[IndicatorDefinition, ...]:
        return self._definitions

    def category_counts(self) -> dict[str, int]:
        return dict(Counter(item.category for item in self._definitions))

    def get(self, code: str) -> IndicatorDefinition:
        normalized = code.strip().upper()
        try:
            return self._by_code[normalized]
        except KeyError as error:
            raise ValueError(f"unknown indicator code: {normalized}") from error

    def resolve_alias(self, text: str) -> IndicatorDefinition | None:
        return self._by_alias.get(_normalize_alias(text))

    def expand(
        self,
        *,
        indicator_codes: list[str],
        category_codes: list[str],
    ) -> tuple[IndicatorDefinition, ...]:
        selected: list[IndicatorDefinition] = []
        selected_codes: set[str] = set()

        for code in indicator_codes:
            definition = self.get(code)
            if definition.code not in selected_codes:
                selected.append(definition)
                selected_codes.add(definition.code)

        valid_categories = set(self.category_counts())
        for category in category_codes:
            normalized = category.strip().upper()
            if normalized not in valid_categories:
                raise ValueError(f"unknown indicator category: {normalized}")
            for definition in self._definitions:
                if definition.category == normalized and definition.code not in selected_codes:
                    selected.append(definition)
                    selected_codes.add(definition.code)

        if len(selected) > self.max_query_indicators:
            raise ValueError(
                f"too many indicators requested: {len(selected)} > {self.max_query_indicators}"
            )
        return tuple(selected)

def _normalize_alias(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold()


_DEFINITIONS = (
    IndicatorDefinition("BMD_QCT", "骨密度", "BONE", "NUMERIC", "mg/cm3"),
    IndicatorDefinition("OSTEOCALCIN", "骨钙素", "BONE", "NUMERIC", "ng/mL"),
    IndicatorDefinition(
        "25OHD", "25-羟基维生素D", "BONE", "NUMERIC", "ng/mL", ("维生素D", "25羟维生素D")
    ),
    IndicatorDefinition("PINP", "血清I型原胶原氨基端前肽", "BONE", "NUMERIC", "ng/mL"),
    IndicatorDefinition("BETA_CTX", "β-胶原降解产物", "BONE", "NUMERIC", "ng/mL", ("β-CTX",)),
    IndicatorDefinition("WBC", "白细胞", "CBC", "NUMERIC", "10^9/L", ("白细胞计数",)),
    IndicatorDefinition("RBC", "红细胞", "CBC", "NUMERIC", "10^12/L", ("红细胞计数",)),
    IndicatorDefinition("HGB", "血红蛋白浓度", "CBC", "NUMERIC", "g/L", ("血红蛋白", "HB")),
    IndicatorDefinition("HCT", "红细胞压积", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("MCV", "平均红细胞体积", "CBC", "NUMERIC", "fL"),
    IndicatorDefinition("MCH", "平均红细胞血红蛋白含量", "CBC", "NUMERIC", "pg"),
    IndicatorDefinition("MCHC", "平均红细胞血红蛋白浓度", "CBC", "NUMERIC", "g/L"),
    IndicatorDefinition("PLT", "血小板计数", "CBC", "NUMERIC", "10^9/L", ("血小板",)),
    IndicatorDefinition("PCT", "血小板压积", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("MPV", "平均血小板体积", "CBC", "NUMERIC", "fL"),
    IndicatorDefinition("LYMPH_PCT", "淋巴细胞百分数", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("NEUT_PCT", "中性粒细胞百分数", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("MONO_PCT", "单核细胞百分数", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("EOS_PCT", "嗜酸性粒细胞百分数", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("BASO_PCT", "嗜碱性粒细胞百分数", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("LYMPH_ABS", "淋巴细胞计数", "CBC", "NUMERIC", "10^9/L"),
    IndicatorDefinition("NEUT_ABS", "中性粒细胞计数", "CBC", "NUMERIC", "10^9/L"),
    IndicatorDefinition("MONO_ABS", "单核细胞计数", "CBC", "NUMERIC", "10^9/L"),
    IndicatorDefinition("EOS_ABS", "嗜酸性粒细胞计数", "CBC", "NUMERIC", "10^9/L"),
    IndicatorDefinition("BASO_ABS", "嗜碱性粒细胞计数", "CBC", "NUMERIC", "10^9/L"),
    IndicatorDefinition("RDW_SD", "红细胞分布宽度", "CBC", "NUMERIC", "fL"),
    IndicatorDefinition("RDW_CV", "红细胞体积分布宽度-CV", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("PDW", "血小板体积分布宽度", "CBC", "NUMERIC", "fL"),
    IndicatorDefinition("P_LCR", "大血小板比率", "CBC", "NUMERIC", "%"),
    IndicatorDefinition("URINE_GLU", "尿葡萄糖", "URINE", "SEMI_QUANT", "mmol/L", ("尿糖",)),
    IndicatorDefinition("URINE_BLD", "尿潜血", "URINE", "SEMI_QUANT", "cells/uL"),
    IndicatorDefinition("URINE_WBC", "尿白细胞", "URINE", "SEMI_QUANT", "cells/uL"),
    IndicatorDefinition("URINE_PRO", "尿蛋白质", "URINE", "SEMI_QUANT", "g/L", ("尿蛋白",)),
    IndicatorDefinition("URINE_MALB", "尿微量白蛋白", "URINE", "SEMI_QUANT", "g/L"),
    IndicatorDefinition("URINE_NIT", "亚硝酸盐", "URINE", "SEMI_QUANT", "mg/L"),
    IndicatorDefinition("URINE_URO", "尿胆原", "URINE", "SEMI_QUANT", "umol/L"),
    IndicatorDefinition("URINE_BIL", "尿胆红素", "URINE", "SEMI_QUANT", "mg/L"),
    IndicatorDefinition("URINE_KET", "尿酮体", "URINE", "SEMI_QUANT", "mg/L"),
    IndicatorDefinition("URINE_PH", "尿酸碱度", "URINE", "NUMERIC", None, ("尿PH",)),
    IndicatorDefinition("URINE_SG", "尿比重", "URINE", "NUMERIC", None),
    IndicatorDefinition("URINE_OTHER", "尿常规其他", "URINE", "TEXT", None),
    IndicatorDefinition("ALT", "丙氨酸氨基转移酶", "LIVER", "NUMERIC", "U/L", ("谷丙转氨酶",)),
    IndicatorDefinition("AST", "天门冬氨酸氨基转移酶", "LIVER", "NUMERIC", "U/L", ("谷草转氨酶",)),
    IndicatorDefinition("GGT", "γ-谷氨酰基转移酶", "LIVER", "NUMERIC", "U/L", ("谷氨酰转肽酶",)),
    IndicatorDefinition("TBIL", "总胆红素", "LIVER", "NUMERIC", "umol/L"),
    IndicatorDefinition("DBIL", "直接胆红素", "LIVER", "NUMERIC", "umol/L"),
    IndicatorDefinition(
        "CREA", "肌酐", "RENAL", "NUMERIC", "umol/L", ("血肌酐", "CR", "CREATININE")
    ),
    IndicatorDefinition("UREA", "尿素", "RENAL", "NUMERIC", "mmol/L", ("血尿素",)),
    IndicatorDefinition("UA", "尿酸", "RENAL", "NUMERIC", "umol/L"),
    IndicatorDefinition("CO2CP", "二氧化碳结合力", "RENAL", "NUMERIC", "mmol/L"),
    IndicatorDefinition("EGFR", "肾小球滤过率", "RENAL", "NUMERIC", "mL/min"),
    IndicatorDefinition("IL6", "白介素6", "CYTOKINE", "NUMERIC", "pg/mL", ("白细胞介素6", "IL-6")),
    IndicatorDefinition("TESTOSTERONE", "睾酮", "HORMONE", "NUMERIC", "ng/mL"),
    IndicatorDefinition("ESTRADIOL", "雌二醇", "HORMONE", "NUMERIC", "pg/mL", ("E2",)),
)

INDICATOR_CATALOG = IndicatorCatalog(_DEFINITIONS)

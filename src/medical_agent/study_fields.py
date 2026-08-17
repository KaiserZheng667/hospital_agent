"""Authoritative mapping for the 78 columns in the osteoporosis visit workbook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from medical_agent.indicator_catalog import INDICATOR_CATALOG

FieldStorage = Literal["patients", "visits", "observations"]
FieldValueType = Literal["INTEGER", "NUMERIC", "DATE", "TEXT", "SEMI_QUANT"]


@dataclass(frozen=True)
class StudyFieldDefinition:
    """Mapping from one Excel source column to normalized database storage."""

    source_column: int
    code: str
    excel_header: str
    storage_table: FieldStorage
    storage_key: str
    value_type: FieldValueType
    unit: str | None = None


_NON_INDICATOR_FIELDS = (
    StudyFieldDefinition(1, "SOURCE_SEQUENCE", "序号", "patients", "source_sequence", "INTEGER"),
    StudyFieldDefinition(2, "PATIENT_NAME", "姓名", "patients", "name", "TEXT"),
    StudyFieldDefinition(3, "CONTACT", "联系方式", "patients", "contact", "TEXT"),
    StudyFieldDefinition(4, "IDENTITY_CARD", "身份证", "patients", "identity_card", "TEXT"),
    StudyFieldDefinition(5, "RESEARCHER", "研究者", "visits", "researcher", "TEXT"),
    StudyFieldDefinition(6, "VISITOR", "接诊访视者", "visits", "visitor", "TEXT"),
    StudyFieldDefinition(7, "VISITED_AT", "入组时间", "visits", "visited_at", "DATE"),
    StudyFieldDefinition(
        8,
        "VISIT_LABEL",
        "备注/治疗前还是治疗后？",
        "visits",
        "visit_label",
        "TEXT",
    ),
    StudyFieldDefinition(9, "CONSENTED_AT", "签知情同意书时间", "patients", "consented_at", "DATE"),
    StudyFieldDefinition(10, "PATIENT_NO", "编号", "patients", "patient_no", "TEXT"),
    StudyFieldDefinition(11, "SEX", "性别", "patients", "sex", "TEXT"),
    StudyFieldDefinition(12, "AGE_YEARS", "年龄(岁)", "visits", "age_years", "NUMERIC", "岁"),
    StudyFieldDefinition(13, "WEIGHT_KG", "体重(kg)", "visits", "weight_kg", "NUMERIC", "kg"),
    StudyFieldDefinition(14, "HEIGHT_M", "身高(m)", "visits", "height_m", "NUMERIC", "m"),
    StudyFieldDefinition(
        15, "SYSTOLIC_BP", "收缩压(mmHg)", "visits", "systolic_bp", "NUMERIC", "mmHg"
    ),
    StudyFieldDefinition(
        16, "DIASTOLIC_BP", "舒张压(mmHg)", "visits", "diastolic_bp", "NUMERIC", "mmHg"
    ),
    StudyFieldDefinition(17, "PULSE", "脉搏(次/分)", "visits", "pulse", "NUMERIC", "次/分"),
    StudyFieldDefinition(18, "TCM_SYNDROME", "中医证候", "visits", "tcm_syndrome", "TEXT"),
    StudyFieldDefinition(19, "TCM_SYMPTOMS", "中医症状", "visits", "tcm_symptoms", "TEXT"),
    StudyFieldDefinition(
        20,
        "TREATMENT",
        "治疗方式（中药方剂、西药治疗等）",
        "visits",
        "treatment",
        "TEXT",
    ),
    StudyFieldDefinition(21, "PAST_HISTORY", "既往史", "visits", "past_history", "TEXT"),
    StudyFieldDefinition(22, "FAMILY_HISTORY", "家族史", "visits", "family_history", "TEXT"),
    StudyFieldDefinition(
        23,
        "ANTIBIOTIC_HISTORY",
        "抗生素使用史",
        "visits",
        "antibiotic_history",
        "TEXT",
    ),
    StudyFieldDefinition(24, "STUDY_GROUP", "分组情况", "visits", "study_group", "TEXT"),
)

_INDICATOR_HEADERS = (
    "骨密度（mg/cm3）",
    "骨钙素(ng/mL)",
    "25-羟基维生素D(ng/mL)",
    "血清I型原胶原氨基端前肽(ng/mL)",
    "β-胶原降解产物测定(ng/mL)",
    "白细胞(*10^9/L)",
    "红细胞(*10^12/L)",
    "血红蛋白浓度（g/L)",
    "红细胞压积(%)",
    "平均红细胞体积（fL）",
    "平均红细胞血红蛋白含量（pg）",
    "平均红细胞血红蛋白浓度（g/L)",
    "血小板计数(*10^9/L)",
    "血小板压积（%）",
    "平均血小板体积(fL)",
    "淋巴细胞百分数（%）",
    "中性粒细胞百分数（%）",
    "单核细胞百分数（%）",
    "嗜酸性粒细胞百分数（%）",
    "嗜碱性粒细胞百分数（%）",
    "淋巴细胞计数(*10^9/L)",
    "中性粒细胞计数(*10^9/L)",
    "单核细胞计数(*10^9/L)",
    "嗜酸性粒细胞计数(*10^9/L)",
    "嗜碱性粒细胞计数(*10^9/L)",
    "红细胞分布宽度（fL)",
    "红细胞体积分布宽度-CV(%)",
    "血小板体积分布宽度(fL)",
    "大血小板比率（%）",
    "尿葡萄糖（mmol/L)",
    "尿潜血（cells/u)",
    "白细胞(cells/u）",
    "蛋白质（g/L）",
    "尿微量白蛋白（g/L）",
    "亚硝酸盐（mg/L）",
    "尿胆原（umol/L）",
    "胆红素（mg/L）",
    "尿酮体（mg/L）",
    "酸碱度",
    "尿比重",
    "其他",
    "丙氨酸氨基转移酶(U/L)",
    "天门冬氨酸氨基转移酶(U/L)",
    "γ-谷氨酰基转移酶(U/L)",
    "总胆红素(μmol/L)",
    "直接胆红素(μmol/L)",
    "  肌酐  (μmol/L)",
    "尿素(mmol/L)",
    "尿酸(μmol/L)",
    "二氧化碳结合力（mmol/L）",
    "肾小球滤过率mL/min",
    "白介素6(pg/mL）",
    "睾酮(ng/mL)",
    "雌二醇(pg/mL)",
)

_INDICATOR_FIELDS = tuple(
    StudyFieldDefinition(
        source_column=source_column,
        code=definition.code,
        excel_header=excel_header,
        storage_table="observations",
        storage_key=definition.code,
        value_type=definition.value_type,
        unit=definition.standard_unit,
    )
    for source_column, (definition, excel_header) in enumerate(
        zip(INDICATOR_CATALOG.definitions, _INDICATOR_HEADERS, strict=True),
        start=25,
    )
)

STUDY_FIELDS = _NON_INDICATOR_FIELDS + _INDICATOR_FIELDS

if len(STUDY_FIELDS) != 78:
    raise RuntimeError("the osteoporosis workbook mapping must contain exactly 78 fields")
if tuple(field.source_column for field in STUDY_FIELDS) != tuple(range(1, 79)):
    raise RuntimeError("the osteoporosis workbook columns must map continuously from 1 to 78")

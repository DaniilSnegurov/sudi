"""Постановление ФССП в XML → поля верхнего уровня (спецификация, раздел 10)."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from defusedxml import ElementTree as SafeET

from .address import parse_dbtr_adr
from .debt_text import extract_debt_fields
from .models import EXTRACTED, FAILED, INFO, NEEDS_REVIEW, DocumentResult, Field, Issue

STAGE = "xml"

NAMESPACES = (
    "http://www.fssprus.ru/namespace/order/2017/2",
    "http://www.fssp.gov.ru/namespace/order_fssp/2022/1",
)

# Код не трактуется по английскому названию: REOPEN_CANCEL в наборе — отказ в возбуждении
DOC_TYPE2 = {
    "O_IP_ACT_END_END": "Постановление об окончании ИП",
    "O_IP_ACT_END_STOP": "Постановление о прекращении ИП",
    "O_IP_ACT_REOPEN_CANCEL": "Постановление об отказе в возбуждении ИП",
    "O_IP_ACT_RETURN": "Постановление об окончании и возвращении ИД",
    "O_IP_RES_REOPEN": "Постановление о возбуждении",
}
_DOC_NAME = {
    "O_IP_ACT_END_END": "постановление об окончании исполнительного производства",
    "O_IP_ACT_END_STOP": "постановление о прекращении исполнительного производства",
    "O_IP_ACT_REOPEN_CANCEL": "постановление об отказе в возбуждении исполнительного производства",
    "O_IP_ACT_RETURN": "постановление об окончании и возвращении ид взыскателю",
    "O_IP_RES_REOPEN": "постановление о возбуждении исполнительного производства",
}

DIRECT = (
    "DocType", "DebtorType", "DocDate", "IdDebtText", "IdDocNo", "IdDocDate", "IdDeloNo",
    "IdDeloDate", "IdDebtSum", "IpNo", "IdType", "DbtrAdr", "DbtrName",
)
DATES = ("DocDate", "IdDocDate", "IdDeloDate")
NUMBERS = ("IdDocNo", "IdDeloNo", "IpNo")
CORE = ("DocType", "DocDate", "DbtrName", "IdDocNo", "IdDebtSum")

_RE_MONEY = re.compile(r"\d+(\.\d{1,2})?")


def _local(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return ns, name
    return "", tag


def _suspicious_number(value: str) -> bool:
    v = value.strip()
    return v.count("(") != v.count(")") or bool(re.search(r"[\[\].,;:]$|^[\[\].,;:)]", v))


def process_fssp_xml(abs_path: Path, rel_path: str, doc_id: str) -> DocumentResult:
    doc = DocumentResult(doc_id=doc_id, path=rel_path, profile="fssp")
    doc.set(Field("FileName", rel_path, EXTRACTED, "registry"))

    def fail(code: str, message: str) -> DocumentResult:
        doc.status = FAILED
        doc.add_issue(Issue(code, message, STAGE, None, rel_path, [], "В таблице только путь файла", "Проверить исходный файл"))
        return doc

    try:
        root = SafeET.parse(str(abs_path)).getroot()
    except Exception as exc:  # повреждённый XML, DTD с сущностями, внешние ресурсы
        return fail("XML_PARSE_ERROR", f"XML не разобран: {type(exc).__name__}: {exc}")

    ns, root_name = _local(root.tag)
    if ns not in NAMESPACES or root_name != "OIp":
        return fail("UNSUPPORTED_XML_SCHEMA", f"Неподдерживаемая схема XML: корень «{root_name}», пространство имён «{ns}».")

    # Только непосредственные дочерние элементы корня: IpNo верхнего уровня и IP/IPNo — разные пути
    top: dict[str, list[tuple[int, str]]] = {}
    counters: dict[str, int] = {}
    nested: list[dict] = []
    for child in root:
        cns, name = _local(child.tag)
        if cns != ns:
            continue
        counters[name] = counters.get(name, 0) + 1
        if name == "IP":
            item = {_local(g.tag)[1]: (g.text or "") for g in child}
            item["_path"] = f"/f:OIp/f:IP[{counters[name]}]"
            nested.append(item)
        elif len(child) == 0:
            top.setdefault(name, []).append((counters[name], child.text or ""))

    for name in DIRECT:
        f = doc.set(Field(name))
        values = top.get(name, [])
        distinct = {v for _, v in values if v.strip()}
        if not distinct:
            continue
        f.source_refs = [f"/f:OIp/f:{name}" + (f"[{i}]" if len(values) > 1 else "") for i, _ in values]
        if len(distinct) > 1:
            f.state = NEEDS_REVIEW
            doc.add_issue(Issue(
                "SOURCE_CONFLICT", f"Элемент {name} повторяется на верхнем уровне с разными значениями.",
                STAGE, name, f"/f:OIp/f:{name}", sorted(distinct), "Ячейка оставлена пустой",
                "Выбрать значение по исходному документу",
            ))
            continue
        f.raw = f.value = next(v for _, v in values if v.strip())
        f.state = EXTRACTED

    for name in DATES:
        f = doc.fields[name]
        if f.value is None:
            continue
        try:
            f.value = date.fromisoformat(f.raw.strip())
        except ValueError:
            doc.add_issue(Issue(
                "INVALID_VALUE_FORMAT", f"{name}: «{f.raw}» не является датой.", STAGE, name,
                f.source_refs[0], [f.raw], "Значение сохранено как в источнике", "Сверить дату с документом",
            ))

    f = doc.fields["IdDebtSum"]
    id_debt_sum: Decimal | None = None
    if f.value is not None:
        if _RE_MONEY.fullmatch(f.raw.strip()):
            f.value = id_debt_sum = Decimal(f.raw.strip())
        else:
            doc.add_issue(Issue(
                "INVALID_VALUE_FORMAT", f"IdDebtSum: «{f.raw}» не является суммой.", STAGE, "IdDebtSum",
                f.source_refs[0], [f.raw], "Значение сохранено как в источнике", "Сверить сумму с документом",
            ))

    for name in NUMBERS:
        f = doc.fields[name]
        if f.value is not None and _suspicious_number(f.value):
            doc.add_issue(Issue(
                "INVALID_VALUE_FORMAT", f"{name}: в значении «{f.value}» лишние знаки.", STAGE, name,
                f.source_refs[0], [f.value], "Значение сохранено как в источнике", "Сверить номер с документом",
            ))

    for name in CORE:
        if doc.fields[name].value is None and doc.fields[name].state != NEEDS_REVIEW:
            doc.add_issue(Issue(
                "MISSING_CORE_FIELD", f"В XML нет обязательного элемента {name}.", STAGE, name,
                f"/f:OIp/f:{name}", [], "Ячейка пустая", "Проверить исходный файл",
            ))

    doc_type = doc.value("DocType")
    doc_name = next((v for _, v in top.get("DocName", []) if v.strip()), None)
    doc.extra["DocName"] = doc_name
    doc.extra["Id"] = next((v.strip() for _, v in top.get("Id", []) if v.strip()), None)
    t2 = doc.set(Field("DocType2", derivation="lookup", source_refs=["/f:OIp/f:DocType"]))
    if doc_type is not None:
        code = doc_type.strip()
        if code in DOC_TYPE2:
            t2.value, t2.state = DOC_TYPE2[code], EXTRACTED
            if doc_name and " ".join(doc_name.lower().split()) != _DOC_NAME[code]:
                doc.add_issue(Issue(
                    "SOURCE_CONFLICT", f"DocName «{doc_name}» не соответствует коду {code} по справочнику.",
                    STAGE, "DocType2", "/f:OIp/f:DocName", [doc_name, DOC_TYPE2[code]],
                    "DocType2 взят из справочника по коду", "Проверить тип постановления",
                ))
        else:
            t2.state = NEEDS_REVIEW
            doc.add_issue(Issue(
                "UNKNOWN_DOC_TYPE", f"Код DocType «{code}» отсутствует в справочнике.", STAGE, "DocType2",
                "/f:OIp/f:DocType", [doc_name] if doc_name else [], "DocType2 пустой: код не угадывается",
                "Добавить код в справочник",
            ))

    adr = doc.value("DbtrAdr")
    for name in ("street", "dom", "kv"):
        doc.set(Field(name, derivation="parsed", source_refs=["/f:OIp/f:DbtrAdr"]))
    if adr is not None:
        parts = parse_dbtr_adr(adr, doc.value("IdDebtText") or "")
        doc.extra["address_layout"] = parts.layout
        if parts.layout == "unknown":
            for name in ("street", "dom", "kv"):
                doc.fields[name].state = NEEDS_REVIEW
            doc.add_issue(Issue(
                "ADDRESS_FORMAT", "Формат DbtrAdr не распознан, адрес не разобран на части.", STAGE, "street",
                "/f:OIp/f:DbtrAdr", [adr], "street, dom, kv пустые; DbtrAdr сохранён", "Разобрать адрес вручную",
            ))
        else:
            for name, value in (("street", parts.street_column), ("dom", parts.house), ("kv", parts.flat)):
                if value:
                    doc.fields[name].value, doc.fields[name].state, doc.fields[name].raw = value, EXTRACTED, adr

    debt_fields, debt_issues, debt_extra = extract_debt_fields(
        doc.value("IdDebtText"), "/f:OIp/f:IdDebtText", id_debt_sum
    )
    for fld in debt_fields.values():
        doc.set(fld)
    for iss in debt_issues:
        doc.add_issue(iss)
    doc.extra.update(debt_extra)

    if nested:
        doc.extra["nested_ip"] = nested
        others = sorted({n.get("DbtrName", "") for n in nested} - {doc.value("DbtrName") or "", ""})
        doc.add_issue(Issue(
            "NESTED_PROCEEDINGS",
            f"В XML {len(nested)} вложенных производств(а)" + (f", в том числе с другим должником: {', '.join(others)}." if others else "."),
            STAGE, None, "/f:OIp/f:IP", others, "Строка построена по верхнему уровню постановления",
            "Действий не требуется", INFO,
        ))

    doc.finalize()
    return doc

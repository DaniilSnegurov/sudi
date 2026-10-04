"""Компоненты адреса должника из DbtrAdr в форме образца таблицы.

Другой адрес документа сюда не подставляется. Свободный текст (IdDebtText) используется
только чтобы подтвердить тип улицы, если та же улица названа там с типом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TYPE_FORMS = {
    "ул": "ул.", "улица": "ул.",
    "пер": "пер.", "переулок": "пер.",
    "пр-кт": "пр-кт", "просп": "пр-кт", "проспект": "пр-кт", "пр": "пр-кт",
    "городок": "городок.",
    "проезд": "проезд", "б-р": "б-р", "бульвар": "б-р",
    "ш": "ш.", "шоссе": "ш.", "наб": "наб.", "набережная": "наб.",
    "пл": "пл.", "площадь": "пл.", "туп": "туп.", "тупик": "туп.", "мкр": "мкр.",
}
_TYPE = "(" + "|".join(sorted(map(re.escape, _TYPE_FORMS), key=len, reverse=True)) + ")"
_L = r"(?<![А-Яа-яЁё\-])"
_R = r"(?![А-Яа-яЁё\-])\.?"
_RE_TYPE_PREFIX = re.compile(rf"^{_L}{_TYPE}{_R}\s+(.+)$", re.I)
_RE_TYPE_SUFFIX = re.compile(rf"^(.+?)\s+{_L}{_TYPE}{_R}$", re.I)
_RE_ORDINAL = re.compile(r"^(\d+)(?:-?(я|й|е))?\s+(\S.*)$", re.I)
_ZIP = re.compile(r"\d{6}")


@dataclass
class AddressParts:
    layout: str  # "codes" | "labeled" | "free" | "unknown"
    city: str = ""
    street_type: str = ""
    street: str = ""
    ordinal: str = ""
    house: str = ""
    flat: str = ""

    @property
    def street_column(self) -> str:
        street = " ".join(p for p in (self.street_type, self.street, self.ordinal) if p)
        return ", ".join(p for p in (self.city, street) if p)


def _strip_label(value: str, label: str) -> str:
    return re.sub(rf"^(?:{label})\.?\s*", "", value.strip(), flags=re.I).strip()


def _city(raw: str) -> str:
    raw = raw.strip()
    name = re.sub(r"^(г\.?|город)\s*(?=[А-ЯЁ])", "", raw)
    return f"г {name}" if name and name != raw or re.fullmatch(r"[А-ЯЁ][\w\- ]*", raw) else raw


def _ordinal_suffix(word: str) -> str:
    w = word.lower()
    if w.endswith(("ая", "яя")):
        return "я"
    if w.endswith(("ый", "ий", "ой")):
        return "й"
    if w.endswith(("ое", "ее")):
        return "е"
    return ""


def _street(raw: str, context: str) -> tuple[str, str, str]:
    """→ (тип, название, порядковый номер). Тип не выдумывается: нет подтверждения — пусто."""
    name, stype = " ".join(raw.split()), ""
    if m := _RE_TYPE_PREFIX.match(name):
        stype, name = _TYPE_FORMS[m.group(1).lower()], m.group(2)
        if inner := _RE_TYPE_PREFIX.match(name):  # «ул. пр. Тест» — действует ближайший к названию тип
            stype, name = _TYPE_FORMS[inner.group(1).lower()], inner.group(2)
    elif (m := _RE_TYPE_SUFFIX.match(name)) and not re.fullmatch(r"\d+(-?[яйе])?", m.group(1)):
        name, stype = m.group(1), _TYPE_FORMS[m.group(2).lower()]  # «2-й Проезд» — название, а не тип

    ordinal = ""
    if m := _RE_ORDINAL.match(name):
        suffix = (m.group(2) or _ordinal_suffix(m.group(3).split()[0])).lower()
        # «5 Образцовая» → «Образцовая 5-я»; «40 лет Примеров» остаётся названием
        if suffix and _ordinal_suffix(m.group(3).split()[0]):
            ordinal, name = f"{m.group(1)}-{suffix}", m.group(3)

    if not stype and name and context:
        esc = re.escape(name)
        after = re.search(rf"{_L}{esc}\s+{_L}{_TYPE}{_R}", context, re.I)
        before = re.search(rf"{_L}{_TYPE}{_R}\s+(?:\d+(?:-?[яйе])?\s+)?{esc}(?![А-Яа-яЁё])", context, re.I)
        if after and _TYPE_FORMS[after.group(1).lower()] != "ул.":
            stype = _TYPE_FORMS[after.group(1).lower()]
        elif before:
            stype = _TYPE_FORMS[before.group(1).lower()]
    return stype, name, ordinal


def _house(house: str, corp: str = "") -> str:
    """Форма образца: буква — «38/Б», цифровой корпус — «20 корп.1»."""
    house = _strip_label(" ".join(house.split()).strip(" .,;"), "дом|д")
    if m := re.fullmatch(r"(\d+\s*/?\s*)([ABCEHKMOPTX])", house):
        house = m.group(1) + m.group(2).translate(str.maketrans("ABCEHKMOPTX", "АВСЕНКМОРТХ"))
    if not corp:
        if m := re.fullmatch(r"(\d+)\s*([А-Яа-яЁё])", house):
            return f"{m.group(1)}/{m.group(2)}"
        if m := re.fullmatch(r"(\d+)\s*к\.?\s*(\d+)", house, re.I):
            return f"{m.group(1)} корп.{m.group(2)}"
        return house
    value = _strip_label(corp, "корпус|корп|к")
    if re.fullmatch(r"[А-Яа-яЁё]", value):
        return f"{house}/{value}"
    if value.isdigit():
        return f"{house} корп.{value}"
    return f"{house} {corp.strip()}"


def parse_dbtr_adr(raw: str, context: str = "") -> AddressParts:
    parts = [p.strip() for p in raw.split(",")]

    # 643,индекс,регион,район,город,нас.пункт,улица,дом,квартира
    if len(parts) == 9 and parts[0] == "643":
        stype, street, ordinal = _street(parts[6], context)
        return AddressParts("codes", _city(parts[4] or parts[5]), stype, street, ordinal, _house(parts[7]), parts[8])

    # индекс, Россия, регион, район, город, нас.пункт, улица, д. N, корп. K, кв. M
    if len(parts) == 10 and parts[1] == "Россия":
        stype, street, ordinal = _street(parts[6], context)
        house = _house(_strip_label(parts[7], "дом|д"), parts[8])
        return AddressParts("labeled", _city(parts[4] or parts[5]), stype, street, ordinal, house, _strip_label(parts[9], "кв"))

    # Свободная запись: «Эталонная ул., д. 9, кв.121, г. Тест, 644110»
    res = AddressParts("free")
    corp, rest = "", []
    for p in parts:
        if not p or _ZIP.fullmatch(p):
            continue
        if re.match(r"(г|город|пгт|пос|с)\.?\s+\S", p, re.I) and not res.city:
            res.city = _city(p)
        elif re.match(r"(д|дом)\.?\s*\d", p, re.I) and not res.house:
            res.house = _strip_label(p, "дом|д")
        elif re.match(r"(кв|ком|пом)\.?\s*\S", p, re.I) and not res.flat:
            res.flat = _strip_label(p, "кв")
        elif re.match(r"(корпус|корп|к)\.?\s*\S", p, re.I) and res.house and not corp:
            corp = p
        else:
            rest.append(p)
    typed = [p for p in rest if _RE_TYPE_PREFIX.match(p) or _RE_TYPE_SUFFIX.match(p)]
    if len(typed) != 1 or not res.house:
        return AddressParts("unknown")
    res.street_type, res.street, res.ordinal = _street(typed[0], context)
    res.house = _house(res.house, corp)
    return res

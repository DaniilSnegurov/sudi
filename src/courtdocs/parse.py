"""Нормализация значений, прочитанных из документа. Утраченные цифры не восстанавливаются."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_RE_DATE_NUM = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
_RE_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_RE_DATE_TXT = re.compile(r"(?<!\d)(\d{1,2})\W{0,3}\s*([а-яё]+)\s+(\d{4})(?!\d)", re.I)

_INT = r"(\d{1,3}(?:[  ]\d{3})+|\d+)"
_RE_RUB_KOP = re.compile(_INT + r"\s*руб\w*\.?\s*(\d{1,2})\s*коп", re.I)
_RE_DECIMAL = re.compile(r"(?<![\d.,])" + _INT + r"[.,](\d{2})(?!\d)")
_RE_INTEGER = re.compile(r"(?<![\d.,])" + _INT + r"(?![\d]|[.,]\d)")
_RE_CASE = re.compile(r"(?<!\d)(\d{1,2}-\d+(?:-\d+)?/\d{4})(?!\d)")

# Латинские двойники кириллических букв: чинятся только в словах, где уже есть кириллица
_HOMOGLYPHS = str.maketrans("ABCEHKMOPTXaceopxyÁáÓó", "АВСЕНКМОРТХасеорхуАаОо")
# Буквы других славянских алфавитов, которых в русском тексте быть не может
_FOREIGN = str.maketrans("ЎўІіЇїЄєҐґ", "УуИиИиЕеГг")
_RE_GLUED = re.compile(r"\d(?!руб|коп|р\b|г\b)[A-Za-zА-Яа-яЁё]|[A-Za-zА-Яа-яЁё]\d")
_RE_WORD = re.compile(r"[A-Za-zÁáÓóА-Яа-яЁё]+")
_RE_CYR = re.compile(r"[А-Яа-яЁё]")


def clean_text(text: str) -> str:
    """Версия для поиска по ключевым словам. Цифры не трогаются; исходный текст хранится отдельно."""
    text = text.translate(_FOREIGN)
    return _RE_WORD.sub(lambda m: m.group(0).translate(_HOMOGLYPHS) if _RE_CYR.search(m.group(0)) else m.group(0), text)


def squash(text: str) -> str:
    return " ".join(text.split())


def parse_date(text: str) -> date | None:
    """Первая однозначная дата в тексте: 06.01.2029, 2029-01-06 или «6 января 2029 г.»."""
    for rx, order in ((_RE_DATE_NUM, (3, 2, 1)), (_RE_DATE_ISO, (1, 2, 3))):
        if m := rx.search(text):
            try:
                return date(*(int(m.group(i)) for i in order))
            except ValueError:
                return None
    if m := _RE_DATE_TXT.search(text):
        word = m.group(2).lower()
        month = next((n for stem, n in _MONTHS.items() if word.startswith(stem) and (stem != "ма" or word in ("мая", "май"))), None)
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None
    return None


def parse_money(text: str) -> Decimal | None:
    """Сумма в рублях: «63008,92 руб.», «114 666 руб. 18 коп.», «4000 руб.»."""
    def num(s: str) -> str:
        return re.sub(r"[  ]", "", s)

    text = re.sub(r"[pр][yу][6б]", "руб", text, flags=re.I)  # «py6», «ру6» — частое чтение слова «руб»
    # Буква вплотную к цифре или ведущий ноль — цифра прочитана неверно; сумма не принимается
    if _RE_GLUED.search(text) or re.search(r"(?<![\d.,])0[  ]?\d", text):
        return None
    if m := _RE_RUB_KOP.search(text):
        return Decimal(f"{num(m.group(1))}.{m.group(2).zfill(2)}")
    if m := _RE_DECIMAL.search(text):
        return Decimal(f"{num(m.group(1))}.{m.group(2)}")
    if m := _RE_INTEGER.search(text):
        return Decimal(num(m.group(1))).quantize(Decimal("0.01"))
    return None


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text)


def parse_passport(text: str) -> str | None:
    """Серия и номер: 4 + 6 цифр. Иная структура не подгоняется."""
    d = digits_only(text)
    return f"{d[:4]} {d[4:]}" if len(d) == 10 else None


def parse_case_number(text: str) -> str | None:
    m = _RE_CASE.search(text.replace(" ", ""))
    return m.group(1) if m else None


def has_foreign_chars(value: str, allowed: str) -> bool:
    return bool(re.search(rf"[^{allowed}]", value))

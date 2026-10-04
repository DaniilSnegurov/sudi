"""ФИО в именительный падеж — только там, где форма определяется однозначно."""

from __future__ import annotations

import re
from functools import lru_cache

_MALE = re.compile(r"(ич)(а|у|ем|ом|е)?$", re.I)
_FEMALE = re.compile(r"(н)(а|ы|е|у|ой|ою)$", re.I)


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def _cap(word: str) -> str:
    return "-".join(p[:1].upper() + p[1:].lower() for p in word.split("-"))


def gender(patronymic: str) -> str | None:
    p = patronymic.strip().lower()
    if re.search(r"(ич|ича|ичу|ичем|иче|ичом)$", p):
        return "masc"
    if re.search(r"(вна|вны|вне|вну|вной|чна|чны|чне|чну|чной)$", p):
        return "femn"
    return None


def _patronymic(word: str, g: str) -> str:
    w = word.strip()
    if g == "masc":
        return _cap(_MALE.sub(r"\1", w))
    return _cap(_FEMALE.sub(r"\1а", w))


def _surname(word: str, g: str) -> tuple[str, bool]:
    w = word.strip()
    low = w.lower()
    rules = (
        (("masc",), r"(ов|ев|ёв|ин|ын)(а|у|ым|е)$", r"\1"),
        (("masc",), r"(ск|цк)(ого|ому|им|ом)$", r"\1ий"),
        (("femn",), r"(ов|ев|ёв|ин|ын)(ой|ою|у)$", r"\1а"),
        (("femn",), r"(ск|цк)(ой|ую|ою)$", r"\1ая"),
    )
    for genders, pattern, repl in rules:
        if g in genders and re.search(pattern, low):
            return _cap(re.sub(pattern, repl, low)), True
    if (g == "masc" and re.search(r"(ов|ев|ёв|ин|ын|ий|ый|ой)$", low)) or (g == "femn" and re.search(r"(ова|ева|ёва|ина|ына|ая)$", low)):
        return _cap(w), True  # уже именительный
    return _cap(w), False


def _first_name(word: str, g: str) -> tuple[str, bool]:
    w = word.strip()
    for p in _morph().parse(w):
        if "Name" in p.tag and p.tag.gender == g:
            form = p.inflect({"nomn", "sing"})
            if form:
                return _cap(form.word.replace("ё", "е") if "ё" not in w.lower() else form.word), True
    return _cap(w), False


def to_nominative(surname: str, first_name: str, patronymic: str) -> tuple[str, str, str, bool]:
    """→ (фамилия, имя, отчество, уверенность). Без распознанного отчества форма не меняется."""
    g = gender(patronymic)
    if g is None:
        return surname.strip(), first_name.strip(), patronymic.strip(), False
    s, ok_s = _surname(surname, g)
    f, ok_f = _first_name(first_name, g)
    return s, f, _patronymic(patronymic, g), ok_s and ok_f


def person_key(surname: str, first_name: str, patronymic: str) -> tuple[str, str, str]:
    return tuple(x.strip().lower().replace("ё", "е") for x in (surname, first_name, patronymic))

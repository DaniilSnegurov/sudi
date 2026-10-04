"""Суммы и период основного долга из свободного текста IdDebtText (правила, без модели).

Текст не переписывается: каждое значение хранит позицию и цитату в исходной строке.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .models import EXTRACTED, NEEDS_REVIEW, NOT_FOUND, Field, Issue

STAGE = "debt_text"

_NUM = r"(?<!\d)(\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?!\d)"
# «в размере N», «пошлины по N руб. с каждого», «N руб. K коп.»; пробелы после OCR могут пропадать
_RE_AMOUNT = re.compile(
    r"(?:в\s*(?:общем\s*)?размере|пошлин\w*\s+(?:по\s*)?)\s*" + _NUM + r"(?:\s*руб\w*\.?\s*(\d{1,2})\s*коп)?", re.I
)
_RE_EACH = re.compile(r"^\W{0,3}\w{0,6}\W{0,3}с\s+каждого", re.I)
_RE_PLACEHOLDER = re.compile(r"в\s+размере\s+_+")
_RE_PERIOD = re.compile(
    r"за\s*период\s*(?:с\s*)?(\d{1,2}\.\d{1,2}\.\d+)\s*(?:года|г\.?)?\s*(?:по|[-–—])\s*(\d{1,2}\.\d{1,2}\.\d+)",
    re.I,
)
_RE_COMPONENT = re.compile(r"(отопление|ГВС|горяч\w*\s+вод\w*)\s*[-–—:]\s*" + _NUM, re.I)
# Любая сумма «N … руб» — для поиска сумм, назначение которых правила не определили
_RE_RUB = re.compile(_NUM + r"\s*(?:[а-яё]+\s+){0,8}?руб", re.I)
_RE_DATE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")

_KEYWORDS = (
    ("principal", re.compile(r"задолженност", re.I)),
    ("peni", re.compile(r"\bпен[иейяю]\b|неустойк", re.I)),
    ("poshlina", re.compile(r"пошлин", re.I)),
    ("post", re.compile(r"почтов|корреспонденц", re.I)),
    ("costs", re.compile(r"судебн\w+\s+расход|издерж", re.I)),
)
# «Судебные расходы по отправке почтовой корреспонденции» — уточнение, а не два назначения
_COMPATIBLE = ({"costs", "post"}, {"costs", "poshlina"})

_FIELD_BY_CATEGORY = {
    "principal": "rub_deb",
    "peni": "rub_peni",
    "poshlina": "rub_poshlina",
    "post": "rub_post",
}


@dataclass
class Period:
    start_raw: str
    end_raw: str
    span: tuple[int, int]
    start: date | None = None
    end: date | None = None

    @property
    def valid(self) -> bool:
        return self.start is not None and self.end is not None and self.start <= self.end

    @property
    def raw(self) -> str:
        return f"{self.start_raw} — {self.end_raw}"


@dataclass
class Amount:
    category: str  # principal | peni | poshlina | post | costs | ambiguous | unknown
    value: Decimal
    span: tuple[int, int]
    quote: str
    periods: list[Period] = field(default_factory=list)
    each: bool = False  # «с каждого»


@dataclass
class ParsedDebtText:
    amounts: list[Amount] = field(default_factory=list)
    components: list[Amount] = field(default_factory=list)
    periods: list[Period] = field(default_factory=list)
    placeholders: list[tuple[str, tuple[int, int]]] = field(default_factory=list)
    unattributed: list[Amount] = field(default_factory=list)


def _money(int_part: str, frac: str | None, kop: str | None = None) -> Decimal:
    digits = re.sub(r"[  ]", "", int_part)
    cents = kop.zfill(2) if kop and not frac else (frac or "0").ljust(2, "0")
    return Decimal(f"{digits}.{cents}")


def _broken_number(int_part: str, tail: str = "") -> bool:
    """«0 538» или «4об0»: цифра потеряна либо прочитана буквой — значение не принимается."""
    glued = re.match(r"(?!\s*(?:руб|py6|ру6|коп|р\b))[A-Za-zА-Яа-яЁё]+\d", tail)
    return bool(re.match(r"0[\d  ]", int_part)) or bool(glued)


def parse_ru_date(raw: str) -> date | None:
    m = _RE_DATE.fullmatch(raw)
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _keywords(text: str, start: int, end: int) -> list[tuple[int, str]]:
    found = [(m.start(), cat) for cat, rx in _KEYWORDS for m in rx.finditer(text, start, end)]
    return sorted(found)


def _category(kws: list[tuple[int, str]]) -> str:
    if not kws:
        return "unknown"
    cats = {c for _, c in kws}
    if len(cats) > 1 and cats not in _COMPATIBLE:
        return "ambiguous"
    if cats == {"costs", "post"}:
        return "post"
    if cats == {"costs", "poshlina"}:
        return "poshlina"
    return kws[-1][1]


def parse(text: str) -> ParsedDebtText:
    res = ParsedDebtText()
    clause_kws: list[list[tuple[int, str]]] = []
    # Прочерк вместо суммы («в размере _») тоже закрывает свой фрагмент текста
    marks = sorted(
        [(m.start(), m, True) for m in _RE_AMOUNT.finditer(text)]
        + [(m.start(), m, False) for m in _RE_PLACEHOLDER.finditer(text)],
        key=lambda t: t[0],
    )
    prev_end = 0
    for _, m, is_amount in marks:
        kws = _keywords(text, prev_end, m.start())
        if is_amount and not _broken_number(m.group(1), text[m.end() : m.end() + 8]):
            clause_kws.append(kws)
            category = "poshlina" if m.group(0).lower().startswith("пошлин") else _category(kws)
            amount = Amount(category, _money(m.group(1), m.group(2), m.group(3)), m.span(), m.group(0))
            amount.each = bool(_RE_EACH.match(text[m.end() : m.end() + 30]))
            res.amounts.append(amount)
        else:
            res.placeholders.append((_category(kws), m.span()))
        prev_end = m.end()

    for m in _RE_COMPONENT.finditer(text):
        res.components.append(Amount("component", _money(m.group(2), m.group(3)), m.span(), m.group(0)))

    for m in _RE_PERIOD.finditer(text):
        per = Period(m.group(1), m.group(2), m.span(), parse_ru_date(m.group(1)), parse_ru_date(m.group(2)))
        res.periods.append(per)
        # Период относится к сумме своего фрагмента, если назван после её ключевого слова;
        # иначе это «хвост» предыдущей суммы («пени в размере X за период …»).
        idx = next((i for i, a in enumerate(res.amounts) if a.span[0] > m.start()), None)
        if idx is None:
            owner = res.amounts[-1] if res.amounts else None
        else:
            before = [c for pos, c in clause_kws[idx] if pos < m.start()]
            if before:
                owner = res.amounts[idx] if res.amounts[idx].category in (before[-1], "ambiguous") else None
            else:
                owner = res.amounts[idx - 1] if idx > 0 else None
        if owner is not None:
            owner.periods.append(per)

    taken = [a.span for a in res.amounts] + [c.span for c in res.components]
    for m in _RE_RUB.finditer(text):
        s, e = m.span(1)[0], m.end()
        if not any(ts <= s < te for ts, te in taken):
            res.unattributed.append(Amount("unknown", _money(m.group(1), m.group(2)), (s, e), m.group(0)))
    return res


def _ref(xml_path: str, span: tuple[int, int]) -> str:
    return f"{xml_path}#chars={span[0]}-{span[1]}"


def _fmt(v: Decimal) -> str:
    return f"{v:.2f}"


def extract_debt_fields(
    text: str | None, xml_path: str, id_debt_sum: Decimal | None
) -> tuple[dict[str, Field], list[Issue], dict]:
    """Возвращает поля date_start, date_end, rub_deb, rub_peni, rub_poshlina, rub_post."""
    names = ("date_start", "date_end", "rub_deb", "rub_peni", "rub_poshlina", "rub_post")
    fields = {n: Field(n, derivation="parsed") for n in names}
    issues: list[Issue] = []
    extra: dict = {}
    if not text or not text.strip():
        return fields, issues, extra

    parsed = parse(text)
    by_cat: dict[str, list[Amount]] = {}
    for a in parsed.amounts:
        by_cat.setdefault(a.category, []).append(a)

    def issue(code: str, message: str, fld: str | None, cands: list[str], effect: str, action: str, span=None):
        issues.append(
            Issue(code, message, STAGE, fld, _ref(xml_path, span) if span else xml_path, cands, effect, action)
        )

    for per in parsed.periods:
        if not per.valid:
            issue(
                "INVALID_PERIOD",
                f"В тексте повреждённый или невозможный период: «{per.raw}».",
                "date_start",
                [per.raw],
                "Период не восстанавливается по догадке",
                "Сверить даты периода с исходным документом",
                per.span,
            )

    principals = by_cat.get("principal", [])
    if len({(a.value, tuple(p.raw for p in a.periods)) for a in principals}) > 1:
        cands = [f"{_fmt(a.value)} ({'; '.join(p.raw for p in a.periods) or 'без периода'})" for a in principals]
        issue(
            "MULTIPLE_OBLIGATIONS",
            f"В тексте {len(principals)} самостоятельных обязательства с разными суммами или периодами.",
            "rub_deb",
            cands,
            "Суммы и период оставлены пустыми: один столбец не передаёт несколько обязательств",
            "Внести суммы вручную по каждому обязательству",
        )
        for n in names:
            fields[n].state = NEEDS_REVIEW
        extra["obligations"] = cands
        return fields, issues, extra

    for cat, fname in _FIELD_BY_CATEGORY.items():
        items = by_cat.get(cat, [])
        f = fields[fname]
        if len({a.value for a in items}) == 1:
            a = items[0]
            f.value, f.state, f.raw = a.value, EXTRACTED, a.quote
            f.source_refs = [_ref(xml_path, x.span) for x in items]
        elif items:
            f.state = NEEDS_REVIEW
            issue(
                "MULTIPLE_OBLIGATIONS",
                f"Для поля {fname} в тексте несколько разных сумм.",
                fname,
                [_fmt(a.value) for a in items],
                "Ячейка оставлена пустой",
                "Выбрать сумму по исходному документу",
            )
        for pcat, span in parsed.placeholders:
            if pcat == cat and f.value is None:
                f.state = NEEDS_REVIEW
                issue(
                    "INVALID_VALUE_FORMAT",
                    f"Вместо суммы для поля {fname} в тексте стоит прочерк.",
                    fname,
                    [text[span[0] : span[1]]],
                    "Ячейка оставлена пустой",
                    "Проверить сумму по исходному документу",
                    span,
                )
                break

    deb = fields["rub_deb"]
    comp_sum = sum((c.value for c in parsed.components), Decimal("0"))
    if parsed.components:
        extra["components"] = [c.quote for c in parsed.components]
        if deb.value is None and deb.state == NOT_FOUND:
            deb.value, deb.state, deb.derivation = comp_sum, EXTRACTED, "sum_of_components"
            deb.operands = [_fmt(c.value) for c in parsed.components]
            deb.source_refs = [_ref(xml_path, c.span) for c in parsed.components]
        elif deb.value is not None and comp_sum != deb.value:
            issue(
                "AMOUNT_RECONCILIATION",
                f"Расшифровка основного долга ({_fmt(comp_sum)}) не равна указанному итогу ({_fmt(deb.value)}).",
                "rub_deb",
                [_fmt(deb.value), _fmt(comp_sum)],
                "Сохранён итог из текста, расшифровка не подставлялась",
                "Сверить составляющие основного долга",
            )

    if len(principals) >= 1 and deb.state == EXTRACTED:
        periods = {(p.start_raw, p.end_raw): p for a in principals for p in a.periods}
        ds, de = fields["date_start"], fields["date_end"]
        if len(periods) == 1:
            per = next(iter(periods.values()))
            if per.valid:
                ds.value, de.value = per.start, per.end
                ds.state = de.state = EXTRACTED
                ds.raw, de.raw = per.start_raw, per.end_raw
                ds.source_refs = de.source_refs = [_ref(xml_path, per.span)]
            else:
                ds.state = de.state = NEEDS_REVIEW
        elif len(periods) > 1:
            ds.state = de.state = NEEDS_REVIEW
            issue(
                "INVALID_PERIOD",
                "У основного долга указано несколько разных периодов.",
                "date_start",
                [p.raw for p in periods.values()],
                "Период оставлен пустым",
                "Выбрать период по исходному документу",
            )

    peni_periods = [p.raw for a in by_cat.get("peni", []) for p in a.periods]
    if peni_periods:
        extra["peni_period"] = peni_periods

    costs = by_cat.get("costs", [])
    if costs:
        extra["court_costs"] = [_fmt(a.value) for a in costs]
        issue(
            "UNSPECIFIED_COSTS",
            "«Судебные расходы» указаны без расшифровки: неясно, это пошлина или почтовые расходы.",
            "rub_post",
            [_fmt(a.value) for a in costs],
            "Сумма не записана ни в rub_poshlina, ни в rub_post",
            "Определить назначение расходов по исходному документу",
            costs[0].span,
        )

    unclear = by_cat.get("ambiguous", []) + by_cat.get("unknown", []) + parsed.unattributed
    if unclear:
        issue(
            "UNCLEAR_OWNER",
            "В тексте есть суммы, назначение которых не удалось определить.",
            "rub_deb" if deb.value is None else None,
            [a.quote.strip() for a in unclear],
            "Эти суммы не попали ни в один столбец",
            "Определить назначение сумм по исходному документу",
            unclear[0].span,
        )

    if id_debt_sum is not None and deb.value is not None and not unclear:
        parts = [fields[n].value for n in _FIELD_BY_CATEGORY.values() if fields[n].value is not None]
        total = sum(parts, Decimal("0")) + sum((a.value for a in costs), Decimal("0"))
        if total != id_debt_sum:
            issue(
                "AMOUNT_RECONCILIATION",
                f"Сумма составляющих из текста ({_fmt(total)}) не равна IdDebtSum ({_fmt(id_debt_sum)}), "
                f"разница {_fmt(abs(total - id_debt_sum))}.",
                None,
                [_fmt(total), _fmt(id_debt_sum)],
                "Значения сохранены как в источнике, ничего не подгонялось",
                "Сверить суммы с исходным документом",
            )
    return fields, issues, extra

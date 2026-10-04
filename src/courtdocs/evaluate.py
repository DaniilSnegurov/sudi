"""Сравнение результата с эталонной разметкой. Эталон — средство оценки, не вход конвейера."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

MONEY = {"IdDebtSum", "rub_deb", "rub_peni", "rub_poshlina", "rub_post", "дз_осн", "дз_пени", "дз_пошлина"}
PEOPLE = {"соответчики_фио"}


def _people_key(value: str) -> str:
    """Состав участников сравнивается по людям, а не по падежу и прежним фамилиям в скобках."""
    import re

    from .names import person_key, to_nominative

    keys = []
    for name in str(value).split(","):
        tokens = re.sub(r"\([^)]*\)", " ", name).split()
        if len(tokens) >= 3:
            keys.append(" ".join(person_key(*to_nominative(*tokens[:3])[:3])))
        elif tokens:
            keys.append(" ".join(tokens).lower())
    return "; ".join(sorted(keys))


def _norm(column: str, value) -> str:
    """Пустота не равна нулю; деньги сравниваются до копейки; текст — без различий в пробелах."""
    if value is None or value == "":
        return ""
    if isinstance(value, date):
        return value.isoformat()
    if column in PEOPLE:
        return _people_key(value)
    if column in MONEY:
        try:
            return f"{Decimal(str(value)):.2f}"
        except InvalidOperation:
            return str(value)
    return " ".join(str(value).split())


def load_labels(path: Path, key: str) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def compare(rows: list[dict], labels: dict[str, dict], key: str, columns: tuple[str, ...]) -> dict:
    """→ {"mismatches": [(файл, столбец, получено, эталон)], "accuracy": {столбец: доля}, ...}."""
    got = {r[key]: r for r in rows}
    mismatches: list[tuple[str, str, str, str]] = []
    correct = {c: 0 for c in columns if c != key}
    common = sorted(set(got) & set(labels))
    for k in common:
        for c in correct:
            a, b = _norm(c, got[k].get(c)), _norm(c, labels[k].get(c))
            if a == b:
                correct[c] += 1
            else:
                mismatches.append((k, c, a, b))
    n = len(common) or 1
    return {
        "compared": len(common),
        "missing": sorted(set(labels) - set(got)),
        "extra": sorted(set(got) - set(labels)),
        "accuracy": {c: correct[c] / n for c in correct},
        "mismatches": mismatches,
    }


def format_report(report: dict) -> str:
    lines = [f"Сопоставлено строк: {report['compared']}"]
    if report["missing"]:
        lines.append("Нет в результате: " + ", ".join(report["missing"]))
    if report["extra"]:
        lines.append("Нет в эталоне: " + ", ".join(report["extra"]))
    total = sum(report["accuracy"].values()) / (len(report["accuracy"]) or 1)
    lines.append(f"Совпадение ячеек с эталоном: {total:.1%}")
    for c, acc in report["accuracy"].items():
        if acc < 1:
            lines.append(f"  {c}: {acc:.1%}")
    lines.append(f"Расхождений: {len(report['mismatches'])}")
    for k, c, a, b in report["mismatches"]:
        lines.append(f"  {k} · {c}: получено «{a[:80]}», в эталоне «{b[:80]}»")
    return "\n".join(lines)

"""Внутреннее представление результата: поля с источниками, проблемы, документ."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import date
from decimal import Decimal
from typing import Any

# Состояния поля (спецификация, раздел 13)
EXTRACTED = "extracted"
NOT_FOUND = "not_found"
NEEDS_REVIEW = "needs_review"

# Состояния документа
PROCESSED = "processed"
DOC_NEEDS_REVIEW = "needs_review"
FAILED = "failed"

# Проблема требует проверки сотрудником либо только информирует
REVIEW = "review"
INFO = "info"


@dataclass
class Issue:
    code: str
    message: str
    stage: str
    field: str | None = None
    location: str | None = None
    candidates: list[str] = dc_field(default_factory=list)
    export_effect: str = ""
    action: str = ""
    severity: str = REVIEW


@dataclass
class Field:
    name: str
    value: Any = None
    state: str = NOT_FOUND
    derivation: str = "direct"
    source_refs: list[str] = dc_field(default_factory=list)
    raw: str | None = None
    operands: list[str] = dc_field(default_factory=list)
    issue_codes: list[str] = dc_field(default_factory=list)


@dataclass
class DocumentResult:
    doc_id: str
    path: str
    profile: str | None = None  # "fssp" | "judicial" | None
    status: str = PROCESSED
    sha256: str = ""
    paired_pdf: str | None = None
    duplicates: list[str] = dc_field(default_factory=list)
    fields: dict[str, Field] = dc_field(default_factory=dict)
    issues: list[Issue] = dc_field(default_factory=list)
    extra: dict[str, Any] = dc_field(default_factory=dict)

    def set(self, f: Field) -> Field:
        self.fields[f.name] = f
        return f

    def value(self, name: str) -> Any:
        f = self.fields.get(name)
        return f.value if f else None

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)
        if issue.field and issue.field in self.fields:
            self.fields[issue.field].issue_codes.append(issue.code)

    def finalize(self) -> None:
        if self.status != FAILED and any(i.severity == REVIEW for i in self.issues):
            self.status = DOC_NEEDS_REVIEW


def to_jsonable(obj: Any) -> Any:
    """Машиночитаемый результат: деньги — десятичной строкой, даты — ISO."""
    if isinstance(obj, Decimal):
        return f"{obj:.2f}"
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj

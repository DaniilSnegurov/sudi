"""Обработка пакета: один сбойный файл не останавливает остальные."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .fssp_xml import process_fssp_xml
from .judicial import Context, process_judicial_pdf
from .models import FAILED, INFO, DocumentResult, Issue
from .registry import Entry, register, safe_extract


@dataclass
class BatchResult:
    root: Path
    entries: list[Entry] = field(default_factory=list)
    documents: list[DocumentResult] = field(default_factory=list)

    def by_profile(self, profile: str) -> list[DocumentResult]:
        return [d for d in self.documents if d.profile == profile]


def _check_pair(doc: DocumentResult, pdf: Entry) -> None:
    """Пара по имени подтверждается идентификатором постановления в тексте PDF."""
    doc.paired_pdf = pdf.rel_path
    xml_id = doc.extra.get("Id")
    try:
        with pymupdf.open(pdf.abs_path) as pdf_doc:
            text = "".join(page.get_text() for page in pdf_doc)
    except Exception as exc:
        doc.add_issue(Issue(
            "PAIR_UNREADABLE", f"Парный PDF не открылся: {type(exc).__name__}.", "registry", None, pdf.rel_path,
            [], "Строка построена по XML", "Действий не требуется", INFO,
        ))
        return
    digits = "".join(text.split())
    if not xml_id or not digits:
        return  # нет идентификатора либо текстового слоя — сверить нечем
    if xml_id not in digits:
        doc.add_issue(Issue(
            "PAIR_MISMATCH", f"В парном PDF не найден идентификатор постановления {xml_id}.", "registry", None,
            pdf.rel_path, [xml_id], "Строка построена по XML", "Проверить, что PDF относится к этому XML",
        ))


def _failed(entry: Entry, code: str, message: str, profile: str | None = None) -> DocumentResult:
    doc = DocumentResult(entry.doc_id, entry.rel_path, profile, FAILED, entry.sha256)
    doc.add_issue(Issue(code, message, "registry", None, entry.rel_path, [], "Файл не попал в таблицы", "Обработать вручную"))
    return doc


def run(input_path: Path, work_dir: Path | None = None, include: tuple[str, ...] = (), ctx: Context | None = None,
        on_document=None) -> BatchResult:
    """ctx — OCR и модели для судебных PDF; без него обрабатываются только XML.
    on_document(документ, запись, сделано, всего) вызывается после каждого документа — для интерфейса."""
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        root = safe_extract(input_path, (work_dir or input_path.parent) / input_path.stem)
    else:
        root = input_path
    batch = BatchResult(root.resolve(), register(root, include))
    todo = [e for e in batch.entries if not (e.duplicate_of or e.paired_with) and e.kind in ("xml", "pdf")]

    for e in batch.entries:
        if e.duplicate_of or e.paired_with:
            continue  # копии и парные PDF не создают отдельных строк
        started = time.time()
        if e.kind == "xml":
            try:
                doc = process_fssp_xml(e.abs_path, e.rel_path, e.doc_id)
                if e.paired_pdf is not None and doc.status != FAILED:
                    _check_pair(doc, e.paired_pdf)
                    doc.finalize()
            except Exception as exc:  # непредвиденный сбой одного файла
                doc = _failed(e, "PROCESSING_ERROR", f"Сбой обработки: {type(exc).__name__}: {exc}", "fssp")
        elif e.kind == "pdf" and ctx is not None:
            try:
                doc = process_judicial_pdf(e.abs_path, e.rel_path, e.doc_id, e.sha256, ctx)
            except Exception as exc:
                doc = _failed(e, "PROCESSING_ERROR", f"Сбой обработки: {type(exc).__name__}: {exc}", "judicial")
        elif e.kind == "pdf":
            doc = _failed(e, "MODELS_DISABLED", "Судебный PDF не обработан: запуск без OCR и моделей.")
        else:
            continue  # не документ (описание набора, таблицы): учитывается только в перечне файлов отчёта
        doc.sha256, doc.duplicates = e.sha256, list(e.duplicates)
        doc.extra["seconds"] = round(time.time() - started, 2)
        batch.documents.append(doc)
        if on_document:
            on_document(doc, e, len(batch.documents), len(todo))
    return batch

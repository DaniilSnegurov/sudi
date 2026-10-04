"""Регистрация пакета: безопасная распаковка, контрольные суммы, дубликаты, пары XML/PDF."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

MAX_UNPACKED_BYTES = 2 * 1024**3
_SKIP_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}


@dataclass
class Entry:
    rel_path: str
    abs_path: Path
    kind: str  # "xml" | "pdf" | "other"
    sha256: str
    paired_pdf: "Entry | None" = None
    paired_with: str | None = None  # для PDF: путь XML, к которому он относится
    duplicate_of: str | None = None
    duplicates: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.sha256[:12]


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    # Без флага UTF-8 русские имена в ZIP обычно записаны в cp866
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp866")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def safe_extract(zip_path: Path, target: Path) -> Path:
    """Распаковывает архив только внутрь target; абсолютные пути и «..» запрещены."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = _zip_member_name(info).replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
                raise ValueError(f"Недопустимый путь в архиве: {name}")
            dest = (target / pure).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f"Недопустимый путь в архиве: {name}")
            total += info.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Архив превышает допустимый объём после распаковки")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def register(root: Path, include: tuple[str, ...] = ()) -> list[Entry]:
    """Обходит каталог пакета. include — подкаталоги относительно корня (по умолчанию всё)."""
    root = root.resolve()
    bases = [root / p for p in include] if include else [root]
    files = sorted({f for b in bases for f in b.rglob("*") if f.is_file()})
    entries: list[Entry] = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        if f.name.lower() in _SKIP_NAMES or any(part.startswith((".", "__")) for part in f.relative_to(root).parts):
            continue
        ext = f.suffix.lower()
        kind = "xml" if ext == ".xml" else "pdf" if ext == ".pdf" else "other"
        entries.append(Entry(rel, f, kind, _sha256(f)))

    first: dict[str, Entry] = {}
    for e in entries:
        if e.sha256 in first:
            e.duplicate_of = first[e.sha256].rel_path
            first[e.sha256].duplicates.append(e.rel_path)
        else:
            first[e.sha256] = e

    # Пара — одинаковое имя в одной папке; реквизиты сверяются на этапе обработки
    primary = {e.rel_path: e for e in entries if not e.duplicate_of}
    for e in primary.values():
        if e.kind == "xml":
            pdf = primary.get(str(PurePosixPath(e.rel_path).with_suffix(".pdf")))
            if pdf is not None:
                e.paired_pdf, pdf.paired_with = pdf, e.rel_path
    return entries

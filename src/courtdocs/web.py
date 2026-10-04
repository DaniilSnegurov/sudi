"""Веб-интерфейс: загрузка документов, проверка результата сотрудником, выгрузка таблиц.

Запуск: python -m uvicorn courtdocs.web:app --port 8765
Рабочая папка задаётся переменной COURTDOCS_WORKDIR (по умолчанию out/).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pymupdf
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from . import judicial
from .export import FSSP_COLUMNS, ISSUE_COLUMNS, JUDICIAL_COLUMNS, _write_table
from .judicial import Context
from .llm import DEFAULT_MODEL, DEFAULT_URL, DEFAULT_VLM, ModelClient
from .models import to_jsonable
from .ocr import RENDER_DPI, OcrEngine
from .parse import parse_date
from .pipeline import run

WORK = Path(os.environ.get("COURTDOCS_WORKDIR", "out")).resolve()
STORE = WORK / "web" / "store.json"
STATIC = Path(__file__).parent / "static"

DATE_COLUMNS = {"дата рождения", "дело_дата", "период_дз_начало", "период_дз_оконч", "DocDate", "IdDocDate", "IdDeloDate", "date_start", "date_end"}
MONEY_COLUMNS = {"дз_осн", "дз_пени", "дз_пошлина", "IdDebtSum", "rub_deb", "rub_peni", "rub_poshlina", "rub_post"}
INT_COLUMNS = {"соответчики_кол-во"}
COLUMNS = {"judicial": JUDICIAL_COLUMNS, "fssp": FSSP_COLUMNS}
PATH_COLUMN = {"judicial": "Файл", "fssp": "FileName"}

app = FastAPI(title="Судебные документы")
_lock = threading.RLock()
_jobs: dict[str, dict] = {}
_ocr = OcrEngine()
_settings = {"model": DEFAULT_MODEL, "vlm_model": DEFAULT_VLM, "model_url": DEFAULT_URL, "use_vlm": True,
             "conf_threshold": judicial.CONF_THRESHOLD}


# --- хранилище ---------------------------------------------------------------

def _load() -> dict:
    if STORE.is_file():
        return json.loads(STORE.read_text(encoding="utf-8"))
    return {"documents": {}, "settings": {}}


def _save(store: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STORE)


with _lock:
    _settings.update(_load().get("settings", {}))


def _value(doc: dict, column: str):
    """Итоговое значение: ручная правка сотрудника важнее автоматического результата."""
    if column in doc.get("manual", {}):
        return doc["manual"][column]["value"]
    if column == PATH_COLUMN.get(doc.get("profile") or "", ""):
        return doc["path"]
    f = doc["fields"].get(column)
    return f["value"] if f else None


def _review_issues(doc: dict) -> list[dict]:
    """Замечания, требующие проверки и ещё не снятые ручной правкой поля."""
    return [i for i in doc["issues"] if i["severity"] == "review" and i.get("field") not in doc.get("manual", {})]


def _status(doc: dict) -> str:
    if doc["status"] == "failed":
        return "failed"
    if doc.get("confirmed"):
        return "confirmed"
    return "needs_review" if _review_issues(doc) else "processed"


def _typed(column: str, value):
    if value is None or value == "":
        return None
    if column in DATE_COLUMNS:
        return parse_date(str(value)) or str(value)
    if column in MONEY_COLUMNS:
        try:
            return Decimal(str(value).replace(",", ".").replace(" ", ""))
        except InvalidOperation:
            return str(value)
    if column in INT_COLUMNS:
        return int(value) if str(value).isdigit() else str(value)
    return str(value)


# --- обработка ---------------------------------------------------------------

def _context() -> Context | None:
    llm = ModelClient(_settings["model"], _settings["model_url"])
    if not llm.available():
        return None
    vlm = None
    if _settings.get("use_vlm"):
        vlm = llm if _settings["vlm_model"] == _settings["model"] else ModelClient(_settings["vlm_model"], _settings["model_url"])
        if vlm is not llm and not vlm.available():
            vlm = None
    judicial.CONF_THRESHOLD = float(_settings["conf_threshold"])
    return Context(_ocr, llm, vlm, WORK / "cache")


def _process(job_id: str, root: Path, include: tuple[str, ...]) -> None:
    job = _jobs[job_id]
    try:
        ctx = _context()
        job["models"] = ctx is not None

        def on_document(doc, entry, done, total):
            data = to_jsonable(doc)
            data.update(abs_path=str(entry.abs_path), root=str(root), batch=job_id, processed_at=datetime.now().isoformat(timespec="seconds"),
                        paired_abs=str(entry.paired_pdf.abs_path) if entry.paired_pdf else None)
            with _lock:
                store = _load()
                old = store["documents"].get(doc.doc_id, {})
                data["manual"] = old.get("manual", {})  # повторная обработка не затирает правки сотрудника
                data["confirmed"] = False
                store["documents"][doc.doc_id] = data
                _save(store)
            job.update(done=done, total=total, current=doc.path)

        run(root, WORK / "web" / "unpacked", include, ctx, on_document)
        job["state"] = "done"
    except Exception as exc:  # задание не должно ронять сервер
        job.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    job["finished"] = time.time()


def _start(root: Path, include: tuple[str, ...] = ()) -> dict:
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"id": job_id, "state": "running", "done": 0, "total": 0, "current": "", "started": time.time(), "root": str(root)}
    threading.Thread(target=_process, args=(job_id, root, include), daemon=True).start()
    return _jobs[job_id]


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    """PDF, XML или ZIP. Имена файлов очищаются: путь из запроса не используется."""
    batch_dir = WORK / "web" / "inbox" / datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "file").name
        if Path(name).suffix.lower() not in (".pdf", ".xml", ".zip"):
            continue
        (batch_dir / name).write_bytes(await f.read())
        saved.append(name)
    if not saved:
        raise HTTPException(400, "Нужны файлы PDF, XML или ZIP")
    zips = [n for n in saved if n.lower().endswith(".zip")]
    root = batch_dir / zips[0] if len(saved) == 1 and zips else batch_dir
    return _start(root)


@app.post("/api/process-folder")
def process_folder(payload: dict = Body(...)) -> dict:
    # только папки внутри каталога запуска или рабочей папки: чужие каталоги сервера через интерфейс недоступны
    root = Path(payload.get("path", "")).expanduser().resolve()
    allowed = (Path.cwd().resolve(), WORK)
    if not root.is_dir() or not any(root == a or a in root.parents for a in allowed):
        raise HTTPException(400, "Папка не найдена или лежит вне рабочей папки сервиса")
    return _start(root, tuple(payload.get("include") or ()))


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404)
    return _jobs[job_id]


# --- документы ---------------------------------------------------------------

def _summary(doc: dict) -> dict:
    profile = doc.get("profile")
    get = lambda c: _value(doc, c)  # noqa: E731
    if profile == "judicial":
        title, case, amount, kind = get("фио"), get("дело_номер"), get("дз_осн"), get("тип документа")
    else:
        title, case, amount, kind = get("DbtrName"), get("IdDeloNo"), get("IdDebtSum"), get("DocType2")
    return {
        "id": doc["doc_id"], "path": doc["path"], "profile": profile, "status": _status(doc), "kind": kind, "debtor": title,
        "case": case, "amount": amount, "review": len(_review_issues(doc)), "notes": len(doc["issues"]),
        "manual": len(doc.get("manual", {})), "seconds": doc.get("extra", {}).get("seconds"),
    }


@app.get("/api/documents")
def documents() -> list[dict]:
    with _lock:
        docs = list(_load()["documents"].values())
    return sorted((_summary(d) for d in docs), key=lambda r: (r["profile"] or "", r["path"]))


def _doc(doc_id: str) -> dict:
    with _lock:
        doc = _load()["documents"].get(doc_id)
    if doc is None:
        raise HTTPException(404, "Документ не найден")
    return doc


def _blocks(doc: dict) -> dict[str, dict]:
    cache = WORK / "cache" / "ocr" / f"{doc['sha256']}.json"
    if not cache.is_file():
        return {}
    return {b["id"]: b for b in json.loads(cache.read_text(encoding="utf-8"))["blocks"]}


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    doc = _doc(doc_id)
    profile = doc.get("profile") or ""
    blocks = _blocks(doc)
    fields = []
    for column in COLUMNS.get(profile, ()):
        f = doc["fields"].get(column) or {}
        refs = f.get("source_refs") or []
        fields.append({
            "column": column, "value": _value(doc, column), "auto": f.get("value"), "state": f.get("state"),
            "derivation": f.get("derivation"), "raw": f.get("raw"), "manual": column in doc.get("manual", {}),
            "refs": refs, "boxes": [{"page": blocks[r]["page"], "bbox": blocks[r]["bbox"]} for r in refs if r in blocks],
            "issues": [i["code"] for i in doc["issues"] if i.get("field") == column],
        })
    pdf = doc["abs_path"] if doc["abs_path"].lower().endswith(".pdf") else doc.get("paired_abs")
    pages = 0
    if pdf and Path(pdf).is_file():
        with pymupdf.open(pdf) as d:
            pages = len(d)
    return {**_summary(doc), "fields": fields, "issues": doc["issues"], "pages": pages, "render_dpi": RENDER_DPI,
            "extra": {k: doc.get("extra", {}).get(k) for k in ("main_debtor_rule", "versions", "mode", "seconds", "DocName", "nested_ip", "state_fee_per_person")},
            "confirmed": doc.get("confirmed", False), "debt_text": _value(doc, "IdDebtText") if profile == "fssp" else None}


@app.get("/api/documents/{doc_id}/page/{page}.png")
def page_image(doc_id: str, page: int, dpi: int = 110) -> Response:
    doc = _doc(doc_id)
    pdf = doc["abs_path"] if doc["abs_path"].lower().endswith(".pdf") else doc.get("paired_abs")
    if not pdf or not Path(pdf).is_file():
        raise HTTPException(404, "Нет изображения страницы")
    with pymupdf.open(pdf) as d:
        if not 1 <= page <= len(d):
            raise HTTPException(404)
        png = d[page - 1].get_pixmap(dpi=min(max(dpi, 50), 200), alpha=False).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.patch("/api/documents/{doc_id}/fields")
def edit_field(doc_id: str, payload: dict = Body(...)) -> dict:
    """Ручная правка: хранится отдельно от автоматического результата и не затирается повторной обработкой."""
    column, value = payload.get("column"), payload.get("value")
    with _lock:
        store = _load()
        doc = store["documents"].get(doc_id)
        if doc is None or column not in COLUMNS.get(doc.get("profile") or "", ()):
            raise HTTPException(404)
        manual = doc.setdefault("manual", {})
        if payload.get("reset"):
            manual.pop(column, None)
        else:
            manual[column] = {"value": (value or "").strip() or None, "at": datetime.now().isoformat(timespec="seconds")}
        _save(store)
    return document(doc_id)


@app.post("/api/documents/{doc_id}/confirm")
def confirm(doc_id: str, payload: dict = Body(default={})) -> dict:
    with _lock:
        store = _load()
        if doc_id not in store["documents"]:
            raise HTTPException(404)
        store["documents"][doc_id]["confirmed"] = bool(payload.get("confirmed", True))
        _save(store)
    return document(doc_id)


@app.delete("/api/documents")
def clear() -> dict:
    with _lock:
        store = _load()
        store["documents"] = {}
        _save(store)
    return {"ok": True}


# --- отчёты -------------------------------------------------------------------

@app.get("/api/issues")
def issues() -> list[dict]:
    with _lock:
        docs = list(_load()["documents"].values())
    rows = []
    for d in sorted(docs, key=lambda d: d["path"]):
        for i in d["issues"]:
            rows.append({"id": d["doc_id"], "path": d["path"], "status": _status(d), **i,
                         "resolved": i.get("field") in d.get("manual", {}) or d.get("confirmed", False)})
    return rows


@app.get("/api/stats")
def stats() -> dict:
    with _lock:
        docs = list(_load()["documents"].values())
    by_status, by_code, by_profile = {}, {}, {}
    seconds = {"judicial": [], "fssp": []}
    vlm_fixed = manual = 0
    for d in docs:
        by_status[_status(d)] = by_status.get(_status(d), 0) + 1
        by_profile[d.get("profile") or "?"] = by_profile.get(d.get("profile") or "?", 0) + 1
        for i in d["issues"]:
            key = (i["code"], i["severity"])
            by_code[key] = by_code.get(key, 0) + 1
        vlm_fixed += sum(1 for f in d["fields"].values() if f.get("derivation") == "vlm_read")
        manual += len(d.get("manual", {}))
        if d.get("profile") in seconds and d.get("extra", {}).get("seconds") is not None:
            seconds[d["profile"]].append(d["extra"]["seconds"])
    return {
        "total": len(docs), "by_status": by_status, "by_profile": by_profile,
        "by_code": [{"code": c, "severity": s, "count": n} for (c, s), n in sorted(by_code.items(), key=lambda kv: -kv[1])],
        "avg_seconds": {k: round(sum(v) / len(v), 1) if v else None for k, v in seconds.items()},
        "vlm_fixed": vlm_fixed, "manual": manual,
    }


@app.get("/api/export/{name}")
def export(name: str) -> FileResponse:
    """Таблицы по шаблонам заказчика; ручные правки сотрудника учтены."""
    stem, _, ext = name.partition(".")
    if stem not in ("judicial", "fssp", "issues") or ext not in ("xlsx", "csv"):
        raise HTTPException(404)
    with _lock:
        docs = sorted(_load()["documents"].values(), key=lambda d: d["path"])
    out = WORK / "web" / "export"
    out.mkdir(parents=True, exist_ok=True)
    if stem == "issues":
        rows = [{
            "Документ": d["doc_id"], "Файл": d["path"], "Статус": _status(d), "Этап": i["stage"], "Поле": i.get("field") or "",
            "Код": i["code"], "Причина": i["message"], "Источник": i.get("location") or "", "Кандидаты": "; ".join(i["candidates"]),
            "Влияние на экспорт": i["export_effect"], "Рекомендуемое действие": i["action"],
            "Важность": "проверить" if i["severity"] == "review" else "к сведению",
        } for d in docs for i in d["issues"]]
        _write_table(rows, ISSUE_COLUMNS, out, "issues")
    else:
        columns = COLUMNS[stem]
        rows = [{c: _typed(c, _value(d, c)) for c in columns} for d in docs if d.get("profile") == stem]
        _write_table(rows, columns, out, stem)
    return FileResponse(out / name, filename=name)


# --- настройки ----------------------------------------------------------------

@app.get("/api/settings")
def get_settings() -> dict:
    client = ModelClient(_settings["model"], _settings["model_url"])
    return {**_settings, "model_available": client.available(), "workdir": str(WORK)}


@app.put("/api/settings")
def put_settings(payload: dict = Body(...)) -> dict:
    for key in ("model", "vlm_model", "model_url"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            _settings[key] = payload[key].strip()
    if "use_vlm" in payload:
        _settings["use_vlm"] = bool(payload["use_vlm"])
    if "conf_threshold" in payload:
        _settings["conf_threshold"] = min(max(float(payload["conf_threshold"]), 0.0), 1.0)
    with _lock:
        store = _load()
        store["settings"] = _settings
        _save(store)
    return get_settings()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")

"""Веб-интерфейс: загрузка, ручная правка, выгрузка. Модели отключены — проверяется маршрут XML."""

import csv
import io
import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from courtdocs import web  # noqa: E402

NS = "http://www.fssprus.ru/namespace/order/2017/2"
XML = (
    f'<?xml version="1.0" encoding="UTF-8"?><f:OIp xmlns:f="{NS}"><f:Id>1</f:Id><f:DocType>O_IP_ACT_END_END</f:DocType>'
    "<f:DocDate>2033-06-10</f:DocDate><f:DebtorType>2</f:DebtorType><f:DbtrName>Пробов Назар Назарович</f:DbtrName>"
    "<f:IdDocNo>0012/2033</f:IdDocNo><f:IdDebtSum>100.00</f:IdDebtSum>"
    "<f:IdDebtText>задолженность за период с 01.01.2031 по 02.12.2032 в размере 90,00 руб., судебные расходы в размере 10,00 руб.</f:IdDebtText>"
    "</f:OIp>"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WORK", tmp_path)
    monkeypatch.setattr(web, "STORE", tmp_path / "web" / "store.json")
    monkeypatch.setattr(web, "_context", lambda: None)  # без OCR и моделей
    return TestClient(web.app)


def upload(client, name="a.xml", body=XML):
    job = client.post("/api/upload", files=[("files", (name, body.encode("utf-8"), "text/xml"))]).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["state"] != "running":
            break
        time.sleep(0.05)
    assert job["state"] == "done"
    return client.get("/api/documents").json()


def test_upload_review_edit_and_export(client):
    docs = upload(client)
    assert [(d["path"], d["status"], d["debtor"]) for d in docs] == [("a.xml", "needs_review", "Пробов Назар Назарович")]
    doc_id = docs[0]["id"]

    detail = client.get(f"/api/documents/{doc_id}").json()
    deb = next(f for f in detail["fields"] if f["column"] == "rub_deb")
    assert deb["value"] == "90.00" and deb["refs"][0].startswith("/f:OIp/f:IdDebtText#chars=")
    assert [i["code"] for i in detail["issues"]] == ["UNSPECIFIED_COSTS"]

    # сотрудник относит «судебные расходы» к почтовым: замечание снято, значение попадает в выгрузку
    edited = client.patch(f"/api/documents/{doc_id}/fields", json={"column": "rub_post", "value": "10,00"}).json()
    assert edited["status"] == "processed"
    row = next(csv.DictReader(io.StringIO(client.get("/api/export/fssp.csv").content.decode("utf-8-sig"))))
    assert (row["rub_deb"], row["rub_post"], row["IdDocNo"]) == ("90.00", "10.00", "0012/2033")

    # повторная загрузка того же файла не затирает ручную правку
    again = upload(client)
    assert again[0]["manual"] == 1 and again[0]["status"] == "processed"

    reset = client.patch(f"/api/documents/{doc_id}/fields", json={"column": "rub_post", "reset": True}).json()
    assert reset["status"] == "needs_review"


def test_upload_rejects_other_files_and_strips_paths(client):
    assert client.post("/api/upload", files=[("files", ("notes.txt", b"x", "text/plain"))]).status_code == 400
    docs = upload(client, name="../../evil.xml")
    assert [d["path"] for d in docs] == ["evil.xml"]


def test_pdf_without_models_is_reported_not_lost(client):
    docs = upload(client, name="order.pdf", body="")  # тип определяется по расширению
    assert docs[0]["status"] == "failed"
    issues = client.get("/api/issues").json()
    assert issues[0]["code"] == "MODELS_DISABLED"
    assert client.get("/api/stats").json()["by_status"] == {"failed": 1}


def test_process_folder_is_limited_to_service_directories(client, tmp_path):
    outside = tmp_path.parent  # не каталог запуска и не рабочая папка
    assert client.post("/api/process-folder", json={"path": str(outside)}).status_code == 400
    inside = tmp_path / "batch"
    inside.mkdir()
    (inside / "a.xml").write_text(XML, encoding="utf-8")
    assert client.post("/api/process-folder", json={"path": str(inside)}).status_code == 200

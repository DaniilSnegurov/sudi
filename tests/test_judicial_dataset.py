"""Прогон 23 судебных PDF учебного набора на локальных моделях против labels/ocr.csv.

Нужен запущенный сервер моделей (start-models.ps1); без него тест пропускается.
"""

from pathlib import Path

import pytest

from courtdocs.evaluate import compare, load_labels
from courtdocs.export import JUDICIAL_COLUMNS, judicial_rows
from courtdocs.judicial import Context
from courtdocs.llm import ModelClient
from courtdocs.models import DOC_NEEDS_REVIEW
from courtdocs.ocr import OcrEngine
from courtdocs.pipeline import run

PROJECT = Path(__file__).resolve().parents[1]

# Реестр расхождений с эталоном, где система следует документу (спецификация, раздел 15)
DOCUMENT_OVER_LABEL = {
    ("ocr_006.pdf", "инн"): "REQ-014: ИНН есть в карточке должника, в эталоне пусто",
    ("ocr_008.pdf", "инн"): "REQ-014",
    ("ocr_011.pdf", "инн"): "REQ-014",
    ("ocr_017.pdf", "снилс"): "REQ-014: СНИЛС есть в карточке должника, в эталоне пусто",
    ("ocr_018.pdf", "снилс"): "REQ-014",
    ("ocr_008.pdf", "паспорт"): "в тексте акта у должницы другой паспорт; в эталоне паспорт второго должника",
    ("ocr_002.pdf", "улица"): "в документе «ул. Разметчиков», в эталоне тип улицы опущен",
    ("ocr_014.pdf", "соответчики_кол-во"): "на стр. 1 и 3 третий ответчик один, в разделе «Должник» — другой: в документе четыре имени",
    ("ocr_014.pdf", "соответчики_фио"): "то же",
}
# Расхождения в записи помещения: значение прочитано, но форма отличается от эталона
FLAT_FORMAT = {("ocr_006.pdf", "кв"), ("ocr_009.pdf", "кв"), ("ocr_010.pdf", "кв")}
# На листе ocr_016 у полей несколько штрихов; взгляды модели на отметку расходятся, должник остаётся
# неустановленным: берётся первый по порядку, документ уходит на проверку
UNSTABLE_MARK = {("ocr_016.pdf", c) for c in ("имя", "отчетство", "фио", "дата рождения")}


@pytest.fixture(scope="module")
def batch(dataset_root):
    llm = ModelClient()
    if not llm.available():
        pytest.skip("локальный сервер моделей не запущен")
    ctx = Context(OcrEngine(), llm, llm, PROJECT / "out" / "cache")
    return run(dataset_root, include=("ocr",), ctx=ctx)


def test_judicial_matches_labels_except_known_deviations(batch, xml_labels):
    labels = load_labels(xml_labels.parent / "ocr.csv", "Файл")
    report = compare(judicial_rows(batch), labels, "Файл", JUDICIAL_COLUMNS)
    assert report["compared"] == 23 and not report["missing"] and not report["extra"]
    got = {(k.rsplit("/", 1)[-1], c) for k, c, _, _ in report["mismatches"]}
    assert got - FLAT_FORMAT - UNSTABLE_MARK == set(DOCUMENT_OVER_LABEL)


def test_main_debtor_rules(batch):
    docs = {d.path.rsplit("/", 1)[-1]: d for d in batch.by_profile("judicial")}
    rules = {name: d.extra.get("main_debtor_rule") for name, d in docs.items()}
    assert all(rules[f"ocr_{n:03d}.pdf"] == "handwritten_mark" for n in range(12, 16))  # сканы листов: галочка у должника
    if rules["ocr_016.pdf"] != "handwritten_mark":  # отметка не установлена — документ обязан уйти на проверку
        assert docs["ocr_016.pdf"].status == DOC_NEEDS_REVIEW
        assert any(i.code == "UNCLEAR_OWNER" for i in docs["ocr_016.pdf"].issues)
    assert all(rules[f"ocr_{n:03d}.pdf"] == "card" for n in (*range(6, 12), *range(17, 24)))  # электронные формы
    assert all(rules[f"ocr_{n:03d}.pdf"] == "first" for n in range(1, 6))


def test_vlm_recovers_values_broken_by_ocr(batch):
    by_name = {d.path.rsplit("/", 1)[-1]: d for d in batch.by_profile("judicial")}
    assert by_name["ocr_006.pdf"].fields["дз_пошлина"].derivation == "vlm_read"  # OCR: «4об0 руб.»
    assert by_name["ocr_023.pdf"].fields["дз_пошлина"].derivation == "vlm_read"  # OCR: «0 538 рублей»
    assert by_name["ocr_009.pdf"].fields["паспорт"].derivation == "vlm_read"  # OCR: «8966 б08781»


def test_source_conflict_is_reported(batch):
    conflicts = {
        (d.path.rsplit("/", 1)[-1], i.field)
        for d in batch.by_profile("judicial") for i in d.issues if i.code == "SOURCE_CONFLICT" and i.severity == "review"
    }
    # карточка должника против цитаты судебного акта; значение берётся из карточки, документ уходит на проверку
    assert conflicts == {("ocr_008.pdf", "дата рождения"), ("ocr_020.pdf", "инн")}
    review = {d.path.rsplit("/", 1)[-1] for d in batch.by_profile("judicial") if d.status == DOC_NEEDS_REVIEW}
    assert review - {"ocr_016.pdf"} == {"ocr_008.pdf", "ocr_020.pdf"}

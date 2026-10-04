"""Судебный маршрут на подставных OCR и моделях: проверяется логика выбора полей, а не качество чтения."""

from datetime import date
from decimal import Decimal

import pymupdf
import pytest

from courtdocs.judicial import Context, process_judicial_pdf
from courtdocs.llm import ModelError
from courtdocs.models import DOC_NEEDS_REVIEW, PROCESSED


class FakeOcr:
    """Строки заданы заранее: (текст, оценка) по страницам."""

    def __init__(self, pages):
        self.pages = list(pages)

    def read(self, image):
        lines = self.pages.pop(0)
        out = []
        for i, item in enumerate(lines):
            text, conf = item if isinstance(item, tuple) else (item, 0.97)
            y = 100 + i * 60
            out.append((text, [(100.0, y), (1500.0, y), (1500.0, y + 40), (100.0, y + 40)], conf))
        return out


class FakeModel:
    model = "fake"

    def __init__(self, answers):
        self.answers, self.calls, self.asked = answers, [], []

    def chat_json(self, system, user, schema, purpose, images=None):
        self.asked.append((purpose, user, bool(images)))
        answer = self.answers.get(purpose)
        if answer is None:
            raise ModelError(f"нет ответа для {purpose}")
        return answer


def person(name, **kw):
    base = dict(name_as_written=name, surname="", first_name="", patronymic="", birth_date=None, passport=None,
                inn=None, snils=None, address=None, street=None, house=None, flat=None)
    return {**base, **kw}


def run(tmp_path, pages, answers, vlm_answers=None, mode="text"):
    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for _ in pages:
        doc.new_page()
    doc.save(pdf)
    if mode == "fused":  # одна модель со зрением и читает страницы, и перечитывает фрагменты
        llm = vlm = FakeModel({**answers, **(vlm_answers or {})})
    else:
        llm = FakeModel(answers)
        vlm = FakeModel(vlm_answers) if vlm_answers is not None else None
    result = process_judicial_pdf(pdf, "ocr/doc.pdf", "id1", "0" * 64, Context(FakeOcr(pages), llm, vlm, None, mode))
    return result, vlm


ORDER_E = [
    "судебный приказ",
    "№ 55MS0091#2-8704/2033#2",
    "Выдан органом: Судебный участок № 91, по делу № 2-8704/2033 от 14.02.2033.",
    "Должником по исполнительному документу является: физическое лицо Тестникова Татьяна Назаровна,",
    "ИНН 863681442718, род. 18.10.1947, адрес: 643,55,Тест,Архивная,68,10, адрес фактический: 643,55,Тест,Архивная,68,10, пол женский.",
    "Орган (должностное лицо), РЕШИЛ (ОПРЕДЕЛИЛ, ПОСТАНОВИЛ):",
    "Точная формулировка предмета исполнения: Взыскать солидарно с должников: Архивнова Николая Петровича,",
    "17.09.1996 года рождения, паспорт: 8431 170331, адрес: г. Тест, ул. Архивная, д. 68, кв. 10, Тестниковой Татьяны Назаровны,",
    "18.10.1947 года рождения, паспорт: 8966 608781, в пользу АО «Тест РТС», ИНН 8016874059,",
    "задолженность за период с 04.12.2031 по 02.01.2033 в размере 37703,74 руб., пени в размере 12384,16 руб.,",
    "а также расходы по уплате государственной пошлины в размере 4000 руб.",
    "Идентификатор 55MS0091#2-8704/2033#2",
]
ANSWERS_E = {
    "facts.case": {"document_type": "судебный приказ", "case_number": "2-8704/2033", "act_date": "14.02.2033"},
    "facts.people": {"card_debtor": None, "debtors": [
        person("Архивнова Николая Петровича", birth_date="17.09.1996", passport="8431 170331",
               address="г. Тест, ул. Архивная, д. 68, кв. 10", street="ул. Архивная", house="68", flat="10"),
        person("Тестниковой Татьяны Назаровны", birth_date="18.10.1947", passport="8966 608781"),
    ]},
    "facts.money": {"principal": "37703,74 руб.", "principal_parts": [], "period_start": "04.12.2031", "period_end": "02.01.2033",
                    "penalty": "12384,16 руб.", "state_fee": "4000 руб.", "state_fee_per_person": False},
    "facts.card": person("Тестникова Татьяна Назаровна", birth_date="18.10.1947", inn="863681442718",
                         address="643,55,Тест,Архивная,68,10", street="Архивная", house="68", flat="10"),
}


def test_req003_card_debtor_wins_over_first_mentioned(tmp_path):
    doc, _ = run(tmp_path, [ORDER_E], ANSWERS_E, {})
    v = doc.value
    assert v("тип документа") == "приказ эл"
    assert v("фио") == "Тестникова Татьяна Назаровна" and doc.extra["main_debtor_rule"] == "card"
    assert (v("дело_номер"), v("дело_дата")) == ("2-8704/2033", date(2033, 2, 14))
    assert v("дата рождения") == date(1947, 10, 18)
    assert v("паспорт") == "8966 608781"  # из судебной части: в карточке паспорта нет
    assert (v("улица"), v("дом"), v("кв")) == ("Архивная ул.", "68", "10")
    assert (v("дз_осн"), v("дз_пени"), v("дз_пошлина")) == (Decimal("37703.74"), Decimal("12384.16"), Decimal("4000.00"))
    assert (v("период_дз_начало"), v("период_дз_оконч")) == (date(2031, 12, 4), date(2033, 1, 2))
    assert doc.status == PROCESSED and doc.fields["дз_осн"].derivation == "llm+rules"


def test_req004_creditor_inn_is_not_assigned_to_debtor(tmp_path):
    answers = {**ANSWERS_E, "facts.card": person("Тестникова Татьяна Назаровна", birth_date="18.10.1947")}
    pages = [[line.replace("ИНН 863681442718, ", "") for line in ORDER_E]]
    doc, _ = run(tmp_path, pages, answers, {})
    assert doc.value("инн") is None  # ИНН 8016874059 принадлежит взыскателю


def test_req005_repeated_mentions_are_one_person(tmp_path):
    doc, _ = run(tmp_path, [ORDER_E], ANSWERS_E, {})
    assert doc.value("соответчики_кол-во") == 2
    assert doc.value("соответчики_фио") == "Архивнова Николая Петровича, Тестниковой Татьяны Назаровны"


def test_low_confidence_value_confirmed_by_vlm(tmp_path):
    pages = [[(t, 0.55) if "8966 608781" in t else t for t in ORDER_E]]
    doc, vlm = run(tmp_path, pages, ANSWERS_E, {"vlm.паспорт": {"status": "read", "value": "8966 608781"},
                                                "vlm.дата рождения": {"status": "read", "value": "18.10.1947"}})
    assert doc.value("паспорт") == "8966 608781" and doc.fields["паспорт"].derivation.endswith("+vlm")
    assert all(images for _, _, images in vlm.asked)  # перечитывание идёт по изображению фрагмента
    assert not any("8966" in question for _, question, _ in vlm.asked)  # кандидат OCR не подсказывается
    assert doc.status == PROCESSED


def test_req009_reading_disagreement_leaves_cell_empty(tmp_path):
    pages = [[(t, 0.55) if "8966 608781" in t else t for t in ORDER_E]]
    doc, _ = run(tmp_path, pages, ANSWERS_E, {"vlm.паспорт": {"status": "read", "value": "8966 608787"},
                                              "vlm.дата рождения": {"status": "read", "value": "18.10.1947"}})
    assert doc.value("паспорт") is None and doc.status == DOC_NEEDS_REVIEW
    issue = next(i for i in doc.issues if i.code == "READING_DISAGREEMENT")
    assert issue.field == "паспорт" and any("8966 608781" in c for c in issue.candidates) and any("608787" in c for c in issue.candidates)


def test_unreadable_and_failed_vlm_leave_cell_empty(tmp_path):
    # неуверенно распознаны строка с паспортом и строка карточки с датой рождения и ИНН
    pages = [[(t, 0.55) if "8966 608781" in t or "род. 18.10.1947" in t else t for t in ORDER_E]]
    doc, _ = run(tmp_path, pages, ANSWERS_E, {"vlm.паспорт": {"status": "unreadable", "value": ""}})
    assert doc.value("паспорт") is None and doc.value("дата рождения") is None and doc.value("инн") is None
    by_field = {i.field: i.code for i in doc.issues if i.severity == "review"}
    assert by_field == {"паспорт": "UNREADABLE", "дата рождения": "VLM_STAGE_FAILED", "инн": "VLM_STAGE_FAILED"}


def test_broken_ocr_amount_is_read_from_image(tmp_path):
    pages = [[t.replace("4000 руб.", "4об0 руб.") for t in ORDER_E]]
    answers = {**ANSWERS_E, "facts.money": {**ANSWERS_E["facts.money"], "state_fee": "4об0 руб."}}
    doc, _ = run(tmp_path, pages, answers, {"vlm.дз_пошлина": {"status": "read", "value": "4000 руб."}})
    assert doc.value("дз_пошлина") == Decimal("4000.00") and doc.fields["дз_пошлина"].derivation == "vlm_read"
    assert [i.severity for i in doc.issues if i.field == "дз_пошлина"] == ["info"] and doc.status == PROCESSED


def test_rules_and_model_conflict_is_settled_by_image(tmp_path):
    answers = {**ANSWERS_E, "facts.money": {**ANSWERS_E["facts.money"], "principal": "12384,16 руб."}}  # модель спутала долг и пени
    doc, _ = run(tmp_path, [ORDER_E], answers, {"vlm.дз_осн": {"status": "read", "value": "37703,74 руб."}})
    assert doc.value("дз_осн") == Decimal("37703.74")
    assert any(i.code == "SOURCE_CONFLICT" and i.severity == "info" for i in doc.issues)


ORDER_S = [
    "СУДЕБНЫЙ ПРИКАЗ",
    "6 января 2029 г.",
    "Производство № 2-8074/2029",
    "рассмотрев заявление о вынесении судебного приказа на взыскание задолженности с Макетнова Леонтия Степановича, 25.05.1969 года",
    "рождения, Макетновой Юлии Николаевны, 25.07.1978 года рождения, место жительства: г. Тест, ул. Пробная, д. 47/А, кв. 28,",
    "РЕШИЛ:",
    "Взыскать солидарно с должников Макетнова Леонтия Степановича, 25.05.1969 года рождения, Макетновой Юлии Николаевны,",
    "25.07.1978 года рождения, задолженность за период с 03.07.2025 г. по 02.08.2028 г. в размере 63008,92 руб.,",
    "пени в размере 59000,96 руб., расходы по оплате государственной пошлины в размере 2030,00 руб.",
]
ANSWERS_S = {
    "facts.case": {"document_type": "судебный приказ", "case_number": "2-8074/2029", "act_date": "6 января 2029 г."},
    "facts.people": {"card_debtor": None, "debtors": [
        person("Макетнова Леонтия Степановича", birth_date="25.05.1969 года"),
        person("Макетновой Юлии Николаевны", birth_date="25.07.1978 года", address="г. Тест, ул. Пробная, д. 47/А, кв. 28",
               street="ул. Пробная", house="47/А", flat="28"),
    ]},
    "facts.money": {"principal": "63008,92 руб.", "principal_parts": [], "period_start": "03.07.2025 г.", "period_end": "02.08.2028 г.",
                    "penalty": "59000,96 руб.", "state_fee": "2030,00 руб.", "state_fee_per_person": False},
}


def test_scanned_order_first_debtor_in_nominative_with_shared_address(tmp_path):
    doc, _ = run(tmp_path, [ORDER_S], ANSWERS_S, {})
    v = doc.value
    assert v("тип документа") == "приказ" and doc.extra["main_debtor_rule"] == "first"
    assert (v("фамилия"), v("имя"), v("отчетство")) == ("Макетнов", "Леонтий", "Степанович")
    assert v("дело_дата") == date(2029, 1, 6) and v("дата рождения") == date(1969, 5, 25)
    assert (v("улица"), v("дом"), v("кв")) == ("Пробная ул.", "47/А", "28")  # адрес назван один раз на всех
    assert v("соответчики_кол-во") == 2 and doc.status == PROCESSED


def test_req018_value_without_source_is_not_exported(tmp_path):
    answers = {**ANSWERS_S, "facts.case": {"document_type": "судебный приказ", "case_number": "2-9999/2029", "act_date": "6 января 2029 г."}}
    doc, _ = run(tmp_path, [ORDER_S], answers, {})
    assert doc.value("дело_номер") is None and doc.status == DOC_NEEDS_REVIEW
    assert any(i.code == "EVIDENCE_NOT_FOUND" and i.field == "дело_номер" for i in doc.issues)


def test_invented_person_is_dropped_and_rules_restore_real_one(tmp_path):
    people = {"card_debtor": None, "debtors": [person("Макетнова Леонтия Степановича"), person("Смоловшего Олега Владимировича")]}
    doc, _ = run(tmp_path, [ORDER_S], {**ANSWERS_S, "facts.people": people}, {})
    assert doc.value("соответчики_фио") == "Макетнова Леонтия Степановича, Макетновой Юлии Николаевны"
    assert doc.value("дата рождения") == date(1969, 5, 25)  # дата рождения рядом с ФИО найдена правилами
    assert any(i.code == "EVIDENCE_NOT_FOUND" and "Смоловшего" in i.message for i in doc.issues)


WRIT = [
    ["ИСПОЛНИТЕЛЬНЫЙ ЛИСТ", "2-80743/2023", "23.03.2023", "Дело №",
     "Гражданское дело по иску ОАО \"Тест РТС\" к Сводниковой Алисе Савельевне, Архивниковой Маргарите Павловне о взыскании задолженности."],
    [],
    ["Взыскать солидарно с Сводниковой Алисы Савельевны, Архивниковой Маргариты Павловны задолженность за период",
     "с 03.12.2021 по 02.01.2023 в размере 41828.90 руб., пени в размере 40593.68 руб.",
     "Взыскать расходы по уплате государственной пошлины по 266.66 руб. с каждого."],
    [],
    ["Судебный акт", "16 мая 2023 года", "Должник",
     "Сводникова Алиса Савельевна, 28.05.2000 г.р.", "Прож.: г. Тест, ул. Макетная, 5-139",
     "Архивникова Маргарита Павловна, 06.01.1999 г.р.", "Прож.: г. Тест, ул. Макетная, 5-139"],
]
ANSWERS_W = {
    "facts.case": {"document_type": "исполнительный лист", "case_number": "2-80743/2023", "act_date": "16 мая 2023 года"},
    "facts.people": {"card_debtor": None, "debtors": [
        person("Сводникова Алиса Савельевна", birth_date="28.05.2000 г.р.", address="г. Тест, ул. Макетная, 5-139", street="ул. Макетная", house="5", flat="139"),
        person("Архивникова Маргарита Павловна", birth_date="06.01.1999 г.р.", address="г. Тест, ул. Макетная, 5-139", street="ул. Макетная", house="5", flat="139"),
    ]},
    "facts.money": {"principal": "41828.90 руб.", "principal_parts": [], "period_start": "03.12.2021", "period_end": "02.01.2023",
                    "penalty": "40593.68 руб.", "state_fee": None, "state_fee_per_person": True},
}


def test_writ_marked_debtor_act_date_and_fee_per_person(tmp_path):
    doc, vlm = run(tmp_path, WRIT, ANSWERS_W, {"vlm.mark": {"mark_found": True, "name": "Архивникова Маргарита Павловна"}})
    v = doc.value
    assert v("тип документа") == "ИЛ"
    assert v("фио") == "Архивникова Маргарита Павловна" and doc.extra["main_debtor_rule"] == "handwritten_mark"
    assert v("дата рождения") == date(1999, 1, 6)
    assert v("дело_дата") == date(2023, 3, 23)  # дата рядом с номером дела, а не дата вступления в силу
    assert v("дз_пошлина") == Decimal("266.66") and doc.extra["state_fee_per_person"]  # REQ-008: не умножается
    assert (v("улица"), v("дом"), v("кв")) == ("Макетная ул.", "5", "139")
    assert v("соответчики_кол-во") == 2  # REQ-005: три упоминания каждого — один человек
    assert doc.status == PROCESSED


def test_writ_without_mark_takes_first_debtor_and_flags_it(tmp_path):
    doc, _ = run(tmp_path, WRIT, ANSWERS_W, {"vlm.mark": {"mark_found": False, "name": ""}})
    assert doc.value("фио") == "Сводникова Алиса Савельевна" and doc.status == DOC_NEEDS_REVIEW
    assert any(i.code == "UNCLEAR_OWNER" for i in doc.issues)


def test_model_failure_marks_document_failed(tmp_path):
    doc, _ = run(tmp_path, [ORDER_S], {}, {})
    assert doc.status == "failed" and doc.value("Файл") == "ocr/doc.pdf"
    assert {i.code for i in doc.issues} == {"LLM_STAGE_FAILED"}


@pytest.mark.parametrize("text,expected", [
    ("4об0 руб.", None), ("0 538 рублей", None), ("2030,00py6.", Decimal("2030.00")),
    ("114 666 руб. 18 коп.", Decimal("114666.18")), ("62 473 рублей 58 копеек", Decimal("62473.58")),
    ("266.66 ру6.", Decimal("266.66")), ("4000 руб", Decimal("4000.00")),
])
def test_money_parsing_rejects_damaged_digits(text, expected):
    from courtdocs.parse import clean_text, parse_money

    assert parse_money(clean_text(text)) == expected


def test_reading_order_survives_page_skew():
    from courtdocs.ocr import _reading_order

    def line(text, x0, x1, y_left, slope=0.05, h=40):
        y0, y1 = y_left, y_left + slope * (x1 - x0)
        return (text, [(x0, y0), (x1, y1), (x1, y1 + h), (x0, y0 + h)], 0.9)

    # соседние строки наклонённого скана перекрываются по высоте рамок
    lines = [line("в размере 284", 200, 1600, 1479), line("518 рублей 60 копеек", 200, 1200, 1530), line("выше", 200, 1600, 1428)]
    assert [t for t, _, _ in _reading_order(lines)] == ["выше", "в размере 284", "518 рублей 60 копеек"]


# --- режим fused: модель со зрением читает страницу вместе с текстом OCR ---------------------

def test_fused_page_and_ocr_text_go_to_the_model_together(tmp_path):
    doc, vlm = run(tmp_path, [ORDER_E], ANSWERS_E, {}, mode="fused")
    facts_calls = [(user, images) for purpose, user, images in vlm.asked if purpose.startswith("facts.")]
    assert len(facts_calls) == 4 and all(images for _, images in facts_calls)
    assert all(user.startswith("Текст OCR:\nПриложены изображения страниц: 1.") and "\n\nЗадание.\n" in user for user, _ in facts_calls)
    assert all("Должником по исполнительному документу является" in user for user, _ in facts_calls)  # текст OCR в запросе
    assert doc.extra["mode"] == "fused" and doc.status == PROCESSED
    assert doc.value("паспорт") == "8966 608781" and doc.fields["паспорт"].derivation == "vlm"


def test_fused_reading_that_corrects_ocr_needs_blind_confirmation(tmp_path):
    pages = [[t.replace("8966 608781", "8966 б08781") for t in ORDER_E]]  # OCR спутал цифру с буквой
    doc, vlm = run(tmp_path, pages, ANSWERS_E, {"vlm.паспорт": {"status": "read", "value": "8966 608781"}}, mode="fused")
    assert doc.value("паспорт") == "8966 608781" and doc.status == PROCESSED
    issue = next(i for i in doc.issues if i.field == "паспорт")
    assert (issue.code, issue.severity) == ("OCR_VLM_DIFFER", "info")
    assert any("б08781" in c for c in issue.candidates)  # в отчёте видно, что прочитал OCR
    blind = next(user for purpose, user, _ in vlm.asked if purpose == "vlm.паспорт")
    assert "8966" not in blind  # при перечитывании фрагмента подсказки нет


def test_fused_blind_reading_can_side_with_ocr(tmp_path):
    card = {**ANSWERS_E["facts.card"], "birth_date": "18.10.1941"}  # модель ошиблась, читая страницу
    doc, _ = run(tmp_path, [ORDER_E], {**ANSWERS_E, "facts.card": card},
                 {"vlm.дата рождения": {"status": "read", "value": "18.10.1947"}}, mode="fused")
    assert doc.value("дата рождения") == date(1947, 10, 18)
    assert [(i.code, i.severity) for i in doc.issues if i.field == "дата рождения"] == [("OCR_VLM_DIFFER", "info")]


def test_fused_three_different_readings_leave_cell_empty(tmp_path):
    card = {**ANSWERS_E["facts.card"], "birth_date": "18.10.1941"}
    doc, _ = run(tmp_path, [ORDER_E], {**ANSWERS_E, "facts.card": card},
                 {"vlm.дата рождения": {"status": "read", "value": "13.10.1947"}}, mode="fused")
    assert doc.value("дата рождения") is None and doc.status == DOC_NEEDS_REVIEW
    issue = next(i for i in doc.issues if i.code == "READING_DISAGREEMENT")
    assert len(issue.candidates) == 3  # чтение страницы, OCR и слепое чтение фрагмента


@pytest.mark.parametrize("mode", ["text", "fused"])
def test_header_total_does_not_replace_principal(tmp_path, mode):
    header = "взыскать (1460000) Задолженность по платежам за газ, тепло и электроэнергию на сумму: 54087.90 в валюте по ОКВ: 643."
    pages = [ORDER_E[:6] + [header] + ORDER_E[6:]]
    answers = {**ANSWERS_E, "facts.money": {**ANSWERS_E["facts.money"], "principal": "54087.90"}}
    doc, _ = run(tmp_path, pages, answers, {}, mode=mode)
    assert doc.value("дз_осн") == Decimal("37703.74") and doc.status == PROCESSED
    assert doc.extra["outside_operative_part"] == ["дз_осн: 54087.90"]


def test_find_fuzzy_binds_corrected_value_to_ocr_text():
    from courtdocs.judicial import DocIndex
    from courtdocs.ocr import Block

    blocks = [Block("p1_b1", 1, "паспорт: 8431 170331, зарегистрирован", (0, 0, 10, 10), 0.9, 0),
              Block("p1_b2", 1, "паспорт: 8966 б08781, место жительства", (0, 20, 10, 30), 0.9, 1)]
    index = DocIndex(blocks)
    assert index.find("8966 608781") is None
    found, ocr = index.find_fuzzy("8966 608781")
    assert [b.id for b in found] == ["p1_b2"] and ocr == "8966 б08781"
    assert index.find_fuzzy("1234 567890") is None

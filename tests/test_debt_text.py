from datetime import date
from decimal import Decimal

from courtdocs.debt_text import extract_debt_fields
from courtdocs.models import EXTRACTED, NEEDS_REVIEW, NOT_FOUND

P = "/f:OIp/f:IdDebtText"


def run(text, total=None):
    fields, issues, extra = extract_debt_fields(text, P, Decimal(total) if total else None)
    return {k: f.value for k, f in fields.items()}, {i.code for i in issues}, fields, extra


def test_standard_order_text():
    v, codes, fields, extra = run(
        "задолженность за потребленную тепловую энергию за период с 03.01.2031 по 02.12.2032 в размере "
        "88300,94 руб., пени в размере 41622,76 руб. за период с 04.01.2031 по 02.12.2032, а также расходы "
        "по уплате государственной пошлины в размере 4000,00 руб., почтовые расходы в размере 364,80 руб.",
        "134288.50",
    )
    assert v == {
        "date_start": date(2031, 1, 3), "date_end": date(2032, 12, 2), "rub_deb": Decimal("88300.94"),
        "rub_peni": Decimal("41622.76"), "rub_poshlina": Decimal("4000.00"), "rub_post": Decimal("364.80"),
    }
    assert codes == set()
    # период пеней идёт после суммы пеней и не подменяет период основного долга
    assert extra["peni_period"] == ["04.01.2031 — 02.12.2032"]
    assert fields["rub_deb"].source_refs[0].startswith(P + "#chars=")


def test_peni_period_before_amount_and_integer_rubles():
    v, codes, *_ = run(
        "задолженность по адресу: г. Тест, за период с 04.04.2030 по 01.09.2032 в размере 110776,92 рублей, "
        "пени за период с 05.04.2030 по 01.09.2032 в размере 27027,36 рублей, а также расходы по оплате "
        "государственной пошлины в размере 4000 рублей."
    )
    assert (v["date_start"], v["date_end"]) == (date(2030, 4, 4), date(2032, 9, 1))
    assert v["rub_poshlina"] == Decimal("4000.00") and v["rub_post"] is None and codes == set()


def test_req013_postal_correspondence_is_post():
    v, codes, *_ = run(
        "задолженность за период с 05.06.2030 по 02.12.2032 в размере 100,00 руб., судебные расходы по "
        "отправке почтовой корреспонденции в размере 182,40 руб."
    )
    assert v["rub_post"] == Decimal("182.40") and codes == set()


def test_unspecified_court_costs_are_not_assigned():
    v, codes, _, extra = run(
        "задолженности за период с 04.04.2025 по 02.08.2032 в размере 122546,36 руб., судебные расходы в "
        "размере 182,40 руб., а также расходы по уплате государственной пошлины в размере 4000,00 руб."
    )
    assert v["rub_post"] is None and v["rub_poshlina"] == Decimal("4000.00")
    assert codes == {"UNSPECIFIED_COSTS"} and extra["court_costs"] == ["182.40"]


def test_req006_components_without_total_are_summed():
    v, codes, fields, _ = run("взыскать за отопление - 37110,50 руб., ГВС - 84852,72 руб.")
    assert v["rub_deb"] == Decimal("121963.22")
    assert fields["rub_deb"].derivation == "sum_of_components"
    assert fields["rub_deb"].operands == ["37110.50", "84852.72"] and len(fields["rub_deb"].source_refs) == 2


def test_req007_total_is_not_added_to_its_breakdown():
    v, codes, *_ = run(
        "задолженность за период с 04.10.2031 по 02.01.2033 в размере 125196,88 руб., "
        "(отопление-88599,26 руб., ГВС-36597,62 руб), пени в размере 108482,18 руб. за период 04.10.2031 по 02.01.2033"
    )
    assert v["rub_deb"] == Decimal("125196.88") and v["rub_peni"] == Decimal("108482.18") and codes == set()


def test_breakdown_mismatch_is_reported_not_fixed():
    v, codes, *_ = run("задолженность в размере 100,00 руб. (отопление - 60,00 руб., ГВС - 30,00 руб.)")
    assert v["rub_deb"] == Decimal("100.00") and codes == {"AMOUNT_RECONCILIATION"}


def test_req012_explicit_zero_differs_from_missing():
    v, codes, fields, _ = run(
        "р/с: 84914922957058406869 19368 девятнадцать тысяч триста шестьдесят восемь рублей 00 копеек "
        "неустойку в размере _, пени в размере _, расходы по уплате государственной пошлины в размере 0 "
        "(ноль рублей 00 копеек)",
        "38735.12",
    )
    assert v["rub_poshlina"] == Decimal("0.00") and fields["rub_poshlina"].state == EXTRACTED
    assert v["rub_deb"] is None  # IdDebtSum не подставляется вместо расшифровки
    assert v["rub_peni"] is None and fields["rub_peni"].state == NEEDS_REVIEW
    assert codes == {"INVALID_VALUE_FORMAT", "UNCLEAR_OWNER"}


def test_req010_multiple_obligations_and_broken_date():
    v, codes, fields, _ = run(
        "задолженность за период с 01.11.20217 по 13.04.2027 в размере 98756,44 рублей, пени в размере 9000 "
        "рублей за период с 05.03.2025 по 13.04.2027, а также расходы по оплате государственной пошлины в "
        "размере 3938 рублей. Взыскать задолженность за период с 14.04.2027 по 16.07.2027 в размере 12987,62 "
        "рублей, пени в размере 366,06 рублей"
    )
    assert all(x is None for x in v.values())
    assert all(f.state == NEEDS_REVIEW for f in fields.values())
    assert codes == {"MULTIPLE_OBLIGATIONS", "INVALID_PERIOD"}


def test_reversed_period_is_not_exported():
    v, codes, *_ = run("задолженность за период с 05.03.2029 по 13.04.2027 в размере 10,00 руб.")
    assert v["rub_deb"] == Decimal("10.00") and v["date_start"] is None and codes == {"INVALID_PERIOD"}


def test_short_text_without_amounts_is_not_an_error():
    v, codes, fields, _ = run("Задолженность по платежам за тепловую энергию, госпошлина", "35637.46")
    assert all(x is None for x in v.values()) and codes == set()
    assert all(f.state == NOT_FOUND for f in fields.values())


def test_total_mismatch_with_id_debt_sum_keeps_values():
    v, codes, *_ = run("задолженность в размере 234156,36 руб., пени в размере 410801,08 руб.", "644958.40")
    assert codes == {"AMOUNT_RECONCILIATION"} and v["rub_deb"] == Decimal("234156.36")

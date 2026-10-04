"""Прогон по учебному набору: 21 XML ФССП против эталона labels/xml.csv."""

from courtdocs.evaluate import compare, load_labels
from courtdocs.export import FSSP_COLUMNS, fssp_rows
from courtdocs.models import DOC_NEEDS_REVIEW, PROCESSED

# Реестр расхождений с эталоном: система следует документу и правилам спецификации.
# Любое расхождение вне этого списка — ошибка извлечения.
KNOWN_DEVIATIONS = {
    ("fssp_001.xml", "street"): "тип улицы в документе не подтверждён; в эталоне номер улицы оставлен впереди",
    ("fssp_009.xml", "street"): "в эталоне номер улицы оставлен впереди («18 Учебная»), в остальных строках — в конце",
    ("fssp_011.xml", "street"): "тип улицы в документе не подтверждён, в эталоне проставлен «ул.»",
    ("fssp_002.xml", "rub_post"): "REQ-013: расходы на отправку корреспонденции — почтовые; в эталоне пусто",
    ("fssp_003.xml", "rub_post"): "REQ-013",
    ("fssp_012.xml", "rub_post"): "REQ-013",
    ("fssp_013.xml", "date_start"): "REQ-010: несколько обязательств и повреждённая дата; в эталоне взято первое",
    ("fssp_013.xml", "date_end"): "REQ-010",
    ("fssp_013.xml", "rub_deb"): "REQ-010",
    ("fssp_013.xml", "rub_peni"): "REQ-010",
    ("fssp_013.xml", "rub_poshlina"): "REQ-010",
    ("fssp_014.xml", "rub_deb"): "в эталоне подставлен IdDebtSum, в тексте основной долг не назван",
    ("fssp_014.xml", "rub_poshlina"): "REQ-012: в тексте явный ноль, в эталоне пусто",
}

NEEDS_REVIEW = {
    "fssp_004.xml": {"UNSPECIFIED_COSTS"},
    "fssp_008.xml": {"UNSPECIFIED_COSTS"},
    "fssp_010.xml": {"AMOUNT_RECONCILIATION"},
    "fssp_011.xml": {"AMOUNT_RECONCILIATION"},  # разница 0,96 руб. не скрывается допуском
    "fssp_013.xml": {"MULTIPLE_OBLIGATIONS", "INVALID_PERIOD"},
    "fssp_014.xml": {"INVALID_VALUE_FORMAT", "UNCLEAR_OWNER"},
    "fssp_016.xml": {"INVALID_VALUE_FORMAT"},
}


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def test_all_xml_processed(fssp_batch):
    docs = fssp_batch.by_profile("fssp")
    assert len(docs) == 21
    assert sum(e.paired_with is not None for e in fssp_batch.entries) == 21  # парные PDF не дают строк
    assert all(d.paired_pdf for d in docs)
    assert not [i for d in docs for i in d.issues if i.code.startswith("PAIR_")]


def test_matches_labels_except_known_deviations(fssp_batch, xml_labels):
    report = compare(fssp_rows(fssp_batch), load_labels(xml_labels, "FileName"), "FileName", FSSP_COLUMNS)
    assert report["compared"] == 21 and not report["missing"] and not report["extra"]
    got = {(_name(k), c) for k, c, _, _ in report["mismatches"]}
    assert got == set(KNOWN_DEVIATIONS)


def test_review_flags(fssp_batch):
    flagged = {
        _name(d.path): {i.code for i in d.issues if i.severity == "review"}
        for d in fssp_batch.by_profile("fssp")
        if d.status == DOC_NEEDS_REVIEW
    }
    assert flagged == NEEDS_REVIEW
    assert sum(d.status == PROCESSED for d in fssp_batch.by_profile("fssp")) == 14


def test_req011_top_level_fields_win_over_nested(fssp_batch):
    doc = next(d for d in fssp_batch.documents if d.path.endswith("fssp_010.xml"))
    assert doc.value("DbtrName") == "Бланкова Анна Романовна"
    assert doc.value("IpNo") == "8096686/33/55006-ИП"
    assert doc.fields["IpNo"].source_refs == ["/f:OIp/f:IpNo"]
    nested = doc.extra["nested_ip"]
    assert [n["DbtrName"] for n in nested] == ["Учебнова Светлана Павловна", "Бланкова Анна Романовна"]
    assert nested[0]["_path"] == "/f:OIp/f:IP[1]"


def test_direct_fields_keep_source_value(fssp_batch):
    doc = next(d for d in fssp_batch.documents if d.path.endswith("fssp_016.xml"))
    assert doc.value("IdDeloNo") == "2-8264/2029)"  # дефект источника не исправляется молча

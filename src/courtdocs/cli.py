"""courtdocs run <пакет> --out <папка> [--labels <папка с эталоном>] [--no-models | --no-vlm]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import compare, format_report, load_labels
from .export import FSSP_COLUMNS, JUDICIAL_COLUMNS, fssp_rows, judicial_rows, write_all
from .judicial import Context
from .llm import DEFAULT_MODEL, DEFAULT_URL, DEFAULT_VLM, ModelClient
from .models import DOC_NEEDS_REVIEW, FAILED, PROCESSED
from .ocr import OcrEngine
from .pipeline import run


def _context(args) -> Context | None:
    if args.no_models:
        return None
    fused = args.mode == "fused"
    # в режиме fused в запрос входят изображения страниц — контекст нужен больше
    llm = ModelClient(args.vlm_model if fused else args.model, args.model_url, num_ctx=20480 if fused else 16384)
    if not llm.available():
        print(f"Модель {args.model} недоступна по адресу {args.model_url}: судебные PDF будут помечены как необработанные.")
        llm = None
    vlm = None
    if llm is not None and (fused or not args.no_vlm):
        vlm = llm if fused or args.vlm_model == args.model else ModelClient(args.vlm_model, args.model_url)
        if vlm is not llm and not vlm.available():
            print(f"Модель со зрением {args.vlm_model} недоступна: перечитывание фрагментов отключено.")
            vlm = None
    return Context(OcrEngine(), llm, vlm, args.out / "cache", args.mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="courtdocs", description="Обработка судебных документов и постановлений ФССП")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="обработать каталог или ZIP")
    p.add_argument("input", type=Path, help="корень пакета (каталог или .zip)")
    p.add_argument("--out", type=Path, default=Path("out"), help="куда записать таблицы и отчёт")
    p.add_argument("--only", action="append", default=[], help="обработать только этот подкаталог пакета (можно несколько)")
    p.add_argument("--labels", type=Path, help="папка с эталоном (xml.csv, ocr.csv) для сравнения результата")
    p.add_argument("--no-models", action="store_true", help="только XML: без OCR и моделей")
    p.add_argument("--no-vlm", action="store_true", help="без перечитывания проблемных мест моделью со зрением")
    p.add_argument("--mode", choices=("text", "fused"), default="text",
                   help="text: модель видит только текст OCR; fused: модель со зрением читает страницы вместе с текстом OCR")
    p.add_argument("--model", default=DEFAULT_MODEL, help="языковая модель (по умолчанию %(default)s)")
    p.add_argument("--vlm-model", default=DEFAULT_VLM, help="модель со зрением (по умолчанию %(default)s)")
    p.add_argument("--model-url", default=DEFAULT_URL, help="адрес локального сервера моделей")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # консоль Windows может не знать отдельных символов

    batch = run(args.input, args.out / "unpacked", tuple(args.only), _context(args))
    write_all(batch, args.out)

    docs = batch.documents
    print(f"Файлов: {len(batch.entries)}, документов: {len(docs)}")
    print(f"  без выявленных проблем: {sum(d.status == PROCESSED for d in docs)}")
    print(f"  требуют проверки: {sum(d.status == DOC_NEEDS_REVIEW for d in docs)}")
    print(f"  не обработаны: {sum(d.status == FAILED for d in docs)}")
    print(f"Результат: {args.out.resolve()}")

    if args.labels:
        for title, name, rows, key, columns in (
            ("ФССП", "xml.csv", fssp_rows(batch), "FileName", FSSP_COLUMNS),
            ("Судебные документы", "ocr.csv", judicial_rows(batch), "Файл", JUDICIAL_COLUMNS),
        ):
            if rows and (args.labels / name).is_file():
                print(f"\n== {title} ==")
                print(format_report(compare(rows, load_labels(args.labels / name, key), key, columns)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

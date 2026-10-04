"""Чтение PDF по страницам и OCR: блоки текста с координатами и оценкой распознавания."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pymupdf

RENDER_DPI = 216  # сканы набора сохранены с разрешением около 216 dpi; увеличение не восстанавливает символы
OCR_VERSION = "rapidocr-3.9.2/PP-OCRv5-server-det/eslav-mobile-rec/order-2"


@dataclass
class Block:
    id: str  # p{страница}_b{номер}
    page: int
    text: str
    bbox: tuple[int, int, int, int]  # пиксели изображения страницы при RENDER_DPI
    conf: float | None
    order: int


@dataclass
class PageInfo:
    page: int
    width: int
    height: int
    text_layer_chars: int
    blocks: int


def render_page(pdf_path: Path, page: int, dpi: int = RENDER_DPI, clip: tuple | None = None) -> np.ndarray:
    """Страница (с 1) в RGB. clip — область в пикселях при RENDER_DPI."""
    with pymupdf.open(pdf_path) as doc:
        p = doc[page - 1]
        rect = None
        if clip is not None:
            k = 72 / RENDER_DPI
            rect = pymupdf.Rect(clip[0] * k, clip[1] * k, clip[2] * k, clip[3] * k) & p.rect
        pix = p.get_pixmap(dpi=dpi, clip=rect, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def crop_png(pdf_path: Path, page: int, clip: tuple[int, int, int, int], dpi: int = RENDER_DPI) -> bytes:
    """Фрагмент страницы для модели со зрением; clip — пиксели при RENDER_DPI."""
    with pymupdf.open(pdf_path) as doc:
        p = doc[page - 1]
        k = 72 / RENDER_DPI
        rect = pymupdf.Rect(clip[0] * k, clip[1] * k, clip[2] * k, clip[3] * k) & p.rect
        return p.get_pixmap(dpi=dpi, clip=rect, alpha=False).tobytes("png")


def page_png(pdf_path: Path, page: int, dpi: int = RENDER_DPI) -> bytes:
    """Страница целиком для модели со зрением."""
    with pymupdf.open(pdf_path) as doc:
        return doc[page - 1].get_pixmap(dpi=dpi, alpha=False).tobytes("png")


class OcrEngine:
    """PP-OCRv5: детектор server, распознавание eslav (русский); исполнение через ONNX на процессоре."""

    def __init__(self) -> None:
        self._ocr = None

    def _load(self):
        if self._ocr is None:
            from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

            self._ocr = RapidOCR(params={
                "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER, "Det.lang_type": LangDet.CH,
                "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.ESLAV,
                "Global.use_cls": False, "Global.log_level": "error",
            })
        return self._ocr

    def read(self, image_rgb: np.ndarray) -> list[tuple[str, list, float]]:
        """→ (текст, четырёхугольник строки, оценка распознавания)."""
        res = self._load()(image_rgb)
        if res.txts is None:
            return []  # страница без текста
        return [
            (text.strip(), [(float(x), float(y)) for x, y in box], float(score))
            for text, score, box in zip(res.txts, res.scores, res.boxes)
            if text.strip()
        ]


def _reading_order(lines: list[tuple[str, list, float]]) -> list[tuple[str, tuple[int, int, int, int], float]]:
    """Строки сверху вниз, внутри строки слева направо; наклон скана компенсируется."""
    if not lines:
        return []
    slopes = [(p[1][1] - p[0][1]) / (p[1][0] - p[0][0]) for _, p, _ in lines if p[1][0] - p[0][0] > 200]
    slope = float(np.median(slopes)) if slopes else 0.0
    items = []
    for text, poly, score in lines:
        xs, ys = [x for x, _ in poly], [y for _, y in poly]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        height = max(1.0, ((poly[3][0] - poly[0][0]) ** 2 + (poly[3][1] - poly[0][1]) ** 2) ** 0.5)
        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        items.append((cy - slope * cx, height, min(xs), (text, bbox, score)))
    items.sort(key=lambda it: it[0])
    rows: list[list] = []
    for it in items:
        if rows and abs(it[0] - rows[-1][-1][0]) < 0.5 * min(it[1], rows[-1][-1][1]):
            rows[-1].append(it)
        else:
            rows.append([it])
    return [it[3] for row in rows for it in sorted(row, key=lambda it: it[2])]


def ocr_document(pdf_path: Path, engine: OcrEngine, cache_dir: Path | None, sha256: str) -> tuple[list[Block], list[PageInfo]]:
    """Все страницы учитываются; результат кэшируется по контрольной сумме файла и версии OCR."""
    cache = cache_dir / f"{sha256}.json" if cache_dir else None
    if cache and cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("version") == OCR_VERSION:
            return [Block(**{**b, "bbox": tuple(b["bbox"])}) for b in data["blocks"]], [PageInfo(**p) for p in data["pages"]]

    blocks: list[Block] = []
    pages: list[PageInfo] = []
    with pymupdf.open(pdf_path) as doc:
        n_pages = len(doc)
        layer = [len(p.get_text().strip()) for p in doc]
    for page in range(1, n_pages + 1):
        image = render_page(pdf_path, page)
        lines = _reading_order(engine.read(image))
        for i, (text, box, score) in enumerate(lines, 1):
            blocks.append(Block(f"p{page}_b{i}", page, text, box, round(score, 4), len(blocks)))
        pages.append(PageInfo(page, image.shape[1], image.shape[0], layer[page - 1], len(lines)))

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": OCR_VERSION, "blocks": [asdict(b) for b in blocks], "pages": [asdict(p) for p in pages]}
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return blocks, pages

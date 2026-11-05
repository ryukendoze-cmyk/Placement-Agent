import logging
import os
import urllib.request
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

TESSDATA_DOWNLOAD_URL = (
    "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata"
)


def _get_tessdata_dir() -> str | None:
    """Locate or prepare directory containing Tesseract `eng.traineddata`."""
    env_dir = os.getenv("TESSDATA_PREFIX")
    if env_dir and os.path.isfile(os.path.join(env_dir, "eng.traineddata")):
        return env_dir

    candidate_dirs = [
        Path(__file__).resolve().parents[2] / "data" / "tessdata",
        Path(__file__).resolve().parents[1] / "data" / "tessdata",
        Path.cwd() / "data" / "tessdata",
    ]
    for d in candidate_dirs:
        if (d / "eng.traineddata").is_file():
            return str(d)

    try:
        import pymupdf
        detected = pymupdf.get_tessdata(None)
        if detected and os.path.isfile(os.path.join(detected, "eng.traineddata")):
            return detected
    except Exception:
        pass

    target_dir = candidate_dirs[0]
    target_file = target_dir / "eng.traineddata"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading eng.traineddata to %s...", target_dir)
        req = urllib.request.Request(
            TESSDATA_DOWNLOAD_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp, open(target_file, "wb") as f:
            f.write(resp.read())
        if target_file.is_file():
            return str(target_dir)
    except Exception as exc:
        logger.warning("Could not auto-download eng.traineddata: %s", exc)

    return None


def parse_resume_bytes(data: bytes, suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        doc = pymupdf.open(stream=BytesIO(data), filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    if suffix == ".docx":
        from docx import Document
        doc = Document(BytesIO(data))
        parts: list[str] = [p.text for p in doc.paragraphs]
        seen_cells: set[int] = set()
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    tc_id = id(getattr(cell, "_tc", cell))
                    if tc_id in seen_cells:
                        continue  # merged cell referenced more than once
                    seen_cells.add(tc_id)
                    cell_text = (cell.text or "").strip()
                    if cell_text:
                        parts.append(cell_text)
        return "\n".join(parts)
    if suffix in (".png", ".jpg", ".jpeg"):
        import pymupdf
        filetype = "jpeg" if suffix in (".jpg", ".jpeg") else "png"
        try:
            doc = pymupdf.open(stream=BytesIO(data), filetype=filetype)
        except Exception as e:
            raise ValueError(f"could not open image: {e}") from e

        try:
            if len(doc) == 0:
                return ""
            try:
                _first_page = doc[0]
            except Exception as e:
                raise ValueError(f"could not parse image data: {e}") from e

            tessdata = _get_tessdata_dir()
            if not tessdata:
                raise RuntimeError(
                    "Tesseract OCR language data not found. Please ensure "
                    "eng.traineddata exists in data/tessdata or set TESSDATA_PREFIX."
                )

            parts: list[str] = []
            for page in doc:
                try:
                    pix = page.get_pixmap(dpi=150)
                except Exception as e:
                    raise ValueError(f"could not render image page: {e}") from e
                ocr_pdf_bytes = pix.pdfocr_tobytes(language="eng", tessdata=tessdata)
                with pymupdf.open(stream=ocr_pdf_bytes, filetype="pdf") as ocr_doc:
                    ocr_text = "\n".join(p.get_text() for p in ocr_doc).strip()
                    if ocr_text:
                        parts.append(ocr_text)
            return "\n".join(parts)
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise ValueError(f"could not parse image: {e}") from e
        finally:
            doc.close()
    if suffix in (".txt", ".text", ".md"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"unsupported resume format: {suffix}")


def parse_resume(path: Path | str, suffix: str) -> str:
    path = Path(path)
    return parse_resume_bytes(path.read_bytes(), suffix)

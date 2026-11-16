import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_DOCX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024  # 25 MB decompressed cap against zip bombs
MAX_DOCX_ENTRIES = 1000
MAX_PDF_PAGES = 40


def _get_tessdata_dir() -> str | None:
    """Locate local directory containing Tesseract `eng.traineddata` without runtime network I/O."""
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

    return None


def _validate_docx_archive(data: bytes) -> None:
    """Validate a DOCX ZIP container against corruption and decompression bombs."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"docx contains too many archive entries (>{MAX_DOCX_ENTRIES})")
            total_uncompressed = 0
            for info in infos:
                if info.file_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise ValueError("docx entry exceeds maximum decompressed size")
                with zf.open(info, "r") as member:
                    while True:
                        chunk = member.read(65536)
                        if not chunk:
                            break
                        total_uncompressed += len(chunk)
                        if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
                            raise ValueError("docx exceeds maximum decompressed size")
    except ValueError:
        raise
    except zipfile.BadZipFile as e:
        raise ValueError(f"invalid or corrupt docx archive: {e}") from e
    except Exception as e:
        raise ValueError(f"could not read docx archive: {e}") from e


def parse_resume_bytes(data: bytes, suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".pdf":
        import pymupdf
        try:
            doc = pymupdf.open(stream=BytesIO(data), filetype="pdf")
        except Exception as e:
            raise ValueError(f"could not open pdf: {e}") from e
        try:
            if len(doc) > MAX_PDF_PAGES:
                raise ValueError(f"pdf exceeds {MAX_PDF_PAGES} page limit")
            return "\n".join(page.get_text() for page in doc)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"could not parse pdf: {e}") from e
        finally:
            doc.close()
    if suffix == ".docx":
        _validate_docx_archive(data)
        from docx import Document
        try:
            doc = Document(BytesIO(data))
        except Exception as e:
            raise ValueError(f"could not parse docx document: {e}") from e
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
            if len(doc) > MAX_PDF_PAGES:
                raise ValueError(f"image document exceeds {MAX_PDF_PAGES} page limit")
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

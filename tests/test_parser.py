from pathlib import Path

import pytest

from app.resume.parser import parse_resume, parse_resume_bytes

FIX = Path(__file__).parent / "fixtures"


def test_parse_txt():
    assert "FastAPI" in parse_resume(FIX / "resume.txt", ".txt")


def test_parse_pdf():
    assert "PostgreSQL" in parse_resume(FIX / "resume.pdf", ".pdf")


def test_parse_docx():
    assert "Bharat Sharma" in parse_resume(FIX / "resume.docx", ".docx")


def test_parse_png():
    assert "Bharat Sharma" in parse_resume(FIX / "resume.png", ".png")


def test_parse_jpeg():
    assert "Bharat Sharma" in parse_resume(FIX / "resume.jpeg", ".jpeg")


def test_parse_jpg():
    assert "Bharat Sharma" in parse_resume(FIX / "resume.jpeg", ".jpg")


def test_parse_corrupt_image_raises():
    with pytest.raises(ValueError):
        parse_resume_bytes(b"\x89PNG\r\n\x1a\ncorrupt", ".png")


def test_unsupported_suffix_raises():
    with pytest.raises(ValueError):
        parse_resume(FIX / "resume.txt", ".xyz")

"""JD ingestion and text normalization (PDF, DOCX, TXT, raw text)."""
from __future__ import annotations

import re
import zipfile

from fastapi import HTTPException

from app.resume.parser import parse_resume_bytes

MAX_JD_BYTES = 2 * 1024 * 1024  # 2 MB limit
SUPPORTED_JD_SUFFIXES = {".pdf", ".docx", ".txt", ".text", ".md"}

_HEADER_PREFIX_RE = re.compile(
    r"^(?:from|to|cc|bcc|sent|date|reply-to|message-id|mime-version|"
    r"content-type|content-transfer-encoding|x-[a-z0-9-]+)\s*:",
    re.IGNORECASE,
)
_FORWARDED_MARKER_RE = re.compile(
    r"^[-_=\s]*(?:forwarded message|begin forwarded message|original message)[-_=\s]*$",
    re.IGNORECASE,
)
_SUBJECT_RE = re.compile(r"^subject\s*:\s*(?:(?:fwd?|re)\s*:\s*)*(.*)$", re.IGNORECASE)
_SIG_DIVIDER_RE = re.compile(r"^\s*(?:--|___|===)\s*$")
_SIG_SALUTATION_RE = re.compile(
    r"^\s*(?:thanks(?:\s*(?:&|and)\s*regards)?|thank\s+you|"
    r"(?:warm|best|kind)\s+regards|regards|sincerely|cheers|"
    r"sent\s+from\s+my\s+(?:iphone|ipad|android|mobile(?:\s+device)?))\s*[,!.]?\s*$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[•●▪◦‣\*–—]|-(?!\d)|\d{1,2}[.)])\s+(.*)$")


def normalize_jd_text(raw_text: str) -> str:
    """Normalize whitespace, standardize bullets to '- ', and strip email headers/signatures."""
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()

    # 1. Strip email headers at top or immediately after forwarded-message markers
    cleaned_lines: list[str] = []
    in_header_block = True
    for line in lines:
        stripped = line.strip()
        if _FORWARDED_MARKER_RE.match(stripped):
            in_header_block = True
            continue
        if in_header_block:
            if not stripped:
                # Blank line after headers ends the header block
                if cleaned_lines:
                    cleaned_lines.append("")
                in_header_block = False
                continue
            subj_m = _SUBJECT_RE.match(stripped)
            if subj_m:
                subj_val = subj_m.group(1).strip()
                if subj_val:
                    cleaned_lines.append(subj_val)
                continue
            if _HEADER_PREFIX_RE.match(stripped):
                continue
            in_header_block = False
        cleaned_lines.append(line)

    # 2. Heuristically strip trailing email signature block
    cut_idx: int | None = None
    n = len(cleaned_lines)
    # Search from near the end (only strip if <= 12 trailing lines remain)
    search_start = max(0, n - 14)
    for idx in range(search_start, n):
        stripped = cleaned_lines[idx].strip()
        if _SIG_DIVIDER_RE.match(stripped) or _SIG_SALUTATION_RE.match(stripped):
            remaining = [ln.strip() for ln in cleaned_lines[idx + 1 :] if ln.strip()]
            # Ensure we don't accidentally chop a real JD section
            if len(remaining) <= 10 and not any(
                kw in " ".join(remaining).lower()
                for kw in ("eligibility", "required skills", "ctc", "stipend", "responsibilities")
            ):
                cut_idx = idx
                break
    if cut_idx is not None:
        cleaned_lines = cleaned_lines[:cut_idx]

    # 3. Normalize whitespace and standardize bullet markers to '- '
    out_lines: list[str] = []
    blank_run = 0
    for line in cleaned_lines:
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        if not collapsed:
            blank_run += 1
            if blank_run <= 1 and out_lines:
                out_lines.append("")
            continue
        blank_run = 0
        bm = _BULLET_RE.match(collapsed)
        if bm:
            body = bm.group(1).strip()
            if body:
                out_lines.append(f"- {body}")
        else:
            out_lines.append(collapsed)

    return "\n".join(out_lines).strip()


def extract_jd_text(data: bytes, suffix: str) -> str:
    """Extract and normalize JD text from uploaded bytes (.pdf, .docx, .txt, .md).

    Enforces 2 MB max size (HTTP 413) and supported file extensions (HTTP 415).
    """
    if data is None:
        return ""
    if len(data) > MAX_JD_BYTES:
        raise HTTPException(413, f"JD file exceeds maximum allowed size ({MAX_JD_BYTES} bytes)")
    ext = (suffix or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if ext not in SUPPORTED_JD_SUFFIXES:
        raise HTTPException(
            415,
            f"Unsupported JD file extension: {ext or '(none)'}. Supported: .pdf, .docx, .txt",
        )
    try:
        raw = parse_resume_bytes(data, ext)
    except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise HTTPException(400, f"Could not extract text from JD file: {exc}") from exc
    return normalize_jd_text(raw)

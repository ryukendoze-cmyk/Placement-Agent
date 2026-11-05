"""PII detection & redaction (Phase 4). Redacts phone/email/Aadhaar/PAN/address in
text BEFORE it is stored (Student.resume_text) or sent to an LLM.

Deterministic regex rules; zero external deps. Returns (redacted_text, redacted_count)
and a per-kind breakdown. Redaction markers are `[REDACTED_<KIND>]`.
"""
from __future__ import annotations

import re

# Order matters: emails before phones (so phone regexes do not eat email digits),
# Aadhaar/PAN before looser number patterns.
_RULES: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d{5}[\s-]?\d{5}|\d{10})(?!\d)")),
    ("ADDRESS", re.compile(
        r"\b\d{1,4}[\s,]+(?:[A-Za-z]+\s+){0,4}"
        r"(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Nagar|Colony|Block|Sector|Phase|"
        r"House|Flat|Apartment|Housing)\b\.?",
        re.IGNORECASE)),
]

# Characters/words we allow around redactions so output stays readable.
_KINDS = ("EMAIL", "AADHAAR", "PAN", "PHONE", "ADDRESS")


def detect_pii(text: str) -> list[dict]:
    """Return every PII hit: {"kind": str, "value": str, "span": (start, end)}."""
    if not isinstance(text, str) or not text:
        return []
    hits: list[dict] = []
    claimed: list[tuple[int, int]] = []
    for kind, pat in _RULES:
        for m in pat.finditer(text):
            start, end = m.span()
            # skip overlaps with an earlier (higher-priority) rule
            if any(not (end <= s or start >= e) for s, e in claimed):
                continue
            claimed.append((start, end))
            hits.append({"kind": kind, "value": m.group(0), "span": (start, end)})
    hits.sort(key=lambda h: h["span"][0])
    return hits


def redact_pii(text: str) -> tuple[str, int, dict[str, int]]:
    """Replace every PII span with [REDACTED_<KIND>].

    Returns (redacted_text, redacted_count, per_kind_counts).
    """
    hits = detect_pii(text)
    if not hits:
        return (text if isinstance(text, str) else ""), 0, {}
    out: list[str] = []
    last = 0
    per_kind: dict[str, int] = {}
    for h in hits:
        start, end = h["span"]
        out.append(text[last:start])
        out.append(f"[REDACTED_{h['kind']}]")
        per_kind[h["kind"]] = per_kind.get(h["kind"], 0) + 1
        last = end
    out.append(text[last:])
    return "".join(out), len(hits), per_kind


def scrub_for_llm(text: str) -> tuple[str, int]:
    """Redact PII in text that is about to be sent to an LLM."""
    redacted, count, _ = redact_pii(text)
    return redacted, count

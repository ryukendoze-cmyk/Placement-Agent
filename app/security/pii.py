"""PII detection & redaction (Phase 4). Redacts phone/email/Aadhaar/PAN/address in
text BEFORE it is stored (Student.resume_text) or sent to an LLM.

Deterministic regex rules; zero external deps. Returns (redacted_text, redacted_count)
and a per-kind breakdown. Redaction markers are `[REDACTED_<KIND>]`.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

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
    ("ADDRESS", re.compile(
        r"\b(?:Flat|House|Plot|Apt|Apartment|Room|Bungalow|Villa)\s*(?:No\.?|#|:)?\s*[A-Za-z0-9-]{1,8}"
        r"(?:\s*,?\s*(?:[A-Za-z0-9.-]+\s+){0,4}(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Nagar|Colony|Vihar|Enclave|Extension|Extn|Marg|Society|Apartments|Residency))?\b\.?",
        re.IGNORECASE)),
    ("ADDRESS", re.compile(
        r"\b(?:Sector|Block)\s*(?:No\.?|#)?\s*[A-Za-z0-9-]{1,6}(?:\s*,?\s*(?:[A-Za-z]+\s+){0,3}(?:Nagar|Colony|Vihar|Enclave|City|Gurgaon|Gurugram|Noida|Jaipur|Delhi))?\b\.?",
        re.IGNORECASE)),
    ("ADDRESS", re.compile(
        r"\b(?:PIN|Pin\s*Code|Pincode|Postal\s*Code)\s*[:#\-]?\s*[1-9]\d{2}\s?\d{3}\b",
        re.IGNORECASE)),
    ("ADDRESS", re.compile(
        r"(?:\b(?:Jaipur|New\s+Delhi|Delhi|Noida|Gurgaon|Gurugram|Bengaluru|Bangalore|Mumbai|Pune|Hyderabad|Chennai|Kolkata|Ahmedabad|Chandigarh|Indore|Surat|Lucknow|Kochi|Rajasthan|Karnataka|Maharashtra|Haryana|Telangana|Gujarat|India)\s*[,:\-]\s*|\b(?:Jaipur|New\s+Delhi|Delhi|Noida|Gurgaon|Gurugram|Bengaluru|Bangalore|Mumbai|Pune|Hyderabad|Chennai|Kolkata|Rajasthan|India)\s+|,\s*|-\s*)[1-9](?!00000)\d{2}\s?\d{3}\b",
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


_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%*+-]+@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\[external-domain\])"
)


def _pseudonym_id(raw: str) -> str:
    digest = hashlib.sha256(raw.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"id-{digest}"


def mask_email(email: str | None) -> str:
    """Return an irreversible ``id-<8hex>`` reference with zero email/domain leakage."""
    cleaned = str(email or "").strip().lower()
    if not cleaned:
        return "anonymous"
    return _pseudonym_id(cleaned)


def audit_actor_ref(email: str | None, user_id: str | None = None) -> str:
    """Return a deterministic pseudonymized ``id-<8hex>`` actor reference for audit logs."""
    cleaned = str(email or "").strip().lower()
    if not cleaned:
        uid = str(user_id or "").strip()
        if not uid:
            return "anonymous"
        return _pseudonym_id(uid)
    return _pseudonym_id(cleaned)


def sanitize_audit_text(text: str | None) -> str:
    """Scrub any raw email pattern from an audit string into ``id-<8hex>``."""
    raw = str(text or "")
    if not raw:
        return ""
    return _EMAIL_PATTERN.sub(lambda m: _pseudonym_id(m.group(0)), raw)


def _scrub_audit_value(val: Any) -> Any:
    if isinstance(val, str):
        return sanitize_audit_text(val)
    if isinstance(val, dict):
        return {k: _scrub_audit_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_scrub_audit_value(item) for item in val]
    if isinstance(val, tuple):
        return tuple(_scrub_audit_value(item) for item in val)
    return val


def sanitize_audit_entry(entity_id: str | None, detail: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Recursively scrub any email addresses out of audit ``entity_id`` and ``detail``."""
    clean_entity = sanitize_audit_text(entity_id)
    clean_detail = _scrub_audit_value(dict(detail or {}))
    return clean_entity, clean_detail

"""Indirect prompt-injection defense (Milestone 15).

Untrusted text (resume uploads, mock-interview answers, scraped descriptions) can
carry adversarial instructions aimed at the LLM behind the CareerOS. Every pattern
below is deterministic regex (zero LLM calls, offline-safe): matches are replaced
with ``[REDACTED_INJECTION_ATTEMPT]`` and reported as flags so callers can write an
``AuditLog(action="security.prompt_injection_blocked")`` row.

Three helpers are exposed:

* :func:`sanitize_untrusted_text` - strip known injection phrasing from free text.
* :func:`wrap_untrusted_xml` - wrap untrusted content in an XML-ish tag with any
  embedded closing-tag broken up, so the payload cannot close the wrapper.
* :func:`verify_extracted_skill_grounding` - drop hallucinated high-level skills an
  LLM claims but that never appear in the resume text (SOP-08 resume integrity).
"""
from __future__ import annotations

import re

#: Adversarial instruction patterns, compiled once (case-insensitive).
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous_instructions",
     re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
                re.IGNORECASE)),
    ("system_prompt",
     re.compile(r"\bsystem\s*prompt\b", re.IGNORECASE)),
    ("you_are_now",
     re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE)),
    ("task_directive",
     re.compile(r"\bTASK:\s*[a-z_]+\b", re.IGNORECASE)),
    ("im_start_marker",
     re.compile(r"\<\|\s*im_start\s*\|\>")),
    ("inst_marker",
     re.compile(r"\[\s*INST\s*\]")),
    ("override_rubric",
     re.compile(r"\boverride\s+rubric\b", re.IGNORECASE)),
    ("disregard_rubric",
     re.compile(r"\b(?:disregard|forget|bypass|skip)\s+(?:all\s+|any\s+|the\s+|this\s+|your\s+|previous\s+|prior\s+|above\s+|scoring\s+|grading\s+|evaluation\s+|interview\s+)*(?:rubric|instructions|guidelines|policy|policies|rules)\b",
                re.IGNORECASE)),
    ("assign_score",
     re.compile(r"\b(?:assign|give|set|force|award)\s+(?:a\s+|an\s+|me\s+|the\s+|candidate\s+|full\s+)?(?:score|grade|rating|marks?)\s*(?:of|to|=|:)?\s*\d{1,3}\b",
                re.IGNORECASE)),
    ("json_score_output",
     re.compile(r"\b(?:output|return|respond\s+with|print|reply)\s*(?:only\s+)?\{\s*[\"']score[\"']",
                re.IGNORECASE)),
    ("forced_score",
     re.compile(r"\bscore\s*[:=]\s*100\b", re.IGNORECASE)),
]

REDACTION = "[REDACTED_INJECTION_ATTEMPT]"


def sanitize_untrusted_text(text: str, *, source: str = "resume") -> tuple[str, list[str]]:
    """Neutralize prompt-injection phrasing inside ``text``.

    Returns ``(sanitized_text, detected_flags)``. ``detected_flags`` is empty for
    clean input; otherwise it lists each pattern name that fired (for audit logging).
    Deterministic and offline-safe - never calls an LLM. ``source`` is accepted for
    caller context but does not change behavior (kept for API symmetry).
    """
    if not isinstance(text, str) or not text:
        return text or "", []
    flags: list[str] = []
    cleaned = text
    for name, pat in INJECTION_PATTERNS:
        if pat.search(cleaned):
            flags.append(name)
            cleaned = pat.sub(REDACTION, cleaned)
    return cleaned, flags


def wrap_untrusted_xml(tag: str, text: str) -> str:
    """Wrap ``text`` in ``<tag>`` with any embedded ``</tag>`` neutralized.

    Breaks up closing tags inside the payload (``</tag>`` -> ``<\\/tag>``) so the
    untrusted content cannot terminate the wrapper early and smuggle instructions
    outside the delimiters.
    """
    safe_tag = re.sub(r"[^A-Za-z0-9_-]", "", str(tag)) or "data"
    body = re.sub(rf"</\s*{re.escape(safe_tag)}\s*>",
                  f"<\\/{safe_tag}>", str(text), flags=re.IGNORECASE)
    return f"<{safe_tag}>\n{body}\n</{safe_tag}>"


def verify_extracted_skill_grounding(resume_text: str, extracted_skills: dict,
                                     taxonomy=None) -> dict:
    """Drop hallucinated high/intermediate-level skills not grounded in the resume.

    A skill entry (dict with ``level`` and optional ``evidence``) is kept when its
    level is ``basic``/``beginner`` (conservative), or when the skill id, one of its
    taxonomy aliases/name, or any evidence substring actually appears in
    ``resume_text`` (case-insensitive). Anything else is dropped, so an LLM cannot
    invent ``advanced`` proficiency the resume never claimed (SOP-08).
    """
    if not isinstance(extracted_skills, dict):
        return {}
    text = str(resume_text or "").lower()
    kept: dict = {}
    for skill_id, record in extracted_skills.items():
        if not isinstance(record, dict):
            continue
        level = str(record.get("level", "basic")).lower()
        if level in ("basic", "beginner", ""):
            kept[skill_id] = record
            continue
        probe_terms = [str(skill_id)]
        if taxonomy is not None:
            try:
                sk = taxonomy.get_opt(skill_id)
                if sk is not None:
                    probe_terms.append(str(getattr(sk, "name", "")))
                    probe_terms.extend(str(a) for a in (getattr(sk, "aliases", None) or []))
            except Exception:  # pragma: no cover - defensive
                pass
        evidence = record.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        grounded = any(term and term.lower() in text for term in probe_terms)
        if not grounded:
            grounded = any(str(ev).lower() in text for ev in evidence if ev)
        if grounded:
            kept[skill_id] = record
    return kept

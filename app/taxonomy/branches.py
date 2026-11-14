"""Canonical branch normalization and eligibility matching.

Resolves naming disparities across AI-extracted resume branches (e.g.
'Computer Science Engineering', 'B.Tech CSE') and job posting requirements
(e.g. ['CSE', 'IT']).
"""
from __future__ import annotations

import re
from typing import Iterable

CANONICAL_BRANCHES: tuple[tuple[str, str], ...] = (
    ("CSE", "CSE - Computer Science & Engineering"),
    ("AI&DS", "AI&DS - Artificial Intelligence & Data Science"),
    ("ECE", "ECE - Electronics & Communication Engineering"),
    ("IT", "IT - Information Technology"),
    ("EE", "EE - Electrical Engineering"),
    ("ME", "ME - Mechanical Engineering"),
    ("CE", "CE - Civil Engineering"),
    ("CHE", "CHE - Chemical Engineering"),
    ("BT", "BT - Biotechnology Engineering"),
)
JKLU_BRANCHES = CANONICAL_BRANCHES


def canonical_branches() -> list[str]:
    """Return ordered canonical JKLU branch codes."""
    return [code for code, _ in CANONICAL_BRANCHES]


def branch_choices() -> list[tuple[str, str]]:
    """Return ordered (code, display_label) tuples for UI branch dropdowns."""
    return list(CANONICAL_BRANCHES)


BRANCH_ALIASES: dict[str, set[str]] = {
    "CSE": {
        "CSE",
        "CS",
        "CS&E",
        "CS AND E",
        "COMPUTER SCIENCE",
        "COMPUTER SCIENCE ENGINEERING",
        "COMPUTER SCIENCE AND ENGINEERING",
        "COMPUTER SCIENCE & ENGINEERING",
        "COMPUTER SCIENCE & DESIGN",
        "COMPUTER SCIENCE AND DESIGN",
        "COMPUTING",
        "COMPUTER ENGINEERING",
        "COMP SCI",
        "CSE (AI)",
        "CSE AI",
        "CSE-AI",
        "CSE (CYBERSECURITY)",
        "CSE CYBERSECURITY",
        "CSE (CYBER SECURITY)",
        "CSE CYBER SECURITY",
        "CSE (DESIGN)",
        "CSE DESIGN",
        "CSE (IOT)",
        "CSE IOT",
        "CSE (CSBS)",
        "BTECH CSE",
        "B.TECH CSE",
        "MTECH CSE",
        "M.TECH CSE",
        "BE CSE",
        "ME CSE",
        "B.E. COMPUTER SCIENCE",
        "BTECH COMPUTER SCIENCE",
        "B.TECH COMPUTER SCIENCE",
        "MTECH COMPUTER SCIENCE",
        "M.TECH COMPUTER SCIENCE",
        "B.TECH COMPUTER ENGINEERING",
        "M.TECH COMPUTER ENGINEERING",
        "B.TECH IN COMPUTER SCIENCE",
        "M.TECH IN COMPUTER SCIENCE",
        "B.TECH IN COMPUTER SCIENCE AND ENGINEERING",
        "B.TECH IN COMPUTER SCIENCE & ENGINEERING",
        "M.TECH IN COMPUTER SCIENCE AND ENGINEERING",
        "M.TECH IN COMPUTER SCIENCE & ENGINEERING",
    },
    "IT": {
        "IT",
        "INFORMATION TECHNOLOGY",
        "INFO TECH",
        "INFORMATION SCIENCE",
        "INFORMATION SCIENCE AND ENGINEERING",
        "INFORMATION SCIENCE & ENGINEERING",
        "ISE",
        "BTECH IT",
        "B.TECH IT",
        "MTECH IT",
        "M.TECH IT",
        "B.TECH IN INFORMATION TECHNOLOGY",
        "BE IT",
        "B.E. IT",
    },
    "ECE": {
        "ECE",
        "ECE (VLSI)",
        "ECE VLSI",
        "ECE-VLSI",
        "VLSI",
        "ELECTRONICS AND COMMUNICATION",
        "ELECTRONICS & COMMUNICATION",
        "ELECTRONICS AND COMMUNICATION ENGINEERING",
        "ELECTRONICS & COMMUNICATION ENGINEERING",
        "ELECTRONICS AND COMMUNICATION (VLSI)",
        "ELECTRONICS & COMMUNICATION (VLSI)",
        "ELECTRONICS AND TELECOMMUNICATION",
        "ELECTRONICS & TELECOMMUNICATION",
        "ELECTRONICS",
        "ETC",
        "ETE",
        "BTECH ECE",
        "B.TECH ECE",
        "MTECH ECE",
        "M.TECH ECE",
        "B.TECH IN ELECTRONICS & COMMUNICATION ENGINEERING",
        "B.TECH IN ELECTRONICS AND COMMUNICATION ENGINEERING",
    },
    "EE": {
        "EE",
        "EEE",
        "ELECTRICAL",
        "ELECTRICAL ENGINEERING",
        "ELECTRICAL AND ELECTRONICS",
        "ELECTRICAL & ELECTRONICS",
        "ELECTRICAL AND ELECTRONICS ENGINEERING",
        "ELECTRICAL & ELECTRONICS ENGINEERING",
        "BTECH EE",
        "B.TECH EE",
        "MTECH EE",
        "M.TECH EE",
        "BTECH EEE",
        "B.TECH EEE",
        "MTECH EEE",
        "M.TECH EEE",
    },
    "ME": {
        "ME",
        "MECHANICAL",
        "MECHANICAL ENGINEERING",
        "MECHANICAL AND AUTOMATION",
        "MECHANICAL & AUTOMATION",
        "BTECH ME",
        "B.TECH ME",
        "MTECH ME",
        "M.TECH ME",
    },
    "CE": {
        "CE",
        "CIVIL",
        "CIVIL ENGINEERING",
        "BTECH CE",
        "B.TECH CE",
        "MTECH CE",
        "M.TECH CE",
    },
    "AI&DS": {
        "AI",
        "DS",
        "AIDS",
        "AI&DS",
        "AI/DS",
        "AI_DS",
        "AI & DS",
        "AI AND DS",
        "ARTIFICIAL INTELLIGENCE",
        "DATA SCIENCE",
        "AI AND DATA SCIENCE",
        "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
        "ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        "AIML",
        "AI ML",
        "AI-ML",
        "CSE AIML",
        "CSE-AIML",
        "CSE (AIML)",
        "CSE (AI & ML)",
        "CSE (AI&DS)",
        "CSE (AI & DS)",
        "CSE-DS",
        "CSE (DS)",
        "CSE (DATA SCIENCE)",
        "BTECH AI&DS",
        "B.TECH AI&DS",
        "MTECH AI&DS",
        "M.TECH AI&DS",
        "B.TECH IN ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
        "B.TECH IN ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
        "B.TECH AIML",
        "M.TECH AIML",
    },
    "CHE": {
        "CHE",
        "CH",
        "CHEMICAL",
        "CHEMICAL ENGINEERING",
        "BTECH CHE",
        "B.TECH CHE",
        "MTECH CHE",
        "M.TECH CHE",
    },
    "BT": {
        "BT",
        "BIOTECH",
        "BIOTECHNOLOGY",
        "BIOTECHNOLOGY ENGINEERING",
        "BTECH BT",
        "B.TECH BT",
        "MTECH BT",
        "M.TECH BT",
    },
}

_LOOKUP: dict[str, str] = {}
for canonical, aliases in BRANCH_ALIASES.items():
    _LOOKUP[canonical.upper()] = canonical
    for alias in aliases:
        _LOOKUP[alias.upper()] = canonical
        # Also index cleaned form with stripped punctuation
        cleaned = re.sub(r"[^\w\s]", " ", alias.upper())
        cleaned = " ".join(cleaned.split())
        _LOOKUP[cleaned] = canonical


def normalize_branch(raw: str | None) -> str:
    """Normalize any branch variant into its canonical uppercase code (e.g. 'CSE').

    If the string cannot be mapped to a known canonical branch, returns a trimmed,
    cleaned uppercase version of the input so custom branch names are preserved.
    """
    if not raw or not str(raw).strip():
        return ""

    s = str(raw).strip().upper()
    if s in _LOOKUP:
        return _LOOKUP[s]

    # Clean punctuation and extra whitespace: e.g. "B.Tech (CSE)" -> "B TECH CSE"
    cleaned = s.replace("&", " AND ").replace("/", " ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    if cleaned in _LOOKUP:
        return _LOOKUP[cleaned]

    stripped_deg = re.sub(
        r"^(?:B\s*TECH|BTECH|M\s*TECH|MTECH|B\s*E|BE|M\s*E|B\s*SC|M\s*SC|DIPLOMA)\s+(?:IN\s+|OF\s+)?",
        "",
        cleaned,
    ).strip()
    if stripped_deg in _LOOKUP:
        return _LOOKUP[stripped_deg]
    # Specialization parenthetical variants e.g. "B.Tech CSE (Cybersecurity)" -> "CSE"
    base_no_paren = re.sub(r"\s*\([^)]*\)", "", s).strip()
    if base_no_paren in _LOOKUP:
        return _LOOKUP[base_no_paren]

    # Substring / pattern matching in priority order
    if any(k in cleaned for k in ["ARTIFICIAL INTELLIGENCE", "DATA SCIENCE", "AI AND DS", "AIML", "AI ML"]):
        return "AI&DS"
    if any(k in cleaned for k in ["COMPUTER SCIENCE", "COMPUTER ENGINEERING", "COMPUTING", "COMP SCI"]) or re.search(r"\bCSE\b", cleaned):
        return "CSE"
    if any(k in cleaned for k in ["INFORMATION TECH", "INFORMATION SCIENCE", "INFO TECH"]):
        return "IT"
    if any(k in cleaned for k in ["ELECTRONIC", "COMMUNICATION ENGINEERING", "TELECOMMUNICATION"]):
        return "ECE"
    if any(k in cleaned for k in ["ELECTRICAL"]):
        return "EE"
    if any(k in cleaned for k in ["MECHANICAL"]):
        return "ME"
    if any(k in cleaned for k in ["CIVIL"]):
        return "CE"
    if any(k in cleaned for k in ["CHEMICAL"]):
        return "CHE"
    if any(k in cleaned for k in ["BIOTECH"]):
        return "BT"

    return cleaned


def is_branch_eligible(student_branch: str | None, allowed_branches: Iterable[str] | None) -> bool:
    """Check if student's branch is allowed for a job.

    Empty allowed_branches means open to all branches.
    Uses canonical branch normalization to resolve variations like
    'Computer Science Engineering' == 'CSE'.
    """
    if not allowed_branches:
        return True

    allowed_list = [str(b).strip() for b in allowed_branches if str(b).strip()]
    if not allowed_list:
        return True

    norm_student = normalize_branch(student_branch)
    norm_allowed = {normalize_branch(b) for b in allowed_list}

    if norm_student and norm_student in norm_allowed:
        return True

    # Fallback to direct raw case-insensitive equality
    raw_student = str(student_branch or "").strip().upper()
    raw_allowed = {str(b).strip().upper() for b in allowed_list}
    if raw_student and raw_student in raw_allowed:
        return True

    return False

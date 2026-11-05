"""Canonical branch normalization and eligibility matching.

Resolves naming disparities across AI-extracted resume branches (e.g.
'Computer Science Engineering', 'B.Tech CSE') and job posting requirements
(e.g. ['CSE', 'IT']).
"""
from __future__ import annotations

import re
from typing import Iterable

BRANCH_ALIASES: dict[str, set[str]] = {
    "CSE": {
        "CSE",
        "CS",
        "COMPUTER SCIENCE",
        "COMPUTER SCIENCE ENGINEERING",
        "COMPUTER SCIENCE AND ENGINEERING",
        "COMPUTER SCIENCE & ENGINEERING",
        "COMPUTING",
        "COMPUTER ENGINEERING",
        "COMP SCI",
        "BTECH CSE",
        "B.TECH CSE",
        "BE CSE",
        "B.E. COMPUTER SCIENCE",
        "B.TECH COMPUTER SCIENCE",
        "B.TECH IN COMPUTER SCIENCE",
        "B.TECH IN COMPUTER SCIENCE AND ENGINEERING",
        "B.TECH IN COMPUTER SCIENCE & ENGINEERING",
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
        "BE IT",
        "B.E. IT",
    },
    "ECE": {
        "ECE",
        "ELECTRONICS AND COMMUNICATION",
        "ELECTRONICS & COMMUNICATION",
        "ELECTRONICS AND COMMUNICATION ENGINEERING",
        "ELECTRONICS & COMMUNICATION ENGINEERING",
        "ELECTRONICS AND TELECOMMUNICATION",
        "ELECTRONICS & TELECOMMUNICATION",
        "ELECTRONICS",
        "ETC",
        "ETE",
        "BTECH ECE",
        "B.TECH ECE",
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
        "BTECH EEE",
        "B.TECH EEE",
    },
    "ME": {
        "ME",
        "MECHANICAL",
        "MECHANICAL ENGINEERING",
        "MECHANICAL AND AUTOMATION",
        "MECHANICAL & AUTOMATION",
        "BTECH ME",
        "B.TECH ME",
    },
    "CE": {
        "CE",
        "CIVIL",
        "CIVIL ENGINEERING",
        "BTECH CE",
        "B.TECH CE",
    },
    "AI&DS": {
        "AI",
        "DS",
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
        "CSE-DS",
        "CSE (DS)",
    },
    "CHE": {
        "CHE",
        "CH",
        "CHEMICAL",
        "CHEMICAL ENGINEERING",
        "BTECH CHE",
        "B.TECH CHE",
    },
    "BT": {
        "BT",
        "BIOTECH",
        "BIOTECHNOLOGY",
        "BIOTECHNOLOGY ENGINEERING",
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

    # Substring / pattern matching in priority order
    if any(k in cleaned for k in ["ARTIFICIAL INTELLIGENCE", "DATA SCIENCE", "AI AND DS", "AIML", "AI ML"]):
        return "AI&DS"
    if any(k in cleaned for k in ["COMPUTER SCIENCE", "COMPUTER ENGINEERING", "COMPUTING", "COMP SCI"]):
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

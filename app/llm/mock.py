import json
import math
import re


class MockLLM:
    def complete(self, system: str, user: str, json_mode: bool = False, **kwargs) -> str:
        tokens = system.split()
        task = ""
        for i, token in enumerate(tokens):
            if not token.startswith("TASK:"):
                continue
            task = token.split(":", 1)[1]
            if not task and i + 1 < len(tokens):
                task = tokens[i + 1]
            break

        if task == "extract_resume":
            return self._extract_resume(user)

        if task == "followup_probe":
            payload = json.loads(user) if user else {}
            missed = [str(k) for k in (payload.get("missed_keywords") or []) if str(k).strip()]
            skill = str(payload.get("skill_id") or "the skill")
            kw = missed[0] if missed else "the concept"
            if json_mode:
                return json.dumps({"question": (
                    f"Thanks - can you go deeper on '{kw}'? Explain how it applies to "
                    f"{skill} and what trade-offs you would consider.")})
            return (f"Can you go deeper on '{kw}'? Explain how it applies to {skill} "
                    f"and what trade-offs you would consider?")

        if task == "resume_probe":
            payload = json.loads(user) if user else {}
            snippet = str(payload.get("evidence") or payload.get("snippet") or "").strip()
            skill = str(payload.get("skill") or "the stack")
            if not snippet:
                snippet = f"work involving {skill}"
            if json_mode:
                return json.dumps({"question": (
                    f"In your resume you mentioned '{snippet}' using {skill}. Walk me "
                    f"through the architecture, the failure modes you guarded against, and "
                    f"how you measured performance under load.")})
            return (f"In your resume you mentioned '{snippet}' using {skill}. Walk me "
                    f"through the architecture, the failure modes you guarded against, and "
                    f"how you measured performance under load?")

        if task == "extract_job_posting":
            return self._extract_job_posting(user)

        if task == "plan_narrative":
            payload = json.loads(user) if user else {}
            gaps = payload.get("gaps", [])
            hours = payload.get("total_hours", 0)
            narrative = (
                f"Focus on {', '.join(gaps) if gaps else 'core fundamentals'} first - "
                f"these are the highest-leverage gaps for your target role. "
                f"Your plan totals {hours} hours; spread across the week "
                f"(about {max(1, round(hours / 4))}h x 4 weeks) keeps it sustainable. "
                "Complete items in order and mark them done to watch your readiness climb."
            )
            if json_mode:
                return json.dumps({"narrative": narrative})
            return narrative

        if task == "evaluate_answer":
            payload = json.loads(user) if user else {}
            answer = (payload.get("answer") or "").lower()
            keywords = [k.lower() for k in payload.get("keywords", [])]
            matched = [k for k in keywords if k in answer]
            missed = [k for k in keywords if k not in answer]
            hits = len(matched)
            denom = max(1, len(keywords))
            keyword_score = round(70 * hits / denom)
            word_list = answer.split()
            words = len(word_list)
            unique_ratio = (len(set(word_list)) / words) if words > 0 else 1.0
            length_bonus = 30 if words >= 12 else max(0, round(30 * words / 12))
            if words >= 6 and unique_ratio < 0.4:
                # Anti-gaming guard: penalize repetitive keyword stuffing
                length_bonus = min(length_bonus, 5)
                keyword_score = round(keyword_score * unique_ratio)
            score = max(0, min(100, keyword_score + length_bonus))
            technical = score
            depth = max(0, min(100, round(0.6 * length_bonus + 0.4 * keyword_score)))
            communication = max(0, min(100, round(0.5 * score + 0.5 * min(100, words * 3))))
            rubric = {
                "technical_accuracy": technical,
                "depth_and_examples": depth,
                "communication_clarity": communication,
            }
            feedback = (
                f"Matched {hits}/{denom} key concepts; answer "
                f"{'is detailed' if length_bonus == 30 else 'could be more detailed'}."
            )
            outline = [str(x) for x in (payload.get("outline") or []) if str(x).strip()]
            if outline:
                ideal = "Key points a strong answer covers: " + "; ".join(outline) + "."
            else:
                ideal = (
                    "A strong answer defines the concept precisely, gives a concrete example, "
                    "and names the trade-offs or edge cases an experienced engineer would raise."
                )
            if json_mode:
                return json.dumps({
                    "score": score,
                    "feedback": feedback,
                    "rubric": rubric,
                    "matched_keywords": matched,
                    "missed_keywords": missed,
                    "ideal_answer_hint": ideal,
                })
            return feedback

        if task == "interview_feedback":
            payload = json.loads(user) if user else {}
            score = payload.get("score", 0)
            per_skill = payload.get("per_skill", {})
            strengths = [s for s, v in per_skill.items() if v >= 70]
            weak = [s for s, v in per_skill.items() if v < 70]
            line = (
                f"Overall {score}/100. "
                + (f"Strong on {', '.join(strengths)}. " if strengths else "")
                + (f"Revisit {', '.join(weak)} before applying. " if weak else "")
                + "Keep practicing - consistency beats cramming."
            )
            if json_mode:
                return json.dumps({"feedback": line})
            return line

        if task == "improve_bullets":
            payload = json.loads(user) if user else {}
            bullets = payload.get("bullets", [])
            skills = payload.get("available_skills", [])
            s1 = skills[0] if skills else "the core skill"
            s2 = skills[1] if len(skills) > 1 else s1
            out = []
            for b in bullets[:3]:
                gist = str(b).strip().rstrip(".")
                improved = (f"Engineered a solution using {s1} (with {s2}) to {gist}. "
                            f"Result: measurably cleaner, tested, and faster output.")
                out.append({
                    "improved": improved,
                    "reason": "Added a strong action verb, a concrete available skill, and impact framing (STAR).",
                })
            if json_mode:
                return json.dumps({"improvements": out})
            return json.dumps(out)

        if task == "copilot_answer":
            return json.dumps(self._copilot_answer(user))

        if task == "plan_steps":
            return json.dumps(self._plan_steps(user))

        if task == "replan_steps":
            return json.dumps(self._replan_steps(user))

        if task == "repair_step":
            payload = json.loads(user) if user else {}
            tool = payload.get("tool", "")
            repaired_args = dict(payload.get("args") or {})
            if payload.get("student_id") and not repaired_args.get("student_id"):
                repaired_args["student_id"] = payload["student_id"]
            return json.dumps({
                "args": repaired_args,
                "thought": f"Retrying {tool} with corrected arguments.",
                "steps": [{"thought": f"Retrying {tool} with corrected arguments.",
                           "tool": tool, "args": repaired_args}],
            })

        if task == "synthesize":
            payload = json.loads(user) if user else {}
            goal = str(payload.get("goal", ""))
            facts = payload.get("facts", {}) or {}
            refs = ", ".join(payload.get("refs", []) or []) or "the records above"
            return json.dumps({
                "summary": (f"Plan for: {goal or 'the stated goal'}. "
                            f"Key findings were grounded in {refs}."),
                "sections": [
                    {"heading": "Executive Summary",
                     "body": f"Analyzed the goal and produced evidence-backed steps."},
                    {"heading": "Evidence & Reasoning",
                     "body": f"Findings: {json.dumps(facts, default=str)[:600]}"},
                    {"heading": "Next Actions",
                     "body": "Review the ranked matches, close the highlighted skill gaps, and track readiness."},
                ],
            })

        # generic fallback
        if json_mode:
            return json.dumps({"echo": user})
        return f"mock response to: {user[:80]}"

    def stream(self, system: str, user: str, json_mode: bool = False):
        """Deterministic offline streaming: same output as ``complete``, yielded
        in realistic 4-6 word chunks (Milestone 17).

        The chunker splits on whitespace and emits 4-6 words at a time so the
        JSON payload is preserved intact (splitting only on whitespace never
        breaks tokens) while producing a believable token cadence for SSE.
        """
        text = self.complete(system, user, json_mode=json_mode)
        words = text.split()
        if not words:
            yield text
            return
        i = 0
        toggle = 0
        while i < len(words):
            n = 4 if toggle % 2 == 0 else 5
            toggle += 1
            chunk_words = words[i:i + n]
            i += len(chunk_words)
            piece = " ".join(chunk_words)
            # re-add the original single-space joining (no trailing space needed
            # except between chunks so streamed concatenation == complete())
            if i < len(words):
                piece += " "
            yield piece

    def _extract_job_posting(self, user: str) -> str:
        """TASK: extract_job_posting - deterministic job posting extractor for MockLLM."""
        data: dict = {}
        try:
            if user and user.strip().startswith("{"):
                data = json.loads(user)
        except Exception:
            data = {}

        if isinstance(data, dict) and data:
            title = str(data.get("title") or "")
            company = str(data.get("company") or "")
            desc = str(data.get("description") or "")
            tags = str(data.get("tags") or "")
            full_text = f"{title} {company} {desc} {tags}"
            min_amt = data.get("min_amount")
            max_amt = data.get("max_amount")
            interval = str(data.get("interval") or "").lower()
        else:
            full_text = user or ""
            min_amt = None
            max_amt = None
            interval = ""

        # 1. Salary / Stipend Extraction
        salary_lpa = [0, 0]
        salary_disclosed = False

        def _to_float(v):
            try:
                amt = float(v)
                return None if math.isnan(amt) or amt <= 0 else amt
            except (TypeError, ValueError):
                return None

        p_min = _to_float(min_amt)
        p_max = _to_float(max_amt)

        if p_min is not None or p_max is not None:
            v_min = p_min if p_min is not None else p_max
            v_max = p_max if p_max is not None else p_min
            if "month" in interval or interval in ("monthly", "mo", "pm"):
                lpa_min = max(1, round(v_min * 12 / 100_000))
                lpa_max = max(lpa_min, round(v_max * 12 / 100_000))
                salary_lpa = [lpa_min, lpa_max]
            elif v_min <= 100:
                salary_lpa = [round(v_min), round(v_max)]
            else:
                salary_lpa = [max(1, round(v_min / 100_000)), max(1, round(v_max / 100_000))]
            salary_disclosed = True
        else:
            # Check text for monthly stipend patterns:
            # e.g., ₹30,000/mo, ₹25k/mo, 30,000 / month, 30k/mo, 25k per month
            stipend_re = re.search(
                r"(?:stipend|salary)?[:\s]*(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(k)?\s*(?:/|-)?\s*(?:mo|month|pm|per month)",
                full_text,
                re.IGNORECASE,
            )
            if not stipend_re:
                stipend_re = re.search(
                    r"(?:₹|rs\.?|inr)\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(k)?\s*(?:/|-)?\s*(?:mo|month|pm|per month)",
                    full_text,
                    re.IGNORECASE,
                )
            if not stipend_re:
                stipend_re = re.search(
                    r"stipend\s*(?:of|is|:)?\s*(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(k)?",
                    full_text,
                    re.IGNORECASE,
                )

            if stipend_re:
                raw_s = stipend_re.group(1).replace(",", "")
                is_k = bool(stipend_re.group(2))
                base_amt = float(raw_s) * (1000 if is_k else 1)
                annual = max(1, round(base_amt * 12 / 100_000))
                salary_lpa = [annual, annual]
                salary_disclosed = True
            else:
                # LPA range patterns: e.g. 8-15 LPA, 10 to 18 Lakhs, 6.5 LPA
                lpa_range = re.search(
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|to)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|lakhs?|lac|lacs)",
                    full_text,
                    re.IGNORECASE,
                )
                if lpa_range:
                    s_min = max(1, round(float(lpa_range.group(1))))
                    s_max = max(s_min, round(float(lpa_range.group(2))))
                    salary_lpa = [s_min, s_max]
                    salary_disclosed = True
                else:
                    single_lpa = re.search(
                        r"([0-9]+(?:\.[0-9]+)?)\s*(?:lpa|lakhs?|lac|lacs)",
                        full_text,
                        re.IGNORECASE,
                    )
                    if single_lpa:
                        val = max(1, round(float(single_lpa.group(1))))
                        salary_lpa = [val, val]
                        salary_disclosed = True

        # 2. Fresher-friendly & Experience Years
        low = full_text.lower()
        fresher_tokens = ("intern", "internship", "trainee", "fresher", "entry level", "entry-level", "junior", "jr.", "sde-1", "sde 1", "sde i", "college", "campus")
        senior_tokens = ("senior", "sr.", "lead", "staff", "principal", "architect", "manager", "head of", "director")

        exp_re = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:\+)?\s*(?:-|to)?\s*([0-9]+(?:\.[0-9]+)?)?\s*(?:years?|yrs?)(?:\s*of)?\s*exp",
            full_text,
            re.IGNORECASE,
        )
        if exp_re:
            exp_val = float(exp_re.group(1))
            experience_years = exp_val
            is_fresher_friendly = (exp_val == 0.0)
        elif any(tok in low for tok in fresher_tokens):
            experience_years = 0.0
            is_fresher_friendly = True
        elif any(tok in low for tok in senior_tokens):
            experience_years = 3.0
            is_fresher_friendly = False
        else:
            experience_years = 0.0
            is_fresher_friendly = True

        # 3. Grounded Skills Extraction
        matched_skills: list[str] = []
        try:
            from ..taxonomy.taxonomy import load_taxonomy
            tax = load_taxonomy()
            for alias, skill_id in tax.alias_map.items():
                pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
                if re.search(pattern, low):
                    resolved = tax.resolve(alias) or skill_id
                    if resolved not in matched_skills:
                        matched_skills.append(resolved)
        except Exception:
            for fallback in ("python", "sql", "git"):
                if fallback in low and fallback not in matched_skills:
                    matched_skills.append(fallback)

        if not matched_skills:
            matched_skills = ["python", "sql", "git"]
        elif len(matched_skills) == 1:
            fallback_pad = [s for s in ("python", "sql", "git") if s != matched_skills[0]][0]
            matched_skills.append(fallback_pad)

        split_idx = min(max(2, min(len(matched_skills), 4)), 6)
        required_skills = matched_skills[:split_idx]
        preferred_skills = matched_skills[split_idx:]

        return json.dumps({
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "salary_lpa": salary_lpa,
            "salary_disclosed": salary_disclosed,
            "is_fresher_friendly": is_fresher_friendly,
            "experience_years": experience_years,
        })

    def _extract_resume(self, user: str) -> str:
        """TASK: extract_resume - deterministic contextual resume parser.

        Parses the ``{"resume_text": ..., "taxonomy": [...]}`` payload, rebuilds
        a lightweight taxonomy view, and analyzes the resume text for candidate
        profile metadata (name/branch/year/gpa/summary) and grounded skills with
        dynamically-assigned proficiency levels (basic|intermediate|advanced)
        based on quantified metrics, engineering verbs, and evidence depth -
        no static keyword triggers.
        """
        from ..skills.extract import (
            _contextual_level,
            _skill_lines,
        )
        from ..taxonomy.taxonomy import Taxonomy

        payload = json.loads(user) if user else {}
        resume_text = str(payload.get("resume_text") or "")

        raw_taxonomy = payload.get("taxonomy")
        if isinstance(raw_taxonomy, list) and raw_taxonomy:
            normalized = [
                {
                    "id": s.get("id", ""),
                    "name": s.get("name", s.get("id", "")),
                    "category": s.get("category", "general"),
                    "aliases": list(s.get("aliases", [])),
                }
                for s in raw_taxonomy if isinstance(s, dict) and s.get("id")
            ]
            taxonomy = Taxonomy(normalized)
        else:
            from ..taxonomy.taxonomy import load_taxonomy
            taxonomy = load_taxonomy()

        skills: list[dict] = []
        for skill in taxonomy.all():
            lines = _skill_lines(skill, resume_text, taxonomy)
            if not lines:
                continue
            skills.append({
                "skill_id": skill.id,
                "level": _contextual_level(lines),
                "mentions": max(1, len(lines)),
                "evidence": list(lines),
            })

        profile = self._mock_profile(resume_text)
        return json.dumps({"profile": profile, "skills": skills})

    def _mock_profile(self, resume_text: str) -> dict:
        """Lightweight candidate-profile guess from resume header/section lines."""
        import re

        profile = {"name": "", "branch": "", "year": None, "gpa": None, "summary": ""}
        if not resume_text.strip():
            return profile

        gpa_match = re.search(r"\b(?:gpa|cgpa)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,3})?)\b", resume_text, re.I)
        if gpa_match:
            try:
                profile["gpa"] = float(gpa_match.group(1))
            except ValueError:
                profile["gpa"] = None

        branch_match = re.search(
            r"\b(CSE|CS|IT|ECE|EEE|ME|CE|AIML|AI|DS|AI&DS|AIDS|BT|CH)\b"
            r"|(computer science(?:\s+&\s*engineering)?|information technology|"
            r"artificial intelligence(?:\s*&\s*data science)?|electronics(?:\s*&\s*communication)?)",
            resume_text, re.I,
        )
        if branch_match:
            profile["branch"] = (branch_match.group(1) or branch_match.group(2) or "").strip()

        year_match = re.search(
            r"\b(?:graduat|batch|class of|year)\s*(?:e?d)?\s*[:\-]?\s*(20\d{2}|\d)\b",
            resume_text, re.I,
        )
        if year_match:
            token = year_match.group(1)
            profile["year"] = int(token)

        summary_match = re.search(r"^summary\s*:\s*(.+)$", resume_text, re.I | re.M)
        if summary_match:
            profile["summary"] = summary_match.group(1).strip()

        contact_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://|www\.|linkedin|github", re.I)
        header_re = re.compile(
            r"^(resume|curriculum vitae|cv|skills?|education|experience|projects?|"
            r"achievements?|summary|objective|contact|profile|certifications?)\b",
            re.I,
        )
        non_name_re = re.compile(
            r"\b(b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|bca|mca|"
            r"year|cgpa|gpa|semester|batch|student|engineer|developer|aspirant)\b",
            re.I,
        )
        verb_start_re = re.compile(
            r"^(built|developed|designed|implemented|created|architected|engineered|"
            r"led|ran|made|worked|spearheaded|launched|optimized|deployed|maintained|"
            r"used|wrote|configured|delivered|automated|integrated|tested)\b",
            re.I,
        )
        for line in resume_text.splitlines()[:8]:
            stripped = line.strip()
            if not stripped or len(stripped) > 60:
                continue
            if contact_re.search(stripped) or header_re.match(stripped):
                continue
            words = re.findall(r"[A-Z][a-z]+", stripped)
            if not (2 <= len(words) <= 4):
                continue
            if non_name_re.search(stripped) or verb_start_re.match(stripped):
                continue
            profile["name"] = " ".join(words)
            break

        # Comma-separated header form: "Rohan Gupta, B.Tech CSE, Year 3, CGPA 8.4"
        if not profile["name"]:
            for line in resume_text.splitlines()[:8]:
                stripped = line.strip()
                if not stripped or len(stripped) > 60:
                    continue
                if contact_re.search(stripped) or header_re.match(stripped):
                    continue
                head, sep, _tail = stripped.partition(",")
                seg = head if sep else stripped
                if non_name_re.search(seg) or verb_start_re.match(seg):
                    continue
                seg_words = re.findall(r"[A-Za-z]+", re.sub(r"[^A-Za-z\s]", " ", seg))
                if not (2 <= len(seg_words) <= 3):
                    continue
                if any(not w[0].isupper() for w in seg_words):
                    continue
                profile["name"] = " ".join(seg_words)
                break
        return profile

    def _copilot_answer(self, user: str) -> dict:
        """Deterministic, intent-driven grounded answer citing only known SOP ids."""
        payload = json.loads(user) if user else {}
        question = str(payload.get("question", ""))
        q = question.lower()
        student = payload.get("student") or {}
        elig = payload.get("eligibility") or {}
        sources = payload.get("sources") or []
        src_ids = [s.get("sop_id") for s in sources if isinstance(s, dict) and s.get("sop_id")]

        gpa = student.get("gpa")
        mock = student.get("mock_interview_score")
        status = elig.get("status")

        def cite(*ids):
            # keep only ids that actually appear in retrieved sources (grounded)
            out = [i for i in ids if i in src_ids]
            return out or src_ids[:2]

        stripped = q.strip().rstrip("?!., ")
        if stripped in ("what", "what do you mean", "why", "hello", "hi", "hey", "who are you", "help", "help me"):
            c = cite("SOP-01", "SOP-04")
            sname = student.get("name", "Student")
            answer = (f"Hello {sname}! I am your Placement Copilot. I can sort and rank your top job matches, "
                      f"analyze your verified competencies and weaknesses, or check institutional placement rules. "
                      f"Under [SOP-01] and [SOP-04], I can guide your pathway toward Dream and Super-Dream placements. "
                      f"Try asking: 'Rank my top job matches' or 'What skills of mine are weak?'.")
            return {"answer": answer, "cited_sops": [i.strip("[]") for i in c]}

        if any(k in q for k in ("mock", "interview")):
            c = cite("SOP-04", "SOP-01")
            answer = ("[SOP-04] requires a mock-interview score of >= 65 to be nominated for "
                      "Dream and Super-Dream drives")
            if mock is not None:
                answer += f"; your current score is {mock}"
            answer += ". [SOP-01] classifies 8-15 LPA as Dream and >= 15 LPA as Super-Dream."
            if status == "blocked":
                answer += " You are currently blocked until a re-interview reaches 65."
        elif any(k in q for k in ("tier", "salary", "ctc", "upgrade", "offer", "lpa")):
            c = cite("SOP-01", "SOP-02")
            answer = ("[SOP-01] tiers: Standard < 8 LPA, Dream 8-14.99 LPA, Super-Dream >= 15 LPA. "
                      "[SOP-02] allows at most one offer per tier and upward-only mobility.")
        elif any(k in q for k in ("backlog", "gpa", "cgpa", "standing", "academic")):
            c = cite("SOP-03")
            answer = "[SOP-03] enforces the recruiter CGPA cutoff and limits active backlogs (0 for Super-Dream, <=1 for Dream)."
            if gpa is not None:
                answer += f" Your GPA is {gpa}."
        elif any(k in q for k in ("resume", "fake", "fabricat", "cheat", "integrity")):
            c = cite("SOP-08")
            answer = "[SOP-08] mandates zero fabrication: every resume claim needs verifiable evidence; violations cause debarment."
        elif any(k in q for k in ("rank", "job", "match", "sort", "recommend", "priorit")):
            top_matches = student.get("top_matches") or []
            c = cite("SOP-01", "SOP-04")
            lines = []
            if top_matches:
                items = []
                for i, m in enumerate(top_matches[:3], 1):
                    co = f" at {m['company']}" if m.get("company") else ""
                    tier = m.get("tier", "Standard")
                    items.append(f"{i}. {m.get('title', 'Role')}{co} (Match Score: {m.get('score', 0)}, Tier: {tier})")
                match_txt = " ".join(items)
                lines.append(f"Here are your top job matches sorted by profile compatibility: {match_txt}. [SOP-01] classifies placement tiers.")
            else:
                skills = student.get("skills_by_level", {}).get("advanced", [])
                skill_str = (", ".join(skills[:4])) if skills else "technical skills"
                lines.append(f"Evaluating your profile with key skills in {skill_str}. [SOP-01] governs job tier classification.")

            if any(k in q for k in ("prepar", "standing", "fit", "ready", "gate")):
                if mock is not None and mock < 65:
                    lines.append(f"On preparedness: your mock interview score is {mock}/100, which is below the 65 verified score required under [SOP-04] for Dream (8-14.99 LPA) and Super-Dream (15+ LPA) roles. You are currently eligible for Standard tier roles; completing a re-interview will unlock recruiter shortlists for higher tiers.")
                elif mock is not None:
                    lines.append(f"On preparedness: your mock interview score is {mock}/100, which satisfies the [SOP-04] readiness requirement for Dream drives.")
            elif mock is not None and mock < 65:
                lines.append(f"Note: Under [SOP-04], you require a mock interview score >= 65 for Dream/Super-Dream nomination (your current score: {mock}).")

            answer = " ".join(lines)
        elif any(k in q for k in ("weak", "skill", "gap", "prepar", "strength", "improv", "learn", "competenc")):
            c = cite("SOP-01", "SOP-04")
            by_level = student.get("skills_by_level") or {}
            basic_skills = by_level.get("basic") or []
            adv_skills = by_level.get("advanced") or []
            top_m = (student.get("top_matches") or [{}])[0]
            missing = top_m.get("missing_required") or []

            lines = []
            if basic_skills:
                lines.append(f"Based on your profile, your verified skills currently at 'Basic' proficiency include: {', '.join(basic_skills[:6])}.")
            if missing:
                lines.append(f"For your lead job match ({top_m.get('title', 'top role')}), key missing required skills include: {', '.join(missing[:4])}.")
            if adv_skills:
                lines.append(f"Your strongest skills (Advanced) are: {', '.join(adv_skills[:5])}.")

            # If the user requested details, provide structured actionable steps
            if any(k in q for k in ("detail", "talk", "explain", "how", "plan", "roadmap", "priorit")):
                lines.append("\nRecommended Upskilling Action Plan:")
                if missing:
                    for i, m_sk in enumerate(missing[:3], 1):
                        lines.append(f"{i}. {m_sk} (High Priority): Build and commit a hands-on project demonstrating core operations, indexing, and REST integration to earn verified evidence under [SOP-08].")
                if basic_skills:
                    for i, b_sk in enumerate(basic_skills[:2], len(missing[:3]) + 1):
                        lines.append(f"{i}. Elevate {b_sk}: Advance from Basic to Intermediate by passing technical mock evaluations and adding production-style tests.")

            if mock is not None and mock < 65:
                lines.append(f"\nMost critically, under [SOP-04], your AI mock interview score is {mock}/100, which is below the mandatory 65 cutoff required to unblock Dream and Super-Dream drive nominations.")
            else:
                lines.append("\nUnder [SOP-01], leveling up these foundational skills will unlock higher placement tiers (Dream 8-14.99 LPA and Super-Dream 15+ LPA).")
            answer = "\n".join(lines)
        else:
            c = src_ids[:2]
            sop_summaries = []
            for s in sources[:2]:
                if isinstance(s, dict) and s.get("rule_text"):
                    sop_summaries.append(f"[{s.get('sop_id')}]: {s.get('rule_text')}")
            if sop_summaries:
                answer = "Based on institutional policy: " + " ".join(sop_summaries)
            else:
                answer = f"Based on JKLU placement policy {', '.join('[' + i + ']' for i in c)}, " \
                         f"your inquiry is governed by the retrieved sections."

        cited = [i.strip("[]") for i in c]
        return {"answer": answer, "cited_sops": cited}

    #: Objective verbs the planner can act on without further clarification.
    _OBJECTIVES = ("rank", "match", "eligib", "apply", "nomina", "drive", "plan",
                   "roadmap", "course", "interview", "resume", "ats", "bullet",
                   "cohort", "bottleneck", "intervention", "workshop", "bootcamp",
                   "policy", "sop", "rule", "handbook", "simulat", "gpa", "backlog",
                   "recommend", "improve", "explain", "show", "find", "search", "build",
                   "generate", "draft", "check", "audit")

    def _is_underspecified(self, goal: str) -> bool:
        """True when the goal names no placement objective the planner can act on.

        Deterministic proxy for a real LLM's judgement: a goal like "help me with my
        career today" carries no actionable objective, while anything mentioning
        ranking, eligibility, plans, interviews, cohorts, policy, etc. is specific
        enough to execute. Never fires once the user's clarification is appended.
        """
        g = (goal or "").lower()
        if not g.strip():
            return True
        if "(user clarification:" in g:
            return False  # clarified on resume
        return not any(k in g for k in self._OBJECTIVES)

    def _has_placement_context(self, goal: str) -> bool:
        """True when an eligibility goal already names a role/company/target."""
        g = (goal or "").lower()
        if "(user clarification:" in g:
            return True
        anchors = ("backend", "frontend", "full stack", "fullstack", "data", "devops",
                   "sde", "engineer", "developer", "analyst", "naukri", "amazon",
                   "google", "microsoft", "top match", "top matched", "job-", "dream",
                   "super", "specific", "company", "target")
        return any(a in g for a in anchors)

    def _plan_steps(self, user: str) -> dict:
        """Deterministic, intent-driven 2-5 step decomposition."""
        payload = json.loads(user) if user else {}
        goal = str(payload.get("goal", "")).lower()
        student_id = payload.get("student_id")
        role = str(payload.get("actor_role", "student")).lower()

        def S(thought, tool, args):
            return {"thought": thought, "tool": tool, "args": args}

        def sid(default=None):
            return student_id or default

        if any(k in goal for k in ("purge live", "remove live", "clear live", "delete live")):
            steps = [
                S("Purge external live-scraped market jobs from the database.",
                  "purge_live_market_jobs", {}),
            ]
        elif any(k in goal for k in ("live market", "external job", "search live", "market job")):
            steps = [
                S("Inspect the student's verified skills and academic standing.",
                  "inspect_student_profile", {"student_id": sid()}),
                S("Search live market jobs matching the student's profile and query.",
                  "search_live_market_jobs", {"student_id": sid(), "query": "software engineer"}),
            ]
        elif any(k in goal for k in ("prerequisite", "dag", "topological", "dependency order")):
            steps = [
                S("Inspect the student's verified skills and current proficiency levels.",
                  "inspect_student_profile", {"student_id": sid()}),
                S("Rank jobs to determine the target missing skills.",
                  "rank_and_explain_jobs", {"student_id": sid(), "top": 3}),
                S("Order missing skills topologically along the prerequisite DAG.",
                  "generate_prerequisite_roadmap", {"target_skills": "$last.missing_required"}),
            ]
        elif any(k in goal for k in ("interview history", "past interview", "rubric breakdown")):
            steps = [
                S("Inspect the student's completed mock interview history and rubric breakdown.",
                  "inspect_interview_history", {"student_id": sid()}),
                S("Inspect the student's profile to correlate weak interview skills with resume evidence.",
                  "inspect_student_profile", {"student_id": sid()}),
            ]
        elif any(k in goal for k in ("persist plan", "save plan", "persist upskilling")):
            steps = [
                S("Inspect the student's verified skills and standing.",
                  "inspect_student_profile", {"student_id": sid()}),
                S("Rank jobs to identify the top target role.",
                  "rank_and_explain_jobs", {"student_id": sid(), "top": 3}),
                S("Build and persist a prerequisite-ordered upskilling plan for the top match.",
                  "persist_upskilling_plan",
                  {"student_id": sid(), "job_id": "$rank_and_explain_jobs.top_job_id"}),
            ]
        elif any(k in goal for k in ("submit application", "submit drive application")):
            steps = [
                S("Inspect the student's profile before submitting a drive application.",
                  "inspect_student_profile", {"student_id": sid()}),
                S("Rank jobs to select the best target drive.",
                  "rank_and_explain_jobs", {"student_id": sid(), "top": 3}),
                S("Submit the campus drive application with SOP eligibility verification.",
                  "submit_drive_application",
                  {"student_id": sid(), "job_id": "$rank_and_explain_jobs.top_job_id"}),
            ]
        elif any(k in goal for k in ("resume", "bullet", "ats", "xray", "improve")):
            steps = [
                S("Retrieve the student's stored resume text and skill evidence.", "inspect_student_profile",
                  {"student_id": sid()}),
                S("Run the ATS X-Ray and draft grounded STAR improvements.", "audit_resume_bullets",
                  {"student_id": sid()}),
            ]
            # Multi-intent composition: if the user ALSO asked for interview history or policy or roadmap
            if any(k in goal for k in ("interview history", "past interview")):
                steps.append(S("Review completed mock interview history.", "inspect_interview_history",
                               {"student_id": sid()}))
            if any(k in goal for k in ("policy", "sop", "handbook")):
                steps.append(S("Retrieve governing placement SOP clauses.", "search_placement_policy_rag",
                               {"query": goal, "top_k": 3}))
        elif any(k in goal for k in ("simulate drive", "prospective drive", "cohort drive")):
            steps = [
                S("Simulate cohort eligibility and bootcamp ROI for the prospective drive.",
                  "simulate_placement_drive",
                  {"title": "Prospective Campus Drive", "min_gpa": 7.0,
                   "required_skills": ["python", "sql"]}),
                S("Query aggregate cohort skill gaps to cross-check bottlenecks.",
                  "query_cohort_bottlenecks", {"top_n": 5}),
            ]
        elif any(k in goal for k in ("cohort", "bottleneck", "intervention", "batch", "workshop", "bootcamp")):
            steps = [
                S("Query aggregate cohort skill gaps and readiness bands.", "query_cohort_bottlenecks",
                  {"top_n": 5}),
                S("Design a targeted intervention for the top gap, grounded in the catalog.",
                  "draft_cohort_intervention", {"skill_id": "$first.top_gap_skill"}),
            ]
            if role in ("tpo_admin", "faculty_mentor"):
                steps.append(S("Route the intervention for human approval before publishing.",
                               "propose_hitl_approval",
                               {"kind": "cohort_bootcamp_publish", "target_id": "$first.top_gap_skill",
                                "title": "Cohort bridge bootcamp", "summary": "Launch a bootcamp for the top cohort skill gap.",
                                "payload": {"skill_id": "$first.top_gap_skill"}}))
        elif ("eligib" in goal and not self._has_placement_context(goal)):
            # underspecified eligibility ask (no role/company named) AND no drive
            # action intent: ask the user first. When the goal also asks to apply /
            # nominate / check a drive, rank first then let the orchestrator's
            # observation-driven injection add the SOP eligibility gate.
            if any(k in goal for k in ("apply", "nomina", "drive")):
                steps = [
                    S("Inspect the student's verified skills and academic standing.",
                      "inspect_student_profile", {"student_id": sid()}),
                    S("Rank jobs so the top match can be SOP-checked for the drive.",
                      "rank_and_explain_jobs", {"student_id": sid(), "top": 3}),
                ]
            else:
                steps = [
                    S("Ask which target role or company the eligibility check should cover.",
                      "ask_clarifying_question",
                      {"question": "Which target role or company should I check drive eligibility for?",
                       "options": ["Top matched job", "A specific company", "Any Dream role"],
                       "missing_fields": ["target_job"]}),
                ]
        elif any(k in goal for k in ("policy", "sop", "copilot", "eligib", "placement rule",
                                     "handbook", "regulation", "mock interview", "offer rule")):
            steps = [
                S("Retrieve the governing institutional placement SOP clauses for the question.",
                  "search_placement_policy_rag", {"query": goal, "top_k": 3}),
                S("Inspect the student's academic standing to ground the policy answer.",
                  "inspect_student_profile", {"student_id": sid()}),
            ]
        elif self._is_underspecified(goal):
            # M16: genuinely ambiguous goal -> pause for clarification first
            steps = [
                S("Ask the user to narrow the goal before spending tool calls.",
                  "ask_clarifying_question",
                  {"question": ("Could you clarify what you would like me to focus on "
                                "for this placement request?"),
                   "options": ["Rank and explain my top job matches",
                               "Check my drive eligibility",
                               "Build an upskilling plan",
                               "Review my interview history"],
                   "missing_fields": ["objective"]}),
            ]
        else:
            # default: job matching & upskilling intent
            steps = [
                S("Inspect the student's verified skills, GPA, and academic standing.",
                  "inspect_student_profile", {"student_id": sid()}),
                S("Rank jobs and extract the missing required competencies for the top match.",
                  "rank_and_explain_jobs", {"student_id": sid(), "top": 3}),
                S("Simulate the score lift if the student acquires the top missing gaps.",
                  "simulate_skill_acquisition",
                  {"student_id": sid(), "job_id": "$last.top_job_id", "skills": "$last.missing_required"}),
                S("Retrieve certified courses that close the identified gaps.",
                  "search_course_catalog", {"skill_ids": "$last.impacted_skills"}),
            ]
            if any(k in goal for k in ("apply", "nominate", "drive", "eligib", "offer")):
                steps.append(S("Draft a nomination approval request for the placement drive.",
                               "propose_hitl_approval",
                               {"kind": "job_application_nomination",
                                "target_id": "$rank_and_explain_jobs.top_job_id",
                                "title": "Drive nomination",
                                "summary": "Nominate the student for the top-matched drive.",
                                "payload": {"student_id": student_id or "",
                                            "job_id": "$rank_and_explain_jobs.top_job_id",
                                            "readiness": 0}}))
        return {"steps": steps[:5]}

    def _replan_steps(self, user: str) -> dict:
        """Observation-driven replanner for TASK: replan_steps."""
        try:
            payload = json.loads(user) if user else {}
        except Exception:
            payload = {}
        goal = str(payload.get("goal") or "").lower()
        student_id = payload.get("student_id")
        last_tool = str(payload.get("last_tool") or "")
        observation = payload.get("observation") if isinstance(payload.get("observation"), dict) else {}
        queued = set(payload.get("queued_tools") or [])

        wants_drive = any(mk in goal for mk in ("eligib", "apply", "nomina", "drive"))
        steps = []
        if (last_tool == "rank_and_explain_jobs" and wants_drive
                and student_id and observation.get("top_job_id")
                and "check_drive_eligibility" not in queued):
            steps.append({
                "thought": "Observation: top match identified; checking institutional SOP eligibility.",
                "tool": "check_drive_eligibility",
                "args": {"student_id": student_id, "job_id": "$rank_and_explain_jobs.top_job_id"},
            })
        elif (last_tool == "check_drive_eligibility"
                and observation.get("eligible") is False
                and student_id):
            suggested = " ".join(str(s) for s in (observation.get("suggested_actions") or []))
            if "SOP-04" in suggested and "inspect_interview_history" not in queued:
                steps.append({
                    "thought": "Observation: blocked by the mock-interview gate (SOP-04); reviewing interview history to close it.",
                    "tool": "inspect_interview_history",
                    "args": {"student_id": student_id},
                })
            elif "persist_upskilling_plan" not in queued:
                steps.append({
                    "thought": "Observation: eligibility blocked; building a proficiency-aware remediation plan.",
                    "tool": "persist_upskilling_plan",
                    "args": {"student_id": student_id, "job_id": "$check_drive_eligibility.job_id"},
                })
        return {"steps": steps[:2]}

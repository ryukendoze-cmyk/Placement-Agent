from datetime import datetime, timezone

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


class Student(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    branch: str = ""
    year: int = 3
    gpa: float = 0.0
    backlogs: int = 0
    pii_redacted_count: int = 0
    resume_reupload_unlocked: bool = Field(default=False)
    resume_text: str = Field(default="", sa_column=Column(Text))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudentSkill(SQLModel, table=True):
    student_id: str = Field(foreign_key="student.id", primary_key=True)
    skill_id: str = Field(foreign_key="skill.id", primary_key=True)
    mentions: int = 0
    level: str = "basic"
    evidence: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    verification_status: str = Field(default="claimed", index=True)
    confidence: float = Field(default=0.5)
    source: str = Field(default="resume")
    last_verified_score: int | None = Field(default=None)
    last_verified_at: datetime | None = Field(default=None)


class Skill(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    category: str
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    company: str
    description: str
    required_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    preferred_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    experience_years: int = 0
    location: str = ""
    remote: bool = False
    salary_lpa: list[int] = Field(default_factory=lambda: [0, 0], sa_column=Column(JSON))
    salary_disclosed: bool = Field(default=False)
    min_gpa: float = 0.0
    allowed_branches: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    max_backlogs: int | None = Field(default=None)
    job_url: str = Field(default="")
    source: str = Field(default="local")
    scraper_engine: str = Field(default="")
    is_live: bool = Field(default=False, index=True)
    owner_student_id: str | None = Field(default=None, index=True)

    @property
    def is_fresher(self) -> bool:
        from app.matching.engine import is_fresher_job

        return is_fresher_job(self)


class JobMatch(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id", index=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    score: int
    band: str
    breakdown: dict = Field(default_factory=dict, sa_column=Column(JSON))
    is_favorite: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobApplication(SQLModel, table=True):
    """Campus drive application row (M19): one per (student, job) drive attempt.

    status lifecycle:
      submitted | awaiting_mentor_endorsement | awaiting_tpo_approval
      | nominated | rejected | withdrawn
    tier: standard | dream | super_dream (job salary band at apply time).
    """
    id: str = Field(primary_key=True)                        # app-<hex8>
    student_id: str = Field(foreign_key="student.id", index=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    status: str = "submitted"
    tier: str = "standard"
    match_score: int = 0
    eligible_at_apply: bool = True
    eligibility_checks: dict = Field(default_factory=dict, sa_column=Column(JSON))
    approval_id: str | None = Field(default=None, foreign_key="approvalrequest.id")
    endorsed_by: str = ""
    approved_by: str = ""
    notes: str = ""
    is_favorite: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Course(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    provider: str
    url: str
    skill_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    level: str = "beginner"
    hours: int = 4
    cost: str = "free"


class UpskillPlan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id", index=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    total_hours: int
    narrative: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlanItem(SQLModel, table=True):
    id: str = Field(primary_key=True)
    plan_id: str = Field(foreign_key="upskillplan.id", index=True)
    order: int
    skill_id: str = Field(index=True)
    course_id: str = Field(foreign_key="course.id")
    hours: int
    status: str = "pending"
    week: int = 1


class ProgressEvent(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id", index=True)
    kind: str
    ref_id: str = ""
    readiness: int
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Interview(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id", index=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    transcript: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    score: int | None = None
    feedback: dict = Field(default_factory=dict, sa_column=Column(JSON))
    finished: bool = False
    role: str | None = Field(default=None)
    #: M18.2: serve resume-grounded project deep-dive probes first.
    include_resume_probes: bool = False
    #: M18.1: enable adaptive follow-up probing for this interview.
    adaptive_followup: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRun(SQLModel, table=True):
    """A single autonomous agent execution (plan -> execute -> verify -> recover)."""
    id: str = Field(primary_key=True)
    student_id: str | None = Field(default=None, foreign_key="student.id")
    actor_role: str = "student"
    goal: str
    status: str = "running"                       # running|completed|failed|awaiting_approval
    plan_json: list = Field(default_factory=list, sa_column=Column(JSON))
    final_answer: str = ""
    citations: dict = Field(default_factory=dict, sa_column=Column(JSON))  # verified/flags/grounding_score
    total_latency_ms: int = 0
    provider_used: str = ""
    fallback_triggered: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentStep(SQLModel, table=True):
    """One tool invocation inside an AgentRun (the reasoning trace)."""
    id: str = Field(primary_key=True)
    run_id: str = Field(foreign_key="agentrun.id", index=True)
    step_order: int
    thought: str = ""
    tool_name: str
    tool_input: dict = Field(default_factory=dict, sa_column=Column(JSON))
    tool_output: dict = Field(default_factory=dict, sa_column=Column(JSON))
    verification_passed: bool = False
    error_recovered: str = ""
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalRequest(SQLModel, table=True):
    """Human-in-the-loop approval for a state-changing action (Phase 4).
    Nothing mutates until status flips pending_review -> approved."""
    id: str = Field(primary_key=True)                        # appr-<hex8>
    kind: str                                                # nomination | cohort_bootcamp_publish | proficiency_waiver | job_application_nomination | resume_reupload_request
    target_id: str = ""
    title: str = ""
    summary: str = ""
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="pending_review", index=True)  # pending_review|approved|rejected
    requested_by: str = "agent"                              # agent | role
    reviewed_by: str = ""                                    # role that approved/rejected
    review_note: str = ""
    run_id: str | None = Field(default=None, foreign_key="agentrun.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: datetime | None = None
    endorsed_by: str = ""
    endorsed_at: datetime | None = None


class AuditLog(SQLModel, table=True):
    """Append-only audit trail. Rows are immutable (no update/delete endpoints;
    an SQLAlchemy before_update/before_delete event refuses modifications)."""
    id: str = Field(primary_key=True)                        # aud-<hex8>
    action: str = Field(index=True)                           # e.g. approval.approved
    entity_type: str = ""                                    # approval | student | plan | ...
    entity_id: str = ""
    actor_role: str = "system"
    detail: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                 index=True)


class CampusDrive(SQLModel, table=True):
    """Campus placement drive intake, AI extraction provenance, and HITL lifecycle."""
    id: str = Field(primary_key=True)                        # cd-<hex8>
    trace_id: str = Field(default="", index=True)            # tr-<hex8>
    company: str = ""
    role_title: str = ""
    location: str = ""
    work_mode: str = "On-site"                               # On-site | Remote | Hybrid
    ctc_lpa_min: float | None = None
    ctc_lpa_max: float | None = None
    salary_disclosed: bool = False
    stipend_inr_per_month: int | None = None
    policy_tier: str | None = None                           # super_dream | dream | standard | None
    policy_tier_source: str | None = None                    # SOP-01 | None
    experience_level: str = "Fresher"                        # Fresher | Internship
    source_portal: str = "On-Campus (Local)"
    source_badge: str = "On-Campus · JKLU"
    origin: str = "tpo_smart_jd"                             # tpo_smart_jd | recruiter_jnf
    eligibility: dict = Field(default_factory=dict, sa_column=Column(JSON))
    required_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    preferred_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    unmapped_skill_mentions: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    raw_jd_text: str = Field(default="", sa_column=Column(Text))
    parser_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    cohort_summary: dict = Field(default_factory=dict, sa_column=Column(JSON))
    approval_status: str = Field(default="DRAFT", index=True)  # DRAFT | PENDING_TPO_APPROVAL | PUBLISHED | REJECTED
    approval_id: str | None = Field(default=None, index=True)
    job_id: str | None = Field(default=None, index=True)
    created_by: str = ""
    reviewed_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

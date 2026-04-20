from __future__ import annotations

from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    name: str = "Unknown Candidate"
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    years_experience: float = 0.0
    highlights: list[str] = Field(default_factory=list)
    raw_text_preview: str = ""


class JobDescriptionProfile(BaseModel):
    title: str = "Target Role"
    required_skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    education_requirements: list[str] = Field(default_factory=list)
    minimum_years_experience: float = 0.0
    raw_text_preview: str = ""


class ScoreBreakdown(BaseModel):
    semantic_similarity: float
    skill_match: float
    experience_match: float
    education_match: float
    overall_score: float


class AIFeedback(BaseModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    improvement_tips: list[str] = Field(default_factory=list)


class ResumeScreeningResult(BaseModel):
    candidate_id: str
    filename: str
    candidate_name: str
    extracted_text_length: int
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown
    resume_profile: ResumeProfile
    ai_feedback: AIFeedback | None = None
    ranking_position: int | None = None


class BatchScreeningResponse(BaseModel):
    processed_count: int
    shortlisted_count: int
    top_n: int
    skipped_files: list[str] = Field(default_factory=list)
    ranked_candidates: list[ResumeScreeningResult] = Field(default_factory=list)
    shortlisted_candidates: list[ResumeScreeningResult] = Field(default_factory=list)

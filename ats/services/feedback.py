from __future__ import annotations

from ats.config import Settings
from ats.schemas import AIFeedback, JobDescriptionProfile, ResumeScreeningResult
from ats.utils import extract_json_object

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


SYSTEM_PROMPT = """
You are an expert ATS resume coach.
Return valid JSON with exactly these keys:
summary, strengths, gaps, improvement_tips
Each list must contain concise, practical bullet-style strings.
""".strip()


class FeedbackService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate_feedback(
        self,
        result: ResumeScreeningResult,
        jd_profile: JobDescriptionProfile,
    ) -> AIFeedback:
        prompt = self._build_prompt(result, jd_profile)

        try:
            if self.settings.llm_provider == "openai":
                return self._generate_openai_feedback(prompt)
            if self.settings.llm_provider == "gemini":
                return self._generate_gemini_feedback(prompt)
        except Exception:
            pass

        return self._heuristic_feedback(result, jd_profile)

    def _generate_openai_feedback(self, prompt: str) -> AIFeedback:
        if OpenAI is None or not self.settings.openai_api_key:
            raise RuntimeError("OpenAI client is unavailable or API key is missing")

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.responses.create(
            model=self.settings.openai_model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            temperature=self.settings.ai_temperature,
        )
        return self._parse_feedback_text(response.output_text)

    def _generate_gemini_feedback(self, prompt: str) -> AIFeedback:
        if genai is None or genai_types is None or not self.settings.gemini_api_key:
            raise RuntimeError("Gemini client is unavailable or API key is missing")

        client = genai.Client(api_key=self.settings.gemini_api_key)
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.settings.ai_temperature,
            ),
        )
        return self._parse_feedback_text(response.text or "")

    def _parse_feedback_text(self, text: str) -> AIFeedback:
        payload = extract_json_object(text)
        if payload:
            return AIFeedback(
                summary=str(payload.get("summary", "The resume has been evaluated against the job description.")),
                strengths=[str(item) for item in payload.get("strengths", [])][:5],
                gaps=[str(item) for item in payload.get("gaps", [])][:5],
                improvement_tips=[str(item) for item in payload.get("improvement_tips", [])][:5],
            )
        return AIFeedback(
            summary=text.strip() or "The resume has been evaluated against the job description.",
            strengths=[],
            gaps=[],
            improvement_tips=[],
        )

    def _heuristic_feedback(
        self,
        result: ResumeScreeningResult,
        jd_profile: JobDescriptionProfile,
    ) -> AIFeedback:
        strengths: list[str] = []
        gaps: list[str] = []
        tips: list[str] = []

        if result.matched_skills:
            strengths.append(f"Matched core skills: {', '.join(result.matched_skills[:5])}.")
        if result.score_breakdown.semantic_similarity >= 70:
            strengths.append("The resume language aligns well with the job description.")
        if result.resume_profile.years_experience >= jd_profile.minimum_years_experience > 0:
            strengths.append("The candidate appears to meet the experience requirement.")

        if result.missing_skills:
            gaps.append(f"Missing or unclear skills: {', '.join(result.missing_skills[:5])}.")
            tips.append("Add measurable project bullets that explicitly demonstrate the missing skills.")
        if jd_profile.minimum_years_experience > result.resume_profile.years_experience:
            gaps.append("The resume does not clearly show enough years of experience for the role.")
            tips.append("Clarify total years of experience and add dates for each relevant position.")
        if jd_profile.education_requirements and not set(result.resume_profile.education).intersection(jd_profile.education_requirements):
            gaps.append("The education requirement is not clearly reflected in the resume.")
            tips.append("Move degree details closer to the top or expand the education section.")

        if not tips:
            tips.append("Tailor the summary and top project bullets to mirror the job description keywords.")
            tips.append("Quantify impact with metrics such as revenue, latency, users, or model performance.")

        summary = (
            f"{result.candidate_name} scored {result.score_breakdown.overall_score:.2f}%. "
            "The resume shows a solid baseline match, with the strongest gains likely coming from tighter keyword alignment."
        )

        return AIFeedback(
            summary=summary,
            strengths=strengths[:5],
            gaps=gaps[:5],
            improvement_tips=tips[:5],
        )

    def _build_prompt(self, result: ResumeScreeningResult, jd_profile: JobDescriptionProfile) -> str:
        return f"""
Evaluate this resume screening result and provide actionable ATS improvement feedback.

Candidate: {result.candidate_name}
Filename: {result.filename}
Overall Score: {result.score_breakdown.overall_score}
Semantic Similarity: {result.score_breakdown.semantic_similarity}
Skill Match: {result.score_breakdown.skill_match}
Experience Match: {result.score_breakdown.experience_match}
Education Match: {result.score_breakdown.education_match}
Matched Skills: {", ".join(result.matched_skills) or "None"}
Missing Skills: {", ".join(result.missing_skills) or "None"}
Resume Skills: {", ".join(result.resume_profile.skills) or "None"}
Resume Experience: {result.resume_profile.years_experience} years
Resume Education: {", ".join(result.resume_profile.education) or "None"}
Resume Highlights: {" | ".join(result.resume_profile.highlights) or "None"}
Job Title: {jd_profile.title}
Job Required Skills: {", ".join(jd_profile.required_skills) or "None"}
Job Experience Requirement: {jd_profile.minimum_years_experience} years
Job Education Requirement: {", ".join(jd_profile.education_requirements) or "None"}

Return concise ATS-focused feedback in JSON only.
""".strip()

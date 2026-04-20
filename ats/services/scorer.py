from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ats.config import Settings
from ats.schemas import JobDescriptionProfile, ResumeProfile, ScoreBreakdown
from ats.utils import clamp_score

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None
    util = None


class ResumeScorer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedding_model = None

    def score_resume(
        self,
        resume_text: str,
        jd_text: str,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
    ) -> tuple[ScoreBreakdown, list[str], list[str]]:
        semantic_score = self._semantic_similarity(resume_text, jd_text)
        matched_skills = sorted(set(resume_profile.skills).intersection(jd_profile.required_skills))
        missing_skills = sorted(set(jd_profile.required_skills).difference(resume_profile.skills))
        skill_score = 100.0 if not jd_profile.required_skills else (len(matched_skills) / len(jd_profile.required_skills)) * 100

        if jd_profile.minimum_years_experience <= 0:
            experience_score = 100.0
        else:
            experience_ratio = resume_profile.years_experience / jd_profile.minimum_years_experience
            experience_score = min(experience_ratio, 1.0) * 100

        if not jd_profile.education_requirements:
            education_score = 100.0
        elif set(resume_profile.education).intersection(jd_profile.education_requirements):
            education_score = 100.0
        else:
            education_score = 40.0

        weights = self.settings.scoring_weights
        overall_score = (
            semantic_score * weights["semantic"]
            + skill_score * weights["skill"]
            + experience_score * weights["experience"]
            + education_score * weights["education"]
        )

        breakdown = ScoreBreakdown(
            semantic_similarity=clamp_score(semantic_score),
            skill_match=clamp_score(skill_score),
            experience_match=clamp_score(experience_score),
            education_match=clamp_score(education_score),
            overall_score=clamp_score(overall_score),
        )
        return breakdown, matched_skills, missing_skills

    def _semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        model = self._load_embedding_model()
        if model is not None and util is not None:
            embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True, normalize_embeddings=True)
            score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
            return max(0.0, score) * 100

        matrix = TfidfVectorizer(stop_words="english").fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(similarity) * 100

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model

        if SentenceTransformer is None:
            return None

        try:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        except Exception:
            self._embedding_model = None
        return self._embedding_model

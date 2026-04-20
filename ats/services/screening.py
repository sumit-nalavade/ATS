from __future__ import annotations

import io
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ats.config import Settings, get_settings
from ats.constants import SUPPORTED_EXTENSIONS
from ats.schemas import BatchScreeningResponse, JobDescriptionProfile, ResumeScreeningResult
from ats.services.extractor import ResumeTextExtractor
from ats.services.feedback import FeedbackService
from ats.services.parser import ResumeParserService
from ats.services.scorer import ResumeScorer


class ScreeningService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.extractor = ResumeTextExtractor()
        self.parser = ResumeParserService(self.settings)
        self.scorer = ResumeScorer(self.settings)
        self.feedback = FeedbackService(self.settings)

    def screen_single_resume(
        self,
        filename: str,
        file_bytes: bytes,
        job_description: str,
        generate_feedback: bool = True,
    ) -> ResumeScreeningResult:
        self._validate_file_size(file_bytes)
        jd_profile = self.parser.parse_job_description(job_description)
        resume_text = self.extractor.extract_text(filename, file_bytes)
        return self._screen_text(
            filename=filename,
            resume_text=resume_text,
            jd_text=job_description,
            jd_profile=jd_profile,
            generate_feedback=generate_feedback,
        )

    def screen_batch_zip(
        self,
        zip_bytes: bytes,
        job_description: str,
        top_n: int,
        generate_feedback: bool = False,
    ) -> BatchScreeningResponse:
        jd_profile = self.parser.parse_job_description(job_description)
        ranked_candidates: list[ResumeScreeningResult] = []
        skipped_files: list[str] = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                filename = Path(member.filename).name
                extension = Path(filename).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    skipped_files.append(f"{filename}: unsupported format")
                    continue

                try:
                    file_bytes = archive.read(member)
                    self._validate_file_size(file_bytes)
                    resume_text = self.extractor.extract_text(filename, file_bytes)
                    result = self._screen_text(
                        filename=filename,
                        resume_text=resume_text,
                        jd_text=job_description,
                        jd_profile=jd_profile,
                        generate_feedback=False,
                    )
                    ranked_candidates.append(result)
                except Exception as exc:
                    skipped_files.append(f"{filename}: {exc}")

        ranked_candidates.sort(key=lambda item: item.score_breakdown.overall_score, reverse=True)
        for index, candidate in enumerate(ranked_candidates, start=1):
            candidate.ranking_position = index

        shortlisted_candidates = ranked_candidates[: max(0, top_n)]

        if generate_feedback:
            for candidate in shortlisted_candidates:
                candidate.ai_feedback = self.feedback.generate_feedback(candidate, jd_profile)

        return BatchScreeningResponse(
            processed_count=len(ranked_candidates),
            shortlisted_count=len(shortlisted_candidates),
            top_n=top_n,
            skipped_files=skipped_files,
            ranked_candidates=ranked_candidates,
            shortlisted_candidates=shortlisted_candidates,
        )

    def results_to_dataframe(self, results: list[ResumeScreeningResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            rows.append(
                {
                    "rank": result.ranking_position,
                    "candidate_name": result.candidate_name,
                    "filename": result.filename,
                    "overall_score": result.score_breakdown.overall_score,
                    "semantic_similarity": result.score_breakdown.semantic_similarity,
                    "skill_match": result.score_breakdown.skill_match,
                    "experience_match": result.score_breakdown.experience_match,
                    "education_match": result.score_breakdown.education_match,
                    "years_experience": result.resume_profile.years_experience,
                    "email": result.resume_profile.email,
                    "matched_skills": ", ".join(result.matched_skills),
                    "missing_skills": ", ".join(result.missing_skills),
                }
            )
        return pd.DataFrame(rows)

    def _screen_text(
        self,
        filename: str,
        resume_text: str,
        jd_text: str,
        jd_profile: JobDescriptionProfile,
        generate_feedback: bool,
    ) -> ResumeScreeningResult:
        resume_profile = self.parser.parse_resume(resume_text)
        score_breakdown, matched_skills, missing_skills = self.scorer.score_resume(
            resume_text=resume_text,
            jd_text=jd_text,
            resume_profile=resume_profile,
            jd_profile=jd_profile,
        )

        result = ResumeScreeningResult(
            candidate_id=uuid4().hex,
            filename=filename,
            candidate_name=resume_profile.name,
            extracted_text_length=len(resume_text),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            score_breakdown=score_breakdown,
            resume_profile=resume_profile,
        )

        if generate_feedback:
            result.ai_feedback = self.feedback.generate_feedback(result, jd_profile)

        return result

    def _validate_file_size(self, file_bytes: bytes) -> None:
        max_size = self.settings.max_resume_mb * 1024 * 1024
        if len(file_bytes) > max_size:
            raise ValueError(f"File exceeds maximum allowed size of {self.settings.max_resume_mb} MB")

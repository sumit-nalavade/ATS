from __future__ import annotations

import re
from collections import Counter

import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ats.config import Settings
from ats.constants import COMMON_SKILLS, DEGREE_KEYWORDS
from ats.schemas import JobDescriptionProfile, ResumeProfile
from ats.utils import keyword_in_text, normalize_whitespace


class ResumeParserService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.nlp = self._load_nlp()

    def parse_resume(self, text: str) -> ResumeProfile:
        doc = self.nlp(text)
        return ResumeProfile(
            name=self._extract_name(doc, text),
            email=self._extract_email(text),
            phone=self._extract_phone(text),
            skills=self._extract_skills(text),
            education=self._extract_education(text),
            years_experience=self._extract_experience_years(text),
            highlights=self._extract_highlights(doc, text),
            raw_text_preview=text[:500],
        )

    def parse_job_description(self, text: str) -> JobDescriptionProfile:
        doc = self.nlp(text)
        required_skills = self._extract_skills(text)
        keywords = self._extract_keywords(doc, text)
        return JobDescriptionProfile(
            title=self._extract_title(text),
            required_skills=required_skills,
            keywords=keywords,
            education_requirements=self._extract_education(text),
            minimum_years_experience=self._extract_experience_years(text),
            raw_text_preview=text[:500],
        )

    def _load_nlp(self):
        try:
            return spacy.load(self.settings.spacy_model)
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    @staticmethod
    def _extract_email(text: str) -> str | None:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else None

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        match = re.search(r"(\+?\d[\d\-\s()]{8,}\d)", text)
        return match.group(0).strip() if match else None

    def _extract_name(self, doc, text: str) -> str:
        for entity in getattr(doc, "ents", []):
            if entity.label_ == "PERSON" and len(entity.text.split()) <= 4:
                return entity.text.strip()

        first_lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in first_lines[:5]:
            if not re.search(r"\d|@|resume|curriculum", line, flags=re.IGNORECASE):
                words = line.split()
                if 1 < len(words) <= 4:
                    return line
        return "Unknown Candidate"

    @staticmethod
    def _extract_skills(text: str) -> list[str]:
        normalized = text.lower()
        matched_skills = [
            skill for skill in sorted(COMMON_SKILLS, key=len, reverse=True) if keyword_in_text(normalized, skill)
        ]
        return sorted(dict.fromkeys(matched_skills))

    @staticmethod
    def _extract_education(text: str) -> list[str]:
        normalized = text.lower()
        matches = [degree for degree in sorted(DEGREE_KEYWORDS) if keyword_in_text(normalized, degree)]
        return sorted(dict.fromkeys(matches))

    @staticmethod
    def _extract_experience_years(text: str) -> float:
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b", text, flags=re.IGNORECASE)
        if not matches:
            return 0.0
        years = [float(match) for match in matches]
        sensible_years = [year for year in years if 0 <= year <= 50]
        return max(sensible_years, default=0.0)

    @staticmethod
    def _extract_title(text: str) -> str:
        first_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return first_lines[0][:120] if first_lines else "Target Role"

    @staticmethod
    def _extract_highlights(doc, text: str) -> list[str]:
        if hasattr(doc, "sents"):
            sentences = [
                normalize_whitespace(sentence.text)
                for sentence in doc.sents
                if 5 <= len(sentence.text.split()) <= 30
            ]
            if sentences:
                return sentences[:3]

        lines = [normalize_whitespace(line) for line in text.splitlines() if len(line.split()) >= 5]
        return lines[:3]

    @staticmethod
    def _extract_keywords(doc, text: str, limit: int = 25) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", text.lower())
        filtered = [
            token
            for token in tokens
            if token not in ENGLISH_STOP_WORDS and len(token) > 2 and not token.isdigit()
        ]
        counter = Counter(filtered)
        keywords = [token for token, _ in counter.most_common(limit)]

        entities = [entity.text.lower() for entity in getattr(doc, "ents", []) if len(entity.text) > 2]
        ordered = keywords + entities + ResumeParserService._extract_skills(text)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in ordered:
            clean = item.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped[:limit]

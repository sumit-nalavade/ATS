from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "ATS Resume Screening System"
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.45"))
    skill_weight: float = float(os.getenv("SKILL_WEIGHT", "0.30"))
    experience_weight: float = float(os.getenv("EXPERIENCE_WEIGHT", "0.15"))
    education_weight: float = float(os.getenv("EDUCATION_WEIGHT", "0.10"))
    top_n_default: int = int(os.getenv("TOP_N_DEFAULT", "5"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    ai_temperature: float = float(os.getenv("AI_TEMPERATURE", "0.2"))
    max_resume_mb: int = int(os.getenv("MAX_RESUME_MB", "10"))

    @property
    def feedback_enabled(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        return False

    @property
    def scoring_weights(self) -> dict[str, float]:
        raw_weights = {
            "semantic": self.semantic_weight,
            "skill": self.skill_weight,
            "experience": self.experience_weight,
            "education": self.education_weight,
        }
        total = sum(raw_weights.values()) or 1.0
        return {key: value / total for key, value in raw_weights.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

import streamlit as st

from ats.config import get_settings
from ats.schemas import AIFeedback, ResumeScreeningResult
from ats.services.screening import ScreeningService

st.set_page_config(page_title="ATS Resume Screening System", page_icon="📄", layout="wide")


@st.cache_resource
def get_screening_service() -> ScreeningService:
    return ScreeningService(get_settings())


def render_feedback(feedback: AIFeedback) -> None:
    st.subheader("AI Feedback")
    st.write(feedback.summary)

    strengths_col, gaps_col = st.columns(2)
    with strengths_col:
        st.markdown("**Strengths**")
        if feedback.strengths:
            for item in feedback.strengths:
                st.write(f"- {item}")
        else:
            st.write("No major strengths captured.")

    with gaps_col:
        st.markdown("**Gaps**")
        if feedback.gaps:
            for item in feedback.gaps:
                st.write(f"- {item}")
        else:
            st.write("No major gaps captured.")

    st.markdown("**Improvement Tips**")
    for item in feedback.improvement_tips:
        st.write(f"- {item}")


def render_single_result(result: ResumeScreeningResult) -> None:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Overall Score", f"{result.score_breakdown.overall_score:.2f}%")
    metric_2.metric("Semantic Match", f"{result.score_breakdown.semantic_similarity:.2f}%")
    metric_3.metric("Skill Match", f"{result.score_breakdown.skill_match:.2f}%")
    metric_4.metric("Experience Match", f"{result.score_breakdown.experience_match:.2f}%")

    st.subheader("Candidate Snapshot")
    st.write(f"**Name:** {result.candidate_name}")
    st.write(f"**Email:** {result.resume_profile.email or 'Not found'}")
    st.write(f"**Phone:** {result.resume_profile.phone or 'Not found'}")
    st.write(f"**Years of Experience:** {result.resume_profile.years_experience}")
    st.write(f"**Education:** {', '.join(result.resume_profile.education) or 'Not detected'}")

    matched_col, missing_col = st.columns(2)
    with matched_col:
        st.markdown("**Matched Skills**")
        if result.matched_skills:
            for skill in result.matched_skills:
                st.write(f"- {skill}")
        else:
            st.write("No explicit matched skills found.")

    with missing_col:
        st.markdown("**Missing Skills**")
        if result.missing_skills:
            for skill in result.missing_skills:
                st.write(f"- {skill}")
        else:
            st.write("No obvious missing skills.")

    st.subheader("Resume Highlights")
    if result.resume_profile.highlights:
        for item in result.resume_profile.highlights:
            st.write(f"- {item}")
    else:
        st.write("No highlights extracted.")

    if result.ai_feedback:
        render_feedback(result.ai_feedback)


def main() -> None:
    settings = get_settings()
    service = get_screening_service()

    st.title(settings.app_name)
    st.caption("Screen a single resume or rank a ZIP of resumes against a job description.")

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**LLM Provider:** `{settings.llm_provider}`")
        if settings.feedback_enabled:
            st.success("AI feedback is enabled.")
        else:
            st.warning("AI key not detected. Heuristic feedback will be used.")
        st.write("Supported resume formats: PDF, DOCX, TXT")

    single_tab, batch_tab = st.tabs(["Single Resume", "Batch Screening"])

    with single_tab:
        single_resume = st.file_uploader(
            "Upload a resume",
            type=["pdf", "docx", "txt"],
            key="single_resume",
        )
        single_jd = st.text_area("Paste the Job Description", height=260, key="single_jd")
        generate_single_feedback = st.checkbox("Generate feedback", value=True, key="single_feedback")

        if st.button("Analyze Resume", type="primary", key="analyze_single"):
            if not single_resume or not single_jd.strip():
                st.error("Upload a resume and paste a job description first.")
            else:
                with st.spinner("Scoring resume..."):
                    result = service.screen_single_resume(
                        filename=single_resume.name,
                        file_bytes=single_resume.getvalue(),
                        job_description=single_jd,
                        generate_feedback=generate_single_feedback,
                    )
                render_single_result(result)

    with batch_tab:
        batch_zip = st.file_uploader("Upload a ZIP of resumes", type=["zip"], key="batch_zip")
        batch_jd = st.text_area("Paste the Job Description", height=260, key="batch_jd")
        top_n = st.slider("Top N shortlisted candidates", min_value=1, max_value=20, value=settings.top_n_default)
        generate_batch_feedback = st.checkbox(
            "Generate AI feedback for shortlisted candidates only",
            value=False,
            key="batch_feedback",
        )

        if st.button("Run Batch Screening", type="primary", key="run_batch"):
            if not batch_zip or not batch_jd.strip():
                st.error("Upload a ZIP file and paste a job description first.")
            else:
                with st.spinner("Ranking resumes..."):
                    result = service.screen_batch_zip(
                        zip_bytes=batch_zip.getvalue(),
                        job_description=batch_jd,
                        top_n=top_n,
                        generate_feedback=generate_batch_feedback,
                    )

                st.subheader("Batch Summary")
                summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
                summary_col_1.metric("Processed", result.processed_count)
                summary_col_2.metric("Shortlisted", result.shortlisted_count)
                summary_col_3.metric("Requested Top N", result.top_n)

                ranked_df = service.results_to_dataframe(result.ranked_candidates)
                if not ranked_df.empty:
                    st.subheader("Ranked Candidates")
                    st.dataframe(ranked_df, use_container_width=True)
                    st.bar_chart(ranked_df.set_index("candidate_name")["overall_score"])
                    st.download_button(
                        "Download ranked CSV",
                        data=ranked_df.to_csv(index=False),
                        file_name="ranked_candidates.csv",
                        mime="text/csv",
                    )

                if result.skipped_files:
                    st.subheader("Skipped Files")
                    for item in result.skipped_files:
                        st.write(f"- {item}")

                st.subheader("Shortlisted Candidates")
                for candidate in result.shortlisted_candidates:
                    label = (
                        f"#{candidate.ranking_position} - {candidate.candidate_name} "
                        f"({candidate.score_breakdown.overall_score:.2f}%)"
                    )
                    with st.expander(label, expanded=False):
                        render_single_result(candidate)


if __name__ == "__main__":
    main()

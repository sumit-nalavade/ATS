from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ats.config import get_settings
from ats.services.screening import ScreeningService

settings = get_settings()
service = ScreeningService(settings)

app = FastAPI(title=settings.app_name, version="1.0.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/screen/single")
async def screen_single_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    generate_feedback: bool = Form(True),
):
    try:
        file_bytes = await resume.read()
        result = service.screen_single_resume(
            filename=resume.filename or "resume.pdf",
            file_bytes=file_bytes,
            job_description=job_description,
            generate_feedback=generate_feedback,
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/screen/batch")
async def screen_batch_resumes(
    resumes_zip: UploadFile = File(...),
    job_description: str = Form(...),
    top_n: int = Form(5),
    generate_feedback: bool = Form(False),
):
    try:
        zip_bytes = await resumes_zip.read()
        result = service.screen_batch_zip(
            zip_bytes=zip_bytes,
            job_description=job_description,
            top_n=top_n,
            generate_feedback=generate_feedback,
        )
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

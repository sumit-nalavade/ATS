# ATS Resume Screening System

A production-ready end-to-end ATS resume screening system built with Python, Streamlit, FastAPI, spaCy, sentence-transformers, and OpenAI or Gemini for AI feedback.

## Features

- Score a single resume against a job description
- Generate AI feedback and improvement tips
- Process multiple resumes from a ZIP upload
- Rank candidates and return the top N shortlist
- Fall back to TF-IDF scoring when sentence-transformers is unavailable
- Fall back to heuristic feedback when no AI key is configured

## Project Structure

```text
.
|-- app.py
|-- api.py
|-- requirements.txt
|-- .env.example
`-- ats/
    |-- config.py
    |-- constants.py
    |-- schemas.py
    |-- utils.py
    `-- services/
        |-- extractor.py
        |-- parser.py
        |-- scorer.py
        |-- feedback.py
        `-- screening.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the spaCy English pipeline:

```bash
python -m spacy download en_core_web_sm
```

4. Update `.env` with your API keys.

Example:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5.2
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview
```

## Run the Streamlit UI

```bash
streamlit run app.py
```

## Run the FastAPI API

```bash
uvicorn api:app --reload
```

## API Endpoints

- `GET /health`
- `POST /screen/single`
- `POST /screen/batch`

### Single Resume Request

Send `multipart/form-data` with:

- `resume`: file
- `job_description`: string
- `generate_feedback`: boolean

### Batch Request

Send `multipart/form-data` with:

- `resumes_zip`: ZIP file
- `job_description`: string
- `top_n`: integer
- `generate_feedback`: boolean

Batch feedback is generated for shortlisted candidates only to reduce latency and token cost.

## Notes

- Supported resume formats: `.pdf`, `.docx`, `.txt`
- ZIP uploads are processed in memory without extracting files to disk
- If no AI key is configured, the app returns deterministic, rules-based feedback

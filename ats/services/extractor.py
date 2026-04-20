from __future__ import annotations

import io
from pathlib import Path

from ats.utils import normalize_whitespace

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None


class ResumeTextExtractor:
    def extract_text(self, filename: str, file_bytes: bytes) -> str:
        extension = Path(filename).suffix.lower()

        if extension == ".pdf":
            text = self._extract_from_pdf(file_bytes)
        elif extension == ".docx":
            text = self._extract_from_docx(file_bytes)
        elif extension == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        cleaned_text = normalize_whitespace(text)
        if not cleaned_text:
            raise ValueError(f"No extractable text found in {filename}")
        return cleaned_text

    def _extract_from_pdf(self, file_bytes: bytes) -> str:
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages).strip()
                if text:
                    return text
            except Exception:
                pass

        if PdfReader is None:
            raise ValueError("PDF support requires pdfplumber or PyPDF2 to be installed")

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _extract_from_docx(self, file_bytes: bytes) -> str:
        if Document is None:
            raise ValueError("DOCX support requires python-docx to be installed")

        document = Document(io.BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs)

"""
PDF text extraction utility.
Uses pypdf to pull text out of uploaded PDF files.
"""

import io
from pypdf import PdfReader

# Vercel Hobby plan limits request bodies to ~4.5MB.
# We enforce 4MB to stay safely under that limit.
MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract plain text from a PDF byte string.

    Returns a tuple of (text, page_count).
    Raises ValueError if the PDF is empty, encrypted, or unreadable.
    """
    if not file_bytes:
        raise ValueError("Empty file received.")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError("This PDF is encrypted. Please upload an unencrypted PDF.")

    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("This PDF has no pages.")

    text_parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            text_parts.append(page_text.strip())

    text = "\n\n".join(text_parts)

    if not text.strip():
        raise ValueError(
            "No extractable text found. The PDF may be scanned/image-based. "
            "Try an OCR'd or text-based PDF."
        )

    # Cap the total text length to keep embedding costs/latency sane.
    MAX_CHARS = 100_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text, page_count

from pathlib import Path

from .models import StudyDocument


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_document(path: str | Path) -> StudyDocument:
    """Load one source file into a normalized StudyDocument."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type {file_path.suffix!r}. Use: {supported}")

    if file_path.suffix.lower() == ".pdf":
        text = _load_pdf(file_path)
    else:
        text = file_path.read_text(encoding="utf-8")

    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        raise ValueError(f"Source file is empty after parsing: {file_path}")

    return StudyDocument(
        text=normalized,
        source=str(file_path),
        metadata={"filename": file_path.name, "extension": file_path.suffix.lower()},
    )


def load_documents(paths: list[str | Path]) -> list[StudyDocument]:
    """Load many source files."""
    if not paths:
        raise ValueError("At least one source file is required.")
    return [load_document(path) for path in paths]


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install it with: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = " ".join(page_text.split())
        if page_text:
            pages.append(f"[page {page_number}]\n{page_text}")

    return "\n\n".join(pages)

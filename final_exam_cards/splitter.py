from .models import StudyDocument, TextChunk


def split_documents(
    documents: list[StudyDocument],
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
) -> list[TextChunk]:
    """Split source documents into review-sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    raw_chunks: list[tuple[str, str, dict]] = []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " ", ""],
        )
        for document in documents:
            for chunk in splitter.split_text(document.text):
                raw_chunks.append((chunk, document.source, dict(document.metadata)))
    except ImportError:
        for document in documents:
            for chunk in _fallback_split(document.text, chunk_size, chunk_overlap):
                raw_chunks.append((chunk, document.source, dict(document.metadata)))

    return [
        TextChunk(text=text.strip(), source=source, index=index, metadata=metadata)
        for index, (text, source, metadata) in enumerate(raw_chunks, start=1)
        if text.strip()
    ]


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Small local splitter used when LangChain text splitters are unavailable."""
    markdown_chunks = _split_markdown_sections(text, chunk_size)
    if markdown_chunks:
        return markdown_chunks

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            best_cut = max(window.rfind("\n\n"), window.rfind("。"), window.rfind("."))
            if best_cut > chunk_size // 2:
                end = start + best_cut + 1
                window = text[start:end]

        chunks.append(window)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _split_markdown_sections(text: str, chunk_size: int) -> list[str]:
    sections: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    sections = [section for section in sections if section]
    if len(sections) <= 1:
        return []

    chunks: list[str] = []
    for section in sections:
        if len(section) > chunk_size:
            chunks.extend(_split_paragraphs(section, chunk_size))
        else:
            chunks.append(section)
    return chunks


def _split_paragraphs(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in [item.strip() for item in text.split("\n\n") if item.strip()]:
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current)
    return chunks

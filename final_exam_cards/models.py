from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StudyDocument:
    """A source document loaded from course notes, PDFs, or markdown files."""

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    """A chunk of source text prepared for summarization and card generation."""

    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Summary:
    """A compact review summary for one text chunk or chapter."""

    title: str
    bullets: list[str]
    source: str


@dataclass
class Flashcard:
    """An Anki-style question-answer review card."""

    question: str
    answer: str
    tags: list[str]
    source: str


@dataclass
class PracticeQuestion:
    """A practice question generated from course material."""

    question: str
    answer: str
    explanation: str
    difficulty: str
    source: str


@dataclass
class ReviewPack:
    """The final artifact exported for exam-week review."""

    course_name: str
    summaries: list[Summary]
    flashcards: list[Flashcard]
    practice_questions: list[PracticeQuestion]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

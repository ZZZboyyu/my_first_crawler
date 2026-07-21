import json
import os
import re
from abc import ABC, abstractmethod

from .models import Flashcard, PracticeQuestion, Summary, TextChunk


class ReviewGenerator(ABC):
    """Interface for turning course chunks into review artifacts."""

    @abstractmethod
    def summarize(self, chunk: TextChunk) -> Summary:
        raise NotImplementedError

    @abstractmethod
    def make_flashcards(self, chunk: TextChunk, count: int = 3) -> list[Flashcard]:
        raise NotImplementedError

    @abstractmethod
    def make_practice_questions(self, chunk: TextChunk, count: int = 2) -> list[PracticeQuestion]:
        raise NotImplementedError


class HeuristicReviewGenerator(ReviewGenerator):
    """Offline generator for demos, tests, and API-free learning."""

    def summarize(self, chunk: TextChunk) -> Summary:
        sentences = _sentences(chunk.text)
        bullets = sentences[:4] or [chunk.text[:120]]
        title = _title_from_text(chunk.text, fallback=f"Review Chunk {chunk.index}")
        return Summary(title=title, bullets=bullets, source=chunk.source)

    def make_flashcards(self, chunk: TextChunk, count: int = 3) -> list[Flashcard]:
        sentences = _sentences(chunk.text)
        keywords = _keywords(chunk.text)
        cards: list[Flashcard] = []
        for index, keyword in enumerate(keywords[:count], start=1):
            answer = _find_sentence_containing(sentences, keyword) or sentences[0] if sentences else chunk.text[:160]
            cards.append(
                Flashcard(
                    question=f"What should you remember about {keyword}?",
                    answer=answer,
                    tags=[_safe_tag(keyword), f"chunk-{chunk.index}"],
                    source=chunk.source,
                )
            )

        while len(cards) < count:
            number = len(cards) + 1
            cards.append(
                Flashcard(
                    question=f"What is key point {number} in chunk {chunk.index}?",
                    answer=(sentences[number - 1] if number - 1 < len(sentences) else chunk.text[:160]),
                    tags=[f"chunk-{chunk.index}"],
                    source=chunk.source,
                )
            )
        return cards

    def make_practice_questions(self, chunk: TextChunk, count: int = 2) -> list[PracticeQuestion]:
        cards = self.make_flashcards(chunk, count=count)
        questions: list[PracticeQuestion] = []
        for index, card in enumerate(cards, start=1):
            questions.append(
                PracticeQuestion(
                    question=f"Explain briefly: {card.question}",
                    answer=card.answer,
                    explanation="This question is generated from the same source sentence used by the review card.",
                    difficulty="basic" if index == 1 else "medium",
                    source=chunk.source,
                )
            )
        return questions


class LangChainReviewGenerator(ReviewGenerator):
    """LLM-backed generator using LangChain prompt templates and ChatOpenAI."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model, temperature=temperature)
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You create concise Chinese final-exam review notes from course material."),
                (
                    "user",
                    "Read the source text and return JSON with keys title and bullets.\n"
                    "bullets must be 3-5 short Chinese review bullets.\n\nSource:\n{text}",
                ),
            ]
        )
        self.card_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You create Anki-style Chinese study flashcards."),
                (
                    "user",
                    "Return JSON array of {count} cards. Each item has question, answer, tags.\n"
                    "Questions must test understanding, not memorized wording.\n\nSource:\n{text}",
                ),
            ]
        )
        self.practice_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You create final-exam practice questions in Chinese."),
                (
                    "user",
                    "Return JSON array of {count} items. Each item has question, answer, explanation, difficulty.\n"
                    "Use only the source text.\n\nSource:\n{text}",
                ),
            ]
        )

    def summarize(self, chunk: TextChunk) -> Summary:
        content = self._invoke_json(self.summary_prompt, {"text": chunk.text})
        return Summary(
            title=str(content.get("title", f"Review Chunk {chunk.index}")),
            bullets=[str(item) for item in content.get("bullets", [])][:5],
            source=chunk.source,
        )

    def make_flashcards(self, chunk: TextChunk, count: int = 3) -> list[Flashcard]:
        items = self._invoke_json(self.card_prompt, {"text": chunk.text, "count": count})
        return [
            Flashcard(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                tags=[str(tag) for tag in item.get("tags", [])],
                source=chunk.source,
            )
            for item in _as_list(items)[:count]
        ]

    def make_practice_questions(self, chunk: TextChunk, count: int = 2) -> list[PracticeQuestion]:
        items = self._invoke_json(self.practice_prompt, {"text": chunk.text, "count": count})
        return [
            PracticeQuestion(
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                explanation=str(item.get("explanation", "")),
                difficulty=str(item.get("difficulty", "medium")),
                source=chunk.source,
            )
            for item in _as_list(items)[:count]
        ]

    def _invoke_json(self, prompt, variables: dict) -> dict | list:
        message = (prompt | self.llm).invoke(variables)
        raw = getattr(message, "content", str(message))
        return _loads_json(raw)


def build_generator(use_llm: bool = False) -> ReviewGenerator:
    if use_llm:
        return LangChainReviewGenerator()
    return HeuristicReviewGenerator()


def _sentences(text: str) -> list[str]:
    content_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", "\n".join(content_lines))
    return [part.strip(" -\t") for part in parts if len(part.strip()) > 10]


def _keywords(text: str) -> list[str]:
    text_without_headings = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}|[\u4e00-\u9fff]{2,}", text_without_headings)
    stopwords = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "about",
        "course",
        "what",
        "when",
        "where",
        "common",
        "follows",
        "first",
        "review",
        "final",
        "structures",
    }
    result: list[str] = []
    for word in words:
        lowered = word.lower()
        if lowered in stopwords or word in result:
            continue
        result.append(word)
    return result[:12]


def _find_sentence_containing(sentences: list[str], keyword: str) -> str:
    for sentence in sentences:
        if keyword in sentence:
            return sentence
    return ""


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = line.strip("# -\t")
        if 4 <= len(clean) <= 80:
            return clean
    return fallback


def _safe_tag(value: str) -> str:
    return re.sub(r"\W+", "-", value.lower()).strip("-") or "review"


def _loads_json(raw: str) -> dict | list:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {raw[:300]}") from exc


def _as_list(value: dict | list) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return [item for item in value["items"] if isinstance(item, dict)]
    return []

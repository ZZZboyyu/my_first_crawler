from pathlib import Path

from .exporter import export_review_pack
from .generator import ReviewGenerator, build_generator
from .loaders import load_documents
from .models import ReviewPack
from .splitter import split_documents


class ReviewPipeline:
    """End-to-end pipeline for creating an exam-week review pack."""

    def __init__(
        self,
        course_name: str,
        generator: ReviewGenerator | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        flashcards_per_chunk: int = 3,
        questions_per_chunk: int = 2,
    ):
        self.course_name = course_name
        self.generator = generator or build_generator(use_llm=False)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.flashcards_per_chunk = flashcards_per_chunk
        self.questions_per_chunk = questions_per_chunk

    def build(self, source_paths: list[str | Path]) -> ReviewPack:
        documents = load_documents(source_paths)
        chunks = split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks were produced from the source documents.")

        summaries = []
        flashcards = []
        practice_questions = []

        for chunk in chunks:
            summaries.append(self.generator.summarize(chunk))
            flashcards.extend(self.generator.make_flashcards(chunk, count=self.flashcards_per_chunk))
            practice_questions.extend(
                self.generator.make_practice_questions(chunk, count=self.questions_per_chunk)
            )

        return ReviewPack(
            course_name=self.course_name,
            summaries=summaries,
            flashcards=flashcards,
            practice_questions=practice_questions,
        )

    def build_and_export(self, source_paths: list[str | Path], output_dir: str | Path) -> dict[str, Path]:
        pack = self.build(source_paths)
        return export_review_pack(pack, output_dir)

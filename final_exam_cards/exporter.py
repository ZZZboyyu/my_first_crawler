import csv
import json
from pathlib import Path

from .models import ReviewPack


def export_review_pack(pack: ReviewPack, output_dir: str | Path) -> dict[str, Path]:
    """Export a review pack as JSON, Markdown, and Anki-friendly CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "review_pack.json"
    markdown_path = output_path / "review_pack.md"
    csv_path = output_path / "flashcards.csv"

    json_path.write_text(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_to_markdown(pack), encoding="utf-8")
    _write_flashcard_csv(pack, csv_path)

    return {"json": json_path, "markdown": markdown_path, "csv": csv_path}


def _to_markdown(pack: ReviewPack) -> str:
    lines = [f"# {pack.course_name} Final Exam Review Pack", ""]

    lines.extend(["## Summaries", ""])
    for summary in pack.summaries:
        lines.append(f"### {summary.title}")
        for bullet in summary.bullets:
            lines.append(f"- {bullet}")
        lines.append(f"_Source: {summary.source}_")
        lines.append("")

    lines.extend(["## Flashcards", ""])
    for index, card in enumerate(pack.flashcards, start=1):
        lines.append(f"### Card {index}")
        lines.append(f"**Q:** {card.question}")
        lines.append("")
        lines.append(f"**A:** {card.answer}")
        lines.append("")
        lines.append(f"Tags: {', '.join(card.tags)}")
        lines.append("")

    lines.extend(["## Practice Questions", ""])
    for index, question in enumerate(pack.practice_questions, start=1):
        lines.append(f"### Question {index} ({question.difficulty})")
        lines.append(question.question)
        lines.append("")
        lines.append(f"**Answer:** {question.answer}")
        lines.append("")
        lines.append(f"**Explanation:** {question.explanation}")
        lines.append("")

    return "\n".join(lines)


def _write_flashcard_csv(pack: ReviewPack, csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "tags", "source"])
        for card in pack.flashcards:
            writer.writerow([card.question, card.answer, " ".join(card.tags), card.source])

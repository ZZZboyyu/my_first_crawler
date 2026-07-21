from pathlib import Path

from final_exam_cards import ReviewPipeline


ROOT = Path(__file__).resolve().parent


def main() -> None:
    pipeline = ReviewPipeline(
        course_name="Data Structures",
        chunk_size=700,
        chunk_overlap=80,
        flashcards_per_chunk=3,
        questions_per_chunk=2,
    )
    outputs = pipeline.build_and_export(
        source_paths=[ROOT / "sample_notes.md"],
        output_dir=ROOT / "generated",
    )
    print("Demo finished. Generated files:")
    for kind, path in outputs.items():
        print(f"- {kind}: {path}")


if __name__ == "__main__":
    main()

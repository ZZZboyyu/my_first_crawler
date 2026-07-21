import argparse
from pathlib import Path

from .generator import build_generator
from .pipeline import ReviewPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final-exam review cards from course files.")
    parser.add_argument("sources", nargs="+", help="Source files: .md, .txt, or .pdf")
    parser.add_argument("--course", default="Course", help="Course name shown in exported files")
    parser.add_argument("--out", default="generated", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    parser.add_argument("--cards", type=int, default=3, help="Flashcards per chunk")
    parser.add_argument("--questions", type=int, default=2, help="Practice questions per chunk")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LangChain + ChatOpenAI. Requires OPENAI_API_KEY or compatible env config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = ReviewPipeline(
        course_name=args.course,
        generator=build_generator(use_llm=args.llm),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        flashcards_per_chunk=args.cards,
        questions_per_chunk=args.questions,
    )
    outputs = pipeline.build_and_export(args.sources, Path(args.out))
    print("Generated review pack:")
    for kind, path in outputs.items():
        print(f"- {kind}: {path}")


if __name__ == "__main__":
    main()

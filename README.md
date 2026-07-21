# Final Exam Card Generator

This is a small LangChain-style project for turning course files into an exam-week review pack.

It supports:

- chapter summaries
- Anki-style flashcards
- practice questions
- JSON, Markdown, and CSV export
- offline demo mode
- optional LangChain + ChatOpenAI mode

## Quick Start

Run the offline demo:

```powershell
cd C:\Users\63186\Documents\Codex\2026-07-21\advanced-rag-agent-team-pytest-100\outputs\final_exam_card_generator
python run_demo.py
```

Generated files appear in:

```text
generated/
  review_pack.json
  review_pack.md
  flashcards.csv
```

Use your own files:

```powershell
python -m final_exam_cards.cli notes.md slides.txt --course "Python" --out generated
```

Use real LLM generation:

```powershell
$env:OPENAI_API_KEY="your_api_key"
python -m final_exam_cards.cli notes.md --course "Python" --out generated --llm
```

## Project Structure

```text
final_exam_cards/
  models.py      # data structures: Summary, Flashcard, PracticeQuestion, ReviewPack
  loaders.py     # loads .md, .txt, .pdf files
  splitter.py    # splits source text into chunks
  generator.py   # offline generator and LangChain LLM generator
  exporter.py    # writes JSON, Markdown, and CSV
  pipeline.py    # connects the whole workflow
  cli.py         # command line entry point
run_demo.py      # one-command demo
sample_notes.md  # sample course material
```

## How It Maps To LangChain

### 1. Document Loading

`loaders.py` turns course files into `StudyDocument` objects.

In a full LangChain project, this maps to document loaders such as `TextLoader`, `PyPDFLoader`, or other loaders from `langchain_community`.

### 2. Text Splitting

`splitter.py` tries to use:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

If that package is missing, it uses a small built-in fallback splitter.

Text splitting matters because long course files must be broken into chunks before they are sent to an LLM.

### 3. Prompting

`generator.py` has `LangChainReviewGenerator`, which uses:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
```

There are three prompt chains:

- summary chain
- flashcard chain
- practice-question chain

Each one asks the model to return JSON so the program can export stable structured data.

### 4. Pipeline

`pipeline.py` is the orchestration layer:

```text
load files -> split chunks -> generate summaries/cards/questions -> export files
```

This is the same mental model you will reuse in RAG and Agent projects.

## Why There Is Offline Mode

The offline generator lets you learn the project structure without needing an API key.

It is not as smart as an LLM, but it proves that:

- files can be loaded
- chunks can be produced
- artifacts can be generated
- exports work

After the pipeline works, switching to `--llm` upgrades only the generation layer.

## Suggested Next Upgrades

1. Add a vector database and let users ask questions about their review pack.
2. Add a wrong-answer log and regenerate cards from weak knowledge points.
3. Add OCR support for screenshots.
4. Export to real Anki `.apkg`.
5. Add a Streamlit web UI.

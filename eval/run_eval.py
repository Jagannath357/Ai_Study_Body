from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from study_buddy.config import load_settings
from study_buddy.qa import answer_question
from study_buddy.retriever import retrieve_chunks


def load_questions(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run basic retrieval and answer eval.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--questions", default="eval/questions.jsonl")
    args = parser.parse_args()

    settings = load_settings(data_dir=Path(args.data_dir), db_dir=Path(args.db_dir))
    question_path = Path(args.questions)
    if not question_path.exists():
        print(f"Questions file not found: {question_path}")
        return 1

    rows = load_questions(question_path)
    if not rows:
        print("No evaluation rows found.")
        return 1

    retrieval_hits = 0
    answer_success = 0
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        expected_sources = set(row.get("expected_sources", []))
        if not question:
            print(f"[{idx}] skipped: empty question")
            continue

        chunks = retrieve_chunks(question=question, settings=settings, k=args.k)
        hit = False
        if expected_sources:
            retrieved_sources = {c.source_file for c in chunks}
            hit = bool(retrieved_sources.intersection(expected_sources))
        else:
            hit = bool(chunks)
        if hit:
            retrieval_hits += 1

        try:
            result = answer_question(question=question, settings=settings, k=args.k)
            if result.answer.strip():
                answer_success += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}] answer generation failed: {exc}")

        print(
            f"[{idx}] hit={hit} retrieved={len(chunks)} "
            f"question={question[:70]}"
        )

    print("")
    print(f"retrieval_hits={retrieval_hits}/{total}")
    print(f"answer_success={answer_success}/{total}")
    if retrieval_hits == total and answer_success == total:
        print("Evaluation passed")
        return 0
    print("Evaluation failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

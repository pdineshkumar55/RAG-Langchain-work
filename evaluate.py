"""
Measure how well each retrieval setup finds the right pages.

Build the index in the app first; this script reads the
saved index from vector_store/.

Usage:
    python evaluate.py
    python evaluate.py --questions my_questions.json --top-n 5

The questions file is a JSON list like:
    [
        {
            "question": "What is the refund period?",
            "expected_source": "policy.pdf",
            "expected_page": 4
        }
    ]

expected_page is optional; without it, any chunk from the
expected file counts as a hit.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

import rag_core


# (name, use_hybrid, use_reranker)
CONFIGURATIONS = [
    ("Vector only", False, False),
    ("Hybrid (BM25 + vector)", True, False),
    ("Vector + re-rank", False, True),
    ("Hybrid + re-rank", True, True),
]


def is_match(document, case):

    if document.metadata.get("source") != case["expected_source"]:
        return False

    expected_page = case.get("expected_page")

    return (
        expected_page is None
        or document.metadata.get("page") == expected_page
    )


def main():

    parser = argparse.ArgumentParser(
        description="Compare retrieval setups on a set of test questions."
    )

    parser.add_argument(
        "--questions",
        default="eval_questions.json",
        help="Path to the questions JSON file."
    )

    parser.add_argument("--top-n", type=int, default=5)

    parser.add_argument("--fetch-k", type=int, default=20)

    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        sys.exit("OPENAI_API_KEY is missing. Add it to your .env file.")

    if not os.path.exists(args.questions):
        sys.exit(
            f"{args.questions} not found. Copy "
            "eval_questions.example.json to eval_questions.json "
            "and fill in questions about your own PDFs."
        )

    with open(args.questions, encoding="utf-8") as file:
        cases = json.load(file)

    saved_index = rag_core.load_index(
        rag_core.get_embeddings(api_key)
    )

    if not saved_index:
        sys.exit("No saved index found. Build one in the app first.")

    vectorstore, _ = saved_index

    bm25_retriever = rag_core.build_bm25(
        rag_core.stored_chunks(vectorstore)
    )

    print(
        f"\n{len(cases)} questions, top {args.top_n} chunks, "
        f"{args.fetch_k} candidates per retriever\n"
    )

    print(f"{'Setup':<26}{'Hit rate':>10}{'MRR':>8}")
    print("-" * 44)

    for name, use_hybrid, use_reranker in CONFIGURATIONS:

        hits = 0
        reciprocal_rank_total = 0.0

        for case in cases:

            documents = rag_core.retrieve(
                case["question"],
                vectorstore,
                bm25_retriever,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                fetch_k=args.fetch_k,
                top_n=args.top_n
            )

            for rank, document in enumerate(documents, start=1):

                if is_match(document, case):

                    hits += 1
                    reciprocal_rank_total += 1.0 / rank

                    break

        print(
            f"{name:<26}"
            f"{hits / len(cases):>10.0%}"
            f"{reciprocal_rank_total / len(cases):>8.3f}"
        )

    print(
        "\nHit rate: share of questions where the right page was retrieved."
        "\nMRR: higher means the right page was ranked nearer the top.\n"
    )


if __name__ == "__main__":
    main()

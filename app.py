"""
CLI chat entry point for the 3GPP RAG chatbot.

Usage:
    python app.py                     # interactive chat loop
    python app.py --q "your question" # single-shot question
"""

import argparse
import json
import sys

from src.pipeline import RAGPipeline


def print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    print("-" * 70)
    print(f"A: {result['answer']}")
    print("-" * 70)
    print(f"Status: {result['status']}")
    if result.get("ts_filter_detected"):
        print(f"TS filter detected in query: {result['ts_filter_detected']}")
    if result["citations"]:
        print("Citations:")
        for c in result["citations"]:
            print(f"  - {c['ts_number']} ({c['release']}), clause {c['clause']}, "
                  f"page {c['page']}  [chunk_id={c['chunk_id']}]")
    else:
        print(f"Gate reason: {result.get('gate_reason')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", help="single question, non-interactive mode")
    parser.add_argument("--json", action="store_true", help="print raw JSON result")
    args = parser.parse_args()

    print("[app] loading indexes...")
    pipeline = RAGPipeline()
    print("[app] ready.")

    if args.q:
        result = pipeline.answer(args.q)
        print(json.dumps(result, indent=2)) if args.json else print_result(result)
        return

    print("3GPP RAG Chatbot -- type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        result = pipeline.answer(question)
        print(json.dumps(result, indent=2)) if args.json else print_result(result)


if __name__ == "__main__":
    main()

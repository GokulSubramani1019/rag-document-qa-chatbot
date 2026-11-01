"""Chat with your documents in the terminal.

Examples:
  python app.py
  python app.py --category hr      # only search documents in data/docs/hr/
"""
import argparse
import sys

from rag import config
from rag.qa import answer_question, get_llm
from rag.store import get_embeddings, load_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Document Q&A chatbot")
    parser.add_argument("--index", default=config.DEFAULT_INDEX_DIR)
    parser.add_argument("--category", help="limit search to one document category (sub-folder)")
    parser.add_argument("-k", type=int, default=4, help="number of chunks to retrieve")
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    store = load_index(args.index, get_embeddings())
    llm = get_llm()
    filters = {"category": args.category} if args.category else None

    print("Ask a question about your documents (type 'exit' to quit).")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = answer_question(llm, store, question, filters=filters, k=args.k)
        print(f"\nBot: {result.text}")
        if result.sources:
            unique = sorted({(s["source"], s["page"]) for s in result.sources})
            print("Sources: " + ", ".join(f"{name} p.{page}" for name, page in unique))
    return 0


if __name__ == "__main__":
    sys.exit(main())

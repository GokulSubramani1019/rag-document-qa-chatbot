"""Build a FAISS index from PDFs.

Examples:
  python ingest.py                                   # semantic chunking -> index/
  python ingest.py --chunking fixed --index index_baseline   # baseline for evaluation
"""
import argparse
import sys

from rag import config
from rag.chunking import fixed_chunk_documents, semantic_chunk_documents
from rag.loader import load_pdfs
from rag.store import build_index, get_embeddings, make_embed_fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the document index.")
    parser.add_argument("--docs", default=config.DEFAULT_DOCS_DIR, help="folder with PDFs")
    parser.add_argument("--index", default=config.DEFAULT_INDEX_DIR, help="where to save the index")
    parser.add_argument("--chunking", choices=["semantic", "fixed"], default="semantic")
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    pages = load_pdfs(args.docs)
    if not pages:
        print(f"No readable PDF pages found in {args.docs}")
        return 1
    print(f"Loaded {len(pages)} pages")

    embeddings = get_embeddings()
    if args.chunking == "semantic":
        chunks = semantic_chunk_documents(pages, make_embed_fn(embeddings))
    else:
        chunks = fixed_chunk_documents(pages)
    print(f"Created {len(chunks)} chunks using {args.chunking} chunking")

    build_index(chunks, args.index, embeddings)
    print(f"Index saved to {args.index}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

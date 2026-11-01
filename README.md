# RAG-Based Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions over private PDF documents. It uses **LangChain**, a **FAISS** vector store and **OpenAI GPT-3.5**, with **semantic chunking** and **metadata filtering** to improve retrieval, and a strict grounded prompt to reduce hallucination. An evaluation script measures the hallucination rate against a baseline.

## Features

- **PDF ingestion:** reads every PDF in `data/docs/`. The sub-folder name becomes the document `category`, and the file name and page number are stored as metadata.
- **Semantic chunking:** text is split into sentences, embedded, and cut where the meaning changes, so each chunk covers one topic.
- **Metadata filtering:** restrict retrieval to a category, for example `--category hr`.
- **Grounded answers with citations:** the prompt allows only the retrieved context, requires citations like `[leave_policy.pdf p.1]`, and tells the model to say "I don't know" when the answer is missing.
- **Evaluation:** compares baseline vs improved setup and reports the hallucination rate using an LLM judge.

## Tech Stack

Python, LangChain, FAISS, OpenAI API (GPT-3.5 for answers, `text-embedding-3-small` for embeddings), pypdf, pytest, GitHub Actions

## How It Works

```
PDFs in data/docs/<category>/
   |  load_pdfs            (pypdf: text + source, page, category)
   v
Semantic chunking          (embed sentences -> split where meaning changes)
   |
   v
Embeddings -> FAISS index  (saved to index/)

Question
   |  retrieve top-k chunks (optional metadata filter)
   v
Strict prompt + context -> GPT-3.5 -> answer with citations / "I don't know"
```

## Project Structure

```
rag/
  loader.py     PDF loading + metadata
  chunking.py   semantic chunking and the fixed-size baseline
  store.py      FAISS build/load helpers
  qa.py         retrieval, prompts, answer generation
ingest.py       build an index
app.py          chat in the terminal
eval/           evaluate.py + questions.json (baseline vs improved)
scripts/        make_sample_pdfs.py (creates 2 fictional sample PDFs)
data/docs/      sample PDFs (hr/, finance/): replace with your own
tests/          10 tests that run offline with fake embeddings and a fake LLM
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your OPENAI_API_KEY

python ingest.py                # build the index from data/docs
python app.py                   # chat
python app.py --category hr     # search only HR documents
```

Sample questions to try: "How many days of annual leave do I get?", "What is the hotel limit in metro cities?", "What is the stock option vesting schedule?" (the last one is not in the documents, so the bot should say it doesn't know).

To use your own documents, put PDFs in `data/docs/<category>/` and run `python ingest.py` again. Scanned PDFs without a text layer need OCR first.

## Measure the Hallucination Rate

```bash
python ingest.py --chunking semantic --index index
python ingest.py --chunking fixed --index index_baseline
python -m eval.evaluate
```

| | Baseline | Improved |
| --- | --- | --- |
| Chunking | fixed-size (800 chars) | semantic |
| Retrieval | no filter | metadata filter by category |
| Prompt | "answer using the context" | strict, cited, can say "I don't know" |

The script asks the judge model whether each answer's claims are supported by the retrieved context, then prints each system's hallucination rate and the relative reduction. Per-question results are saved to `eval/results.json` so you can check them by hand.

**Use your own result.** The number you get depends on your documents, your questions and the model. Run the evaluation on your own PDFs with a larger question set (50 or more, including questions the documents cannot answer), and only quote the figure you actually measure. The 12 sample questions are just a starting point.

## Run the Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design Decisions (good interview talking points)

- **Why semantic chunking?** Fixed-size chunks can cut a topic in half or mix two topics, which hurts retrieval. Splitting where the embedding distance between sentences spikes keeps each chunk on one topic.
- **Why metadata filtering?** It stops the retriever from pulling a similar-sounding chunk from the wrong document type.
- **Why a strict prompt?** Most hallucinations happen when the model answers without evidence. Allowing "I don't know" removes that pressure.
- **Why LLM-as-judge?** It scales better than manual checking, but it is imperfect, so spot-check the saved results.
- **Known limits:** chunking works within a page, so a topic that spans two pages is split. FAISS is in-memory and local. `langchain-community` is being sunset, so FAISS may move to a standalone package later.

## Author

Gokul Subramani, Chennai, India

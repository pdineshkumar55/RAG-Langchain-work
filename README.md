# 📚 Multi-PDF RAG with LangChain

A Streamlit app for chatting with your PDF documents. Upload several PDFs, build a searchable index, and ask questions. Each answer is based only on what the documents say and cites the file and page it came from.

## Features

- **Multiple PDFs.** Ask questions across all uploaded documents at once.
- **Hybrid search.** Combines BM25 keyword search with FAISS vector search, so both exact terms (codes, names, IDs) and differently worded questions find the right passages.
- **Re-ranking.** A local cross-encoder model ([FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)) re-scores the candidate passages and keeps only the most relevant ones for the LLM.
- **Chat with memory.** Follow-up questions such as "what about page 5?" are rewritten into complete questions before searching.
- **Streaming answers** with sources: each answer lists the files and pages used, and you can inspect the retrieved passages with their relevance scores.
- **Saved index.** The index is stored on disk and reloads automatically, so the same PDFs are not embedded (and paid for) twice.
- **OCR for scanned PDFs** (optional). PDFs with no selectable text can be read from their page images.
- **Retrieval evaluation.** A script compares retrieval setups on your own test questions.
- **LangSmith tracing** (optional), for inspecting each LLM call.

## How it works

```
PDFs ─► pages ─► chunks (1000 chars, 150 overlap) ─┬─► FAISS index (OpenAI embeddings)
                                                   └─► BM25 keyword index

Question ─► rewrite using chat history ─► FAISS top 20 ┐
                                       ─► BM25 top 20  ┴─► merge (Reciprocal Rank Fusion)
         ─► re-rank with cross-encoder ─► top 5 chunks ─► gpt-4o-mini ─► streamed answer + sources
```

## Requirements

- Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Optional: a [LangSmith API key](https://smith.langchain.com/) for tracing

## Setup

```bash
git clone https://github.com/pdineshkumar55/RAG-Langchain-work.git
cd RAG-Langchain-work

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Copy the example environment file and add your keys:

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

```env
OPENAI_API_KEY=your-openai-key

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Multi-PDF-RAG
```

To turn tracing off, set `LANGCHAIN_TRACING_V2=false`. `.env` is listed in `.gitignore`. Never commit your API keys.

## Usage

```bash
streamlit run rag_app.py
```

1. Upload one or more PDFs and click **Build Vector Database**.
2. Ask questions in the chat box at the bottom of the page.
3. Expand **View Retrieved Chunks** under an answer to see which passages it was based on.

The first question takes a little longer, because the re-ranker model (about 30–40 MB) is downloaded into `.model_cache/`. After that it loads from disk.

### Sidebar settings

| Setting | Default | What it does |
|---|---|---|
| Hybrid search (BM25 + vector) | On | Adds keyword search to vector search |
| Re-rank results | On | Re-scores candidates with the cross-encoder |
| Candidates per retriever | 20 | How many chunks each retriever fetches before re-ranking |
| Chunks sent to the LLM | 5 | How many chunks are used to write the answer |
| OCR for scanned PDFs | Off | Reads text from page images when a PDF has no selectable text |

The sidebar also has buttons to clear the chat and delete the saved index. Building a new index replaces the old one.

### OCR for scanned PDFs

OCR needs one extra package, which is not installed by default:

```bash
pip install rapidocr-onnxruntime
```

It may not yet have builds for the newest Python versions. If it fails to install, use Python 3.12 or run your PDFs through an OCR tool first.

## Evaluating retrieval

`evaluate.py` measures how often each retrieval setup finds the right page for a set of test questions:

1. Build the index in the app first.
2. Copy `eval_questions.example.json` to `eval_questions.json`.
3. Replace the examples with 10–20 real questions about your PDFs, each with the file (and optionally the page) that contains the answer:

   ```json
   [
       {
           "question": "What is the refund period?",
           "expected_source": "policy.pdf",
           "expected_page": 4
       }
   ]
   ```

4. Run:

   ```bash
   python evaluate.py
   ```

Example output (illustrative numbers; yours depend on your documents and questions):

```
Setup                       Hit rate     MRR
--------------------------------------------
Vector only                      70%   0.512
Hybrid (BM25 + vector)           80%   0.590
Vector + re-rank                 85%   0.733
Hybrid + re-rank                 90%   0.781
```

- **Hit rate** is the share of questions where the right page was among the chunks retrieved.
- **MRR** (mean reciprocal rank) is higher when the right page is ranked nearer the top.

Options: `--questions <file>`, `--top-n 5`, `--fetch-k 20`. The script only queries the embedding model, so it does not call the chat model.

## Project structure

```
├── rag_app.py                    # Streamlit app (interface)
├── rag_core.py                   # Loading, indexing, retrieval, re-ranking, prompts
├── evaluate.py                   # Retrieval evaluation script
├── eval_questions.example.json   # Template for evaluation questions
├── requirements.txt
├── .env.example                  # Template for .env
└── .env                          # Your API keys (not committed)
```

Generated at runtime (not committed): `vector_store/` (saved index) and `.model_cache/` (re-ranker model).

## Configuration

The model names are constants at the top of `rag_core.py`:

| Constant | Default |
|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHAT_MODEL` | `gpt-4o-mini` |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` |

If you change `EMBEDDING_MODEL`, delete the saved index and build it again, because existing vectors will not match the new model.

## Tech stack

[Streamlit](https://streamlit.io/) · [LangChain](https://python.langchain.com/) · [OpenAI](https://platform.openai.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) · [LangSmith](https://smith.langchain.com/)

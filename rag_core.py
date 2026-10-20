"""
Shared RAG logic used by the Streamlit app (rag_app.py)
and the evaluation script (evaluate.py).
"""

import json
import os
import shutil
from functools import lru_cache

from flashrank import Ranker, RerankRequest
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# Configuration
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_DIR = os.path.join(BASE_DIR, "vector_store")

INDEX_INFO_FILE = os.path.join(INDEX_DIR, "index_info.json")

RERANKER_CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")

EMBEDDING_MODEL = "text-embedding-3-small"

CHAT_MODEL = "gpt-4o-mini"

RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"


def get_embeddings(api_key):

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )


def get_llm(api_key):

    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=api_key
    )


# =========================================================
# Loading and splitting PDFs
# =========================================================

def load_pdf(file_path, display_name, use_ocr=False):
    """
    Load a PDF into one Document per page.

    Returns (documents, used_ocr). OCR is only attempted
    when the PDF has no selectable text at all, since it
    is much slower than normal text extraction.
    """

    documents = PyPDFLoader(file_path).load()

    used_ocr = False

    has_text = any(
        document.page_content.strip()
        for document in documents
    )

    if use_ocr and not has_text:

        # Reads text from page images. Requires the
        # rapidocr-onnxruntime package.
        documents = PyPDFLoader(
            file_path,
            extract_images=True
        ).load()

        used_ocr = True

    for document in documents:

        document.metadata["source"] = display_name

        # PyPDFLoader page numbers are zero-based.
        # Convert to 1-based.
        if "page" in document.metadata:
            document.metadata["page"] += 1

    return documents, used_ocr


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Remove completely empty chunks
    return [
        chunk
        for chunk in chunks
        if chunk.page_content
        and chunk.page_content.strip()
    ]


# =========================================================
# Indexes (FAISS + BM25) and persistence
# =========================================================

def create_vectorstore(chunks, embeddings):

    return FAISS.from_documents(chunks, embeddings)


def build_bm25(chunks):

    return BM25Retriever.from_documents(chunks)


def stored_chunks(vectorstore):

    # The FAISS docstore keeps every indexed chunk, so the
    # BM25 index can be rebuilt from it after loading.
    return list(vectorstore.docstore._dict.values())


def save_index(vectorstore, indexed_files, chunk_count):

    vectorstore.save_local(INDEX_DIR)

    with open(INDEX_INFO_FILE, "w", encoding="utf-8") as file:

        json.dump(
            {
                "files": indexed_files,
                "chunk_count": chunk_count
            },
            file,
            indent=2
        )


def load_index(embeddings):
    """Return (vectorstore, info), or None if no index is saved."""

    if not os.path.exists(INDEX_INFO_FILE):
        return None

    # FAISS indexes are stored with pickle. Loading is safe
    # here because the app only loads files it wrote itself.
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    with open(INDEX_INFO_FILE, encoding="utf-8") as file:
        info = json.load(file)

    return vectorstore, info


def delete_index():

    shutil.rmtree(INDEX_DIR, ignore_errors=True)


# =========================================================
# Retrieval (hybrid search + re-ranking)
# =========================================================

@lru_cache(maxsize=1)
def load_reranker():

    # Downloaded once on first use, then loaded
    # from the local cache directory.
    return Ranker(
        model_name=RERANKER_MODEL,
        cache_dir=RERANKER_CACHE_DIR
    )


def document_key(document):

    return (
        document.metadata.get("source"),
        document.metadata.get("page"),
        document.page_content
    )


def reciprocal_rank_fusion(result_lists, k=60):

    # Merge several ranked lists into one. A chunk
    # scores higher the nearer the top it appears in
    # each list, so chunks found by both keyword
    # (BM25) and semantic (FAISS) search rise to the top.
    scores = {}
    documents = {}

    for results in result_lists:

        for rank, document in enumerate(results):

            key = document_key(document)

            documents[key] = document

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (k + rank + 1)
            )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [documents[key] for key in ranked_keys]


def rerank_documents(question, documents, top_n):

    ranker = load_reranker()

    passages = [
        {"id": index, "text": document.page_content}
        for index, document in enumerate(documents)
    ]

    results = ranker.rerank(
        RerankRequest(
            query=question,
            passages=passages
        )
    )

    reranked = []

    for result in results[:top_n]:

        document = documents[result["id"]]

        reranked.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rerank_score": float(result["score"])
                }
            )
        )

    return reranked


def retrieve(
    question,
    vectorstore,
    bm25_retriever,
    use_hybrid=True,
    use_reranker=True,
    fetch_k=20,
    top_n=5
):

    # Fetch a wide candidate pool, then narrow it
    # down. Without re-ranking, fetch only what
    # will be sent to the LLM.
    candidate_k = fetch_k if use_reranker else top_n

    candidates = vectorstore.similarity_search(
        question,
        k=candidate_k
    )

    if use_hybrid and bm25_retriever:

        bm25_retriever.k = candidate_k

        keyword_results = bm25_retriever.invoke(question)

        candidates = reciprocal_rank_fusion(
            [candidates, keyword_results]
        )

    if use_reranker and candidates:
        return rerank_documents(question, candidates, top_n)

    return candidates[:top_n]


# =========================================================
# Conversation: question rewriting and answering
# =========================================================

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the conversation so far and a follow-up question,
rewrite the follow-up as a standalone question that can
be understood without the conversation.

Keep all names, terms and numbers. If the question is
already standalone, return it unchanged.

Return only the question.
"""
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

Answer the user's question ONLY using the
provided document context. Use the conversation
history only to understand what the user means.

If the answer cannot be found in the
provided context, say:

"I couldn't find the answer in the uploaded documents."

Do not make up information.

Be concise and clear.
"""
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            """
Question:

{question}

Document Context:

{context}
"""
        )
    ]
)


def rewrite_question(llm, question, history):
    """
    Turn a follow-up like "what about page 5?" into a
    standalone search query. history is a list of
    (role, content) tuples.
    """

    if not history:
        return question

    chain = REWRITE_PROMPT | llm | StrOutputParser()

    rewritten = chain.invoke(
        {
            "question": question,
            "history": history
        }
    ).strip()

    return rewritten or question


def format_context(documents):

    context_parts = []

    for document in documents:

        context_parts.append(
            f"""
File: {document.metadata.get("source", "Unknown file")}
Page: {document.metadata.get("page", "Unknown")}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)


def stream_answer(llm, question, documents, history):

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    return chain.stream(
        {
            "question": question,
            "context": format_context(documents),
            "history": history
        }
    )

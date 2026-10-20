import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

import rag_core


# =========================================================
# Configuration
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Number of earlier chat messages sent to the LLM
# so follow-up questions have context.
MAX_HISTORY_MESSAGES = 6

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is missing. "
        "Please add it to your .env file."
    )
    st.stop()


# =========================================================
# LangSmith configuration
# =========================================================

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
    "LANGCHAIN_TRACING_V2",
    "true"
)

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT",
    "Multi-PDF-RAG"
)


# =========================================================
# Streamlit configuration
# =========================================================

st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Application")

st.write(
    "Upload multiple PDF documents and ask questions "
    "across all of them."
)


# =========================================================
# Session State
# =========================================================

session_defaults = {
    "vectorstore": None,
    "bm25_retriever": None,
    "indexed_files": [],
    "chunk_count": 0,
    "messages": [],
    "index_checked": False,
}

for key, value in session_defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value

embeddings = rag_core.get_embeddings(OPENAI_API_KEY)


# =========================================================
# Load saved index (once per session)
# =========================================================

if not st.session_state.index_checked:

    st.session_state.index_checked = True

    try:

        saved_index = rag_core.load_index(embeddings)

    except Exception as e:

        saved_index = None

        st.warning(
            f"Could not load the saved index: {e}"
        )

    if saved_index:

        vectorstore, index_info = saved_index

        st.session_state.vectorstore = vectorstore

        st.session_state.bm25_retriever = rag_core.build_bm25(
            rag_core.stored_chunks(vectorstore)
        )

        st.session_state.indexed_files = index_info["files"]

        st.session_state.chunk_count = index_info["chunk_count"]


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Retrieval Settings")

    use_hybrid = st.toggle(
        "Hybrid search (BM25 + vector)",
        value=True,
        help=(
            "Combine keyword search (BM25) with semantic "
            "vector search. Helps with exact terms, codes "
            "and names that embeddings can miss."
        )
    )

    use_reranker = st.toggle(
        "Re-rank results",
        value=True,
        help=(
            "Score each candidate chunk against the question "
            "with a cross-encoder model and keep the best."
        )
    )

    fetch_k = st.slider(
        "Candidates per retriever",
        min_value=5,
        max_value=50,
        value=20,
        help="How many chunks each retriever fetches."
    )

    top_n = st.slider(
        "Chunks sent to the LLM",
        min_value=1,
        max_value=15,
        value=5
    )

    st.divider()

    st.header("📄 Indexing")

    use_ocr = st.toggle(
        "OCR for scanned PDFs",
        value=False,
        help=(
            "If a PDF has no selectable text, read the text "
            "from its page images instead. Slower. Requires "
            "the rapidocr-onnxruntime package."
        )
    )

    st.divider()

    st.header("💾 Saved Index")

    if st.session_state.vectorstore:

        st.caption(
            f"{len(st.session_state.indexed_files)} document(s), "
            f"{st.session_state.chunk_count} chunks. "
            "Reloaded automatically next time."
        )

        if st.button("🗑️ Delete saved index"):

            rag_core.delete_index()

            st.session_state.vectorstore = None
            st.session_state.bm25_retriever = None
            st.session_state.indexed_files = []
            st.session_state.chunk_count = 0
            st.session_state.messages = []

            st.rerun()

    else:

        st.caption(
            "No index yet. Upload PDFs to build one."
        )

    if st.session_state.messages:

        if st.button("🧹 Clear chat"):

            st.session_state.messages = []

            st.rerun()


# =========================================================
# Upload PDFs
# =========================================================

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# Build Vector Database
# =========================================================

if uploaded_files:

    st.info(
        f"{len(uploaded_files)} PDF file(s) selected."
    )

    if st.button(
        "🔨 Build Vector Database",
        type="primary",
        help="Replaces any previously saved index."
    ):

        # All build logs go inside this status box. If a
        # step fails, st.stop() leaves it open in the error
        # state so the logs stay visible. On success the
        # page reruns, which clears the logs.
        with st.status(
            "Building vector database...",
            expanded=True
        ) as build_status:

            all_documents = []

            progress = st.progress(0)

            # ---------------------------------------------
            # Step 1: Load PDFs
            # ---------------------------------------------

            st.write("**1️⃣ Reading PDFs**")

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                temp_file_path = None

                try:

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        temp_file_path = temp_file.name

                    documents, used_ocr = rag_core.load_pdf(
                        temp_file_path,
                        uploaded_file.name,
                        use_ocr=use_ocr
                    )

                    all_documents.extend(documents)

                    st.write(
                        f"✓ {uploaded_file.name}: "
                        f"{len(documents)} page(s)"
                        + (" (OCR)" if used_ocr else "")
                    )

                except Exception as e:

                    st.error(
                        f"Error reading {uploaded_file.name}: "
                        f"{str(e)}"
                    )

                finally:

                    if (
                        temp_file_path
                        and os.path.exists(temp_file_path)
                    ):
                        os.remove(temp_file_path)

                progress.progress(
                    int(
                        ((index + 1) /
                         len(uploaded_files)) * 30
                    )
                )

            # ---------------------------------------------
            # Check extracted documents
            # ---------------------------------------------

            if not all_documents:

                st.error(
                    "❌ No pages were extracted from the "
                    "uploaded PDFs."
                )

                st.stop()

            # ---------------------------------------------
            # Step 2: Split documents
            # ---------------------------------------------

            st.write("**2️⃣ Creating Chunks**")

            chunks = rag_core.split_documents(all_documents)

            if not chunks:

                st.error(
                    "❌ No text chunks were created."
                )

                st.info(
                    "The uploaded PDFs may contain scanned "
                    "images instead of selectable text. "
                    "Turn on **OCR for scanned PDFs** in the "
                    "sidebar and build again."
                )

                st.stop()

            st.write(
                f"✓ {len(all_documents)} pages → "
                f"{len(chunks)} chunks"
            )

            progress.progress(50)

            # ---------------------------------------------
            # Step 3: Create embeddings
            # ---------------------------------------------

            st.write("**3️⃣ Checking Embedding Model**")

            try:

                # Test the embedding model explicitly.
                test_embedding = embeddings.embed_query(
                    "test"
                )

                if not test_embedding:

                    st.error(
                        "❌ OpenAI returned an empty embedding."
                    )

                    st.stop()

                st.write(
                    f"✓ Embedding model working "
                    f"({len(test_embedding)} dimensions)"
                )

            except Exception as e:

                st.error(
                    "❌ Failed to create embeddings."
                )

                st.exception(e)

                st.info(
                    "Check your OPENAI_API_KEY, OpenAI "
                    "account/billing, and internet connection."
                )

                st.stop()

            progress.progress(60)

            # ---------------------------------------------
            # Step 4: Create FAISS Vector Store
            # ---------------------------------------------

            st.write("**4️⃣ Building FAISS Vector Database**")

            try:

                vectorstore = rag_core.create_vectorstore(
                    chunks,
                    embeddings
                )

            except Exception as e:

                st.error(
                    "❌ Failed to create FAISS vector database."
                )

                st.exception(e)

                st.stop()

            progress.progress(85)

            # ---------------------------------------------
            # Step 5: Create BM25 keyword index
            # ---------------------------------------------

            st.write("**5️⃣ Building BM25 Keyword Index**")

            bm25_retriever = rag_core.build_bm25(chunks)

            # ---------------------------------------------
            # Step 6: Save index to disk
            # ---------------------------------------------

            st.write("**6️⃣ Saving Index**")

            indexed_files = [
                file.name
                for file in uploaded_files
            ]

            build_message = (
                f"🎉 Vector database ready: "
                f"{len(indexed_files)} file(s), "
                f"{len(chunks)} chunks."
            )

            try:

                rag_core.save_index(
                    vectorstore,
                    indexed_files,
                    len(chunks)
                )

            except Exception as e:

                build_message += (
                    f" It could not be saved to disk ({e}), "
                    f"so it will only last for this session."
                )

            progress.progress(100)

            build_status.update(
                label="Vector database ready",
                state="complete",
                expanded=False
            )

        # -------------------------------------------------
        # Store in Streamlit session state
        # -------------------------------------------------

        st.session_state.vectorstore = vectorstore

        st.session_state.bm25_retriever = bm25_retriever

        st.session_state.indexed_files = indexed_files

        st.session_state.chunk_count = len(chunks)

        # Old answers refer to the old documents.
        st.session_state.messages = []

        st.session_state.build_message = build_message

        # Rerun so the build logs are cleared before
        # the user starts chatting.
        st.rerun()


# Shown once, on the run right after a successful build.
if "build_message" in st.session_state:

    st.toast(st.session_state.pop("build_message"))


# =========================================================
# Display Indexed Documents
# =========================================================

if st.session_state.vectorstore:

    st.divider()

    with st.expander(
        f"📊 Indexed Documents "
        f"({len(st.session_state.indexed_files)} files, "
        f"{st.session_state.chunk_count} chunks)"
    ):

        for filename in st.session_state.indexed_files:

            st.write(f"📄 {filename}")


# =========================================================
# Chat
# =========================================================

def show_retrieval_details(message):

    if message["search_query"] != message["question"]:

        st.caption(
            f"🔎 Searched for: {message['search_query']}"
        )

    sources = []

    for chunk in message["chunks"]:

        source = (chunk["source"], chunk["page"])

        if source not in sources:
            sources.append(source)

    st.markdown(
        "**Sources:** "
        + " · ".join(
            f"📄 {filename} (p. {page})"
            for filename, page in sources
        )
    )

    with st.expander("🔎 View Retrieved Chunks"):

        for index, chunk in enumerate(
            message["chunks"],
            start=1
        ):

            st.markdown(
                f"**Chunk {index}** — {chunk['source']}, "
                f"page {chunk['page']}"
            )

            if chunk["score"] is not None:

                st.caption(
                    f"Relevance score: {chunk['score']:.3f}"
                )

            st.write(chunk["content"])

            st.divider()


if st.session_state.vectorstore:

    st.subheader("💬 Chat with your documents")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":
                show_retrieval_details(message)

    question = st.chat_input(
        "Ask a question about your documents"
    )

    if question:

        history = [
            (message["role"], message["content"])
            for message in st.session_state.messages[
                -MAX_HISTORY_MESSAGES:
            ]
        ]

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        llm = rag_core.get_llm(OPENAI_API_KEY)

        with st.chat_message("assistant"):

            # ---------------------------------------------
            # Retrieve relevant chunks
            # ---------------------------------------------

            try:

                with st.spinner("Searching documents..."):

                    search_query = rag_core.rewrite_question(
                        llm,
                        question,
                        history
                    )

                    retrieved_documents = rag_core.retrieve(
                        search_query,
                        st.session_state.vectorstore,
                        st.session_state.bm25_retriever,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                        fetch_k=fetch_k,
                        top_n=top_n
                    )

            except Exception as e:

                st.error(
                    "Failed to retrieve documents."
                )

                st.exception(e)

                st.stop()

            if not retrieved_documents:

                st.warning(
                    "No relevant information was found "
                    "in the uploaded documents."
                )

                st.stop()

            # ---------------------------------------------
            # Stream the answer
            # ---------------------------------------------

            try:

                answer = st.write_stream(
                    rag_core.stream_answer(
                        llm,
                        question,
                        retrieved_documents,
                        history
                    )
                )

            except Exception as e:

                st.error(
                    "Failed to generate the answer."
                )

                st.exception(e)

                st.stop()

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "question": question,
                "search_query": search_query,
                "chunks": [
                    {
                        "source": document.metadata.get(
                            "source",
                            "Unknown file"
                        ),
                        "page": document.metadata.get(
                            "page",
                            "Unknown"
                        ),
                        "score": document.metadata.get(
                            "rerank_score"
                        ),
                        "content": document.page_content
                    }
                    for document in retrieved_documents
                ]
            }

            show_retrieval_details(assistant_message)

        st.session_state.messages.append(
            assistant_message
        )

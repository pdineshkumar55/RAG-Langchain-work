import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# Configuration
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

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

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


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
        type="primary"
    ):

        all_documents = []

        progress = st.progress(0)

        # -------------------------------------------------
        # Step 1: Load PDFs
        # -------------------------------------------------

        st.subheader("1️⃣ Reading PDFs")

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            st.write(
                f"Reading: **{uploaded_file.name}**"
            )

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

                # Load PDF
                loader = PyPDFLoader(
                    temp_file_path
                )

                documents = loader.load()

                # Add useful metadata
                for document in documents:

                    document.metadata[
                        "source"
                    ] = uploaded_file.name

                    # PyPDFLoader page numbers are
                    # zero-based. Convert to 1-based.
                    if "page" in document.metadata:

                        document.metadata[
                            "page"
                        ] = (
                            document.metadata["page"] + 1
                        )

                all_documents.extend(documents)

                st.success(
                    f"✓ {uploaded_file.name}: "
                    f"{len(documents)} page(s)"
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

        # -------------------------------------------------
        # Check extracted documents
        # -------------------------------------------------

        if not all_documents:

            st.error(
                "❌ No pages were extracted from the "
                "uploaded PDFs."
            )

            st.info(
                "Please check that your PDFs contain "
                "selectable text. Scanned/image-only PDFs "
                "need OCR before they can be indexed."
            )

            st.stop()

        st.success(
            f"✓ Total pages loaded: "
            f"{len(all_documents)}"
        )

        # -------------------------------------------------
        # Step 2: Split documents
        # -------------------------------------------------

        st.subheader("2️⃣ Creating Chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )

        chunks = text_splitter.split_documents(
            all_documents
        )

        # Remove completely empty chunks
        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content
            and chunk.page_content.strip()
        ]

        if not chunks:

            st.error(
                "❌ No text chunks were created."
            )

            st.info(
                "The uploaded PDFs may contain scanned "
                "images instead of selectable text."
            )

            st.stop()

        st.success(
            f"✓ Created {len(chunks)} chunks"
        )

        progress.progress(50)

        # -------------------------------------------------
        # Step 3: Create embeddings
        # -------------------------------------------------

        st.subheader("3️⃣ Creating Embeddings")

        try:

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            )

            # Test the embedding model explicitly.
            test_embedding = embeddings.embed_query(
                "test"
            )

            if not test_embedding:

                st.error(
                    "❌ OpenAI returned an empty embedding."
                )

                st.stop()

            st.success(
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

        progress.progress(70)

        # -------------------------------------------------
        # Step 4: Create FAISS Vector Store
        # -------------------------------------------------

        st.subheader("4️⃣ Building FAISS Vector Database")

        try:

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

        except Exception as e:

            st.error(
                "❌ Failed to create FAISS vector database."
            )

            st.exception(e)

            st.stop()

        # -------------------------------------------------
        # Store in Streamlit session state
        # -------------------------------------------------

        st.session_state.vectorstore = vectorstore

        st.session_state.indexed_files = [
            file.name
            for file in uploaded_files
        ]

        st.session_state.chunk_count = len(
            chunks
        )

        progress.progress(100)

        st.success(
            f"🎉 Vector database created successfully!"
        )

        st.metric(
            "Chunks Indexed",
            len(chunks)
        )


# =========================================================
# Display Indexed Documents
# =========================================================

if st.session_state.vectorstore:

    st.divider()

    st.subheader("📊 Indexed Documents")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Documents",
            len(st.session_state.indexed_files)
        )

    with col2:

        st.metric(
            "Chunks",
            st.session_state.chunk_count
        )

    st.write("**Files:**")

    for filename in st.session_state.indexed_files:

        st.write(f"📄 {filename}")


# =========================================================
# Question Answering
# =========================================================

if st.session_state.vectorstore:

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is the main topic "
            "of these documents?"
        )
    )

    if st.button("🔍 Ask Question") and question:

        vectorstore = (
            st.session_state.vectorstore
        )

        # -------------------------------------------------
        # Retrieve relevant chunks
        # -------------------------------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
        )

        try:

            retrieved_documents = (
                retriever.invoke(question)
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

        # -------------------------------------------------
        # Create context
        # -------------------------------------------------

        context_parts = []

        for document in retrieved_documents:

            filename = document.metadata.get(
                "source",
                "Unknown file"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
File: {filename}
Page: {page}

Content:
{document.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a document question-answering assistant.

Answer the user's question ONLY using the
provided document context.

If the answer cannot be found in the
provided context, say:

"I couldn't find the answer in the uploaded documents."

Do not make up information.

Be concise and clear.
"""
                ),
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

        # -------------------------------------------------
        # RAG Chain
        # -------------------------------------------------

        chain = prompt | llm

        try:

            response = chain.invoke(
                {
                    "question": question,
                    "context": context
                }
            )

        except Exception as e:

            st.error(
                "Failed to generate the answer."
            )

            st.exception(e)

            st.stop()

        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.subheader("📝 Answer")

        st.write(
            response.content
        )

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        st.subheader("📚 Sources")

        sources = []

        for document in retrieved_documents:

            filename = document.metadata.get(
                "source",
                "Unknown file"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            source = (
                filename,
                page
            )

            if source not in sources:

                sources.append(source)

        for filename, page in sources:

            st.write(
                f"📄 **{filename}** — "
                f"Page **{page}**"
            )

        # -------------------------------------------------
        # Retrieved chunks
        # -------------------------------------------------

        with st.expander(
            "🔎 View Retrieved Chunks"
        ):

            for index, document in enumerate(
                retrieved_documents,
                start=1
            ):

                filename = document.metadata.get(
                    "source",
                    "Unknown file"
                )

                page = document.metadata.get(
                    "page",
                    "Unknown"
                )

                st.markdown(
                    f"### Chunk {index}"
                )

                st.write(
                    f"**File:** {filename}"
                )

                st.write(
                    f"**Page:** {page}"
                )

                st.write(
                    document.page_content
                )

                st.divider()
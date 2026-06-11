import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄"
)

st.title("📄 DocuMind AI")
st.write("Chat with your PDF using RAG")

# =====================================
# API KEY
# =====================================

api_key = st.text_input(
    "Gemini API Key",
    type="password"
)

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# PDF UPLOAD
# =====================================

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# =====================================
# MAIN APP
# =====================================

if uploaded_files:

    # =====================================
    # READ PDFs
    # =====================================

    with st.spinner("Reading PDFs..."):

        text = ""

        for uploaded_file in uploaded_files:

            reader = PdfReader(uploaded_file)

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

    # =====================================
    # CHUNKING
    # =====================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    st.success(
        f"PDF Loaded Successfully! ({len(chunks)} chunks)"
    )

    # =====================================
    # EMBEDDINGS
    # =====================================

    with st.spinner("Creating embeddings..."):

        model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        vectors = model.encode(chunks)

        vectors = np.array(
            vectors
        ).astype("float32")

        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        index.add(vectors)

    # =====================================
    # DISPLAY OLD CHAT
    # =====================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    # =====================================
    # CHAT INPUT
    # =====================================

    question = st.chat_input(
        "Ask your question..."
    )

    if question:

        # =====================================
        # USER MESSAGE
        # =====================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # =====================================
        # RETRIEVAL
        # =====================================

        with st.spinner(
            "Searching document..."
        ):

            query_vector = model.encode(
                [question]
            )

            query_vector = np.array(
                query_vector
            ).astype("float32")

            faiss.normalize_L2(
                query_vector
            )

            scores, indices = index.search(
                query_vector,
                k=7
            )

        # =====================================
        # CONTEXT
        # =====================================

        context = ""
        source_chunks = []

        for idx in indices[0]:

            context += (
                chunks[idx] + "\n\n"
            )

            source_chunks.append(
                chunks[idx]
            )

        # =====================================
        # CHAT HISTORY
        # =====================================

        chat_history = ""

        for msg in st.session_state.messages[-6:]:

            chat_history += (
                f"{msg['role']}: {msg['content']}\n"
            )

        # =====================================
        # PROMPT
        # =====================================

        prompt = f"""
Previous Conversation:
{chat_history}

Context:
{context}

Current Question:
{question}

Answer ONLY from the provided context.

Use previous conversation if needed.

If answer is partially available,
use available information.

Only say "Information not found"
when context contains nothing relevant.
"""

        # =====================================
        # GEMINI
        # =====================================

        try:

            if not api_key:

                st.warning(
                    "Please enter Gemini API Key."
                )

            else:

                with st.spinner(
                    "Generating answer..."
                ):

                    client = genai.Client(
                        api_key=api_key
                    )

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                    answer = response.text

                # =====================================
                # SAVE AI MESSAGE
                # =====================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                # =====================================
                # DISPLAY AI MESSAGE
                # =====================================

                with st.chat_message(
                    "assistant"
                ):
                    st.write(answer)

                # =====================================
                # SOURCES USED
                # =====================================

                with st.expander(
                    "📚 Sources Used"
                ):

                    for i, chunk in enumerate(
                        source_chunks
                    ):

                        st.markdown(
                            f"**Source {i+1}**"
                        )

                        st.write(
                            chunk[:500]
                        )

                        st.divider()

                # =====================================
                # RETRIEVED CHUNKS
                # =====================================

                with st.expander(
                    "Retrieved Chunks"
                ):

                    for rank, idx in enumerate(
                        indices[0]
                    ):

                        st.write(
                            f"Chunk {rank + 1}"
                        )

                        st.write(
                            chunks[idx]
                        )

                        st.divider()

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )
"""
DocuMind AI — RAG engine.

Adapted from the original resume_rag.py script:
  * Now accepts ANY uploaded PDF (not just resume.txt).
  * Uses Pinecone namespaces so each upload is isolated (multi-user safe).
  * Streams the LLM answer token-by-token for a premium chat UX.
  * Pure langchain-core + langchain-google-genai + langchain-groq + langchain-pinecone
    (no langchain.chains / langchain_community), exactly like the original.
  * Supports both Gemini and Groq models.
"""

import os
import uuid
from typing import Any, Generator, Literal

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_NAME = "documind-rag"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIMENSION = 3072
LLM_MODEL_GEMINI = "gemini-2.5-flash"
LLM_MODEL_GROQ = "llama-3.3-70b-versatile"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
INDEX_BATCH = 50  # documents per Pinecone upsert batch

def _get_pinecone(api_key: str) -> Pinecone:
    if not api_key:
        raise ValueError("Pinecone API key is required.")
    return Pinecone(api_key=api_key)


def _get_embeddings(google_api_key: str) -> GoogleGenerativeAIEmbeddings:
    if not google_api_key:
        raise ValueError("Google API key is required.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=google_api_key,
    )


def _get_llm(
    model: Literal["gemini", "groq"] = "gemini",
    *,
    google_api_key: str | None = None,
    groq_api_key: str | None = None,
):
    """Get the appropriate LLM instance based on the model parameter."""
    if model == "groq":
        if not groq_api_key:
            raise ValueError("Groq API key is required for Groq models.")
        return ChatGroq(
            model=LLM_MODEL_GROQ,
            temperature=0.2,
            api_key=groq_api_key,
        )
    else:  # gemini (default)
        if not google_api_key:
            raise ValueError("Google API key is required for Gemini models.")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL_GEMINI,
            temperature=0.2,
            google_api_key=google_api_key,
        )


def ensure_index(pinecone_api_key: str) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    pc = _get_pinecone(pinecone_api_key)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


# ---------------------------------------------------------------------------
# Chunking (manual — no langchain_text_splitters needed, like the original)
# ---------------------------------------------------------------------------
def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split text into overlapping Documents, breaking on sentence boundaries."""
    chunks: list[Document] = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        if end < n:
            # Try to break at a natural boundary inside the chunk window.
            for sep in [". ", "? ", "! ", "\n\n", "\n", " "]:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"chunk_id": len(chunks), "start_char": start},
                )
            )

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def index_document(
    text: str,
    *,
    pinecone_api_key: str,
    google_api_key: str,
) -> str:
    """
    Chunk + embed + upsert the document into a fresh Pinecone namespace.

    Returns the namespace (session_id) the client should use for queries.
    """
    session_id = str(uuid.uuid4())
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    ensure_index(pinecone_api_key)

    splits = split_text(text)
    if not splits:
        raise ValueError("No text chunks could be created from the document.")

    embeddings = _get_embeddings(google_api_key)
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=session_id,
    )

    # Upsert in batches to stay within Pinecone request limits.
    for i in range(0, len(splits), INDEX_BATCH):
        vectorstore.add_documents(splits[i : i + INDEX_BATCH])

    return session_id


# ---------------------------------------------------------------------------
# Querying (streaming)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are DocuMind AI, an expert document analyst. \
Answer the user's question using ONLY the provided context from the uploaded document.

Rules:
- If the answer is in the context, give a clear, well-structured answer. Use Markdown.
- Quote or paraphrase relevant parts of the document when helpful.
- If the answer is NOT in the context, say: "I couldn't find this information in the document."
- Do not invent facts or use outside knowledge.
- Be concise but complete."""


def query_document_stream(
    session_id: str,
    question: str,
    model: Literal["gemini", "groq"] = "gemini",
    *,
    pinecone_api_key: str,
    google_api_key: str | None = None,
    groq_api_key: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Run a manual RAG chain (retrieve -> prompt -> generate) and yield
    Server-Sent-Event payloads:

      {"type": "sources",  "sources": [...]}   # once, before tokens
      {"type": "token",    "content": "..."}   # many times
      {"type": "done"}                        # once, at the end
    """
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    embeddings = _get_embeddings(google_api_key or "")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=session_id,
    )

    # --- Retrieve ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    sources = [
        {
            "chunk_id": doc.metadata.get("chunk_id", i),
            "preview": (
                doc.page_content[:220] + "…"
                if len(doc.page_content) > 220
                else doc.page_content
            ),
        }
        for i, doc in enumerate(docs)
    ]
    yield {"type": "sources", "sources": sources}

    # --- Generate ---
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = _get_llm(
        model,
        google_api_key=google_api_key,
        groq_api_key=groq_api_key,
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM DOCUMENT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    for chunk in llm.stream(prompt):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    yield {"type": "done"}


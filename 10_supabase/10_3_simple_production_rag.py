import os
import shutil
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


load_dotenv()


DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("SUPABASE_DATABASE_URL is missing from .env")


# ---------------------------------------------------------
# 1. Example document
# ---------------------------------------------------------
DOCUMENT = """
# Authentication Guide

## Overview
This guide explains how developers can authenticate with the Example API, manage access tokens, handle rate limits,
process errors, and configure webhooks for real-time event notifications.

The API uses OAuth 2.0 for secure authentication. All requests to protected endpoints must include a valid access token
in the Authorization header.

## OAuth2 Authentication
To authenticate with our API, you need OAuth 2.0 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
If the credentials are valid, the API returns an access token.
Use the access token in future API requests using the Authorization header:
Authorization: Bearer your_access_token.
Access tokens expire after one hour.

## Rate Limiting
The API applies rate limits to protect service stability and prevent abuse.
Each application can make up to 1,000 requests per hour.
If the limit is exceeded, the API returns a 429 Too Many Requests response.
The retry_after value indicates how many seconds your application should wait before sending another request.

## Error Handling
The API uses standard HTTP status codes to indicate request results.
Common status codes include 200 for success, 400 for invalid requests, 401 for missing or invalid authentication,
403 for permission denied, 404 for resource not found, 429 for rate limit exceeded, and 500 for internal server errors.
Error responses usually include an error code, message, and request_id.

## Webhooks
Webhooks allow your application to receive real-time notifications when specific events occur.
To configure a webhook, register a webhook URL in the developer portal.
Your webhook endpoint should return a 200 OK response after successfully processing the event.

## Webhook Security
Each webhook request includes a signature header.
Your application should verify the signature before processing the webhook event.
Recommended webhook security practices include using HTTPS, verifying every webhook signature,
rejecting invalid requests, and storing processed event IDs to prevent duplicate processing.
"""


# ---------------------------------------------------------
# 2. Debug database URL safely
# ---------------------------------------------------------
def debug_database_url(database_url: str):
    parsed = urlparse(database_url)

    print("=" * 80)
    print("Database URL Debug")
    print("=" * 80)
    print(f"Scheme   : {parsed.scheme}")
    print(f"Username : {unquote(parsed.username or '')}")
    print(f"Password : {'FOUND' if parsed.password else 'MISSING'}")
    print(f"Host     : {parsed.hostname}")
    print(f"Port     : {parsed.port}")
    print(f"Database : {parsed.path}")
    print("=" * 80)


# ---------------------------------------------------------
# 3. Create pooled database engine
# ---------------------------------------------------------
def create_pooled_engine():
    """
    Production idea:
    Use connection pooling so your app reuses database connections
    instead of opening a new connection for every request.
    """

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=3,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False,
    )

    return engine


# ---------------------------------------------------------
# 4. Smart chunker
# ---------------------------------------------------------
def smart_chunker(
    text: str,
    source_name: str,
    embeddings: OpenAIEmbeddings,
    use_semantic: bool = True,
    semantic_threshold_amount: int = 90,
    recursive_chunk_size: int = 500,
    recursive_chunk_overlap: int = 80,
    min_chunks: int = 2,
    max_chunk_chars: int = 1200,
) -> Tuple[List[str], List[Document], Dict[str, Any]]:
    """
    Production-style chunking:
    - Try semantic chunking first.
    - If semantic chunking fails or produces poor chunks, use recursive splitting.
    """

    report = {
        "source": source_name,
        "selected_strategy": None,
        "fallback_used": False,
        "fallback_reason": None,
        "num_chunks": 0,
    }

    def recursive_fallback(reason: str) -> List[str]:
        print(f"[Fallback] {reason}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        report["selected_strategy"] = "recursive"
        report["fallback_used"] = True
        report["fallback_reason"] = reason

        return splitter.split_text(text)

    def validate_chunks(chunks: List[str]) -> Optional[str]:
        if not chunks:
            return "No chunks created."

        if len(chunks) < min_chunks:
            return f"Too few chunks created: {len(chunks)}"

        longest_chunk = max(len(chunk) for chunk in chunks)

        if longest_chunk > max_chunk_chars:
            return f"Chunk too large: {longest_chunk} characters."

        return None

    if use_semantic:
        try:
            print("[Chunking] Trying SemanticChunker...")

            semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=semantic_threshold_amount,
            )

            chunks = semantic_chunker.split_text(text)
            validation_error = validate_chunks(chunks)

            if validation_error:
                chunks = recursive_fallback(validation_error)
            else:
                report["selected_strategy"] = "semantic"

        except Exception as e:
            chunks = recursive_fallback(f"Semantic chunking failed: {e}")

    else:
        chunks = recursive_fallback("Semantic chunking disabled.")

    documents = []

    for i, chunk in enumerate(chunks, start=1):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "chunk_id": i,
                    "chunking_strategy": report["selected_strategy"],
                    "chunk_length": len(chunk),
                },
            )
        )

    report["num_chunks"] = len(chunks)

    return chunks, documents, report


# ---------------------------------------------------------
# 5. Create Supabase PGVector store
# ---------------------------------------------------------
def create_vectorstore(engine, embeddings):
    """
    PGVector stores chunks and embeddings inside PostgreSQL.
    Supabase PostgreSQL supports pgvector, which is used for vector similarity search.
    """

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="simple_production_rag_demo",
        connection=engine,
        use_jsonb=True,

        # Good for learning/testing.
        # It resets only this collection each run.
        pre_delete_collection=True,
    )

    return vectorstore


# ---------------------------------------------------------
# 6. Ingest documents into Supabase
# ---------------------------------------------------------
def ingest_documents(vectorstore, documents: List[Document]):
    print("\n[Ingestion] Adding documents to Supabase PGVector...")

    ids = vectorstore.add_documents(documents)

    print(f"[OK] Inserted {len(ids)} chunks into Supabase.")
    return ids


# ---------------------------------------------------------
# 7. Build RAG chain
# ---------------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a single context string.
    """

    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        formatted.append(
            f"[Source {i}] source={source}, chunk_id={chunk_id}\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def answer_question(retriever, question: str):
    """
    Simple production RAG flow:
    1. Retrieve relevant chunks from Supabase.
    2. Put chunks into prompt.
    3. Ask LLM to answer only from context.
    4. Return answer and sources.
    """

    print("\n" + "=" * 80)
    print("RAG Query")
    print("=" * 80)
    print(f"Question: {question}")

    retrieved_docs = retriever.invoke(question)

    print(f"\nRetrieved chunks: {len(retrieved_docs)}")

    context = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful RAG assistant.

Answer the user's question using only the provided context.

Rules:
- If the answer is in the context, answer clearly.
- If the answer is not in the context, say you do not know based on the provided documents.
- Keep the answer concise.
- Mention which source chunk supports the answer.
""",
            ),
            (
                "human",
                """
Question:
{question}

Context:
{context}
""",
            ),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    print("\n" + "-" * 80)
    print("Answer")
    print("-" * 80)
    print(response.content)

    print("\n" + "-" * 80)
    print("Retrieved Sources")
    print("-" * 80)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\nSource {i}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {doc.page_content[:300]}...")

    return response.content, retrieved_docs


# ---------------------------------------------------------
# 8. Main
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("Simple Production RAG with Supabase PGVector")
    print("=" * 80)

    debug_database_url(DATABASE_URL)

    engine = create_pooled_engine()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        chunks, documents, chunk_report = smart_chunker(
            text=DOCUMENT,
            source_name="authentication_guide.md",
            embeddings=embeddings,
            use_semantic=True,
            semantic_threshold_amount=90,
            recursive_chunk_size=500,
            recursive_chunk_overlap=80,
            min_chunks=2,
            max_chunk_chars=1200,
        )

        print("\n" + "=" * 80)
        print("Chunking Report")
        print("=" * 80)
        print(chunk_report)

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
            print(chunk[:500])

        vectorstore = create_vectorstore(engine, embeddings)

        inserted_ids = ingest_documents(vectorstore, documents)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        test_questions = [
            "How do I authenticate with OAuth2?",
            "What happens if I exceed the rate limit?",
            "How should webhook security be handled?",
            "How long does the access token last?",
            "What status code means permission denied?",
        ]

        for question in test_questions:
            answer_question(retriever, question)

        print("\n" + "=" * 80)
        print("[SUCCESS] Supabase RAG demo completed")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] RAG demo failed")
        print("=" * 80)
        print(f"Error type   : {type(e).__name__}")
        print(f"Error message: {e}")

    finally:
        print("\nDisposing database engine...")
        engine.dispose()
        print("[OK] Engine disposed")


if __name__ == "__main__":
    main()
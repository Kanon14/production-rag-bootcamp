from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

# ---------------------------------------------------------
# 1. Embedding model
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# 2. Sample document
# ---------------------------------------------------------
document = """
# Authentication Guide

## Overview
This guide explains how developers can authenticate with the Example API, manage access tokens, handle rate limits, 
process errors, and configure webhooks for real-time event notifications.
The API uses OAuth 2.0 for secure authentication. All requests to protected endpoints must include a valid access token in the Authorization header.

## OAuth2 Authentication
To authenticate with our API, you need OAuth 2.0 credentials.
First, obtain a client_id and client_secret from the developer portal. These credentials identify your application and allow it to request access tokens.
Make a POST request to /oauth/token with grant_type=client_credentials.
If the credentials are valid, the API returns an access token.
Use the access token in future API requests using the Authorization header:
Authorization: Bearer your_access_token
Access tokens expire after one hour. Your application should request a new token before the current token expires.

## Rate Limiting
The API applies rate limits to protect service stability and prevent abuse.
Each application can make up to 1,000 requests per hour. If the limit is exceeded, the API returns a 429 Too Many Requests response.
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
If your endpoint does not respond successfully, the API may retry delivery several times.

## Webhook Security
Each webhook request includes a signature header.
Your application should verify the signature before processing the webhook event.
Recommended webhook security practices include using HTTPS, verifying every webhook signature,
rejecting invalid requests, and storing processed event IDs to prevent duplicate processing.
"""


# ---------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------
def print_chunks(title, chunks):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print("\n" + "-" * 100)
        print(f"Chunk {i}")
        print(f"Length: {len(chunk)} characters")
        print("-" * 100)
        print(chunk)


def get_chunk_stats(chunks):
    lengths = [len(chunk) for chunk in chunks]

    return {
        "num_chunks": len(chunks),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths)
    }


def convert_chunks_to_documents(chunks, splitter_name):
    """
    Convert plain text chunks into LangChain Document objects.
    Metadata is useful for checking which splitter created the chunk.
    """
    docs = []

    for i, chunk in enumerate(chunks, start=1):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "splitter": splitter_name,
                    "chunk_id": i
                }
            )
        )

    return docs


def reset_chroma_dir(path):
    """
    Remove old Chroma database folder to avoid mixing old test results.
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def print_retrieval_results(title, query, results):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print("\n" + "-" * 100)
        print(f"Result {i}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 100)
        print(doc.page_content)


# ---------------------------------------------------------
# 4. Recursive Character Splitting
# ---------------------------------------------------------
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

recursive_chunks = recursive_splitter.split_text(document)

print_chunks(
    title="Recursive Character Text Splitter Result",
    chunks=recursive_chunks
)


# ---------------------------------------------------------
# 5. Semantic Chunking
# ---------------------------------------------------------
semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(document)

print_chunks(
    title="Semantic Chunker Result",
    chunks=semantic_chunks
)


# ---------------------------------------------------------
# 6. Chunk statistics comparison
# ---------------------------------------------------------
recursive_stats = get_chunk_stats(recursive_chunks)
semantic_stats = get_chunk_stats(semantic_chunks)

print("\n" + "=" * 100)
print("Chunking Comparison Summary")
print("=" * 100)

print(f"{'Method':<30} {'Chunks':<10} {'Min':<10} {'Max':<10} {'Average':<10}")
print("-" * 100)

print(
    f"{'Recursive Splitter':<30} "
    f"{recursive_stats['num_chunks']:<10} "
    f"{recursive_stats['min_length']:<10} "
    f"{recursive_stats['max_length']:<10} "
    f"{recursive_stats['avg_length']:<10.2f}"
)

print(
    f"{'Semantic Chunker':<30} "
    f"{semantic_stats['num_chunks']:<10} "
    f"{semantic_stats['min_length']:<10} "
    f"{semantic_stats['max_length']:<10} "
    f"{semantic_stats['avg_length']:<10.2f}"
)


# ---------------------------------------------------------
# 7. Convert chunks into LangChain Documents
# ---------------------------------------------------------
recursive_docs = convert_chunks_to_documents(
    chunks=recursive_chunks,
    splitter_name="recursive"
)

semantic_docs = convert_chunks_to_documents(
    chunks=semantic_chunks,
    splitter_name="semantic"
)


# ---------------------------------------------------------
# 8. Create separate Chroma vector stores
# ---------------------------------------------------------
recursive_db_path = "./chroma_recursive_db"
semantic_db_path = "./chroma_semantic_db"

reset_chroma_dir(recursive_db_path)
reset_chroma_dir(semantic_db_path)

recursive_vectorstore = Chroma.from_documents(
    documents=recursive_docs,
    embedding=embeddings,
    collection_name="recursive_chunks",
    persist_directory=recursive_db_path
)

semantic_vectorstore = Chroma.from_documents(
    documents=semantic_docs,
    embedding=embeddings,
    collection_name="semantic_chunks",
    persist_directory=semantic_db_path
)

print("\n" + "=" * 100)
print("Vector Store Created")
print("=" * 100)
print(f"Recursive Chroma DB path: {recursive_db_path}")
print(f"Semantic Chroma DB path: {semantic_db_path}")


# ---------------------------------------------------------
# 9. Create retrievers
# ---------------------------------------------------------
recursive_retriever = recursive_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

semantic_retriever = semantic_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# ---------------------------------------------------------
# 10. Query test
# ---------------------------------------------------------
test_queries = [
    "How do I authenticate with OAuth2?",
    "What happens if I exceed the rate limit?",
    "How should webhook security be handled?",
    "What HTTP status code means permission denied?",
    "How long does an access token last?"
]

for query in test_queries:
    recursive_results = recursive_retriever.invoke(query)
    semantic_results = semantic_retriever.invoke(query)

    print_retrieval_results(
        title="Recursive Retriever Results",
        query=query,
        results=recursive_results
    )

    print_retrieval_results(
        title="Semantic Retriever Results",
        query=query,
        results=semantic_results
    )


# ---------------------------------------------------------
# 11. Optional: similarity search with scores
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("Similarity Search With Scores Example")
print("=" * 100)

score_query = "What should I do when the API returns 429?"

recursive_scored_results = recursive_vectorstore.similarity_search_with_score(
    query=score_query,
    k=3
)

semantic_scored_results = semantic_vectorstore.similarity_search_with_score(
    query=score_query,
    k=3
)

print("\nRecursive Vector Store Scored Results")
for i, (doc, score) in enumerate(recursive_scored_results, start=1):
    print("\n" + "-" * 100)
    print(f"Result {i}")
    print(f"Score: {score}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 100)
    print(doc.page_content)

print("\nSemantic Vector Store Scored Results")
for i, (doc, score) in enumerate(semantic_scored_results, start=1):
    print("\n" + "-" * 100)
    print(f"Result {i}")
    print(f"Score: {score}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 100)
    print(doc.page_content)


# ---------------------------------------------------------
# 12. Learning notes
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("Learning Notes")
print("=" * 100)

print("""
What this script teaches:

1. Recursive splitting
   - Splits text based on length and separators.
   - More predictable chunk size.
   - Good default choice for many RAG systems.

2. Semantic chunking
   - Uses embeddings to detect meaning changes.
   - Tries to keep related ideas together.
   - More expensive because it needs embedding calls during chunking.

3. Vector store
   - Stores embedded chunks.
   - Allows similarity search over your document chunks.
   - In this example, recursive chunks and semantic chunks are stored separately.

4. Retriever
   - A retriever is the search layer used in RAG.
   - It receives a query and returns the most relevant chunks.
   - Later, these retrieved chunks can be passed to an LLM to generate an answer.

5. Score comparison
   - similarity_search_with_score shows the distance/similarity score.
   - Use it to inspect whether the retrieved chunks are actually relevant.

Practical observation:
- Recursive splitter may return smaller, more controlled chunks.
- Semantic chunker may return larger chunks grouped by meaning.
- The best method depends on your document type and your user queries.
""")
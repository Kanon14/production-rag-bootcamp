from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()


def smart_chunker(
    text: str,
    source_name: str = "unknown_source",
    embeddings: Optional[OpenAIEmbeddings] = None,
    semantic_threshold_type: str = "percentile",
    semantic_threshold_amount: int = 90,
    recursive_chunk_size: int = 500,
    recursive_chunk_overlap: int = 80,
    min_chunks: int = 2,
    max_chunk_chars: int = 1200,
    use_semantic: bool = True,
    verbose: bool = True
) -> Tuple[List[str], List[Document], Dict[str, Any]]:
    """
    Production-style smart chunker for RAG experiments.

    Primary strategy:
        1. Try SemanticChunker first.
        2. Validate the semantic chunks.
        3. If semantic chunking fails or produces poor chunks, fall back to recursive splitting.

    Fallback strategy:
        RecursiveCharacterTextSplitter.

    Parameters
    ----------
    text : str
        Raw input document text.

    source_name : str
        Name of the source document, file, or page.

    embeddings : Optional[OpenAIEmbeddings]
        Embedding model used by SemanticChunker.
        If None, the function will create OpenAIEmbeddings using text-embedding-3-small.

    semantic_threshold_type : str
        SemanticChunker breakpoint threshold type.
        Common values: "percentile", "standard_deviation", "interquartile".

    semantic_threshold_amount : int
        Threshold amount for semantic splitting.
        For percentile, higher value usually means fewer chunks.

    recursive_chunk_size : int
        Maximum character size for recursive chunks.

    recursive_chunk_overlap : int
        Character overlap between recursive chunks.

    min_chunks : int
        Minimum acceptable number of chunks.
        If semantic chunking returns fewer than this, fallback is triggered.

    max_chunk_chars : int
        Maximum acceptable character length for any semantic chunk.
        If any semantic chunk is larger than this, fallback is triggered.

    use_semantic : bool
        Whether to try semantic chunking first.
        Set False to force recursive splitting.

    verbose : bool
        Whether to print logs.

    Returns
    -------
    chunks : List[str]
        Final selected chunks as plain strings.

    documents : List[Document]
        Final selected chunks as LangChain Document objects.

    report : Dict[str, Any]
        Metadata about the chunking decision.
    """

    if not text or not text.strip():
        raise ValueError("Input text is empty. Please provide valid document text.")

    report = {
        "source_name": source_name,
        "selected_strategy": None,
        "semantic_attempted": False,
        "semantic_success": False,
        "fallback_used": False,
        "fallback_reason": None,
        "num_chunks": 0,
        "chunk_lengths": [],
        "min_chunk_length": None,
        "max_chunk_length": None,
        "avg_chunk_length": None,
    }

    def log(message: str):
        if verbose:
            print(message)

    def build_recursive_chunks(reason: str) -> List[str]:
        """
        Fallback chunking strategy.
        """
        log(f"\n[Fallback Triggered] {reason}")
        log("[Using RecursiveCharacterTextSplitter]")

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = recursive_splitter.split_text(text)

        report["selected_strategy"] = "recursive"
        report["fallback_used"] = True
        report["fallback_reason"] = reason

        return chunks

    def validate_chunks(chunks: List[str]) -> Optional[str]:
        """
        Validate whether chunk output is acceptable.
        Returns None if valid.
        Returns reason string if invalid.
        """
        if not chunks:
            return "No chunks were created."

        if len(chunks) < min_chunks:
            return f"Too few chunks were created: {len(chunks)} chunks."

        longest_chunk = max(len(chunk) for chunk in chunks)

        if longest_chunk > max_chunk_chars:
            return (
                f"Semantic chunk too large: {longest_chunk} characters. "
                f"Max allowed: {max_chunk_chars}."
            )

        return None

    final_chunks = []

    # ---------------------------------------------------------
    # 1. Try semantic chunking first
    # ---------------------------------------------------------
    if use_semantic:
        report["semantic_attempted"] = True

        try:
            log("\n[Trying SemanticChunker]")

            if embeddings is None:
                if not os.getenv("OPENAI_API_KEY"):
                    raise EnvironmentError(
                        "OPENAI_API_KEY is missing. Semantic chunking requires embeddings."
                    )

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

            semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=semantic_threshold_type,
                breakpoint_threshold_amount=semantic_threshold_amount
            )

            semantic_chunks = semantic_chunker.split_text(text)

            validation_error = validate_chunks(semantic_chunks)

            if validation_error is None:
                log("[SemanticChunker Success]")
                final_chunks = semantic_chunks
                report["selected_strategy"] = "semantic"
                report["semantic_success"] = True
            else:
                final_chunks = build_recursive_chunks(validation_error)

        except Exception as e:
            final_chunks = build_recursive_chunks(
                reason=f"Semantic chunking failed: {str(e)}"
            )

    else:
        final_chunks = build_recursive_chunks(
            reason="Semantic chunking disabled by use_semantic=False."
        )

    # ---------------------------------------------------------
    # 2. Convert final chunks into Document objects
    # ---------------------------------------------------------
    documents = []

    for i, chunk in enumerate(final_chunks, start=1):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "chunk_id": i,
                    "chunking_strategy": report["selected_strategy"],
                    "chunk_length": len(chunk),
                }
            )
        )

    # ---------------------------------------------------------
    # 3. Final report
    # ---------------------------------------------------------
    chunk_lengths = [len(chunk) for chunk in final_chunks]

    report["num_chunks"] = len(final_chunks)
    report["chunk_lengths"] = chunk_lengths
    report["min_chunk_length"] = min(chunk_lengths) if chunk_lengths else 0
    report["max_chunk_length"] = max(chunk_lengths) if chunk_lengths else 0
    report["avg_chunk_length"] = (
        sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    )

    return final_chunks, documents, report
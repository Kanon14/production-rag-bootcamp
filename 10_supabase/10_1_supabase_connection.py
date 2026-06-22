import os
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


load_dotenv()


# Supabase connection string format:
# postgresql://postgres:[YOUR-PASSWORD]@[PROJECT-REF].supabase.co:5432/postgres
#
# Recommended .env:
# SUPABASE_DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@[PROJECT-REF].supabase.co:5432/postgres
# OPENAI_API_KEY=your_openai_api_key

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    """
    Connect to Supabase PostgreSQL with pgvector through LangChain PGVector.
    """

    print("\n[1] Initializing embedding model...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print("[OK] Embedding model initialized")

    print("\n[2] Connecting to PGVector vector store...")
    print(f"Database source: {'Supabase' if SUPABASE_URL else 'Local PostgreSQL fallback'}")
    print("Collection name: production_docs")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print("[OK] PGVector object created")

    return vectorstore


def verify_connection(vectorstore):
    """
    Verify Supabase / PostgreSQL pgvector connection by:
    1. Adding a test document
    2. Running similarity search
    3. Printing retrieved result
    4. Cleaning up the test document
    """

    print("\n[3] Creating test document...")

    test_doc = Document(
        page_content=(
            "This is a test document to verify the connection of Supabase "
            "PostgreSQL with pgvector and LangChain."
        ),
        metadata={
            "test": True,
            "source": "connection_test",
            "purpose": "verify_supabase_pgvector"
        }
    )

    print("[OK] Test document created")

    try:
        print("\n[4] Adding test document to vector store...")

        ids = vectorstore.add_documents([test_doc])

        if not ids:
            print("[FAILED] No document ID returned after insertion")
            return False

        test_doc_id = ids[0]

        print("[OK] Test document added successfully")
        print(f"Inserted document ID: {test_doc_id}")

        print("\n[5] Running similarity search...")

        query = "verify Supabase pgvector connection"

        results = vectorstore.similarity_search(
            query=query,
            k=3
        )

        print(f"Query: {query}")
        print(f"Number of results returned: {len(results)}")

        if not results:
            print("[FAILED] No search results returned")
            return False

        print("\n[OK] Similarity search works")

        for i, doc in enumerate(results, start=1):
            print("\n" + "-" * 80)
            print(f"Result {i}")
            print("-" * 80)
            print("Content:")
            print(doc.page_content)
            print("\nMetadata:")
            print(doc.metadata)

        print("\n[6] Cleaning up test document...")

        vectorstore.delete(ids=[test_doc_id])

        print("[OK] Cleanup completed")

        print("\n[7] Connection verification completed successfully")
        return True

    except Exception as e:
        print("\n[ERROR] Connection verification failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        return False


def check_environment():
    """
    Check required environment variables before connecting.
    """

    print("=" * 80)
    print("Environment Check")
    print("=" * 80)

    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        print("[OK] OPENAI_API_KEY found")
    else:
        print("[MISSING] OPENAI_API_KEY is not set")
        return False

    if SUPABASE_URL:
        print("[OK] SUPABASE_DATABASE_URL found")
        print("Using Supabase database")
    else:
        print("[WARNING] SUPABASE_DATABASE_URL not found")
        print("Using local DATABASE_URL fallback")
        print(f"DATABASE_URL: {DATABASE_URL}")

    return True


def main():
    """
    Main function to verify Supabase PGVector connection.
    """

    print("=" * 80)
    print("Supabase PGVector Connection Test")
    print("=" * 80)

    env_ok = check_environment()

    if not env_ok:
        print("\n[STOPPED] Environment check failed")
        print("Please check your .env file before running again.")
        return

    try:
        vectorstore = connect_to_supabase()

        success = verify_connection(vectorstore)

        print("\n" + "=" * 80)
        print("Final Result")
        print("=" * 80)

        if success:
            print("[SUCCESS] Supabase / PGVector connection is working.")
        else:
            print("[FAILED] Supabase / PGVector connection test failed.")

    except Exception as e:
        print("\n[ERROR] Failed before verification step")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")


if __name__ == "__main__":
    main()
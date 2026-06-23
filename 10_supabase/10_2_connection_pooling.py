import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document


load_dotenv()


DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("SUPABASE_DATABASE_URL is missing from .env")


# ---------------------------------------------------------
# 1. Debug database URL without exposing password
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
# 2. Create SQLAlchemy engine with connection pooling
# ---------------------------------------------------------
def create_pooled_engine():
    """
    Create a SQLAlchemy engine with a local application-side connection pool.

    pool_size:
        Number of persistent database connections kept open.

    max_overflow:
        Extra temporary connections allowed when pool_size is fully used.

    pool_timeout:
        How many seconds to wait for a connection before raising an error.

    pool_recycle:
        Recycle connections after this many seconds to avoid stale connections.

    pool_pre_ping:
        Checks whether a connection is still alive before using it.
    """

    engine = create_engine(
        DATABASE_URL,

        # QueuePool is SQLAlchemy's standard connection pool.
        poolclass=QueuePool,

        # Keep this small for Supabase learning/demo.
        pool_size=3,

        # Allow 2 extra temporary connections during spikes.
        max_overflow=2,

        # Wait up to 30 seconds for a free connection.
        pool_timeout=30,

        # Recycle old connections after 30 minutes.
        pool_recycle=1800,

        # Test connection before using it.
        pool_pre_ping=True,

        # Set True if you want to see SQL logs.
        echo=False,

        # Set to "debug" if you want detailed pool logs.
        echo_pool=False,
    )

    return engine


# ---------------------------------------------------------
# 3. Basic connection test
# ---------------------------------------------------------
def test_basic_connection(engine):
    print("\n[1] Basic Connection Test")
    print("-" * 80)

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT 
                    current_database() AS database_name,
                    current_user AS user_name,
                    version() AS postgres_version,
                    now() AS server_time;
            """)
        )

        row = result.fetchone()

        print("[OK] Connected to database")
        print(f"Database     : {row.database_name}")
        print(f"User         : {row.user_name}")
        print(f"Server time  : {row.server_time}")
        print(f"Postgres ver : {row.postgres_version[:80]}...")


# ---------------------------------------------------------
# 4. Show pool status
# ---------------------------------------------------------
def print_pool_status(engine, label):
    print("\n" + label)
    print("-" * 80)
    print(engine.pool.status())


# ---------------------------------------------------------
# 5. Simulate multiple concurrent database requests
# ---------------------------------------------------------
def simulated_db_task(engine, task_id: int, sleep_seconds: int = 2):
    """
    Simulates one request using one database connection.

    The connection is borrowed from the pool inside the 'with' block.
    After the block ends, the connection is returned to the pool.
    """

    start_time = time.time()

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT 
                    pg_backend_pid() AS connection_pid,
                    now() AS query_time,
                    pg_sleep(:sleep_seconds);
            """),
            {"sleep_seconds": sleep_seconds}
        )

        row = result.fetchone()

    end_time = time.time()

    return {
        "task_id": task_id,
        "connection_pid": row.connection_pid,
        "query_time": row.query_time,
        "duration": round(end_time - start_time, 2),
    }


def test_connection_pooling(engine):
    print("\n[2] Connection Pooling Concurrency Test")
    print("-" * 80)

    print_pool_status(engine, "[Before running tasks]")

    total_tasks = 8
    max_workers = 8

    print(f"\nRunning {total_tasks} simulated database tasks...")
    print("Pool size = 3, max_overflow = 2")
    print("This means up to 5 database connections can be active at once.\n")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(simulated_db_task, engine, task_id, 2)
            for task_id in range(1, total_tasks + 1)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    results = sorted(results, key=lambda x: x["task_id"])

    for item in results:
        print(
            f"Task {item['task_id']} | "
            f"Connection PID: {item['connection_pid']} | "
            f"Duration: {item['duration']}s"
        )

    print_pool_status(engine, "[After running tasks]")


# ---------------------------------------------------------
# 6. Use the pooled engine with LangChain PGVector
# ---------------------------------------------------------
def create_pgvector_store(engine):
    print("\n[3] Creating PGVector Store With Pooled Engine")
    print("-" * 80)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="connection_pooling_demo",
        connection=engine,
        use_jsonb=True,
    )

    print("[OK] PGVector store created using SQLAlchemy pooled engine")

    return vectorstore


# ---------------------------------------------------------
# 7. Test PGVector insert and search
# ---------------------------------------------------------
def test_pgvector_with_pool(vectorstore):
    print("\n[4] PGVector Insert + Search Test")
    print("-" * 80)

    docs = [
        Document(
            page_content="Connection pooling reuses database connections instead of opening a new one every time.",
            metadata={"topic": "connection_pooling", "level": "basic"}
        ),
        Document(
            page_content="Supabase provides PostgreSQL in the cloud and supports pgvector for vector search.",
            metadata={"topic": "supabase", "level": "basic"}
        ),
        Document(
            page_content="In a RAG system, vector stores are used to retrieve relevant chunks before sending context to an LLM.",
            metadata={"topic": "rag", "level": "basic"}
        ),
    ]

    print("Adding documents to PGVector...")
    ids = vectorstore.add_documents(docs)

    print("[OK] Documents inserted")
    print(f"Inserted IDs: {ids}")

    query = "Why is connection pooling useful?"

    print(f"\nRunning similarity search: {query}")

    results = vectorstore.similarity_search(
        query=query,
        k=2
    )

    print(f"\nReturned results: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(f"Result {i}")
        print("-" * 80)
        print("Content:")
        print(doc.page_content)
        print("\nMetadata:")
        print(doc.metadata)

    print("\nCleaning up inserted test documents...")
    vectorstore.delete(ids=ids)
    print("[OK] Cleanup completed")


# ---------------------------------------------------------
# 8. Main function
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("Supabase Connection Pooling Learning Demo")
    print("=" * 80)

    debug_database_url(DATABASE_URL)

    engine = create_pooled_engine()

    try:
        test_basic_connection(engine)

        test_connection_pooling(engine)

        vectorstore = create_pgvector_store(engine)

        test_pgvector_with_pool(vectorstore)

        print("\n" + "=" * 80)
        print("[SUCCESS] Connection pooling demo completed")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] Demo failed")
        print("=" * 80)
        print(f"Error type   : {type(e).__name__}")
        print(f"Error message: {e}")

    finally:
        print("\nDisposing SQLAlchemy engine...")
        engine.dispose()
        print("[OK] Engine disposed")


if __name__ == "__main__":
    main()
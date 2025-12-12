"""
Resumable embedding generation utilities
"""

import json
import sqlite3

from src.config import settings


def _validate_table_name(table_name: str) -> str:
    """Validate table name to prevent SQL injection - only allow known safe values"""
    valid_tables = {"enhanced_code_chunks"}
    if table_name not in valid_tables:
        error_msg = f"Invalid table name: {table_name}"
        raise ValueError(error_msg)
    return table_name


def update_embedding(record_id: str, embedding: list[float]):
    """
    Update or insert embedding for a chunk in the database.

    Args:
        record_id: The ID of the chunk to update
        embedding: The embedding vector to store (as a list of floats)
    """
    db_path = settings.get_project_db_path()
    table_name = _validate_table_name("enhanced_code_chunks")

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Serialize the embedding as JSON string
        embedding_json = json.dumps(embedding)

        # Use direct string concatenation since table name is validated
        query = f"UPDATE {table_name} SET embedding = ? WHERE id = ?"  # noqa: S608
        cur.execute(query, (embedding_json, record_id))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"⚠️ Failed to update embedding for {record_id}: {e}")


def get_embedded_chunks_ids() -> list[str]:
    """Get IDs of chunks that already have embeddings stored in the database"""
    db_path = settings.get_project_db_path()
    table_name = _validate_table_name("enhanced_code_chunks")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Use direct string concatenation since table name is validated
    query = f"SELECT id FROM {table_name} WHERE embedding IS NOT NULL AND embedding != ?"  # noqa: S608
    cur.execute(query, ("",))
    results = [row[0] for row in cur.fetchall()]

    conn.close()
    return results


def stats_check(setl: int, chunkl: int, nchunkl: int):
    """Print embedding statistics"""
    print(f"Chunks with existing embeddings: {setl}")
    print(f"Total chunks count: {chunkl}")
    print(f"Chunks to embed: {nchunkl}")


def add_embedding_column_to_db():
    """Add embedding column to the enhanced_code_chunks table if it doesn't exist"""
    db_path = settings.get_project_db_path()
    table_name = _validate_table_name("enhanced_code_chunks")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if the embedding column already exists
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cur.fetchall()]

    if "embedding" not in columns:
        # Add the embedding column if it doesn't exist
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN embedding TEXT")
        conn.commit()
        print("✅ Added 'embedding' column to the database table")
    else:
        print("ℹ️ 'embedding' column already exists in the database table")  # noqa: RUF001

    conn.close()


def get_embedding_from_db(chunk_id: str) -> list[float] | None:
    """
    Get embedding for a specific chunk from the database.

    Args:
        chunk_id: The ID of the chunk to retrieve embedding for

    Returns:
        The embedding as a list of floats, or None if not found
    """
    db_path = settings.get_project_db_path()
    table_name = _validate_table_name("enhanced_code_chunks")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Use direct string concatenation since table name is validated
    query = f"SELECT embedding FROM {table_name} WHERE id = ?"  # noqa: S608
    cur.execute(query, (chunk_id,))
    result = cur.fetchone()

    conn.close()

    if result and result[0]:
        try:
            return json.loads(result[0])
        except json.JSONDecodeError:
            return None

    return None

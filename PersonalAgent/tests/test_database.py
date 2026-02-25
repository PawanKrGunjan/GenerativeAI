from Agent.database import (
    compute_file_hash,
    load_document_content,
    semantic_search,
    sync_documents_to_db,
)

# ────────────────────────────────────────────────
# Hashing
# ────────────────────────────────────────────────


def test_compute_file_hash(temp_data_dir):
    file_path = temp_data_dir / "test.txt"
    file_path.write_text("hello world\n")

    file_hash = compute_file_hash(file_path)

    assert isinstance(file_hash, str)
    assert len(file_hash) == 64
    assert file_hash == (
        "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    )


# ────────────────────────────────────────────────
# Document Loading
# ────────────────────────────────────────────────


def test_load_document_content_txt(temp_data_dir):
    file_path = temp_data_dir / "test.txt"
    file_path.write_text("Line 1\nLine 2\n")

    content = load_document_content(file_path)

    assert "Line 1" in content
    assert "Line 2" in content


def test_load_document_content_pdf(real_pdf_path):
    content = load_document_content(real_pdf_path)

    assert isinstance(content, str)
    assert len(content.strip()) > 0


# ────────────────────────────────────────────────
# Sync Logic
# ────────────────────────────────────────────────


def test_sync_documents_inserts_file(temp_db_conn, temp_data_dir):
    file_path = temp_data_dir / "sample.txt"
    file_path.write_text("This is a test document.")

    sync_documents_to_db(temp_db_conn, None, folder=temp_data_dir)

    cursor = temp_db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]

    assert count == 1


def test_sync_updates_modified_file(temp_db_conn, temp_data_dir):
    file_path = temp_data_dir / "sample.txt"
    file_path.write_text("Version 1")

    sync_documents_to_db(temp_db_conn, None, folder=temp_data_dir)

    # Modify file
    file_path.write_text("Version 2")

    sync_documents_to_db(temp_db_conn, None, folder=temp_data_dir)

    cursor = temp_db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]

    assert count == 1


# ────────────────────────────────────────────────
# Semantic Search (Mocked – No Ollama Required)
# ────────────────────────────────────────────────


def test_semantic_search_returns_list(temp_db_conn, monkeypatch):
    import numpy as np

    # Patch embed_text WHERE IT IS USED
    def fake_embed(text):
        return np.ones(768, dtype=np.float32)

    monkeypatch.setattr("Agent.database.embed_text", fake_embed)

    # Insert fake chunk manually
    emb = np.ones(768, dtype=np.float32).tobytes()

    temp_db_conn.execute(
        """
        INSERT INTO document_chunks
        (file_path, chunk_text, embedding, chunk_index, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("test.txt", "hello world", emb, 0, "2025-01-01T00:00:00"),
    )
    temp_db_conn.commit()

    results = semantic_search(temp_db_conn, "test query", top_k=3)

    assert isinstance(results, list)
    assert len(results) >= 1

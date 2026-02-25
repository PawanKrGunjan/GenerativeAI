"""
Shared pytest fixtures for the Agent project
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from Agent.database import (
    setup_document_chunks_table,
    setup_documents_table,
    setup_facts_table,
)


@pytest.fixture(scope="function")
def temp_db_path():
    """Create a temporary SQLite database file"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    yield str(db_path)
    db_path.unlink(missing_ok=True)


@pytest.fixture(scope="function")
def temp_db_conn(temp_db_path):
    """Return a fresh in-memory or file-based connection + clean tables"""
    conn = sqlite3.connect(temp_db_path, check_same_thread=False)

    # Setup schema
    setup_facts_table(conn, None)  # pass None for log (or mock)
    setup_documents_table(conn, None)
    setup_document_chunks_table(conn, None)

    yield conn

    conn.close()


@pytest.fixture(scope="function")
def temp_data_dir():
    """Create a temporary data directory with dummy files"""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def real_pdf_path():
    pdf = Path("data/pawankrgunjan.pdf")
    if not pdf.exists():
        pytest.skip("real PDF not found")
    return pdf

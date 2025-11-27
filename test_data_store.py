#!/usr/bin/env python3
"""
Test script to verify data_store.py handles list of Documents correctly
"""

import tempfile
from pathlib import Path

from llama_index.core import Document

from src.config.data_store import save_data


def test_save_list_of_documents():
    """Test saving a list of LlamaIndex Documents"""

    # Create sample documents
    docs = [
        Document(
            text="This is the first document",
            doc_id="doc1",
            metadata={"source": "file1.py", "author": "test"},
        ),
        Document(
            text="This is the second document with different content",
            doc_id="doc2",
            metadata={"source": "file2.py", "author": "test"},
        ),
        Document(
            text="Third document for testing purposes",
            doc_id="doc3",
            metadata={"source": "file3.py", "author": "test"},
        ),
    ]

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test saving the list of documents
        try:
            file_path = save_data(docs, method="test", language="python", ext="json")
            print(f"✓ Successfully saved list of documents to: {file_path}")

            # Read back the file to verify content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                print(f"Content length: {len(content)} characters")
                print("First 200 characters:", content[:200])

            return True
        except Exception as e:
            print(f"✗ Error saving list of documents: {e}")
            return False


def test_save_single_document():
    """Test saving a single LlamaIndex Document"""

    # Create a single document
    doc = Document(
        text="This is a single test document",
        doc_id="single_doc",
        metadata={"source": "single.py", "author": "test"},
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test saving the single document
        try:
            file_path = save_data(doc, method="test", language="python", ext="json")
            print(f"✓ Successfully saved single document to: {file_path}")

            # Read back the file to verify content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                print(f"Content length: {len(content)} characters")
                print("First 200 characters:", content[:200])

            return True
        except Exception as e:
            print(f"✗ Error saving single document: {e}")
            return False


def test_save_mixed_list():
    """Test saving a list with mixed content (not just documents)"""

    # Create list with mixed content
    mixed_data = [{"name": "test1", "value": 1}, {"name": "test2", "value": 2}, "plain string", 42]

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test saving the mixed list
        try:
            file_path = save_data(mixed_data, method="test", language="python", ext="json")
            print(f"✓ Successfully saved mixed list to: {file_path}")

            # Read back the file to verify content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                print(f"Content length: {len(content)} characters")
                print("Content:", content)

            return True
        except Exception as e:
            print(f"✗ Error saving mixed list: {e}")
            return False


if __name__ == "__main__":
    print("Testing data_store.py with Documents...")

    print("\n1. Testing list of Documents:")
    test1_result = test_save_list_of_documents()

    print("\n2. Testing single Document:")
    test2_result = test_save_single_document()

    print("\n3. Testing mixed list:")
    test3_result = test_save_mixed_list()

    print(f"\nResults: {test1_result and test2_result and test3_result}")

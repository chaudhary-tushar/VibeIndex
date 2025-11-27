import pytest
import tempfile
from pathlib import Path
from llama_index.core import Document

from src.config.data_store import save_data, _determine_file_extension, _save_json_data


def test_determine_file_extension_with_document_list():
    """Test that the function correctly identifies a list of Documents"""
    docs = [
        Document(text="Test document 1", doc_id="doc1"),
        Document(text="Test document 2", doc_id="doc2")
    ]
    
    result = _determine_file_extension(docs)
    assert result == "json"


def test_determine_file_extension_with_single_document():
    """Test that the function correctly identifies a single Document"""
    doc = Document(text="Test document", doc_id="doc1")
    
    result = _determine_file_extension(doc)
    assert result == "json"


def test_save_json_data_with_document_list():
    """Test saving a list of Documents to JSON"""
    docs = [
        Document(
            text="First document",
            doc_id="doc1",
            metadata={"source": "test1.py"}
        ),
        Document(
            text="Second document", 
            doc_id="doc2",
            metadata={"source": "test2.py"}
        )
    ]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Save the list of documents
        _save_json_data(docs, tmp_path)
        
        # Read the file back to make sure it saved correctly
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verify content contains expected elements
        assert "First document" in content
        assert "Second document" in content
        assert "doc1" in content
        assert "doc2" in content
        
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def test_save_json_data_with_single_document():
    """Test saving a single Document to JSON"""
    doc = Document(
        text="Single test document",
        doc_id="single_doc",
        metadata={"source": "single.py"}
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Save the single document
        _save_json_data(doc, tmp_path)
        
        # Read the file back to make sure it saved correctly
        with open(tmp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Verify content contains expected elements
        assert "Single test document" in content
        assert "single_doc" in content
        assert "single.py" in content
        
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


def test_save_document_list_integration():
    """Integration test using the main save_data function with Document list"""
    docs = [
        Document(
            text="Integration test document 1",
            doc_id="int_doc1",
            metadata={"source": "integration1.py"}
        ),
        Document(
            text="Integration test document 2", 
            doc_id="int_doc2",
            metadata={"source": "integration2.py"}
        )
    ]
    
    # Test the main save_data function
    file_path = save_data(docs, method="test_integration", language="python", ext="json")
    
    # Verify the file was created
    assert file_path.exists()
    
    # Read and verify content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Verify content contains expected elements
    assert "Integration test document 1" in content
    assert "Integration test document 2" in content
    assert "int_doc1" in content
    assert "int_doc2" in content
    
    # Clean up
    if file_path.exists():
        file_path.unlink()


if __name__ == "__main__":
    # Run the tests manually if executed directly
    test_determine_file_extension_with_document_list()
    print("✓ test_determine_file_extension_with_document_list passed")
    
    test_determine_file_extension_with_single_document()
    print("✓ test_determine_file_extension_with_single_document passed")
    
    test_save_json_data_with_document_list()
    print("✓ test_save_json_data_with_document_list passed")
    
    test_save_json_data_with_single_document()
    print("✓ test_save_json_data_with_single_document passed")
    
    test_save_document_list_integration()
    print("✓ test_save_document_list_integration passed")
    
    print("\nAll tests passed!")
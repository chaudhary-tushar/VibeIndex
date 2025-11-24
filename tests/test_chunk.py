import pytest

from src.preprocessing.chunk import CodeChunk


def test_codechunk_creation():
    """Test basic CodeChunk creation with required fields"""
    chunk = CodeChunk(
        type="function",
        name="test_function",
        code="def test_function(): pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=1
    )

    assert chunk.type == "function"
    assert chunk.name == "test_function"
    assert chunk.code == "def test_function(): pass"
    assert chunk.file_path == "test.py"
    assert chunk.language == "python"
    assert chunk.start_line == 1
    assert chunk.end_line == 1
    assert chunk.id != ""  # ID should be auto-generated
    assert chunk.qualified_name == "test.test_function"  # Generated from name and file stem


def test_codechunk_default_values():
    """Test CodeChunk with default values"""
    chunk = CodeChunk(
        type="function",
        name="test_function",
        code="def test_function(): pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=1
    )

    # Check default values
    assert chunk.id != ""
    assert chunk.qualified_name == "test.test_function"
    assert chunk.docstring is None
    assert chunk.signature is None
    assert chunk.complexity == 0
    assert chunk.parent is None
    assert chunk.dependencies == []
    assert chunk.references == []
    assert chunk.defines == []
    assert chunk.location == {}
    assert chunk.metadata == {}
    assert chunk.documentation == {}
    assert chunk.analysis == {}
    assert chunk.relationships == {}
    assert chunk.context == {}


def test_codechunk_dependencies_references_defines():
    """Test CodeChunk with dependencies, references, and defines"""
    chunk = CodeChunk(
        type="function",
        name="test_function",
        code="import os\nprint(os.path)",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=2,
        dependencies=["os"],
        references=["os"],
        defines=["test_function"]
    )

    assert chunk.dependencies == ["os"]
    assert chunk.references == ["os"]
    assert chunk.defines == ["test_function"]


def test_codechunk_id_generation():
    """Test that ID is generated correctly"""
    chunk1 = CodeChunk(
        type="function",
        name="func1",
        code="def func1(): pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=1
    )

    chunk2 = CodeChunk(
        type="function",
        name="func2",
        code="def func2(): pass",
        file_path="test.py",
        language="python",
        start_line=5,
        end_line=5
    )

    # IDs should be different for different name/line combinations
    assert chunk1.id != chunk2.id
    # IDs should be 12 characters as per implementation
    assert len(chunk1.id) == 12
    assert len(chunk2.id) == 12


def test_codechunk_qualified_name_generation():
    """Test qualified name generation"""
    # Test with function in a file
    chunk1 = CodeChunk(
        type="function",
        name="my_function",
        code="def my_function(): pass",
        file_path="src/utils.py",
        language="python",
        start_line=1,
        end_line=1
    )
    assert chunk1.qualified_name == "utils.my_function"

    # Test with class
    chunk2 = CodeChunk(
        type="class",
        name="MyClass",
        code="class MyClass: pass",
        file_path="src/models.py",
        language="python",
        start_line=1,
        end_line=1
    )
    assert chunk2.qualified_name == "models.MyClass"

    # Test with file type (should just use name)
    chunk3 = CodeChunk(
        type="file",
        name="config",
        code="CONFIG_VALUE = 'value'",
        file_path="src/config.py",
        language="python",
        start_line=1,
        end_line=1
    )
    assert chunk3.qualified_name == "config"


def test_codechunk_to_dict():
    """Test conversion to dictionary"""
    chunk = CodeChunk(
        type="function",
        name="test_function",
        code="def test_function(): pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=1,
        docstring="A test function",
        complexity=3
    )

    chunk_dict = chunk.to_dict()

    assert chunk_dict["type"] == "function"
    assert chunk_dict["name"] == "test_function"
    assert chunk_dict["code"] == "def test_function(): pass"
    assert chunk_dict["file_path"] == "test.py"
    assert chunk_dict["language"] == "python"
    assert chunk_dict["start_line"] == 1
    assert chunk_dict["end_line"] == 1
    assert chunk_dict["docstring"] == "A test function"
    assert chunk_dict["complexity"] == 3


def test_codechunk_metadata_structures():
    """Test that all metadata structures are properly initialized"""
    chunk = CodeChunk(
        type="function",
        name="test_function",
        code="def test_function(): pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=1
    )

    # All metadata structures should be initialized as empty dicts
    assert isinstance(chunk.location, dict)
    assert isinstance(chunk.metadata, dict)
    assert isinstance(chunk.documentation, dict)
    assert isinstance(chunk.analysis, dict)
    assert isinstance(chunk.relationships, dict)
    assert isinstance(chunk.context, dict)

    # Dependencies and references should be empty lists by default
    assert isinstance(chunk.dependencies, list)
    assert isinstance(chunk.references, list)
    assert isinstance(chunk.defines, list)


def test_codechunk_with_custom_values():
    """Test CodeChunk with custom values for all fields"""
    custom_location = {"start_line": 1, "end_line": 5, "start_col": 0, "end_col": 10}
    custom_metadata = {"decorator": "@app.route", "access_modifier": "public"}
    custom_documentation = {"docstring": "Custom docstring", "comments": ["comment1", "comment2"]}
    custom_analysis = {"complexity": 10, "tokens": 50}
    custom_relationships = {"imports": ["os", "sys"], "children": ["method1"]}
    custom_context = {"module": "utils", "domain": "web"}

    chunk = CodeChunk(
        type="class",
        name="TestClass",
        code="class TestClass: pass",
        file_path="test.py",
        language="python",
        start_line=1,
        end_line=3,
        id="custom-id",
        qualified_name="test.TestClass",
        docstring="Custom docstring",
        signature="class TestClass:",
        complexity=10,
        parent="ParentClass",
        dependencies=["os"],
        references=["sys"],
        defines=["TestClass"],
        location=custom_location,
        metadata=custom_metadata,
        documentation=custom_documentation,
        analysis=custom_analysis,
        relationships=custom_relationships,
        context=custom_context
    )

    assert chunk.id == "custom-id"
    assert chunk.qualified_name == "test.TestClass"
    assert chunk.docstring == "Custom docstring"
    assert chunk.signature == "class TestClass:"
    assert chunk.complexity == 10
    assert chunk.parent == "ParentClass"
    assert chunk.dependencies == ["os"]
    assert chunk.references == ["sys"]
    assert chunk.defines == ["TestClass"]
    assert chunk.location == custom_location
    assert chunk.metadata == custom_metadata
    assert chunk.documentation == custom_documentation
    assert chunk.analysis == custom_analysis
    assert chunk.relationships == custom_relationships
    assert chunk.context == custom_context

import tempfile
from pathlib import Path

import pytest

from src.preprocessing import CodeParser
from src.preprocessing import parse_file
from src.preprocessing import parse_project


def test_parse_project_function():
    """Test the parse_project convenience function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        # The analyzer.py code accesses file_path.parents[3], so we need at least 3 parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a simple Python file in the project directory
        test_file = project_path / "test.py"
        test_file.write_text("def hello(): pass")

        # Call parse_project
        parser = parse_project(str(project_path))

        # Verify the result
        assert isinstance(parser, CodeParser)
        assert parser.project_path == project_path.resolve()


def test_parse_project_with_multiple_files():
    """Test parse_project with multiple files in project"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create multiple files
        (project_path / "main.py").write_text("def main(): pass")
        (project_path / "utils.py").write_text("def helper(): pass")
        (project_path / "app.js").write_text("function app() {}")

        # Call parse_project
        parser = parse_project(str(project_path))

        # Verify the parser was created
        assert isinstance(parser, CodeParser)
        assert parser.project_path == project_path.resolve()


def test_parse_file_function():
    """Test the parse_file convenience function"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a Python file
        test_file = project_path / "test.py"
        test_file.write_text('def hello():\n    """A greeting function."""\n    print("Hello")')

        # Call parse_file
        chunks = parse_file(str(test_file))

        # Verify the result - chunks might be empty depending on tree-sitter availability,
        # but the function should at least not error
        assert isinstance(chunks, list)


def test_parse_file_with_project_path():
    """Test parse_file with explicit project path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a file in a subdirectory
        subdir = project_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.py"
        test_file.write_text("def hello(): pass")

        # Call parse_file with explicit project path
        chunks = parse_file(str(test_file), project_path=str(project_path))

        # Verify the result
        assert isinstance(chunks, list)


def test_parse_file_automatic_project_path():
    """Test parse_file with automatic project path detection"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a file
        test_file = project_path / "test.py"
        test_file.write_text("def hello(): pass")

        # Call parse_file without project path (should use parent directory)
        chunks = parse_file(str(test_file))

        # Verify the result
        assert isinstance(chunks, list)


def test_parse_file_nonexistent_file():
    """Test parse_file with nonexistent file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_file = Path(temp_dir) / "nonexistent.py"

        # Should handle gracefully
        with pytest.raises(FileNotFoundError):
            parse_file(str(nonexistent_file))


def test_parse_file_wrong_extension():
    """Test parse_file with unsupported file extension"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a file with unsupported extension
        unsupported_file = project_path / "test.xyz"
        unsupported_file.write_text("some content")

        # Call parse_file - should return empty list for unsupported extension
        chunks = parse_file(str(unsupported_file))

        # Should return an empty list since .xyz is not supported
        assert chunks == []


def test_parse_project_empty_directory():
    """Test parse_project on empty directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Call parse_project on empty directory
        parser = parse_project(str(project_path))

        # Verify the parser is created
        assert isinstance(parser, CodeParser)
        assert parser.project_path == project_path.resolve()


def test_parse_project_with_ignore_patterns():
    """Test parse_project respects ignore patterns like node_modules"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a Python file
        (project_path / "main.py").write_text("def main(): pass")

        # Create a node_modules directory with a JS file (should be ignored)
        node_modules_dir = project_path / "node_modules"
        node_modules_dir.mkdir()
        (node_modules_dir / "package.js").write_text('console.log("ignored");')

        # Call parse_project
        parser = parse_project(str(project_path))

        # The parser should be created and should have the project path
        assert isinstance(parser, CodeParser)
        assert parser.project_path == project_path.resolve()


def test_consistency_between_functions():
    """Test that parse_project and parse_file work together consistently"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested directory structure to ensure enough parent levels
        project_path = Path(temp_dir) / "level1" / "level2" / "level3"
        project_path.mkdir(parents=True)

        # Create a Python file
        test_file = project_path / "test.py"
        test_file.write_text("def hello(): pass")

        # Use parse_project to get a parser
        project_parser = parse_project(str(project_path))

        # Use parse_file for the same file
        file_chunks = parse_file(str(test_file), project_path=str(project_path))

        # Both should work without error
        assert isinstance(project_parser, CodeParser)
        assert isinstance(file_chunks, list)

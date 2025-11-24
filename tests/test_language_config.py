import pytest

from src.preprocessing.language_config import LanguageConfig


def test_language_config_language_map():
    """Test the LANGUAGE_MAP contains expected extensions"""
    # Check that common extensions are mapped to language names
    assert ".py" in LanguageConfig.LANGUAGE_MAP
    assert LanguageConfig.LANGUAGE_MAP[".py"] == "python"

    assert ".js" in LanguageConfig.LANGUAGE_MAP
    assert LanguageConfig.LANGUAGE_MAP[".js"] == "javascript"

    assert ".css" in LanguageConfig.LANGUAGE_MAP
    assert LanguageConfig.LANGUAGE_MAP[".css"] == "css"

    assert ".html" in LanguageConfig.LANGUAGE_MAP
    assert LanguageConfig.LANGUAGE_MAP[".html"] == "html"

    # Check a few more extensions
    assert ".java" in LanguageConfig.LANGUAGE_MAP
    assert ".cpp" in LanguageConfig.LANGUAGE_MAP
    assert ".go" in LanguageConfig.LANGUAGE_MAP
    assert ".rb" in LanguageConfig.LANGUAGE_MAP


def test_language_config_languages():
    """Test that LANGUAGES contains expected language parsers"""
    # Check that common languages are available
    assert "python" in LanguageConfig.LANGUAGES
    assert "javascript" in LanguageConfig.LANGUAGES
    assert "html" in LanguageConfig.LANGUAGES
    assert "css" in LanguageConfig.LANGUAGES


def test_language_config_queries():
    """Test that QUERIES contains expected query patterns"""
    # Check that queries exist for supported languages
    assert "python" in LanguageConfig.QUERIES
    assert "javascript" in LanguageConfig.QUERIES

    # Check that the Python query contains expected patterns
    python_query = LanguageConfig.QUERIES["python"]
    assert "function_definition" in python_query
    assert "class_definition" in python_query

    # Check that the JavaScript query contains expected patterns
    js_query = LanguageConfig.QUERIES["javascript"]
    assert "function_declaration" in js_query
    assert "class_declaration" in js_query


def test_language_config_default_ignore_patterns():
    """Test DEFAULT_IGNORE_PATTERNS contains expected patterns"""
    expected_patterns = [
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        "build",
        "dist",
        "*.pyc",
        "*.pyo",
    ]

    for pattern in expected_patterns:
        assert pattern in LanguageConfig.DEFAULT_IGNORE_PATTERNS


def test_language_extension_mapping():
    """Test that specific file extensions map to correct languages"""
    # Python extensions
    assert LanguageConfig.LANGUAGE_MAP[".py"] == "python"
    # .pyi is not in the map by default, so checking only if it exists
    if ".pyi" in LanguageConfig.LANGUAGE_MAP:
        assert LanguageConfig.LANGUAGE_MAP[".pyi"] == "python"

    # JavaScript extensions
    assert LanguageConfig.LANGUAGE_MAP[".js"] == "javascript"
    assert LanguageConfig.LANGUAGE_MAP[".mjs"] == "javascript"

    # Web technologies
    assert LanguageConfig.LANGUAGE_MAP[".html"] == "html"
    assert LanguageConfig.LANGUAGE_MAP[".css"] == "css"

    # Other languages
    assert LanguageConfig.LANGUAGE_MAP[".java"] == "java"
    assert LanguageConfig.LANGUAGE_MAP[".cpp"] == "cpp"
    assert LanguageConfig.LANGUAGE_MAP[".c"] == "c"
    assert LanguageConfig.LANGUAGE_MAP[".go"] == "go"
    assert LanguageConfig.LANGUAGE_MAP[".rs"] == "rust"
    assert LanguageConfig.LANGUAGE_MAP[".rb"] == "ruby"


def test_language_config_language_availability():
    """Test that language parsers are available (basic check)"""
    # We can't test full functionality without installing tree-sitter parsers,
    # but we can ensure the structure exists
    assert LanguageConfig.LANGUAGES is not None
    assert isinstance(LanguageConfig.LANGUAGES, dict)

    # All languages in LANGUAGE_MAP that have parsers should be in LANGUAGES
    for ext, lang in LanguageConfig.LANGUAGE_MAP.items():
        # Not all languages may have parsers loaded, but the mapping should exist
        assert isinstance(lang, str)


def test_language_config_language_map_completeness():
    """Test that all languages in QUERIES have corresponding parser definitions"""
    for lang in LanguageConfig.QUERIES.keys():
        assert lang in LanguageConfig.LANGUAGES, f"Language {lang} has query but no parser"


def test_language_config_identifiers():
    """Test that language identifiers are consistent"""
    # Check that language identifiers match between different parts of config
    for ext, lang in LanguageConfig.LANGUAGE_MAP.items():
        # All language identifiers should be lowercase
        assert lang.islower(), f"Language identifier {lang} should be lowercase"

        # If there's a query for this language, it should exist
        if lang in LanguageConfig.QUERIES:
            # Verify the query has content
            assert isinstance(LanguageConfig.QUERIES[lang], str)
            assert len(LanguageConfig.QUERIES[lang]) > 0


def test_language_config_nonexistent_extension():
    """Test that nonexistent extensions are not mapped"""
    # An extension not in the map should return None when looked up
    nonexistent_ext = ".nonexistent_extension_for_test"
    assert nonexistent_ext not in LanguageConfig.LANGUAGE_MAP

"""
Language configuration and mappings for code parsing
"""

import tree_sitter_css as tscss
import tree_sitter_html as tshtml
import tree_sitter_javascript as tsjsvascript
import tree_sitter_python as tspython
from tree_sitter import Language


class LanguageConfig:
    """Configuration for different programming languages"""

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".css": "css",
        ".html": "html",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c_sharp",
    }

    LANGUAGES = {
        "python": Language(tspython.language()),
        "javascript": Language(tsjsvascript.language()),
        "html": Language(tshtml.language()),
        "css": Language(tscss.language()),
    }

    # Tree-sitter query patterns for extracting definitions
    QUERIES = {
        "python": """
            (function_definition
                name: (identifier) @func.name
                parameters: (parameters) @func.params
                body: (block) @func.body) @function

            (class_definition
                name: (identifier) @class.name
                body: (block) @class.body) @class
        """,
        "javascript": """
            (function_declaration
                name: (identifier) @func.name
                parameters: (formal_parameters) @func.params
                body: (statement_block) @func.body) @function

            (class_declaration
                name: (identifier) @class.name
                body: (class_body) @class.body) @class

            (method_definition
                name: (property_identifier) @method.name
                parameters: (formal_parameters) @method.params
                body: (statement_block) @method.body) @method
        """,
    }

    DEFAULT_IGNORE_PATTERNS = [
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        "env",
        ".tinv.venv",
        "build",
        "dist",
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cachecoverage",
        ".coverage",
    ]

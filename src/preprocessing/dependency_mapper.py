"""
Dependency analysis and mapping for code files
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Set
from collections import defaultdict

from .language_config import LanguageConfig


class DependencyMapper:
    """Handles dependency analysis and mapping between code elements"""

    def __init__(self):
        self.symbol_tables = {}  # cache for symbol lookups
        self.dependency_graph = defaultdict(set)

    def extract_dependencies(self, code: str, language: str, symbol_index: Optional[dict] = None) -> List[str]:
        """Extract import/dependency statements and referenced symbols"""
        deps = set()

        if language == 'python':
            deps.update(self._extract_python_imports(code))
            deps.update(self._extract_python_symbol_usage(code, symbol_index))

        elif language in ['javascript', 'typescript']:
            deps.update(self._extract_js_imports(code))
            deps.update(self._extract_js_symbol_usage(code, symbol_index))

        elif language == 'html':
            deps.update(self._extract_html_dependencies(code))

        elif language == 'css':
            deps.update(self._extract_css_dependencies(code))

        return sorted(deps)[:10]  # limit for practicality

    def _extract_python_imports(self, code: str) -> Set[str]:
        """Extract Python import statements"""
        deps = set()

        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                # import A, B as C → extract A, B
                parts = line[7:].split(',')
                for part in parts:
                    module_name = part.split()[0].strip()
                    # Get root module (e.g., 'collections.abc' → 'collections')
                    root_module = module_name.split('.')[0]
                    deps.add(root_module)

            elif line.startswith('from '):
                # from X import Y → extract X
                match = re.match(r"from\s+([a-zA-Z_][\w\.]*)\s+import", line)
                if match:
                    root_module = match.group(1).split('.')[0]
                    deps.add(root_module)

        return deps

    def _extract_python_symbol_usage(self, code: str, symbol_index: Optional[dict] = None) -> Set[str]:
        """Extract symbol usage that might reference project symbols"""
        deps = set()

        # Look for CapitalizedWords (likely classes/models)
        symbols = re.findall(r'\b[A-Z][a-zA-Z_]\w*\b', code)

        # Filter out common built-ins/keywords
        common = {'None', 'True', 'False', 'Exception', 'ValueError',
                 'TypeError', 'AttributeError', 'str', 'int', 'list', 'dict',
                 'tuple', 'set', 'self', 'cls', 'super'}

        filtered_symbols = []
        for sym in symbols:
            if sym not in common and not sym.startswith('_'):
                filtered_symbols.append(sym)

        # If we have a symbol index, only include symbols that are defined in the project
        if symbol_index:
            for sym in filtered_symbols:
                if sym in symbol_index:
                    deps.add(sym)
        else:
            # Without symbol index, add all filtered symbols as potential dependencies
            deps.update(filtered_symbols)

        return deps

    def _extract_js_imports(self, code: str) -> Set[str]:
        """Extract JavaScript/TypeScript import statements"""
        deps = set()

        # Handle import X from 'Y' and require('Y')
        import_re = re.compile(r'import\s+(?:[\w{}\*\s,]+\s+from\s+)?[\'"]([^\'"]+)[\'"]')
        require_re = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')

        for line in code.split('\n'):
            line = line.strip()
            # Skip comments
            if line.startswith('//') or line.startswith('/*'):
                continue

            # Extract imports
            for match in import_re.findall(line):
                deps.add(match.split('/')[0])  # e.g., 'lodash/map' → 'lodash'
            for match in require_re.findall(line):
                deps.add(match.split('/')[0])

        return deps

    def _extract_js_symbol_usage(self, code: str, symbol_index: Optional[dict] = None) -> Set[str]:
        """Extract JavaScript symbol usage"""
        deps = set()

        # Simple heuristic: find word characters followed by parentheses (function calls)
        # or capitalized words (classes)
        words = re.findall(r'\b([A-Za-z_]\w*)\b', code)
        calls = re.findall(r'\b([A-Za-z_]\w*)\s*\(', code)

        candidates = set()
        for word in words:
            # Add capitalized words (likely classes/components)
            if word[0].isupper():
                candidates.add(word)
        # Add called functions
        candidates.update(calls)

        # Filter known built-ins
        builtins = {'console', 'window', 'document', 'Object', 'Array', 'String', 'Number',
                   'Boolean', 'Math', 'Date', 'JSON', 'Promise', 'Error'}

        candidates = candidates - builtins

        if symbol_index:
            for sym in candidates:
                if sym in symbol_index:
                    deps.add(sym)
        else:
            deps.update(candidates)

        return deps

    def _extract_html_dependencies(self, code: str) -> Set[str]:
        """Extract dependencies from HTML (scripts, styles)"""
        deps = set()

        # Extract script sources
        script_re = re.compile(r'<script[^>]*src=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)
        for match in script_re.findall(code):
            deps.add(match.split('/')[0])

        # Extract link hrefs (stylesheets)
        link_re = re.compile(r'<link[^>]*href=["\']([^"\']+\.css[^"\']*)[^>]*>', re.IGNORECASE)
        for match in link_re.findall(code):
            deps.add(match.split('/')[0])

        return deps

    def _extract_css_dependencies(self, code: str) -> Set[str]:
        """Extract dependencies from CSS (@imports, url())"""
        deps = set()

        # Extract @import statements
        import_re = re.compile(r'@import\s+["\']([^"\']+)["\']', re.IGNORECASE)
        for match in import_re.findall(code):
            deps.add(match.split('/')[0])

        # Extract url() references
        url_re = re.compile(r'url\(["\']?([^"\']+)["\']?\)', re.IGNORECASE)
        for match in url_re.findall(code):
            if '.' in match:  # likely a file reference
                deps.add(match.split('/')[0])

        return deps

    def build_dependency_graph(self, chunks: List, symbol_index: dict) -> Dict[str, Set[str]]:
        """Build a dependency graph from chunks"""
        graph = defaultdict(set)

        for chunk in chunks:
            chunk_id = chunk.id
            graph[chunk_id] = set()

            # Add dependencies from the chunk's dependency list
            for dep in chunk.dependencies:
                # If dependency is another chunk in our project, add the relationship
                if dep in symbol_index:
                    # Find the chunk(s) that define this symbol
                    for def_chunk in chunks:
                        if def_chunk.name == dep or dep in def_chunk.defines:
                            graph[chunk_id].add(def_chunk.id)

        self.dependency_graph = graph
        return dict(graph)

    def find_reverse_dependencies(self, chunk_id: str) -> Set[str]:
        """Find all chunks that depend on the given chunk"""
        dependents = set()
        for other_id, deps in self.dependency_graph.items():
            if chunk_id in deps:
                dependents.add(other_id)
        return dependents

    def get_dependency_hierarchy(self, chunk_id: str) -> Dict[str, List[str]]:
        """Get dependency hierarchy for a chunk"""
        return {
            'direct_dependencies': list(self.dependency_graph.get(chunk_id, set())),
            'dependents': list(self.find_reverse_dependencies(chunk_id))
        }

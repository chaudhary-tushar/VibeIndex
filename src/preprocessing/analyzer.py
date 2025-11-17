"""
Code analysis and parsing logic for different languages
"""

import re
import libcst as cst
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from tree_sitter import Node

from .chunk import CodeChunk

console = Console()


class Analyzer:
    """Handles code analysis and parsing for different languages"""

    def extract_js_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> List[CodeChunk]:
        """Extracts functions and classes from JavaScript/TypeScript"""
        chunks = []

        def traverse(node, parent_name=None):
            if node.type in ['function_definition', 'function_declaration']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    called_symbols = self.find_called_symbols(code, language, {})

                    chunk = CodeChunk(
                        type='method' if parent_name else 'function',
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=None,  # JS doesn't have standard docstrings like Python
                        signature=code.split('\n')[0],
                        complexity=self._calculate_complexity(code),
                        dependencies=self._extract_dependencies(code, language),
                        parent=parent_name,
                        defines=[name],
                        references=called_symbols
                    )
                    chunks.append(chunk)

            elif node.type in ['class_definition', 'class_declaration']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    called_symbols = self.find_called_symbols(code, language, {})
                    chunk = CodeChunk(
                        type='class',
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=None,
                        complexity=self._calculate_complexity(code),
                        dependencies=self._extract_dependencies(code, language),
                        defines=[name],
                        references=called_symbols
                    )
                    chunks.append(chunk)

                    # Parse methods within class
                    for child in node.children:
                        traverse(child, parent_name=name)
                    return

            for child in node.children:
                traverse(child, parent_name)

        traverse(node)
        return chunks

    def extract_html_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> List[CodeChunk]:
        """Extract meaningful chunks from HTML"""
        chunks = []

        def walk(n):
            if n.type == 'element':
                tag_name_node = n.child_by_field_name('tag_name')
                if tag_name_node:
                    tag = code_bytes[tag_name_node.start_byte:tag_name_node.end_byte].decode('utf8')
                    # Only chunk meaningful containers
                    if tag in {'div', 'section', 'article', 'template', 'main'}:
                        code = code_bytes[n.start_byte:n.end_byte].decode('utf8')
                        if len(code.strip()) > 50:  # avoid tiny tags
                            called_symbols = self.find_called_symbols(code, language, {})
                            chunk = CodeChunk(
                                type='html_element',
                                name=tag,
                                code=code,
                                file_path=relative_path,
                                language=language,
                                start_line=n.start_point[0] + 1,
                                end_line=n.end_point[0] + 1,
                                docstring=None,
                                signature=None,
                                complexity=1,
                                dependencies=self._extract_dependencies(code, language),
                                parent=None,
                                defines=[],
                                references=called_symbols
                            )
                            chunks.append(chunk)
            for child in n.children:
                walk(child)

        walk(node)

        # Optional: if no chunks, add full file as fallback
        if not chunks:
            full_code = code_bytes.decode('utf8')
            if len(full_code.strip()) > 0:
                chunks.append(CodeChunk(
                    type='html_file',
                    name=relative_path,
                    code=full_code,
                    file_path=relative_path,
                    language=language,
                    start_line=1,
                    end_line=full_code.count('\n') + 1,
                    docstring=None,
                    signature=None,
                    complexity=1,
                    dependencies=[],
                    parent=None,
                    context=f"Full HTML template: {relative_path}"
                ))

        return chunks

    def extract_css_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> List[CodeChunk]:
        """Extract chunks from CSS"""
        chunks = []

        def walk(n):
            if n.type == 'rule_set':
                code = code_bytes[n.start_byte:n.end_byte].decode('utf8')
                # Extract selector as name
                selector_node = n.child_by_field_name('selectors')
                name = "unknown_selector"
                if selector_node:
                    name = code_bytes[selector_node.start_byte:selector_node.end_byte].decode('utf8').strip()
                called_symbols = self.find_called_symbols(code, language, {})

                chunk = CodeChunk(
                    type='css_rule',
                    name=name,
                    code=code,
                    file_path=relative_path,
                    language=language,
                    start_line=n.start_point[0] + 1,
                    end_line=n.end_point[0] + 1,
                    docstring=None,
                    signature=None,
                    complexity=1,
                    dependencies=self._extract_dependencies(code, language),
                    parent=None,
                    context=f"CSS rule for '{name}'",
                    defines=[],
                    references=called_symbols
                )
                chunks.append(chunk)
            for child in n.children:
                walk(child)

        walk(node)
        return chunks

    def extract_generic_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> List[CodeChunk]:
        """Fallback chunking for unsupported languages"""
        full_code = code_bytes.decode('utf8', errors='ignore')
        if len(full_code.strip()) == 0:
            return []

        # Create a single chunk for the entire file
        return [CodeChunk(
            type='file',
            name=relative_path.split('/')[-1],
            code=full_code,
            file_path=relative_path,
            language=language,
            start_line=1,
            end_line=full_code.count('\n') + 1,
            docstring=None,
            signature=None,
            complexity=1,
            dependencies=self._extract_dependencies(full_code, language),
            defines=[],
            references=[]
        )]

    def parse_python_file_libcst(self, file_path: Path) -> List[CodeChunk]:
        """Parse Python file using libCST for better accuracy"""
        if file_path.suffix != '.py':
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        try:
            # Use MetadataWrapper to get position info
            from libcst.metadata import PositionProvider
            wrapper = cst.MetadataWrapper(cst.parse_module(code))
            module = wrapper.module
        except Exception as e:
            console.print(f"[yellow]LibCST parse error for {file_path}: {e}[/yellow]")
            return []

        relative_path = str(file_path.relative_to(file_path.parents[3]))  # Go up to project root
        chunks = []

        class FunctionVisitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def __init__(self, analyzer_instance):
                self.chunks = []
                self.current_class = None
                self.analyzer = analyzer_instance

            def visit_ClassDef(self, node: cst.ClassDef):
                self.current_class = node.name.value

            def leave_ClassDef(self, original_node: cst.ClassDef):
                start_line, end_line = self._get_position(original_node)
                code_block = module.code_for_node(original_node)
                called_symbols = self.analyzer.find_called_symbols(code_block, "python", {})

                chunk = CodeChunk(
                    type='class',
                    name=original_node.name.value,
                    code=code_block,
                    file_path=relative_path,
                    language='python',
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=self.analyzer._extract_dependencies(code_block, language="python"),
                    complexity=self._calculate_complexity(code_block),
                    defines=[original_node.name.value],
                    references=called_symbols
                )
                self.chunks.append(chunk)
                self.current_class = None

            def leave_FunctionDef(self, original_node: cst.FunctionDef):
                start_line, end_line = self._get_position(original_node)
                code_block = module.code_for_node(original_node)
                called_symbols = self.analyzer.find_called_symbols(code_block, "python", {})
                chunk = CodeChunk(
                    type='method' if self.current_class else 'function',
                    name=original_node.name.value,
                    code=code_block,
                    file_path=relative_path,
                    language='python',
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=self.analyzer._extract_dependencies(code_block, language="python"),
                    signature=code_block.split('\n')[0],
                    complexity=self._calculate_complexity(code_block),
                    parent=self.current_class,
                    defines=[original_node.name.value],
                    references=called_symbols
                )
                self.chunks.append(chunk)

            def _get_position(self, node):
                pos = self.get_metadata(PositionProvider, node)
                return pos.start.line, pos.end.line

            def _get_docstring(self, node) -> Optional[str]:
                if hasattr(node, 'body') and isinstance(node.body, cst.IndentedBlock):
                    if node.body.body:
                        first = node.body.body[0]
                        if isinstance(first, cst.SimpleStatementLine):
                            for stmt in first.body:
                                if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.BaseString):
                                    # Handle f-strings, raw strings, etc.
                                    value = stmt.value.evaluated_value
                                    if value is not None:
                                        return value.strip()
                return None

            def _calculate_complexity(self, code: str) -> int:
                complexity = 1
                for keyword in ['if ', 'elif ', 'else', 'for ', 'while ', ' and ', ' or ', 'except ', 'case ']:
                    complexity += code.count(keyword)
                return complexity

        visitor = FunctionVisitor(self)
        wrapper.visit(visitor)
        return visitor.chunks

    def find_called_symbols(self, code: str, language: str, symbol_index: dict) -> List[str]:
        """Find symbols in code that are defined in the current project."""
        if not symbol_index:
            return []

        references = set()

        if language == "python":
            import re
            # Match function calls: func(), obj.method(), Class()
            call_pattern = r'\b([a-zA-Z_]\w*)\s*(?:\(|$)'
            attr_pattern = r'\b([a-zA-Z_]\w*)\.\w+'

            candidates = set()
            candidates.update(re.findall(call_pattern, code))
            candidates.update(re.findall(attr_pattern, code))

            for name in candidates:
                if name in symbol_index and name not in {'self', 'cls', 'super'}:
                    references.add(name)

        elif language in ("javascript", "typescript"):
            import re
            words = re.findall(r'\b([A-Za-z_]\w*)\s*\(', code)
            for name in words:
                if name in symbol_index:
                    references.add(name)

        return sorted(references)

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1
        keywords = ['if', 'elif', 'else', 'for', 'while', 'and', 'or', 'catch', 'case']
        for keyword in keywords:
            complexity += code.count(f' {keyword} ') + code.count(f' {keyword}(')
        return complexity

    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        deps = set()  # use set to avoid duplicates

        if language == 'python':
            # Parse imports
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith('import '):
                    # import A, B as C → extract A, B
                    parts = line[7:].split(',')
                    for part in parts:
                        name = part.split()[0].split('.')[0]  # get root module
                        deps.add(name)
                elif line.startswith('from '):
                    # from X import Y → extract X
                    match = re.match(r"from\s+([a-zA-Z_][\w\.]*)\s+import", line)
                    if match:
                        root_module = match.group(1).split('.')[0]
                        deps.add(root_module)

            # Optional: Extract symbol usage (heuristic for models/utils)
            # Look for CapitalizedWords (likely classes/models)
            symbols = re.findall(r'\b[A-Z][a-zA-Z_]\w*\b', code)
            # Filter out common built-ins/keywords
            common = {'None', 'True', 'False', 'self', 'cls', 'Exception', 'str', 'int'}
            for sym in symbols:
                if sym not in common and not sym.startswith('_'):
                    deps.add(sym)

        elif language in ['javascript', 'typescript']:
            # Handle import X from 'Y' and require('Y')
            import_re = re.compile(r'import\s+(?:[\w{}\*\s,]+\s+from\s+)?[\'"]([^\'"]+)[\'"]')
            require_re = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')

            for line in code.split('\n'):
                line = line.strip()
                # Skip comments
                if line.startswith('//') or line.startswith('/*'):
                    continue
                for match in import_re.findall(line):
                    deps.add(match.split('/')[0])  # e.g., 'lodash/map' → 'lodash'
                for match in require_re.findall(line):
                    deps.add(match.split('/')[0])

        return sorted(deps)[:10]  # return list, deduped

"""
Code analysis and parsing logic for different languages
"""

import re
import hashlib
import libcst as cst
from pathlib import Path
from typing import List, Optional, Dict, Any
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

    # ============================================================================
    # ENHANCED METADATA EXTRACTION METHODS (Migrated from enhanced.py)
    # ============================================================================

    def add_location_metadata(self, chunk: CodeChunk, node=None) -> None:
        """Add detailed location info from a Tree-sitter node (from enhanced.py)"""
        if chunk.language == "python":
            chunk.location = {
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "start_column": 0,
                "end_column": 0,
            }
            return

        if node is None:
            chunk.location = {
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "start_column": 0,
                "end_column": 0
            }
            return

        start_point = getattr(node, "start_point", None)
        end_point = getattr(node, "end_point", None)

        if not start_point or not end_point:
            chunk.location = {
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "start_column": 0,
                "end_column": 0
            }
            return

        start_line, start_col = start_point
        end_line, end_col = end_point

        chunk.location = {
            "start_line": start_line + 1,
            "end_line": end_line + 1,
            "start_column": start_col + 1,
            "end_column": end_col + 1
        }

    def add_code_metadata(self, chunk: CodeChunk, node=None, code_bytes: bytes = None) -> None:
        """Extract code-specific metadata (from enhanced.py)"""
        metadata = {}

        if chunk.language == 'python':
            metadata['decorators'] = self._extract_decorators(node, code_bytes) if node else []
            metadata['base_classes'] = self._extract_base_classes(node, code_bytes) if node else []
            metadata['access_modifier'] = self._determine_access_modifier(chunk.code)
            metadata['is_abstract'] = self._is_abstract(chunk.code)
            metadata['is_final'] = self._is_final(chunk.code)

        elif chunk.language in ['javascript', 'typescript']:
            metadata['export_type'] = self._extract_export_type(chunk.code)
            metadata['is_async'] = 'async' in chunk.code

        chunk.metadata.update(metadata)

    def _extract_decorators(self, node, code_bytes: bytes) -> List[str]:
        """Extract decorators from Python AST or CST nodes (from enhanced.py)"""
        decorators = []

        if hasattr(node, "decorators"):
            for deco in node.decorators:
                try:
                    decorators.append(cst.Module([]).code_for_node(deco).strip())
                except Exception:
                    continue
            return decorators

        if not hasattr(node, 'children'):
            return decorators

        for child in node.children:
            if hasattr(child, 'type') and child.type == 'decorator':
                try:
                    deco = code_bytes[child.start_byte:child.end_byte].decode('utf-8').strip()
                    decorators.append(deco)
                except Exception:
                    continue

        return decorators

    def _extract_base_classes(self, node, code_bytes: bytes) -> List[str]:
        """Extract base classes from class definitions (from enhanced.py)"""
        base_classes = []

        if hasattr(node, "bases"):
            for base in node.bases:
                try:
                    base_classes.append(cst.Module([]).code_for_node(base.value).strip())
                except Exception:
                    continue
            return base_classes

        if hasattr(node, 'children'):
            for child in node.children:
                if hasattr(child, 'type') and child.type == 'argument_list':
                    for arg in child.children:
                        if arg.type not in ['(', ')']:
                            try:
                                base_classes.append(code_bytes[arg.start_byte:arg.end_byte].decode('utf-8').strip())
                            except Exception:
                                continue

        return base_classes

    def _determine_access_modifier(self, code: str) -> str:
        """Determine access modifier (from enhanced.py)"""
        if code.startswith('_') and not code.startswith('__'):
            return 'protected'
        elif code.startswith('__'):
            return 'private'
        return 'public'

    def _is_abstract(self, code: str) -> bool:
        """Check if class/method is abstract (from enhanced.py)"""
        return 'abstract' in code or 'ABC' in code

    def _is_final(self, code: str) -> bool:
        """Check if class/method is final (from enhanced.py)"""
        return 'final' in code or '@final' in code

    def _extract_export_type(self, code: str) -> str:
        """Extract export type for JS/TS (from enhanced.py)"""
        if 'export default' in code:
            return 'default'
        elif 'export' in code:
            return 'named'
        return 'none'

    def add_analysis_metadata(self, chunk: CodeChunk) -> None:
        """Add accurate, dynamic analysis metadata (from enhanced.py)"""
        complexity = self._calculate_complexity(chunk.code)
        token_count = self._count_tokens(chunk.code, chunk.language)
        embedding_size = getattr(self, "embedding_size", None) or 768
        semantic_hash = self._generate_semantic_hash(chunk.code)

        start = chunk.location.get("start_line", 1)
        end = chunk.location.get("end_line", start)
        line_count = max(1, end - start + 1)

        chunk.analysis = {
            "complexity": complexity,
            "token_count": token_count,
            "embedding_size": embedding_size,
            "semantic_hash": semantic_hash,
            "line_count": line_count,
        }

    def _count_tokens(self, code: str, language: str) -> int:
        """Estimate or compute token count (from enhanced.py)"""
        if language == "python":
            try:
                import tokenize
                from io import BytesIO
                tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
                return len(tokens)
            except Exception:
                return len(code.split())
        else:
            return len(code.split())

    def _generate_semantic_hash(self, code: str) -> str:
        """Generate a semantic hash (from enhanced.py)"""
        clean_code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        clean_code = re.sub(r'\s+', ' ', clean_code).strip()
        return hashlib.md5(clean_code.encode()).hexdigest()[:10]

    def add_relationship_metadata(self, chunk: CodeChunk, all_chunks: List[CodeChunk] = None) -> None:
        """Add comprehensive relationship metadata (from enhanced.py)"""
        chunk.relationships = {
            "imports": self._extract_imports(chunk.code, chunk.language),
            "dependencies": chunk.dependencies or [],
            "parent": chunk.parent,
            "children": self._find_child_chunks(chunk, all_chunks or []),
            "references": chunk.references or [],
            "called_functions": self._extract_called_functions(chunk.code, chunk.language),
            "defined_symbols": chunk.defines or []
        }

    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements (from enhanced.py)"""
        imports = []
        lines = code.split('\n')

        if language == 'python':
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'from ')):
                    imports.append(line)
        elif language in ['javascript', 'typescript']:
            for line in lines:
                line = line.strip()
                if line.startswith(('import ', 'export ', 'require(')):
                    imports.append(line)

        return imports

    def _extract_called_functions(self, code: str, language: str) -> List[str]:
        """Extract function calls from code (from enhanced.py)"""
        called_functions = []

        if language == 'python':
            pattern = r'(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            matches = re.findall(pattern, code)
            keywords = {'if', 'for', 'while', 'with', 'def', 'class', 'return'}
            called_functions = [m for m in matches if m not in keywords]

        return called_functions

    def _find_child_chunks(self, chunk: CodeChunk, all_chunks: List[CodeChunk]) -> List[str]:
        """Find IDs of child chunks (from enhanced.py)"""
        children = []
        if chunk.type == 'class':
            for c in all_chunks:
                if getattr(c, 'parent', None) == chunk.name and c.file_path == chunk.file_path:
                    children.append(c.id)
        return children

    def add_context_metadata(self, chunk: CodeChunk, file_path: Path, project_path: Path = None) -> None:
        """Add contextual information (from enhanced.py)"""
        chunk.context = {
            "module_context": self._get_module_context(file_path),
            "project_context": self._get_project_context(project_path),
            "file_hierarchy": self._get_file_hierarchy(file_path),
            "domain_context": self._infer_domain_context(chunk, file_path)
        }

    def _get_module_context(self, file_path: Path) -> str:
        """Get module-level context (from enhanced.py)"""
        parent_dir = file_path.parent.name
        if parent_dir:
            return f"{parent_dir} module"
        return "root module"

    def _get_project_context(self, project_path: Path = None) -> str:
        """Infer project context from structure (from enhanced.py)"""
        return "Project codebase"

    def _get_file_hierarchy(self, file_path: Path) -> List[str]:
        """Get file hierarchy as list (from enhanced.py)"""
        return list(file_path.parts)

    def _infer_domain_context(self, chunk: CodeChunk, file_path: Path) -> str:
        """Infer domain context from file path and content (from enhanced.py)"""
        path_str = str(file_path).lower()

        if 'admin' in path_str:
            return "Django admin configuration"
        elif 'model' in path_str or 'models' in path_str:
            return "Data models and schema"
        elif 'view' in path_str or 'controller' in path_str:
            return "Application logic and controllers"
        elif 'test' in path_str:
            return "Testing code"
        elif any(ui in path_str for ui in ['template', 'component', 'ui', 'css']):
            return "User interface"

        return "General application code"

    def enhance_chunk_completely(self, chunk: CodeChunk, node=None, code_bytes: bytes = None, file_path: Path = None, project_path: Path = None, all_chunks: List[CodeChunk] = None) -> None:
        """Full enhancement pipeline for a chunk (from enhanced.py)"""
        self.add_location_metadata(chunk, node)
        self.add_code_metadata(chunk, node, code_bytes)
        self.add_analysis_metadata(chunk)
        if file_path:
            self.add_relationship_metadata(chunk, all_chunks or [])
            self.add_context_metadata(chunk, file_path, project_path)

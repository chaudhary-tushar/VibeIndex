"""
Code Parser & Chunker - Multi-project support with visualization
Supports: Python, JavaScript, TypeScript, Java, C++, Go, Rust

Prerequisites:
pip install tree-sitter tree-sitter-languages libcst rope rich pathspec gitignore-parser

For ctags support:
- Ubuntu/Debian: sudo apt-get install universal-ctags
- macOS: brew install universal-ctags
- Windows: Download from https://github.com/universal-ctags/ctags
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import libcst as cst
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjsvascript
import tree_sitter_html as tshtml
import tree_sitter_css as tscss
from tree_sitter import Language, Parser
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich import print as rprint
import re

console = Console()

PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjsvascript.language())
HTML_LANGUAGE = Language(tshtml.language())
CSS_LANGUAGE = Language(tscss.language())

@dataclass
class CodeChunk:
    """Represents a parsed code chunk"""

    type: str  # 'function', 'class', 'method', 'file'
    name: str
    code: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    id: str = ""
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: int = 0
    dependencies: List[str] = None
    parent: Optional[str] = None
    context: Optional[Any] = None
    references: List[str] = None  # ←←← NEW
    defines: List[str] = None     # ←←← NEW

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.references is None:   # ←←← NEW
            self.references = []
        if self.defines is None:      # ←←← NEW
            self.defines = []
        if not self.id:
            hash_input = f"{self.file_path}:{self.name}:{self.start_line}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

class LanguageConfig:
    """Configuration for different programming languages"""

    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.mjs': 'javascript',
        '.css': 'css',
        '.html': 'html',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'c_sharp',
    }
    LANGUAGES = {
        'python': PY_LANGUAGE,
        'javascript': JS_LANGUAGE,
        'html': HTML_LANGUAGE,
        'css': CSS_LANGUAGE
    }

    # Tree-sitter query patterns for extracting definitions
    QUERIES = {
        'python': """
            (function_definition
                name: (identifier) @func.name
                parameters: (parameters) @func.params
                body: (block) @func.body) @function

            (class_definition
                name: (identifier) @class.name
                body: (block) @class.body) @class
        """,
        'javascript': """
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
        '.git',
        '__pycache__',
        'node_modules',
        'venv',
        'env',
        '.tinv'
        '.venv',
        'build',
        'dist',
        '*.pyc',
        '*.pyo',
        '*.egg-info',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache'
        'coverage',
        '.coverage',
    ]


class CodeParser:
    """Main parser using Tree-sitter for multiple languages"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.parsers = {}
        self.chunks: List[CodeChunk] = []
        self.stats = defaultdict(int)
        self.ignore_spec = self._load_gitignore()
        self.symbol_index = {}

    def _load_gitignore(self) -> Optional[PathSpec]:
        """Load .gitignore patterns"""
        gitignore_path = self.project_path / '.gitignore'
        patterns = LanguageConfig.DEFAULT_IGNORE_PATTERNS.copy()

        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                patterns.extend(line.strip() for line in f if line.strip() and not line.startswith('#'))

        return PathSpec.from_lines(GitWildMatchPattern, patterns)

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        try:
            relative_path = path.relative_to(self.project_path)
            return self.ignore_spec.match_file(str(relative_path))
        except ValueError:
            return True

    def _get_parser(self, language: str):
        """Get or create parser for language"""
        if language not in self.parsers:
            try:
                lang_obj = LanguageConfig.LANGUAGES.get(language)
                parser = Parser(lang_obj)
                # parser.language(lang_obj)
                # self.parsers[language] = get_parser(language)
                self.parsers[language] = parser
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load parser for {language}: {e}[/yellow]")
                return None
        return self.parsers[language]

    def discover_files(self) -> List[Path]:
        """Discover all code files in project"""
        files = []

        for ext, lang in LanguageConfig.LANGUAGE_MAP.items():
            for file_path in self.project_path.rglob(f'*{ext}'):
                if not self._should_ignore(file_path):
                    files.append(file_path)
                    self.stats[f'files_{lang}'] += 1

        return files

    def _extract_docstring(self, node, code_bytes: bytes) -> Optional[str]:
        """Extract docstring from node"""
        if node.type in ['function_definition', 'class_definition']:
            body = node.child_by_field_name('body')
            if body and body.children:
                first_child = body.children[0]
                if first_child.type == 'expression_statement':
                    expr = first_child.children[0]
                    if expr.type == 'string':
                        return code_bytes[expr.start_byte:expr.end_byte].decode('utf-8').strip('"\' ')
        return None

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1
        keywords = ['if', 'elif', 'else', 'for', 'while', 'and', 'or', 'catch', 'case']
        for keyword in keywords:
            complexity += code.count(f' {keyword} ') + code.count(f' {keyword}(')
        return complexity

    def _low_extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract import/dependency statements"""
        deps = []
        lines = code.split('\n')

        if language == 'python':
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    deps.append(line)
        elif language in ['javascript', 'typescript']:
            for line in lines:
                line = line.strip()
                if 'import ' in line or 'require(' in line:
                    deps.append(line)

        return deps[:10]  # Limit to first 10

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

    def extract_html_chunks(self, node, code_bytes, relative_path, language):
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
                            called_symbols = self._find_called_symbols(full_code, language)
                            chunk = CodeChunk(
                                id="",
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
                                dependencies=self._extract_dependencies(code, language) + self._extract_dependencies_from_code(code, relative_path),
                                parent=None,
                                defines=[],               # ←←← NEW
                                references=called_symbols
                                # context=self._generate_context({"type": "html_element", "name": tag, "code": code, "language": language})
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
                    id="",
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

    def extract_css_chunks(self, node, code_bytes, relative_path, language):
        chunks = []

        def walk(n):
            if n.type == 'rule_set':
                code = code_bytes[n.start_byte:n.end_byte].decode('utf8')
                # Extract selector as name
                selector_node = n.child_by_field_name('selectors')
                name = "unknown_selector"
                if selector_node:
                    name = code_bytes[selector_node.start_byte:selector_node.end_byte].decode('utf8').strip()
                called_symbols = self._find_called_symbols(code, language)

                chunk = CodeChunk(
                    id="",
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
                    dependencies=self._extract_dependencies(code, language) + self._extract_dependencies_from_code(code, relative_path),
                    parent=None,
                    context=f"CSS rule for '{name}'",
                    defines=[],               # ←←← NEW
                    references=called_symbols
                )
                chunks.append(chunk)
            for child in n.children:
                walk(child)

        walk(node)
        return chunks

    def extract_js_chunks(self, node, code_bytes, relative_path, language):
        """Extracts functions and classes"""

        chunks = []

        def traverse(node, parent_name=None):
            if node.type in ['function_definition', 'function_declaration']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    called_symbols = self._find_called_symbols(code, language)

                    chunk = CodeChunk(
                        id="",
                        type='method' if parent_name else 'function',
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=self._extract_docstring(node, code_bytes),
                        signature=code.split('\n')[0],
                        complexity=self._calculate_complexity(code),
                        dependencies=self._extract_dependencies(code, language) + self._extract_dependencies_from_code(code, relative_path),
                        parent=parent_name,
                        defines=[name],          # ←←← NEW
                        references=called_symbols
                    )
                    chunks.append(chunk)
                    self.stats[f'functions_{language}'] += 1

            elif node.type in ['class_definition', 'class_declaration']:
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')
                    code = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
                    called_symbols = self._find_called_symbols(code, language)
                    chunk = CodeChunk(
                        id="",
                        type='class',
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=self._extract_docstring(node, code_bytes),
                        complexity=self._calculate_complexity(code),
                        dependencies=self._extract_dependencies(code, language)  + self._extract_dependencies_from_code(code, relative_path),
                        defines=[name],          # ←←← NEW
                        references=called_symbols
                    )
                    chunks.append(chunk)
                    self.stats[f'classes_{language}'] += 1

                    # Parse methods within class
                    for child in node.children:
                        traverse(child, parent_name=name)
                    return

            for child in node.children:
                traverse(child, parent_name)

        traverse(node)

        return chunks

    def _extract_dependencies_from_code(self, code: str, file_path: Path) -> List[str]:
        """Use ctags-style parsing or symbol_index to find referenced symbols"""
        deps = set()

        # Simple: extract all capitalized words (heuristic for classes/models)
        import re
        candidates = re.findall(r'\b[A-Z][a-zA-Z_]\w*\b', code)

        for name in candidates:
            # Is this symbol defined in OUR codebase? (not built-in)
            if name in self.symbol_index:
                deps.add(name)
            # Or is it a known Django symbol? (optional allowlist)
            elif name in {'Model', 'View', 'APIView', 'admin', 'models'}:
                deps.add(name)

        return list(deps)

    def _find_called_symbols(self, code: str, language: str) -> List[str]:
        """Find symbols in `code` that are defined in the current project."""
        if not hasattr(self, 'symbol_index') or not self.symbol_index:
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
                if name in self.symbol_index and name not in {'self', 'cls', 'super'}:
                    references.add(name)

        elif language in ("javascript", "typescript"):
            import re
            words = re.findall(r'\b([A-Za-z_]\w*)\s*\(', code)
            for name in words:
                if name in self.symbol_index:
                    references.add(name)

        return sorted(references)

    def parse_file_treesitter(self, file_path: Path) -> List[CodeChunk]:
        """Parse file using Tree-sitter"""
        ext = file_path.suffix
        language = LanguageConfig.LANGUAGE_MAP.get(ext)
        code_bytes = ""
        tree= ""

        if not language:
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        with open(file_path, 'rb') as f:
            code_bytes = f.read()

        try:
            tree = parser.parse(code_bytes)
        except Exception as e:
            console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            return []

        chunks = []
        relative_path = str(file_path.relative_to(self.project_path))

        if language == "javascript":
            chunks = self.extract_js_chunks(tree.root_node, code_bytes, relative_path, language)
        elif language == "html":
            chunks = self.extract_html_chunks(tree.root_node, code_bytes, relative_path, language)
        elif language == "css":
            chunks = self.extract_css_chunks(tree.root_node, code_bytes, relative_path, language)
        else:
            # Fallback: capture top-level meaningful nodes or full file
            chunks = self.extract_generic_chunks(tree.root_node, code_bytes, relative_path, language)


        return chunks

    def parse_file_libcst(self, file_path: Path) -> List[CodeChunk]:
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
            return self.parse_file_treesitter(file_path)

        relative_path = str(file_path.relative_to(self.project_path))

        class FunctionVisitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def __init__(self, parser):
                self.chunks = []
                self.current_class = None
                self.parser = parser

            def _get_position(self, node):
                pos = self.get_metadata(PositionProvider, node)
                return pos.start.line, pos.end.line

            def visit_ClassDef(self, node: cst.ClassDef):
                self.current_class = node.name.value

            def leave_ClassDef(self, original_node: cst.ClassDef):
                start_line, end_line = self._get_position(original_node)
                code = module.code_for_node(original_node)
                called_symbols = self.parser._find_called_symbols(code, language="python")

                chunk = CodeChunk(
                    id="",
                    type='class',
                    name=original_node.name.value,
                    code=code,
                    file_path=relative_path,
                    language='python',
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=self.parser._extract_dependencies(code, language="python") + self.parser._extract_dependencies_from_code(code, file_path),
                    complexity=self._calc_complexity(code),
                    defines=[original_node.name.value],          # ←←← NEW
                    references=called_symbols
                )
                self.chunks.append(chunk)
                self.current_class = None

            def leave_FunctionDef(self, original_node: cst.FunctionDef):
                start_line, end_line = self._get_position(original_node)
                code = module.code_for_node(original_node)
                called_symbols = self.parser._find_called_symbols(code, language="python")
                chunk = CodeChunk(
                    id="",
                    type='method' if self.current_class else 'function',
                    name=original_node.name.value,
                    code=code,
                    file_path=relative_path,
                    language='python',
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=self.parser._extract_dependencies(code, language="python") + self.parser._extract_dependencies_from_code(code, file_path),
                    signature=code.split('\n')[0],
                    complexity=self._calc_complexity(code),
                    parent=self.current_class,
                    defines=[original_node.name.value],          # ←←← NEW
                    references=called_symbols
                )
                self.chunks.append(chunk)

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

            def _calc_complexity(self, code: str) -> int:
                complexity = 1
                for keyword in ['if ', 'elif ', 'else', 'for ', 'while ', ' and ', ' or ', 'except ', 'case ']:
                    complexity += code.count(keyword)
                return complexity

        visitor = FunctionVisitor(parser=self)
        wrapper.visit(visitor)
        return visitor.chunks

    def run_ctags(self, file_path: Path) -> List[Dict]:
        """Extract symbols using universal-ctags"""
        try:
            result = subprocess.run(
                ['ctags', '-f', '-', '--output-format=json', str(file_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                symbols = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            symbols.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return symbols
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []


    def parse_project(self, use_libcst: bool = True):
        """Parse entire project"""
        console.print(Panel.fit(
            f"[bold cyan]Parsing Project: {self.project_path}[/bold cyan]",
            border_style="cyan"
        ))

        files = self.discover_files()
        console.print(f"[green]Found {len(files)} code files[/green]\n")
        self.symbol_index = {}

        with console.status("[bold green]Parsing files...") as status:
            for i, file_path in enumerate(files, 1):
                status.update(f"[bold green]Parsing {i}/{len(files)}: {file_path.name}")

                # Try libCST for Python, else Tree-sitter
                if use_libcst and file_path.suffix == '.py':
                    chunks = self.parse_file_libcst(file_path)
                else:
                    chunks = self.parse_file_treesitter(file_path)

                self.chunks.extend(chunks)
                ctag_symbols = self.run_ctags(file_path)
                self.stats['ctags_symbols'] += len(ctag_symbols)

                # 🔥 Build global symbol index
                for sym in ctag_symbols:
                    name = sym['name']
                    if name not in self.symbol_index:
                        self.symbol_index[name] = []
                    self.symbol_index[name].append({
                        'file': sym['path'],
                        'line': sym.get('line', sym.get('address', '1')),
                        'kind': sym.get('kind', 'unknown'),
                        'scope': sym.get('scope', None),
                        'path': str(file_path)
                    })

        console.print(f"[bold green]✓ Parsing complete![/bold green]")
        console.print(f"[cyan]Total chunks extracted: {len(self.chunks)}[/cyan]\n")

    def save_results(self, output_path: str = "parsed_chunks.json"):
        """Save parsed chunks to JSON"""
        output_file = Path(output_path)

        data = {
            'project_path': str(self.project_path),
            'total_chunks': len(self.chunks),
            'statistics': dict(self.stats),
            'chunks': [asdict(chunk) for chunk in self.chunks]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]")

    def visualize_results(self):
        """Display beautiful visualization of results"""
        # Summary Statistics
        stats_table = Table(title="📊 Parsing Statistics", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right", style="green")

        stats_table.add_row("Total Chunks", str(len(self.chunks)))
        for key, value in sorted(self.stats.items()):
            stats_table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(stats_table)
        console.print()

        # Language Distribution
        lang_dist = defaultdict(int)
        type_dist = defaultdict(int)

        for chunk in self.chunks:
            lang_dist[chunk.language] += 1
            type_dist[chunk.type] += 1

        dist_table = Table(title="📚 Distribution", show_header=True, header_style="bold blue")
        dist_table.add_column("Language", style="cyan")
        dist_table.add_column("Chunks", justify="right", style="yellow")
        dist_table.add_column("Type", style="magenta")
        dist_table.add_column("Count", justify="right", style="yellow")

        max_rows = max(len(lang_dist), len(type_dist))
        lang_items = list(lang_dist.items())
        type_items = list(type_dist.items())

        for i in range(max_rows):
            lang = lang_items[i] if i < len(lang_items) else ("", "")
            typ = type_items[i] if i < len(type_items) else ("", "")
            dist_table.add_row(lang[0], str(lang[1]) if lang[1] else "", typ[0], str(typ[1]) if typ[1] else "")

        console.print(dist_table)
        console.print()

        #     # Create tree view for each chunk
        #     tree = Tree(f"[bold cyan]{i}. {chunk.name}[/bold cyan] ({chunk.type})")
        #     tree.add(f"[dim]File:[/dim] {chunk.file_path}")
        #     tree.add(f"[dim]Lines:[/dim] {chunk.start_line}-{chunk.end_line} ({chunk.end_line - chunk.start_line + 1} lines)")
        #     tree.add(f"[dim]Language:[/dim] {chunk.language}")
        #     tree.add(f"[dim]Complexity:[/dim] {chunk.complexity}")

        #     if chunk.docstring:
        #         tree.add(f"[dim]Docstring:[/dim] {chunk.docstring[:100]}...")

        #     if chunk.dependencies:
        #         tree.add(f"[dim]Dependencies:[/dim] {len(chunk.dependencies)} imports")

        #     if chunk.parent:
        #         tree.add(f"[dim]Parent:[/dim] {chunk.parent}")

        #     console.print(tree)

        #     # Show code preview (first 10 lines)
        #     code_preview = '\n'.join(chunk.code.split('\n')[:10])
        #     syntax = Syntax(code_preview, chunk.language, theme="monokai", line_numbers=True)
        #     console.print(Panel(syntax, title=f"Code Preview", border_style="green", padding=(0, 1)))
        #     console.print()



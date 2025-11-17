"""
Main parser orchestration using Tree-sitter for multiple languages
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from rich.console import Console
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from tree_sitter import Parser

from .chunk import CodeChunk
from .language_config import LanguageConfig
from .analyzer import Analyzer
from .metadata_extractor import MetadataExtractor
from .dependency_mapper import DependencyMapper

console = Console()


class CodeParser:
    """Main parser using Tree-sitter for multiple languages"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.parsers = {}
        self.chunks: List[CodeChunk] = []
        self.stats = defaultdict(int)
        self.ignore_spec = self._load_gitignore()
        self.symbol_index = {}
        self.analyzer = Analyzer()
        self.metadata_extractor = MetadataExtractor()
        self.dependency_mapper = DependencyMapper()

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
                if lang_obj:
                    parser = Parser()
                    parser.language = lang_obj  # Use property instead of set_language
                    self.parsers[language] = parser
                else:
                    console.print(f"[yellow]Warning: Language {language} not supported[/yellow]")
                    return None
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

    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file using the appropriate method"""
        ext = file_path.suffix
        language = LanguageConfig.LANGUAGE_MAP.get(ext)
        code_bytes = ""

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

        if language == "python":
            # Use specialized Python parsing with libCST for better accuracy
            chunks = self.analyzer.parse_python_file_libcst(file_path)
        elif language == "javascript":
            chunks = self.analyzer.extract_js_chunks(tree.root_node, code_bytes, relative_path, language)
        elif language == "html":
            chunks = self.analyzer.extract_html_chunks(tree.root_node, code_bytes, relative_path, language)
        elif language == "css":
            chunks = self.analyzer.extract_css_chunks(tree.root_node, code_bytes, relative_path, language)
        else:
            # Fallback for other languages using Tree-sitter
            chunks = self.analyzer.extract_generic_chunks(tree.root_node, code_bytes, relative_path, language)

        # Enhance all chunks with additional metadata
        for chunk in chunks:
            # Add dependencies and references if not already present
            if not chunk.dependencies:
                chunk.dependencies = self.dependency_mapper.extract_dependencies(chunk.code, language)

            # Extract references
            if not chunk.references:
                chunk.references = self.analyzer.find_called_symbols(chunk.code, language, self.symbol_index)

        return chunks

    def run_ctags(self, file_path: Path) -> List[Dict]:
        """Extract symbols using universal-ctags"""
        try:
            import subprocess
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

    def parse_project(self):
        """Parse entire project"""
        console.print(f"Parsing Project: {self.project_path}")

        files = self.discover_files()
        console.print(f"[green]Found {len(files)} code files[/green]\n")

        self.symbol_index = {}

        with console.status("[bold green]Parsing files...") as status:
            for i, file_path in enumerate(files, 1):
                relative_path = str(file_path.relative_to(self.project_path))
                status.update(f"[bold green]Parsing {i}/{len(files)}: {relative_path}")

                # Parse the file
                chunks = self.parse_file(file_path)
                self.chunks.extend(chunks)

                # Build symbol index for cross-references
                ctag_symbols = self.run_ctags(file_path)
                self.stats['ctags_symbols'] += len(ctag_symbols)

                for sym in ctag_symbols:
                    name = sym['name']
                    if name not in self.symbol_index:
                        self.symbol_index[name] = []
                    self.symbol_index[name].append({
                        'file': relative_path,
                        'line': sym.get('line', sym.get('address', '1')),
                        'kind': sym.get('kind', 'unknown'),
                        'scope': sym.get('scope', None),
                        'path': str(file_path)
                    })

        console.print("✓ Parsing complete!")
        console.print(f"[cyan]Total chunks extracted: {len(self.chunks)}[/cyan]\n")

    def save_results(self, output_path: str = "parsed_chunks.json"):
        """Save parsed chunks to JSON"""
        output_file = Path(output_path)

        data = {
            'project_path': str(self.project_path),
            'total_chunks': len(self.chunks),
            'statistics': dict(self.stats),
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]")

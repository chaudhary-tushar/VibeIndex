"""
Main parser orchestration using Tree-sitter for multiple languages
Enhanced with metadata enrichment and visualization from enhanced.py
"""

import json
from collections import defaultdict
from pathlib import Path

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from rich.console import Console
from rich.table import Table
from tree_sitter import Parser
from tree_sitter import Tree

from src.config.data_store import DATA_DIR
from src.config.data_store import save_data

from .analyzer import Analyzer
from .chunk import CodeChunk
from .dependency_mapper import DependencyMapper
from .language_config import LanguageConfig
from .metadata_extractor import MetadataExtractor

console = Console()


class CodeParser:
    """Main parser using Tree-sitter for multiple languages"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.parsers = {}
        self.chunks: list[CodeChunk] = []
        self.stats = defaultdict(int)
        self.ignore_spec = self._load_gitignore()
        self.symbol_index = {}
        self.analyzer = Analyzer()
        self.metadata_extractor = MetadataExtractor()
        self.dependency_mapper = DependencyMapper()

    def _load_gitignore(self) -> PathSpec | None:
        """Load .gitignore patterns"""
        gitignore_path = self.project_path / ".gitignore"
        patterns = LanguageConfig.DEFAULT_IGNORE_PATTERNS.copy()

        if gitignore_path.exists():
            with Path(gitignore_path).open(encoding="utf-8") as f:
                patterns.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))

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
            except OSError as e:
                console.print(f"[yellow]Warning: Could not load parser for {language}: {e}[/yellow]")
                return None
        return self.parsers[language]

    def discover_files(self) -> list[Path]:
        """Discover all code files in project"""
        files = []
        # TODO Add metadata to the filpath list like done by SimpleDirectory_reader.
        for ext, lang in LanguageConfig.LANGUAGE_MAP.items():
            for file_path in self.project_path.rglob(f"*{ext}"):
                if not self._should_ignore(file_path):
                    files.append(file_path)
                    self.stats[f"files_{lang}"] += 1
        save_data(files, method="collecting-reading", ext="txt")
        return files

    # def discover_files(self) -> list[Path]:
    #     """Discover all code files in project"""
    #     files1 = SimpleDirectoryReader(
    #         input_dir=self.project_path,
    #         recursive=True,
    #         )
    #     files = files1.load_data()
    #     print(type(files[0]))
    #     res = files1.list_resources_with_info()
    #     save_data(files, method="collecting-reading")
    #     return files

    @staticmethod
    def _determine_language(file_path: Path) -> tuple[str | None, bytes]:
        """Determine language and read file content"""
        ext = file_path.suffix
        language = LanguageConfig.LANGUAGE_MAP.get(ext)

        if not language:
            return None, b""

        with Path(file_path).open("rb") as f:
            code_bytes = f.read()

        return language, code_bytes

    @staticmethod
    def _parse_with_tree_sitter(parser, code_bytes: bytes, file_path: Path) -> list[CodeChunk]:
        """Parse file content using Tree-sitter"""
        try:
            tree = parser.parse(code_bytes)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            return []

        return tree

    def _get_language_chunks(
        self, tree, code_bytes: bytes, relative_path: str, language: str, file_path: Path
    ) -> list[CodeChunk]:
        """Get chunks based on language"""
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
        return chunks

    def _enhance_chunks(self, chunks: list[CodeChunk], language: str, file_path: Path):
        """Enhance all chunks with additional metadata"""
        for chunk in chunks:
            # Add dependencies and references if not already present
            if not chunk.dependencies:
                chunk.dependencies = self.dependency_mapper.extract_dependencies(chunk.code, language)

            # Extract references
            if not chunk.references:
                chunk.references = self.analyzer.find_called_symbols(chunk.code, language, self.symbol_index)

            # ENHANCED: Add comprehensive metadata (from enhanced.py)
            self.analyzer.enhance_chunk_completely(
                chunk,
                node=None,
                code_bytes=None,
                file_path=file_path,
                project_path=self.project_path,
                all_chunks=self.chunks,
            )

    def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a single file using the appropriate method"""
        language, code_bytes = self._determine_language(file_path)

        if not language:
            return []

        parser = self._get_parser(language)
        if not parser:
            return []

        tree: Tree = self._parse_with_tree_sitter(parser, code_bytes, file_path)
        save_data(tree.root_node, language=language, method="tree_sitter_parsing")

        if isinstance(tree, list):  # Error occurred
            return tree

        relative_path = str(file_path.relative_to(self.project_path))
        chunks = self._get_language_chunks(tree, code_bytes, relative_path, language, file_path)

        # Enhance all chunks with additional metadata
        self._enhance_chunks(chunks, language, file_path)

        return chunks

    @staticmethod
    def run_ctags(file_path: Path) -> list[dict]:
        """Extract symbols using universal-ctags"""
        try:
            import subprocess  # noqa: S404

            result = subprocess.run(  # noqa: S603
                ["ctags", "-f", "-", "--output-format=json", str(file_path)],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                symbols = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            symbols.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            console.print(f"[yellow]Warning: Could not decode JSON line: {e}[/yellow]")
                            continue
                return symbols
        except subprocess.TimeoutExpired:
            console.print("[red]Error: ctags command timed out[/red]")
            return []
        except FileNotFoundError:
            console.print("[red]Error: ctags command not found. Please install universal-ctags.[/red]")
            return []
        return []

    def save_results(self):
        """Save parsed chunks to JSON"""
        output_path = DATA_DIR / self.project_path.name
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "chunks.json"

        data = {
            "project_path": str(self.project_path),
            "total_chunks": len(self.chunks),
            "statistics": dict(self.stats),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

        with Path(output_file).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]")

    def visualize_results(self):
        """Display beautiful visualization of results (from enhanced.py)"""
        # Summary Statistics
        stats_table = Table(title="📊 Parsing Statistics", show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right", style="green")

        stats_table.add_row("Total Chunks", str(len(self.chunks)))
        for key, value in sorted(self.stats.items()):
            stats_table.add_row(key.replace("_", " ").title(), str(value))

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

    def save_symbol_index(self):
        """Save symbol index to JSON (from enhanced.py)"""
        output_path = DATA_DIR / self.project_path.name
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "symbol_index.json"
        with Path(output_file).open("w", encoding="utf-8") as f:
            json.dump(self.symbol_index, f, indent=4)
        console.print(f"[green]✓ Symbol index saved to: {output_file}[/green]")

    def parse_project(self):
        """Parse entire project"""
        console.print(f"Parsing Project: {self.project_path}")

        files = self.discover_files()
        console.print(f"[green]Found {len(files)} code files[/green]\n")
        # sys.exit()

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
                self.stats["ctags_symbols"] += len(ctag_symbols)

                for sym in ctag_symbols:
                    name = sym["name"]
                    if name not in self.symbol_index:
                        self.symbol_index[name] = []
                    self.symbol_index[name].append({
                        "file": relative_path,
                        "line": sym.get("line", sym.get("address", "1")),
                        "kind": sym.get("kind", "unknown"),
                        "scope": sym.get("scope", None),
                        "path": str(file_path),
                    })
        self.save_results()
        self.save_symbol_index()
        console.print("✓ Parsing complete!")
        console.print(f"[cyan]Total chunks extracted: {len(self.chunks)}[/cyan]\n")
        self.visualize_results()

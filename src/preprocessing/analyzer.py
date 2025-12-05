"""
Code analysis and parsing logic for different languages
"""

import builtins
import contextlib
import hashlib
import re
from pathlib import Path

import libcst as cst
from radon.visitors import ComplexityVisitor
from rich.console import Console
from tree_sitter import Node

from src.config import settings
from src.config.data_store import save_data

from .chunk import CodeChunk

console = Console()

TINY_TAGS = 50


class Analyzer:
    """Handles code analysis and parsing for different languages"""

    def extract_js_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> list[CodeChunk]:  # noqa: C901, PLR0915
        """Extracts functions, classes, and other significant chunks from JavaScript/TypeScript"""
        chunks = []

        def traverse(node: Node, parent_name=None):  # noqa: C901, PLR0912, PLR0915
            # Handle function declarations and expressions
            if node.type in {  # noqa: PLR1702
                "function_declaration",
                "function_expression",
                "generator_function",
                "generator_function_declaration",
            }:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                    code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                    called_symbols = self.find_called_symbols(code, language, {})

                    chunk = CodeChunk(
                        type="method" if parent_name else "function",
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=None,  # JS doesn't have standard docstrings like Python
                        signature=code.split("\n")[0],
                        complexity=self._calculate_complexity(code, language),
                        dependencies=self._extract_dependencies(code, language),
                        parent=parent_name,
                        defines=[name],
                        references=called_symbols,
                    )
                    chunks.append(chunk)
                else:
                    # Handle anonymous functions
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                    called_symbols = self.find_called_symbols(code, language, {})

                    chunk = CodeChunk(
                        type="anonymous_function",
                        name=f"anonymous_func_{start_line}_{end_line}",
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=start_line,
                        end_line=end_line,
                        docstring=None,
                        signature=code.split("\n")[0],
                        complexity=self._calculate_complexity(code, language),
                        dependencies=self._extract_dependencies(code, language),
                        parent=parent_name,
                        defines=[],
                        references=called_symbols,
                    )
                    chunks.append(chunk)

            # Handle class declarations and expressions
            elif node.type in {"class", "class_declaration", "class_expression"}:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                    code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                    called_symbols = self.find_called_symbols(code, language, {})
                    chunk = CodeChunk(
                        type="class",
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=None,
                        complexity=self._calculate_complexity(code, language),
                        dependencies=self._extract_dependencies(code, language),
                        defines=[name],
                        references=called_symbols,
                    )
                    chunks.append(chunk)

                    # Parse methods within class (no return here)
                    for child in node.children:
                        traverse(child, parent_name=name)

            # Handle method definitions (including getters/setters)
            elif node.type in {"method_definition", "public_field_definition", "private_field_definition"}:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                    code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                    called_symbols = self.find_called_symbols(code, language, {})

                    chunk_type = "method"
                    if node.type in {"public_field_definition", "private_field_definition"}:
                        chunk_type = "property"
                    elif name.startswith("get "):
                        chunk_type = "getter"
                    elif name.startswith("set "):
                        chunk_type = "setter"

                    chunk = CodeChunk(
                        type=chunk_type,
                        name=name,
                        code=code,
                        file_path=relative_path,
                        language=language,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        docstring=None,
                        signature=code.split("\n")[0],
                        complexity=self._calculate_complexity(code, language),
                        dependencies=self._extract_dependencies(code, language),
                        parent=parent_name,  # Parent will be set by the recursive call
                        defines=[name],
                        references=called_symbols,
                    )
                    chunks.append(chunk)

            # Handle variable declarations (for important constants/objects)
            elif node.type in {"variable_declaration", "lexical_declaration"}:
                # Process each declarator in the declaration
                for child in node.children:
                    if child.type in {"variable_declarator"}:
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            name = code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                            value_node = child.child_by_field_name("value")
                            if value_node:
                                code = code_bytes[value_node.start_byte : value_node.end_byte].decode("utf-8")
                                # Only chunk objects, functions, and arrays that are significant
                                if value_node.type in {"object", "array", "arrow_function", "function", "class"}:
                                    called_symbols = self.find_called_symbols(code, language, {})
                                    chunk = CodeChunk(
                                        type="variable",
                                        name=name,
                                        code=code,
                                        file_path=relative_path,
                                        language=language,
                                        start_line=value_node.start_point[0] + 1,
                                        end_line=value_node.end_point[0] + 1,
                                        docstring=None,
                                        signature=f"const {name} = ...",
                                        complexity=self._calculate_complexity(code, language),
                                        dependencies=self._extract_dependencies(code, language),
                                        parent=parent_name,
                                        defines=[name],
                                        references=called_symbols,
                                    )
                                    chunks.append(chunk)

            # Handle import statements
            elif node.type in {"import_statement", "import_declaration"}:
                code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                chunk = CodeChunk(
                    type="import",
                    name=code.strip(),
                    code=code,
                    file_path=relative_path,
                    language=language,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=None,
                    signature=code.split("\n")[0],
                    complexity=0,  # Imports don't add complexity
                    dependencies=[],
                    parent=parent_name,
                    defines=[],  # Imports don't define new symbols in the same file
                    references=[],  # References are resolved during indexing
                )
                chunks.append(chunk)

            # Handle export statements
            elif node.type in {"export_statement", "export_declaration"}:
                code = code_bytes[node.start_byte : node.end_byte].decode("utf-8")
                chunk = CodeChunk(
                    type="export",
                    name=code.strip(),
                    code=code,
                    file_path=relative_path,
                    language=language,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=None,
                    signature=code.split("\n")[0],
                    complexity=0,
                    dependencies=[],
                    parent=parent_name,
                    defines=[],  # Exports don't define new symbols
                    references=[],  # References are resolved during indexing
                )
                chunks.append(chunk)

            for child in node.children:
                traverse(child, parent_name)

        traverse(node)
        return chunks

    def extract_html_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> list[CodeChunk]:
        """Extract meaningful chunks from HTML"""
        chunks = []

        def walk(n):
            # TODO find if tree-sitter-html even outputs .child_by_field_name method
            if n.type == "element":
                tag_name_node = n.child_by_field_name("tag_name")
                if tag_name_node:
                    tag = code_bytes[tag_name_node.start_byte : tag_name_node.end_byte].decode("utf8")
                    # Only chunk meaningful containers
                    if tag in {"div", "section", "article", "template", "main"}:
                        code = code_bytes[n.start_byte : n.end_byte].decode("utf8")
                        if len(code.strip()) > TINY_TAGS:  # avoid tiny tags
                            called_symbols = self.find_called_symbols(code, language, {})
                            chunk = CodeChunk(
                                type="html_element",
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
                                references=called_symbols,
                            )
                            chunks.append(chunk)
            for child in n.children:
                walk(child)

        walk(node)

        # Optional: if no chunks, add full file as fallback
        if not chunks:
            full_code = code_bytes.decode("utf8")
            if len(full_code.strip()) > 0:
                chunks.append(
                    CodeChunk(
                        type="html_file",
                        name=relative_path,
                        code=full_code,
                        file_path=relative_path,
                        language=language,
                        start_line=1,
                        end_line=full_code.count("\n") + 1,
                        docstring=None,
                        signature=None,
                        complexity=1,
                        dependencies=[],
                        parent=None,
                        context=f"Full HTML template: {relative_path}",
                    )
                )

        return chunks

    def extract_css_chunks(self, node: Node, code_bytes: bytes, relative_path: str, language: str) -> list[CodeChunk]:
        """Extract chunks from CSS"""
        chunks = []

        def walk(n: Node):
            if n.type in {"rule_set", "at_rule", "keyframes_statement"}:
                code = code_bytes[n.start_byte : n.end_byte].decode("utf8")
                # Extract selector as name
                selector_node = ""
                for child in n.children:
                    if child.type in {"selectors", "at_keyword"}:
                        selector_node = child
                name = "unknown_selector"
                if selector_node:
                    name = code_bytes[selector_node.start_byte : selector_node.end_byte].decode("utf8").strip()
                called_symbols = self.find_called_symbols(code, language, {})

                chunk = CodeChunk(
                    type="css_rule",
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
                    references=called_symbols,
                )
                chunks.append(chunk)
            for child in n.children:
                walk(child)

        walk(node)
        return chunks

    def extract_generic_chunks(
        self, node: Node, code_bytes: bytes, relative_path: str, language: str
    ) -> list[CodeChunk]:
        """Fallback chunking for unsupported languages"""
        full_code = code_bytes.decode("utf8", errors="ignore")
        if len(full_code.strip()) == 0:
            return []

        # Create a single chunk for the entire file
        return [
            CodeChunk(
                type="file",
                name=relative_path.rsplit("/", maxsplit=1)[-1],
                code=full_code,
                file_path=relative_path,
                language=language,
                start_line=1,
                end_line=full_code.count("\n") + 1,
                docstring=None,
                signature=None,
                complexity=1,
                dependencies=self._extract_dependencies(full_code, language),
                defines=[],
                references=[],
            )
        ]

    def parse_python_file_libcst(self, file_path: Path, symbol_index: dict) -> list[CodeChunk]:  # noqa: C901
        """Parse Python file using libCST for better accuracy"""
        if file_path.suffix != ".py":
            return []

        with Path(file_path).open(encoding="utf-8") as f:
            code = f.read()

        try:
            # Use MetadataWrapper to get position info
            from libcst.metadata import PositionProvider

            wrapper = cst.MetadataWrapper(cst.parse_module(code))
            module = wrapper.module

        except Exception as e:
            console.print(f"[yellow]LibCST parse error for {file_path}: {e}[/yellow]")
            return []

        # First, collect all imports in the file
        class ImportVisitor(cst.CSTVisitor):
            def __init__(self):
                self.imports = {}  # Maps local name to module source
                self.from_imports = {}  # Maps local name to (module, attribute) for "from X import Y"
                self.django_model_imports = {}  # Django-specific: maps local names to model classes

            def visit_Import(self, node: cst.Import) -> None:
                for alias in node.names:
                    # Handle "import module" or "import module as name"
                    local_name = alias.asname.value if alias.asname else alias.name.value
                    if isinstance(alias.name, cst.Name):
                        module_name = alias.name.value
                    elif isinstance(alias.name, cst.Attribute):
                        module_name = code_for_node(alias.name)
                    else:
                        module_name = code_for_node(alias.name)
                    self.imports[local_name] = module_name
                    # Check for Django models import
                    if module_name in {"django.db.models", "django.core.models"}:
                        self.django_model_imports[local_name] = module_name

            def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
                # Get the module name being imported from
                if isinstance(node.module, cst.Name):
                    module_name = node.module.value
                elif isinstance(node.module, cst.Attribute):  # e.g., django.contrib.gis
                    module_name = code_for_node(node.module)
                elif isinstance(node.module, cst.RelativeImport):  # relative imports like "from .models import X"
                    # For relative imports, we'll store the relative path
                    module_name = "." * len(node.module.dots) + (node.module.module.value if node.module.module else "")
                else:
                    return

                # Process each imported name
                for alias in node.names:
                    local_name = alias.asname.value if alias.asname else alias.name.value
                    if isinstance(alias.name, cst.Name):
                        # Handle "from module import name"
                        self.from_imports[local_name] = (module_name, alias.name.value)
                        # Check for Django-specific imports
                        imported_name = alias.name.value
                        if (
                            module_name in {"django.db", "django.db.models", "django.core.models"}
                            and imported_name == "models"
                        ):
                            self.django_model_imports[local_name] = f"{module_name}.{imported_name}"
                    elif isinstance(alias.name, cst.Attribute):
                        # Handle complex imports like "from module import attr.subattr"
                        attr_name = code_for_node(alias.name)
                        self.from_imports[local_name] = (module_name, attr_name)

        # Helper function to extract code for a node
        def code_for_node(node):
            """Extract the code representation of a node."""
            try:
                return module.code_for_node(node)
            except:
                # Fallback for cases where code_for_node fails
                if hasattr(node, "value") and node.value:
                    return node.value
                return str(node)

        import_visitor = ImportVisitor()
        wrapper.visit(import_visitor)

        relative_path = str(file_path.relative_to(settings.project_path))  # Go up to project root

        class FunctionVisitor(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def __init__(
                self,
                analyzer_instance: Analyzer,
                file_imports: dict,
                file_from_imports: dict,
                django_model_imports: dict,
            ):
                self.chunks = []
                self.current_class = None
                self.current_function = None
                self.analyzer = analyzer_instance
                self.file_imports = file_imports
                self.file_from_imports = file_from_imports
                self.django_model_imports = django_model_imports
                self.current_function_imports = set()  # Track imports used in current function
                self.current_class_imports = set()  # Track imports used in current class
                self.django_model_relationships = {}  # Track Django model relationships within this class
                self.current_class_inheritance = []  # Track class inheritance
                self.django_meta_classes = {}  # Track Django Meta classes
                self.django_model_managers = {}  # Track Django Model Managers
                self.django_model_metadata = {}  # Track Django model metadata (like fields, options)
                self.current_class_decorators = []  # Track decorators for the current class
                self.current_class_type = None  # Track the current class type

            def visit_ClassDef(self, node: cst.ClassDef):  # noqa: C901, N802, PLR0912, PLR0915
                class_name = node.name.value
                if self.current_class is None:
                    # This is a top-level class
                    self.current_class = class_name
                    # Store decorators for the top-level class
                    self.current_class_decorators = []
                    if hasattr(node, "decorators") and node.decorators:
                        for deco in node.decorators:
                            try:
                                deco_code = module.code_for_node(deco.decorator)
                                self.current_class_decorators.append(deco_code)
                            except:
                                continue
                else:
                    # This is a nested class
                    class_name = f"{self.current_class}.{class_name}"

                # Check if this is a Django Meta class
                if class_name.endswith(".Meta"):
                    if self.current_class:
                        # Track this as a Meta class for the parent class
                        self.django_meta_classes[self.current_class] = node
                        # Process the content of the Meta class for Django-specific attributes
                        self._process_django_meta_class(node, class_name)
                elif any(
                    base_class in self.django_model_imports
                    or any(django_import in base_class for django_import in self.django_model_imports)
                    for base_class in [
                        code_for_node(base.value) if hasattr(base.value, "value") else base.value.value
                        for base in node.bases or []
                    ]
                ):
                    # This is a Django model class
                    self.current_class = class_name
                    # Store decorators for this Django model class
                    self.current_class_decorators = []
                    if hasattr(node, "decorators") and node.decorators:
                        for deco in node.decorators:
                            try:
                                deco_code = module.code_for_node(deco.decorator)
                                self.current_class_decorators.append(deco_code)
                            except:
                                continue
                    # Track class inheritance
                    self.current_class_inheritance = []
                    if node.bases:
                        for base in node.bases:
                            if isinstance(base.value, cst.Name):
                                self.current_class_inheritance.append(base.value.value)
                            elif isinstance(base.value, cst.Attribute):
                                # Handle attribute inheritance like models.Model
                                self.current_class_inheritance.append(code_for_node(base.value))

                    # Extract Django model-specific metadata
                    self.django_model_metadata = {"model_fields": [], "meta_options": {}, "managers": []}

                # Check if this is a Django admin class
                elif any(
                    "admin" in base_class.lower()
                    or "Admin" in base_class
                    or base_class in {"admin.ModelAdmin", "admin.StackedInline", "admin.TabularInline"}
                    for base_class in [
                        code_for_node(base.value) if hasattr(base.value, "value") else base.value.value
                        for base in node.bases or []
                    ]
                ):
                    # This is a Django admin class
                    self.current_class = class_name
                    self.current_class_type = "django_admin"
                    # Track admin-specific attributes
                    self.django_admin_attributes = {}
                    self.django_admin_metadata = {
                        "list_display": [],
                        "list_filter": [],
                        "search_fields": [],
                        "readonly_fields": [],
                    }
                    # Extract admin-specific metadata from the class body
                    self._process_django_admin_class(node)

                # Check if this is a Django form class
                elif any(
                    "Form" in base_class or "form" in base_class.lower()
                    for base_class in [
                        code_for_node(base.value) if hasattr(base.value, "value") else base.value.value
                        for base in node.bases or []
                    ]
                ):
                    # This is a Django form class
                    self.current_class = class_name
                    self.current_class_type = "django_form"
                    # Track form-specific attributes
                    self.django_form_fields = {}
                    self.django_form_metadata = {"form_fields": [], "widgets": []}
                    # Extract form-specific metadata
                    self._process_django_form_class(node)

                # Check if this is a Django view class
                elif any(
                    "View" in base_class or "view" in base_class.lower()
                    for base_class in [
                        code_for_node(base.value) if hasattr(base.value, "value") else base.value.value
                        for base in node.bases or []
                    ]
                ):
                    # This is a Django view class
                    self.current_class = class_name
                    self.current_class_type = "django_view"
                    # Track view-specific attributes
                    self.django_view_methods = []
                    self.django_view_metadata = {"http_methods": [], "decorators": []}
                    # Extract view-specific metadata
                    self._process_django_view_class(node)

                # Handle other Django class types like managers
                elif any(
                    "Manager" in base_class
                    for base_class in [
                        code_for_node(base.value) if hasattr(base.value, "value") else base.value.value
                        for base in node.bases or []
                    ]
                ):
                    # This is a Django manager class
                    self.current_class = class_name
                    self.current_class_type = "django_manager"
                    self.django_manager_metadata = {"custom_methods": []}
                    # Process manager-specific metadata
                    self._process_django_manager_class(node)

                # Handle Django model classes with more detailed field detection
                if class_name not in self.django_model_imports and any(
                    django_import in base_class.lower()
                    for django_import in ["models", "model"]
                    for base_class in self.current_class_inheritance
                ):
                    # Process model fields in the class body
                    self._process_django_model_fields(node)

            def _process_django_model_fields(self, node: cst.ClassDef):  # noqa: C901
                """Process Django model fields in the class definition"""
                if not node.body:
                    return

                model_fields = []
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:  # noqa: PLR1702
                    if isinstance(stmt, cst.SimpleStatementLine):
                        for expr in stmt.body:
                            if isinstance(expr, cst.Assign):
                                for target in expr.targets:
                                    if isinstance(target.target, cst.Name):
                                        field_name = target.target.value
                                        # Check if the assignment is a field definition like CharField, IntegerField, etc.
                                        if isinstance(expr.value, cst.Call) and isinstance(expr.value.func, cst.Name):
                                            field_type = expr.value.func.value
                                            if field_type in {
                                                "CharField",
                                                "IntegerField",
                                                "TextField",
                                                "EmailField",
                                                "ForeignKey",
                                                "ManyToManyField",
                                                "OneToOneField",
                                                "DateTimeField",
                                                "DateField",
                                                "BooleanField",
                                                "FloatField",
                                            }:
                                                # This is a Django model field
                                                model_fields.append({
                                                    "name": field_name,
                                                    "type": field_type,
                                                    "parameters": self._extract_call_arguments(expr.value),
                                                })
                                        # Check for attribute-based field definitions like models.CharField
                                        elif (
                                            isinstance(expr.value, cst.Call)
                                            and isinstance(expr.value.func, cst.Attribute)
                                            and isinstance(expr.value.func.value, cst.Name)
                                        ):
                                            base_name = expr.value.func.value.value
                                            field_type = expr.value.func.attr.value
                                            if base_name in self.django_model_imports and field_type in {
                                                "CharField",
                                                "IntegerField",
                                                "TextField",
                                                "EmailField",
                                                "ForeignKey",
                                                "ManyToManyField",
                                                "OneToOneField",
                                                "DateTimeField",
                                                "DateField",
                                                "BooleanField",
                                                "FloatField",
                                            }:
                                                # This is a Django model field using models.CharField format
                                                model_fields.append({
                                                    "name": field_name,
                                                    "type": field_type,
                                                    "base": base_name,
                                                    "parameters": self._extract_call_arguments(expr.value),
                                                })

                # Store the detected model fields in the current class relationships
                if self.current_class:
                    self.django_model_relationships[self.current_class] = model_fields

            def _process_django_admin_class(self, node: cst.ClassDef):
                """Process Django admin class attributes"""
                if not node.body:
                    return

                admin_attrs = {}
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:
                    if isinstance(stmt, cst.SimpleStatementLine):
                        for expr in stmt.body:
                            if isinstance(expr, cst.Assign):
                                for target in expr.targets:
                                    if isinstance(target.target, cst.Name):
                                        attr_name = target.target.value
                                        if attr_name in {
                                            "list_display",
                                            "list_filter",
                                            "search_fields",
                                            "readonly_fields",
                                            "exclude",
                                            "fields",
                                        }:
                                            try:
                                                attr_value = module.code_for_node(expr.value)
                                                admin_attrs[attr_name] = attr_value
                                            except:
                                                continue

                # Store admin attributes
                if self.current_class:
                    self.django_admin_attributes = admin_attrs

            def _process_django_form_class(self, node: cst.ClassDef):
                """Process Django form class fields"""
                if not node.body:
                    return

                form_fields = []
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:
                    if isinstance(stmt, cst.SimpleStatementLine):
                        for expr in stmt.body:
                            if isinstance(expr, cst.Assign):
                                for target in expr.targets:
                                    if isinstance(target.target, cst.Name):
                                        field_name = target.target.value
                                        # Check if the assignment is a form field definition
                                        if isinstance(expr.value, cst.Call):
                                            if isinstance(expr.value.func, cst.Name):
                                                field_type = expr.value.func.value
                                                if field_type in {
                                                    "CharField",
                                                    "IntegerField",
                                                    "TextField",
                                                    "EmailField",
                                                    "ChoiceField",
                                                    "ModelChoiceField",
                                                    "ModelMultipleChoiceField",
                                                }:
                                                    # This is a Django form field
                                                    form_fields.append({
                                                        "name": field_name,
                                                        "type": field_type,
                                                        "parameters": self._extract_call_arguments(expr.value),
                                                    })
                                            elif isinstance(expr.value.func, cst.Attribute) and isinstance(
                                                expr.value.func.value, cst.Name
                                            ):
                                                base_name = expr.value.func.value.value
                                                field_type = expr.value.func.attr.value
                                                if base_name in {"forms", "django.forms"} and field_type in {
                                                    "CharField",
                                                    "IntegerField",
                                                    "TextField",
                                                    "EmailField",
                                                    "ChoiceField",
                                                    "ModelChoiceField",
                                                    "ModelMultipleChoiceField",
                                                }:
                                                    # This is a Django form field using forms.CharField format
                                                    form_fields.append({
                                                        "name": field_name,
                                                        "type": field_type,
                                                        "base": base_name,
                                                        "parameters": self._extract_call_arguments(expr.value),
                                                    })

                # Store form fields
                if self.current_class:
                    self.django_form_fields = form_fields

            def _process_django_view_class(self, node: cst.ClassDef):
                """Process Django view class methods and decorators"""
                if not node.body:
                    return

                view_methods = []
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:
                    if isinstance(stmt, cst.FunctionDef):
                        method_name = stmt.name.value
                        # Check if this is a standard HTTP method or Django-specific method
                        if method_name in {
                            "get",
                            "post",
                            "put",
                            "delete",
                            "dispatch",
                            "setup",
                            "get_context_data",
                            "get_queryset",
                        }:
                            decorators = []
                            if hasattr(stmt, "decorators") and stmt.decorators:
                                for deco in stmt.decorators:
                                    try:
                                        decorators.append(module.code_for_node(deco.decorator))
                                    except:
                                        continue
                            view_methods.append({"name": method_name, "decorators": decorators})

                # Store view methods
                if self.current_class:
                    self.django_view_methods = view_methods

            def _process_django_manager_class(self, node: cst.ClassDef):
                """Process Django manager class custom methods"""
                if not node.body:
                    return

                custom_methods = []
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:
                    if isinstance(stmt, cst.FunctionDef):
                        method_name = stmt.name.value
                        # For manager classes, consider all methods as custom methods
                        custom_methods.append(method_name)

                # Store custom methods
                if self.current_class:
                    self.django_manager_metadata["custom_methods"] = custom_methods

            def _extract_call_arguments(self, call_node: cst.Call) -> dict:
                """Extract arguments from a function call"""
                args_info = {}
                if call_node.args:
                    for i, arg in enumerate(call_node.args):
                        try:
                            if arg.keyword:
                                # Named argument like name=value
                                args_info[arg.keyword.value] = module.code_for_node(arg.value)
                            else:
                                # Positional argument
                                args_info[f"pos_{i}"] = module.code_for_node(arg.value)
                        except:
                            continue
                return args_info

            def _process_django_meta_class(self, node: cst.ClassDef, class_name: str):
                """Process Django Meta class content for model options"""
                if not node.body:
                    return

                meta_attrs = {}
                for stmt in node.body.body if hasattr(node.body, "body") else node.body:
                    if isinstance(stmt, cst.SimpleStatementLine):
                        for expr in stmt.body:
                            if isinstance(expr, cst.Assign):
                                for target in expr.targets:
                                    if isinstance(target.target, cst.Name):
                                        attr_name = target.target.value
                                        try:
                                            attr_value = module.code_for_node(expr.value)
                                            meta_attrs[attr_name] = attr_value
                                        except:
                                            continue
                    elif hasattr(stmt, "child_by_field_name"):
                        # Handle cases where the assignment is more complex
                        name_node = stmt.child_by_field_name("name")
                        if name_node and hasattr(name_node, "value"):
                            attr_name = name_node.value
                            value_node = stmt.child_by_field_name("value")
                            if value_node:
                                try:
                                    attr_value = module.code_for_node(value_node)
                                    meta_attrs[attr_name] = attr_value
                                except:
                                    continue

                # Store Meta class attributes for the parent class
                parent_class_name = class_name[:-5]  # Remove '.Meta' suffix
                if parent_class_name in self.django_meta_classes:
                    self.django_meta_classes[parent_class_name] = meta_attrs
                else:
                    self.django_meta_classes[parent_class_name] = meta_attrs

                # Process common Django Meta options
                for attr_name, attr_value in meta_attrs.items():
                    if attr_name == "db_table":
                        self.django_model_metadata["db_table"] = attr_value
                    elif attr_name == "verbose_name":
                        self.django_model_metadata["verbose_name"] = attr_value
                    elif attr_name == "verbose_name_plural":
                        self.django_model_metadata["verbose_name_plural"] = attr_value
                    elif attr_name == "ordering":
                        self.django_model_metadata["ordering"] = attr_value
                    elif attr_name == "indexes":
                        self.django_model_metadata["indexes"] = attr_value
                    elif attr_name == "unique_together":
                        self.django_model_metadata["unique_together"] = attr_value
                    elif attr_name == "abstract":
                        self.django_model_metadata["is_abstract"] = "True" in attr_value
                    elif attr_name == "managed":
                        self.django_model_metadata["is_managed"] = "True" in attr_value

            def visit_FunctionDef(self, node: cst.FunctionDef):
                self.current_function = node.name.value
                # Reset imports tracking for this function
                self.current_function_imports = set()

            def leave_FunctionDef(self, original_node: cst.FunctionDef):
                start_line, end_line = self._get_position(original_node)
                code_block = module.code_for_node(original_node)

                # Combine general references with imports used in this function
                general_refs = self.analyzer.find_called_symbols(code_block, "python", {})
                all_references = list(set(general_refs) | self.current_function_imports)

                # Extract dependencies from imports used in this function
                dependencies = self.analyzer._extract_dependencies(code_block, language="python")
                for import_name in self.current_function_imports:
                    if import_name in self.file_imports:
                        dependencies.append(self.file_imports[import_name])
                    elif import_name in self.file_from_imports:
                        # Add module part of the from import
                        dependencies.append(self.file_from_imports[import_name][0])

                # Build more accurate function signature
                signature = self._get_function_signature(original_node)

                # Create the chunk first
                chunk = CodeChunk(
                    type="method" if self.current_class else "function",
                    name=original_node.name.value,
                    code=code_block,
                    file_path=relative_path,
                    language="python",
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=sorted(set(dependencies)),
                    signature=signature,
                    complexity=self.analyzer._calculate_complexity(code_block, language="python"),
                    parent=self.current_class,
                    defines=[original_node.name.value],
                    references=sorted(set(all_references)),
                )

                # Enhance relationships metadata
                chunk.relationships.update({
                    "imports_used": sorted(self.current_function_imports),
                    "class_inheritance": [] if not self.current_class else self.current_class_inheritance,
                    "django_model_fields": []
                    if not self.current_class
                    else self.django_model_relationships.get(self.current_class, []),
                })

                self.chunks.append(chunk)
                self.current_function = None

            def leave_ClassDef(self, original_node: cst.ClassDef):
                start_line, end_line = self._get_position(original_node)
                code_block = module.code_for_node(original_node)

                # Combine general references with imports used in this class
                general_refs = self.analyzer.find_called_symbols(code_block, "python", symbol_index)
                all_references = list(set(general_refs) | self.current_class_imports)

                # Extract dependencies from imports used in this class
                dependencies = self.analyzer._extract_dependencies(code_block, language="python")
                for import_name in self.current_class_imports:
                    if import_name in self.file_imports:
                        dependencies.append(self.file_imports[import_name])
                    elif import_name in self.file_from_imports:
                        # Add module part of the from import
                        dependencies.append(self.file_from_imports[import_name][0])

                # Add Django-specific dependencies if this is a model class
                if self.current_class_inheritance:
                    for inheritance in self.current_class_inheritance:
                        if any(django_import in inheritance for django_import in self.django_model_imports):
                            dependencies.append("django.db.models")

                # Build class signature
                signature = self._get_class_signature(original_node)

                # Determine the type of Django class if applicable
                class_type = "class"
                if self.current_class_inheritance:
                    # Check if it's a Django model class
                    for inheritance in self.current_class_inheritance:
                        if "models" in inheritance.lower():
                            class_type = "django_model"
                            break
                        if "admin" in inheritance.lower():
                            class_type = "django_admin"
                            break
                        if "Form" in inheritance or "form" in inheritance.lower():
                            class_type = "django_form"
                            break
                        if "View" in inheritance or "view" in inheritance.lower():
                            class_type = "django_view"
                            break

                # Create the chunk first
                chunk = CodeChunk(
                    type=class_type,
                    name=original_node.name.value,
                    code=code_block,
                    file_path=relative_path,
                    language="python",
                    start_line=start_line,
                    end_line=end_line,
                    docstring=self._get_docstring(original_node),
                    dependencies=sorted(set(dependencies)),
                    signature=signature,
                    complexity=self.analyzer._calculate_complexity(code_block, language="python"),  # noqa: SLF001
                    parent=None,  # Set parent if class is nested
                    defines=[original_node.name.value],
                    references=sorted(set(all_references)),
                )

                # Extract decorators from the original node
                decorators = []
                if hasattr(original_node, "decorators") and original_node.decorators:
                    for deco in original_node.decorators:
                        try:
                            decorators.append(module.code_for_node(deco.decorator))
                        except:
                            continue
                chunk.metadata["decorators"] = decorators

                # Extract base classes from the node
                base_classes = []
                if original_node.bases:
                    for base in original_node.bases:
                        try:
                            base_class_str = module.code_for_node(base.value)
                            base_classes.append(base_class_str)
                        except:
                            continue
                chunk.metadata["base_classes"] = base_classes

                # Extract Django model fields if this is a model class
                if class_type == "django_model":
                    if self.current_class in self.django_model_relationships:
                        django_fields = [f for _, f in self.django_model_relationships[self.current_class]]
                        if "django_model_fields" not in chunk.relationships:
                            chunk.relationships["django_model_fields"] = []
                        chunk.relationships["django_model_fields"].extend(django_fields)

                    # Extract Django model managers if any
                    if self.current_class in self.django_model_managers:
                        django_managers = [m for _, m in self.django_model_managers[self.current_class]]
                        if "django_model_managers" not in chunk.relationships:
                            chunk.relationships["django_model_managers"] = []
                        chunk.relationships["django_model_managers"].extend(django_managers)

                # Enhance relationships metadata
                chunk.relationships.update({
                    "imports": sorted(self.current_class_imports),
                    "class_inheritance": self.current_class_inheritance,
                    "django_model_fields": self.django_model_relationships.get(self.current_class, []),
                    "django_model_inheritance": [
                        base
                        for base in self.current_class_inheritance
                        if any(django_import in base for django_import in self.django_model_imports)
                    ],
                    "django_model_managers": self.django_model_managers.get(self.current_class, []),
                    "django_meta_class": self.django_meta_classes.get(self.current_class, {}),
                    "django_admin_registration": getattr(self, "current_class_metadata", {}).get(
                        "admin_registers", None
                    ),
                })

                self.chunks.append(chunk)
                self.current_class = None

            def visit_Name(self, node: cst.Name) -> None:
                # Check if this name matches any imported names
                name = node.value
                if name in self.file_imports or name in self.file_from_imports:
                    if self.current_function:  # If we're inside a function
                        self.current_function_imports.add(name)
                    elif self.current_class:  # If we're inside a class
                        self.current_class_imports.add(name)

            def visit_Attribute(self, node: cst.Attribute) -> None:
                # Handle attribute access like "module.function" or "obj.method"
                if isinstance(node.value, cst.Name):
                    name = node.value.value
                    # Check if the base of the attribute is an imported name
                    if name in self.file_imports or name in self.file_from_imports:
                        if self.current_function:
                            self.current_function_imports.add(name)
                        elif self.current_class:
                            self.current_class_imports.add(name)
                    # Special handling for Django field definitions in classes
                    if self.current_class and name in self.django_model_imports:
                        # This might be something like models.CharField
                        field_type = code_for_node(node.attr)
                        if field_type in {
                            "CharField",
                            "IntegerField",
                            "ForeignKey",
                            "ManyToManyField",
                            "OneToOneField",
                        }:
                            # Track Django model fields if we're in a class definition
                            if self.current_class not in self.django_model_relationships:
                                self.django_model_relationships[self.current_class] = []
                            self.django_model_relationships[self.current_class].append((name, field_type))
                        # Track Django model managers
                        elif field_type in {"Manager", "RelatedManager"}:
                            if self.current_class not in self.django_model_managers:
                                self.django_model_managers[self.current_class] = []
                            self.django_model_managers[self.current_class].append((name, field_type))

                # Also handle deeply nested attribute access like 'django.contrib.admin.site'
                elif isinstance(node.value, cst.Attribute):
                    # This handles cases like admin.site.register
                    attr_node = node.value
                    if isinstance(attr_node.value, cst.Name):
                        base_name = attr_node.value.value
                        if base_name in self.file_imports or base_name in self.file_from_imports:
                            if self.current_function:
                                self.current_function_imports.add(base_name)
                            elif self.current_class:
                                self.current_class_imports.add(base_name)
                    # Handle more complex attribute access patterns like models.ForeignKey
                    elif isinstance(attr_node.value, cst.Attribute):
                        # This could be something like models.fields.CharField
                        nested_attr = attr_node.value
                        if isinstance(nested_attr.value, cst.Name):
                            nested_base_name = nested_attr.value.value
                            if nested_base_name in self.django_model_imports or nested_base_name in self.file_imports:
                                if self.current_class:
                                    # Track more specific Django field types
                                    field_name = code_for_node(node.attr)
                                    full_attr_chain = f"{nested_base_name}.{code_for_node(attr_node.attr)}.{field_name}"
                                    if field_name in [
                                        "CharField",
                                        "IntegerField",
                                        "ForeignKey",
                                        "ManyToManyField",
                                        "OneToOneField",
                                    ]:
                                        if self.current_class not in self.django_model_relationships:
                                            self.django_model_relationships[self.current_class] = []
                                        self.django_model_relationships[self.current_class].append((
                                            nested_base_name,
                                            field_name,
                                        ))

            def visit_Call(self, node: cst.Call) -> None:
                # Track function calls
                if isinstance(node.func, cst.Name):
                    name = node.func.value
                    # Check if it's an imported function
                    if name in self.file_imports or name in self.file_from_imports:
                        if self.current_function:
                            self.current_function_imports.add(name)
                        elif self.current_class:
                            self.current_class_imports.add(name)
                elif isinstance(node.func, cst.Attribute):
                    # Handle attribute calls like "module.function()"
                    if isinstance(node.func.value, cst.Name):
                        name = node.func.value.value
                        if name in self.file_imports or name in self.file_from_imports:
                            if self.current_function:
                                self.current_function_imports.add(name)
                            elif self.current_class:
                                self.current_class_imports.add(name)
                    # Handle calls like "admin.site.register(Model)"
                    elif isinstance(node.func.value, cst.Attribute):
                        # This could be something like admin.site.register
                        attr_node = node.func.value
                        if isinstance(attr_node.value, cst.Name):
                            base_name = attr_node.value.value
                            if base_name in self.file_imports or base_name in self.file_from_imports:
                                if self.current_function or self.current_class:
                                    # Add the base import to be tracked
                                    if self.current_function:
                                        self.current_function_imports.add(base_name)
                                    elif self.current_class:
                                        self.current_class_imports.add(base_name)
                    # Special handling for Django admin registrations
                    if code_for_node(node.func) == "admin.site.register":
                        if node.args and len(node.args) >= 1:
                            model_name = code_for_node(node.args[0].value)
                            if self.current_class:
                                # Track that this class registers a model with admin
                                if hasattr(self, "current_class_metadata") and self.current_class_metadata:
                                    self.current_class_metadata["admin_registers"] = model_name
                                else:
                                    self.current_class_metadata = {"admin_registers": model_name}

            def _get_position(self, node):
                pos = self.get_metadata(PositionProvider, node)
                return pos.start.line, pos.end.line

            def _get_docstring(self, node) -> str | None:
                """
                Extract docstring from a function or class node.
                Handles various string types and improves detection of docstrings.
                """
                if not hasattr(node, "body"):
                    return None

                # Get the first statement in the body
                body = node.body
                if isinstance(body, cst.IndentedBlock):
                    statements = body.body
                elif isinstance(body, (list, tuple)):
                    statements = body
                else:
                    return None

                if not statements:
                    return None

                first_stmt = statements[0]

                # Handle SimpleStatementLine containing the docstring
                if isinstance(first_stmt, cst.SimpleStatementLine):
                    for expr in first_stmt.body:
                        docstring = self._extract_docstring_from_expr(expr)
                        if docstring:
                            return docstring.strip()
                else:
                    # Handle direct expressions in the body
                    docstring = self._extract_docstring_from_expr(first_stmt)
                    if docstring:
                        return docstring.strip()

                return None

            def _extract_docstring_from_expr(self, expr) -> str | None:
                """
                Extract docstring from an expression node.
                """
                if not isinstance(expr, cst.Expr):
                    return None

                value = expr.value
                # Handle different types of string literals
                if isinstance(value, cst.SimpleString):
                    # For regular strings, f-strings, raw strings, etc.
                    try:
                        # Use evaluated_value to get the actual string content
                        docstring = value.evaluated_value
                        if docstring is not None:
                            return docstring
                    except Exception:
                        # If evaluated_value fails, try raw value
                        raw_value = value.value
                        # Remove string delimiters to get content
                        if len(raw_value) >= 2:
                            return raw_value[1:-1]  # Remove first and last characters (quotes)
                elif isinstance(value, cst.ConcatenatedString):
                    # Handle concatenated strings
                    parts = []
                    for part in [value.left, value.right]:
                        if isinstance(part, cst.SimpleString):
                            try:
                                evaluated = part.evaluated_value
                                if evaluated is not None:
                                    parts.append(evaluated)
                            except Exception:
                                continue
                    if parts:
                        return "".join(parts)

                return None

            def _get_function_signature(self, node: cst.FunctionDef) -> str:
                """
                Extract and build a function signature from its definition.
                """
                # Start with def keyword and function name
                name = node.name.value
                parameters = self._get_parameters_str(node.params)

                # Handle return annotation if present
                return_annotation = ""
                if node.returns:
                    print("checking if node retuens")
                    try:
                        return_annotation = f" -> {module.code_for_node(node.returns)}"
                    except:
                        return_annotation = ""

                return f"def {name}{parameters}{return_annotation}:"

            def _get_parameters_str(self, params: cst.Parameters) -> str:
                """
                Convert function parameters to string representation.
                """
                param_strings = []

                # Handle regular parameters
                for param in params.params:
                    # Check if param is a sentinel value
                    if isinstance(param, cst.Param):
                        # Get parameter name - it could be a string directly or a Name node
                        param_str = param.name if isinstance(param.name, str) else param.name.value

                        # Add type annotation if present
                        if param.annotation and not isinstance(param.annotation, cst.MaybeSentinel):
                            with contextlib.suppress(builtins.BaseException):
                                param_str += f": {module.code_for_node(param.annotation)}"

                        # Add default value if present
                        if param.default:
                            with contextlib.suppress(builtins.BaseException):
                                param_str += f" = {module.code_for_node(param.default)}"

                        param_strings.append(param_str)

                # Handle star args (*args)
                if params.star_arg and not isinstance(params.star_arg, cst.MaybeSentinel):
                    star_arg = params.star_arg
                    # Handle star_arg name
                    star_str = f"*{star_arg.name}" if isinstance(star_arg.name, str) else f"*{star_arg.name.value}"

                    if star_arg.annotation and not isinstance(star_arg.annotation, cst.MaybeSentinel):
                        with contextlib.suppress(builtins.BaseException):
                            star_str += f": {module.code_for_node(star_arg.annotation)}"
                    param_strings.append(star_str)

                # Handle keyword-only parameters
                for param in params.kwonly_params:
                    if isinstance(param, cst.Param):
                        # Get parameter name
                        param_str = param.name if isinstance(param.name, str) else param.name.value

                        # Add type annotation if present
                        if param.annotation and not isinstance(param.annotation, cst.MaybeSentinel):
                            with contextlib.suppress(builtins.BaseException):
                                param_str += f": {module.code_for_node(param.annotation)}"

                        # Add default value if present
                        if param.default:
                            with contextlib.suppress(builtins.BaseException):
                                param_str += f" = {module.code_for_node(param.default)}"

                        param_strings.append(param_str)

                # Handle double-star args (**kwargs)
                if params.star_kwarg and not isinstance(params.star_kwarg, cst.MaybeSentinel):
                    star_kwarg = params.star_kwarg
                    # Handle star_kwarg name
                    if isinstance(star_kwarg.name, str):
                        star_str = f"**{star_kwarg.name}"
                    else:
                        star_str = f"**{star_kwarg.name.value}"

                    if star_kwarg.annotation and not isinstance(star_kwarg.annotation, cst.MaybeSentinel):
                        with contextlib.suppress(builtins.BaseException):
                            star_str += f": {module.code_for_node(star_kwarg.annotation)}"
                    param_strings.append(star_str)

                return f"({', '.join(param_strings)})"

            def _get_class_signature(self, node: cst.ClassDef) -> str:
                """
                Extract and build a class signature from its definition.
                """
                # Start with class keyword and class name
                name = node.name.value

                # Extract base classes if any
                base_classes = []
                if node.bases:
                    for base in node.bases:
                        try:
                            base_class_str = module.code_for_node(base.value)
                            base_classes.append(base_class_str)
                        except:
                            continue

                # Build the signature string
                bases_str = f"({', '.join(base_classes)})" if base_classes else "()"

                return f"class {name}{bases_str}:"

        visitor = FunctionVisitor(
            self, import_visitor.imports, import_visitor.from_imports, import_visitor.django_model_imports
        )
        wrapper.visit(visitor)
        save_data(module, method="cst")
        return visitor.chunks

    def find_called_symbols(self, code: str, language: str, symbol_index: dict) -> list[str]:
        """Find symbols in code that are defined in the current project."""
        if not symbol_index:
            return []

        references = set()

        if language == "python":
            import re

            # Match function calls: func(), obj.method(), Class()
            call_pattern = r"\b([a-zA-Z_]\w*)\s*(?:\(|$)"
            attr_pattern = r"\b([a-zA-Z_]\w*)\.\w+"

            candidates = set()
            candidates.update(re.findall(call_pattern, code))
            candidates.update(re.findall(attr_pattern, code))

            for name in candidates:
                if name in symbol_index and name not in {"self", "cls", "super"}:
                    references.add(name)

        elif language in {"javascript", "typescript"}:
            import re

            words = re.findall(r"\b([A-Za-z_]\w*)\s*\(", code)
            for name in words:
                if name in symbol_index:
                    references.add(name)

        return sorted(references)

    def _calculate_complexity(self, code: str, language: str) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        try:
            if language == "python":
                try:
                    visitor = ComplexityVisitor.from_code(code)
                    return sum(block.complexity for block in visitor.blocks)
                except Exception:
                    print("radon failed, falling back to keyword counting")
            complexity = 1
            # General keywords that increase complexity
            complexity_keywords = [
                "if",
                "else",
                "elif",
                "for",
                "while",
                "for(",
                "while(",
                "switch",
                "case",
                "try",
                "catch",
                "finally",
                "except",
                "&&",
                "||",
                "?",
                ":",
                "=>",
            ]

            for keyword in complexity_keywords:
                # Count occurrences, adjusting for common patterns
                if keyword in {"if", "for", "while"}:
                    # Special handling to avoid double counting in "else if"
                    if keyword == "if":
                        count = code.count("if ") + code.count("if(") - code.count("else if")
                    elif keyword == "for":
                        count = code.count("for ") + code.count("for(")
                    elif keyword == "while":
                        count = code.count("while ") + code.count("while(")
                    complexity += count
                else:
                    complexity += code.count(keyword)

            return max(complexity, 1)  # Ensure at least 1

        except Exception as e:
            print(f"Complexity calculation failed: {e}")
            return 1

    def _extract_dependencies(self, code: str, language: str) -> list[str]:
        deps = set()  # use set to avoid duplicates

        if language == "python":
            # Parse imports
            for fline in code.split("\n"):
                line = fline.strip()
                if line.startswith("import "):
                    # import A, B as C → extract A, B
                    parts = line[7:].split(",")
                    for part in parts:
                        name = part.split()[0].split(".")[0]  # get root module
                        deps.add(name)
                elif line.startswith("from "):
                    # from X import Y → extract X
                    match = re.match(r"from\s+([a-zA-Z_][\w\.]*)\s+import", line)
                    if match:
                        root_module = match.group(1).split(".")[0]
                        deps.add(root_module)

            # Optional: Extract symbol usage (heuristic for models/utils)
            # Look for CapitalizedWords (likely classes/models)
            symbols = re.findall(r"\b[A-Z][a-zA-Z_]\w*\b", code)
            # Filter out common built-ins/keywords
            common = {"None", "True", "False", "self", "cls", "Exception", "str", "int"}
            for sym in symbols:
                if sym not in common and not sym.startswith("_"):
                    deps.add(sym)

        elif language in {"javascript", "typescript"}:
            # Handle import X from 'Y' and require('Y')
            import_re = re.compile(r'import\s+(?:[\w{}\*\s,]+\s+from\s+)?[\'"]([^\'"]+)[\'"]')
            require_re = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)')

            for fline in code.split("\n"):
                line = fline.strip()
                # Skip comments
                if line.startswith(("//", "/*")):
                    continue
                for match in import_re.findall(line):
                    deps.add(match.split("/")[0])  # e.g., 'lodash/map' → 'lodash'
                for match in require_re.findall(line):
                    deps.add(match.split("/")[0])

        return sorted(deps)[:10]  # return list, deduped

    # ============================================================================
    # ENHANCED METADATA EXTRACTION METHODS (Migrated from enhanced.py)
    # ============================================================================

    def add_code_metadata(self, chunk: CodeChunk, node=None, code_bytes: bytes | None = None) -> None:
        """Extract code-specific metadata (from enhanced.py)"""
        metadata = {}

        if chunk.language == "python":
            if chunk.metadata.get("decorators") is None:
                metadata["decorators"] = self._extract_decorators(node, code_bytes) if node else []
            if chunk.metadata.get("base_classes") is None:
                metadata["base_classes"] = self._extract_base_classes(node, code_bytes) if node else []
            metadata["access_modifier"] = self._determine_access_modifier(chunk.code)
            metadata["is_abstract"] = self._is_abstract(chunk.code)
            metadata["is_final"] = self._is_final(chunk.code)

        elif chunk.language in {"javascript", "typescript"}:
            metadata["export_type"] = self._extract_export_type(chunk.code)
            metadata["is_async"] = "async" in chunk.code

        chunk.metadata.update(metadata)

    def _extract_decorators(self, node, code_bytes: bytes) -> list[str]:
        """Extract decorators from Python CST nodes (from enhanced.py)"""
        decorators = []

        if not node or not hasattr(node, "decorators") or not node.decorators:
            return decorators

        # This function assumes 'node' is a LibCST node, consistent with
        # the parsing strategy used in `parse_python_file_libcst`.
        for deco in node.decorators:
            try:
                # `deco` is a `cst.Decorator` node. The actual expression
                # that represents the decorator is in `deco.decorator`.
                # We use an empty Module to call code_for_node, which works for
                # self-contained nodes like decorators.
                decorator_str = cst.Module([]).code_for_node(deco.decorator).strip()
                decorators.append(decorator_str)
            except Exception:
                # If code_for_node fails, we skip this decorator.
                # This is more robust than the previous implementation.
                continue
        return decorators

    def _extract_base_classes(self, node, code_bytes: bytes) -> list[str]:
        """Extract base classes from class definitions (from enhanced.py)"""
        base_classes = []

        if hasattr(node, "bases"):
            for base in node.bases:
                try:
                    base_classes.append(cst.Module([]).code_for_node(base.value).strip())
                except Exception:
                    continue
            return base_classes

        if hasattr(node, "children"):
            for child in node.children:
                if hasattr(child, "type") and child.type == "argument_list":
                    for arg in child.children:
                        if arg.type not in {"(", ")"}:
                            try:
                                base_classes.append(code_bytes[arg.start_byte : arg.end_byte].decode("utf-8").strip())
                            except Exception:
                                continue

        return base_classes

    def _determine_access_modifier(self, code: str) -> str:
        """Determine access modifier (from enhanced.py)"""
        if code.startswith("_") and not code.startswith("__"):
            return "protected"
        if code.startswith("__"):
            return "private"
        return "public"

    def _is_abstract(self, code: str) -> bool:
        """Check if class/method is abstract (from enhanced.py)"""
        return "abstract" in code or "ABC" in code

    def _is_final(self, code: str) -> bool:
        """Check if class/method is final (from enhanced.py)"""
        return "final" in code or "@final" in code

    def _extract_export_type(self, code: str) -> str:
        """Extract export type for JS/TS (from enhanced.py)"""
        if "export default" in code:
            return "default"
        if "export" in code:
            return "named"
        return "none"

    def add_analysis_metadata(self, chunk: CodeChunk) -> None:
        """Add accurate, dynamic analysis metadata (from enhanced.py)"""
        complexity = self._calculate_complexity(chunk.code, chunk.language)
        token_count = self._count_tokens(chunk.code, chunk.language)
        embedding_size = getattr(self, "embedding_size", None) or 768
        semantic_hash = self._generate_semantic_hash(chunk.code)

        start = chunk.start_line
        end = chunk.end_line
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
        clean_code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        clean_code = re.sub(r"\s+", " ", clean_code).strip()
        return hashlib.md5(clean_code.encode()).hexdigest()[:10]

    def add_relationship_metadata(self, chunk: CodeChunk, all_chunks: list[CodeChunk] | None = None) -> None:
        """Add comprehensive relationship metadata (from enhanced.py)"""

        relationships = {}
        if chunk.relationships.get("imports") is None:
            relationships["imports"] = self._extract_imports(chunk.code, chunk.language)
        if chunk.relationships.get("dependencies") is None:
            relationships["dependencies"] = chunk.dependencies or []
        if chunk.relationships.get("parent") is None:
            relationships["parent"] = chunk.parent
        if chunk.relationships.get("children") is None:
            relationships["children"] = self._find_child_chunks(chunk, all_chunks or [])
        if chunk.relationships.get("references") is None:
            relationships["references"] = chunk.references or []
        if chunk.relationships.get("called_functions") is None:
            relationships["called_functions"] = self._extract_called_functions(chunk.code, chunk.language)
        if chunk.relationships.get("defined_symbols") is None:
            relationships["defined_symbols"] = chunk.defines or []

        chunk.relationships.update(relationships)

    def _extract_imports(self, code: str, language: str) -> list[str]:
        """Extract import statements (from enhanced.py)"""
        imports = []
        lines = code.split("\n")

        if language == "python":
            for fline in lines:
                line = fline.strip()
                if line.startswith(("import ", "from ")):
                    imports.append(line)
        elif language in {"javascript", "typescript"}:
            for fline in lines:
                line = fline.strip()
                if line.startswith(("import ", "export ", "require(")):
                    imports.append(line)

        return imports

    def _extract_called_functions(self, code: str, language: str) -> list[str]:
        """Extract function calls from code (from enhanced.py)"""
        called_functions = []

        if language == "python":
            pattern = r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            matches = re.findall(pattern, code)
            keywords = {"if", "for", "while", "with", "def", "class", "return"}
            called_functions = [m for m in matches if m not in keywords]

        return called_functions

    def _find_child_chunks(self, chunk: CodeChunk, all_chunks: list[CodeChunk]) -> list[str]:
        """Find IDs of child chunks (from enhanced.py)"""
        children = []
        if chunk.type == "class":
            children.extend(
                c.id for c in all_chunks if getattr(c, "parent", None) == chunk.name and c.file_path == chunk.file_path
            )
        return children

    def add_context_metadata(self, chunk: CodeChunk, file_path: Path, project_path: Path | None = None) -> None:
        """Add contextual information (from enhanced.py)"""
        chunk.context = {
            "module_context": self._get_module_context(file_path),
            "project_context": self._get_project_context(file_path),
            "file_hierarchy": self._get_file_hierarchy(file_path),
            "domain_context": self._infer_domain_context(chunk, file_path),
        }

    def _get_module_context(self, file_path: Path) -> str:
        """Get module-level context (from enhanced.py)"""
        parent_dir = file_path.parent.name
        if parent_dir:
            return f"{parent_dir} module"
        return "root module"

    def _get_project_context(self, project_path: Path = None) -> str:  # noqa: RUF013
        """Infer project context from structure (from enhanced.py)"""
        pcontext = list(project_path.relative_to(settings.project_path).parts)
        if "migrations" in pcontext:
            return "Database changes"
        if pcontext[-1].endswith(".py"):
            return f"{pcontext[0]} Backend App"
        if pcontext[-1].endswith(".html") or pcontext[-1].endswith(".css"):
            return "Project Templates"
        if pcontext[-1].endswith(".js") or pcontext[-1].endswith(".mjs"):
            return "Tipsy Javascript Frontend"
        return "Project codebase"

    def _get_file_hierarchy(self, file_path: Path) -> list[str]:
        """Get file hierarchy as list, relative to the project root."""
        return list(file_path.relative_to(settings.project_path).parts)

    def _infer_domain_context(self, chunk: CodeChunk, file_path: Path) -> str:
        """Infer domain context from file path and content (from enhanced.py)"""
        path_str = str(file_path).lower()

        if "admin" in path_str:
            return "Django admin configuration"
        if "model" in path_str or "models" in path_str:
            return "Data models and schema"
        if "view" in path_str or "controller" in path_str:
            return "Application logic and controllers"
        if "test" in path_str:
            return "Testing code"
        if any(ui in path_str for ui in ["template", "component", "ui", "css"]):
            return "User interface"

        return "General application code"

    def enhance_chunk_completely(
        self,
        chunk: CodeChunk,
        node=None,
        code_bytes: bytes | None = None,
        file_path: Path | None = None,
        project_path: Path | None = None,
        all_chunks: list[CodeChunk] | None = None,
    ) -> None:
        """Full enhancement pipeline for a chunk (from enhanced.py)"""
        self.add_code_metadata(chunk, node, code_bytes)
        self.add_analysis_metadata(chunk)
        if file_path:
            self.add_relationship_metadata(chunk, all_chunks or [])
            self.add_context_metadata(chunk, file_path, project_path)

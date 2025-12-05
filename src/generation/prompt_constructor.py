"""
Prompt construction and context building for generation
"""

import pathlib
from abc import ABC
from abc import abstractmethod
from textwrap import dedent
from typing import Any
from typing import Optional


class ContextPromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, chunk: dict[str, Any], symbol_index: Optional["SymbolIndex"] = None) -> str:
        pass


class SymbolIndex:
    """Enhanced symbol index for lookup with improved info extraction"""

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data or {}

    def get_info(self, symbol: str) -> str:
        entries = self.data.get(symbol, [])
        if not entries:
            return "No additional info"

        # Take first entry (usually sufficient)
        entry = entries[0]
        kind = entry.get("kind", "unknown")
        file_path = entry.get("file", "unknown file")
        line = entry.get("line", "unknown line")

        # Make relative or just show basename
        short_file = pathlib.Path(file_path).name

        # Enhanced info based on symbol kind
        if kind == "class":
            return f"Class defined in {short_file}:{line}"
        if kind == "function":
            return f"Function defined in {short_file}:{line}"
        if kind == "method":
            return f"Method defined in {short_file}:{line}"
        return f"{kind} defined in {short_file}:{line}"


class MultiLanguageContextPrompt(ContextPromptBuilder):
    """Enhanced prompt builder that handles multiple languages and chunk types"""

    def build_prompt(self, chunk: dict[str, Any], symbol_index: SymbolIndex | None = None) -> str:
        # Build dependency context
        deps_info = self._build_dependency_context(chunk, symbol_index)

        # Get domain-specific context
        domain_context = self._get_domain_specific_context(chunk)

        # Build language-specific instructions
        language_instructions = self._get_language_instructions(chunk.get("language", "unknown"))

        # Build chunk-type specific context
        chunk_type_context = self._get_chunk_type_context(chunk)

        return dedent(f"""You are an expert software developer specializing in {chunk.get("language", "unknown")} and web development.
        Summarize the following code chunk in one clear, concise sentence.
        Focus on its purpose, behavior, and role in the application.

        {domain_context}

        File: {chunk.get("file_path", "unknown")}
        Type: {chunk.get("type", "unknown")}
        Name: {chunk.get("name", "unknown")}
        Qualified Name: {chunk.get("qualified_name", "unknown")}
        Location: Lines {chunk.get("start_line", "?")}-{chunk.get("end_line", "?")}

        {chunk_type_context}

        Dependencies: {", ".join(chunk.get("dependencies", [])) or "None"}
        {deps_info}

        {language_instructions}

        Code:
        ```{chunk.get("language", "unknown")}
        {chunk.get("code", "").strip()}
        ```

        Summary (one sentence, no markdown, no prefix):
        """).strip()

    def _build_dependency_context(self, chunk: dict[str, Any], symbol_index: SymbolIndex | None) -> str:
        """Build context about dependencies and relationships"""
        deps_info = []
        if symbol_index:
            for dep in chunk.get("dependencies", [])[:5]:  # Limit to top 5 for brevity
                info = symbol_index.get_info(dep)
                if info != "No additional info":
                    deps_info.append(f"- {dep}: {info}")

        # Add relationship context
        relationship_context = []
        relationships = chunk.get("relationships", {})
        if relationships:
            imports = relationships.get("imports", [])
            if imports:
                relationship_context.append(f"Imports: {', '.join(imports[:3])}")

            called_functions = relationships.get("called_functions", [])
            if called_functions:
                relationship_context.append(f"Calls: {', '.join(called_functions[:3])}")

            references = relationships.get("references", [])
            if references:
                relationship_context.append(f"References: {', '.join(references[:3])}")

        context_parts = []
        if deps_info:
            context_parts.append("Dependency details:\n" + "\n".join(deps_info))
        if relationship_context:
            context_parts.append("Relationships:\n" + "; ".join(relationship_context))

        return "\n".join(context_parts) if context_parts else "No additional dependency or relationship info."

    def _get_domain_specific_context(self, chunk: dict[str, Any]) -> str:
        """Get domain-specific context based on file path and content"""
        context = chunk.get("context", {})
        domain_ctx = context.get("domain_context", "General code")
        module_ctx = context.get("module_context", "Unknown module")
        project_ctx = context.get("project_context", "Project codebase")

        # Enhanced domain context based on file path
        file_path = chunk.get("file_path", "").lower()
        if any(term in file_path for term in ["admin", "modeladmin"]):
            domain_ctx = "Django admin configuration and model management"
        elif any(term in file_path for term in ["model", "schema"]):
            domain_ctx = "Data models and database schema"
        elif any(term in file_path for term in ["view", "controller"]):
            domain_ctx = "Application logic and request handling"
        elif any(term in file_path for term in ["template", "html"]):
            domain_ctx = "User interface templates and presentation"
        elif any(term in file_path for term in ["static", "css"]):
            domain_ctx = "Styling and user interface design"
        elif any(term in file_path for term in ["static", "js"]):
            domain_ctx = "Client-side functionality and interactivity"

        return f"Module: {module_ctx}\nDomain: {domain_ctx}\nProject: {project_ctx}"

    def _get_language_instructions(self, language: str) -> str:
        """Get language-specific instructions for the LLM"""
        instructions = {
            "python": "Focus on Python-specific patterns, Django conventions if applicable, and the object's role in the application.",
            "javascript": "Focus on JavaScript patterns, DOM manipulation if applicable, and the function's role in client-side logic.",
            "html": "Focus on the template's structure, included components, and its role in the page layout and user interface.",
            "css": "Focus on the styling rules, layout impact, and visual design role in the application.",
            "java": "Focus on Java patterns, object-oriented design, and the class/method's role in the application architecture.",
            "cpp": "Focus on C++ patterns, memory management considerations, and performance characteristics.",
            "go": "Focus on Go patterns, concurrency if applicable, and the function's role in the system.",
            "rust": "Focus on Rust patterns, ownership system, and safety characteristics.",
        }
        return instructions.get(language, "Focus on the code's purpose and behavior in the application.")

    def _get_chunk_type_context(self, chunk: dict[str, Any]) -> str:
        """Get context specific to the chunk type"""
        chunk_type = chunk.get("type", "unknown")
        type_contexts = {
            "class": "This is a class definition. Describe its responsibility and how it might be used.",
            "function": "This is a function. Describe what it does and its input/output behavior.",
            "method": "This is a method within a class. Describe its specific role and how it modifies object state.",
            "html_file": "This is a complete HTML template. Describe its overall structure and purpose in the UI.",
            "html_element": "This is an HTML element or component. Describe its role in the page layout.",
            "css_rule": "This is a CSS rule. Describe its styling purpose and visual impact.",
        }

        base_context = type_contexts.get(chunk_type, "Describe the purpose and behavior of this code.")

        # Add metadata-specific context
        metadata_context = []
        metadata = chunk.get("metadata", {})
        if metadata:
            if metadata.get("decorators"):
                metadata_context.append(f"Decorators: {', '.join(metadata['decorators'])}")
            if metadata.get("base_classes"):
                metadata_context.append(f"Inherits from: {', '.join(metadata['base_classes'])}")
            if metadata.get("is_abstract"):
                metadata_context.append("This is an abstract class/method.")
            if metadata.get("export_type") and metadata["export_type"] != "none":
                metadata_context.append(f"Export type: {metadata['export_type']}")

        if metadata_context:
            return base_context + " " + "; ".join(metadata_context)
        return base_context


class DjangoCodeContextPrompt(ContextPromptBuilder):
    def build_prompt(self, chunk: dict[str, Any], symbol_index: SymbolIndex | None = None) -> str:
        deps_info = []
        if symbol_index:
            for dep in chunk.get("dependencies", []):
                info = symbol_index.get_info(dep)
                if info != "No additional info":
                    deps_info.append(f"- {dep}: {info}")

        deps_context = "\n".join(deps_info) if deps_info else "No additional dependency info."

        context = chunk.get("context", {})
        domain_ctx = context.get("domain_context", "General code")
        module_ctx = context.get("module_context", "Unknown module")

        prompt = dedent(f"""You are an expert Python and Django developer.
        Summarize the following code chunk in one clear, concise sentence.
        Focus on its purpose, behavior, and role in a Django web application.

        Module: {module_ctx}
        Domain context: {domain_ctx}
        File: {chunk.get("file_path", "unknown")}
        Type: {chunk.get("type", "unknown")}
        Name: {chunk.get("name", "unknown")}
        Dependencies: {", ".join(chunk.get("dependencies", [])) or "None"}

        {deps_context}

        Code:
        ```{chunk.get("language", "unknown")}
        {chunk.get("code", "").strip()}```
        Summary (one sentence, no markdown, no prefix):
        """).strip()
        return prompt  # noqa: RET504

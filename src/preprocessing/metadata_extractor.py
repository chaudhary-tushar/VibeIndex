"""
Metadata extraction from code chunks
"""

from .chunk import CodeChunk


class MetadataExtractor:
    """Handles extraction of semantic metadata from code"""

    def extract_docstring(self, node, code_bytes: bytes) -> str | None:
        """Extract docstring from AST node based on the language"""
        if node.type in ["function_definition", "class_definition"]:
            code_str = code_bytes.decode('utf-8')

            # For Python, docstrings are stored as the first expression in the function/class
            if node.type == "function_definition" or node.type == "class_definition":
                # Look for the first child that is an expression statement containing a string
                for child in node.children:
                    if child.type == "expression_statement":
                        # Check if it's a string literal
                        if len(child.children) > 0 and child.children[0].type == "string":
                            # Extract the string content
                            string_node = child.children[0]
                            start_byte = string_node.start_byte
                            end_byte = string_node.end_byte
                            docstring = code_bytes[start_byte:end_byte].decode('utf-8')
                            # Remove quotes
                            docstring = docstring.strip().strip('"""').strip("'''").strip('"').strip("'")
                            return docstring
                    elif child.type == "string":
                        # Direct string child
                        start_byte = child.start_byte
                        end_byte = child.end_byte
                        docstring = code_bytes[start_byte:end_byte].decode('utf-8')
                        # Remove quotes
                        docstring = docstring.strip().strip('"""').strip("'''").strip('"').strip("'")
                        return docstring

        return None

    def extract_signature(self, code: str) -> str | None:
        """Extract function/method signature"""
        lines = code.strip().split("\n")
        if lines:
            return lines[0].strip()
        return None

    def extract_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        keywords = ["if ", "elif ", "else ", "for ", "while ", "case ", "catch ", " && ", " || "]
        for keyword in keywords:
            complexity += code.count(keyword)
        return complexity

    def extract_tags_and_categories(self, chunk: CodeChunk) -> list:
        """Generate tags based on code analysis"""
        tags = []

        # Language-based tags
        tags.append(f"language:{chunk.language}")

        # Type-based tags
        tags.append(f"type:{chunk.type}")

        # Complexity-based tags
        if chunk.complexity >= 10:
            tags.append("complexity:high")
        elif chunk.complexity >= 5:
            tags.append("complexity:medium")
        else:
            tags.append("complexity:low")

        # Size-based tags
        line_count = chunk.end_line - chunk.start_line + 1
        if line_count > 50:
            tags.append("size:large")
        elif line_count > 20:
            tags.append("size:medium")
        else:
            tags.append("size:small")

        # Dependency-based tags
        if chunk.dependencies:
            tags.append("has_dependencies")

        return tags

    def enhance_chunk_metadata(self, chunk: CodeChunk) -> CodeChunk:
        """Add additional metadata to a chunk"""
        # Add tags
        if not hasattr(chunk, "tags") or not chunk.tags:
            chunk.tags = self.extract_tags_and_categories(chunk)

        # Ensure all required fields are present
        if chunk.docstring is None:
            chunk.docstring = ""

        if chunk.signature is None:
            chunk.signature = self.extract_signature(chunk.code)

        if not hasattr(chunk, "metadata"):
            chunk.metadata = {
                "line_count": chunk.end_line - chunk.start_line + 1,
                "has_tests": any("test" in chunk.name.lower() for chunk in [chunk]),  # simplistic check
                "is_entry_point": chunk.name in ["main", "__main__", "app", "application"],
            }

        return chunk

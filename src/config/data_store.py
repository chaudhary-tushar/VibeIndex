import csv
import json
from pathlib import Path
from typing import Any

from llama_index.core import Document
from tree_sitter import Node

BASE_PATH = Path(__file__).parent.parent.parent
DATA_DIR = BASE_PATH / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_file_path(method: str, language: str = None, ext: str = "txt") -> Path:
    """
    Constructs a file path for storing results based on method and language.

    The path structure will be:
    DATA_DIR / <method> / [<language>] / (<language> or <method>).<ext>

    Args:
        method (str): The name of the processing method, used as a subdirectory.
        language (str, optional): The language, used as a further subdirectory. Defaults to None.
        ext (str, optional): The file extension (e.g., "txt", "json"). Defaults to "txt".

    Returns:
        pathlib.Path: The constructed and ensured file path.
    """
    # Start with the base data directory and append the method
    target_directory = DATA_DIR / method

    # If a language is provided, add it as another subdirectory
    if language:
        target_directory = target_directory / language

    # Ensure the complete directory structure exists
    target_directory.mkdir(parents=True, exist_ok=True)

    # Determine the base file name
    base_file_name = language if language else method

    # Construct the full file path
    file_path = target_directory / f"{base_file_name}.{ext}"

    return file_path


def save_data(data: Any, method: str, language: str = None, ext: str = None) -> Path:
    """
    Save data to an appropriate file format based on data type.

    Args:
        data: The data to save. Can be various types (dict, list, str, llama_index Document, etc.)
        method (str): The processing method name (e.g., 'collecting-reading', 'parsing', 'chunking')
        language (str, optional): The programming language if applicable
        ext (str, optional): Forced file extension. If not provided, auto-determined from data type

    Returns:
        pathlib.Path: The path where the data was saved
    """
    # Determine the appropriate extension based on data type
    if ext is None:
        ext = _determine_file_extension(data[0]) if isinstance(data, list) else _determine_file_extension(data)

    # Create the file path
    file_path = get_file_path(method, language, ext)

    # Save data based on its type
    if ext == "json":
        _save_json_data(data, file_path)
    elif ext == "csv":
        _save_csv_data(data, file_path)
    elif ext == "txt":
        _save_txt_data(data, file_path)
    elif ext == ".sexp":
        _save_tree_data(data, file_path)
    else:
        # Default to txt if extension is unknown
        _save_txt_data(data, file_path)

    return file_path


def _determine_file_extension(data: Any) -> str:
    """
    Determine the appropriate file extension based on data type.

    Args:
        data: The data to analyze

    Returns:
        str: The appropriate file extension
    """
    # Check for specific data types
    if isinstance(data, Node):
        return "txt"
    if isinstance(data, dict):
        return "json"
    if isinstance(data, list):
        # If it's a list, check the first element to determine the appropriate extension
        if len(data) > 0:
            first_element = data[0]
            # If the list contains Documents, use JSON
            if (
                isinstance(first_element, Document)
                or (hasattr(first_element, "__class__") and "Document" in first_element.__class__.__name__)
                or isinstance(first_element, dict)
            ):
                return "json"
            if isinstance(first_element, str):
                # For strings, check if it looks like structured data
                if first_element.strip().startswith(("{", "[")):
                    try:
                        # Try to parse as JSON
                        json.loads(first_element)
                        return "json"
                    except json.JSONDecodeError:
                        pass
                return "txt"
            return "json"  # Default to JSON for lists of other types
        # Empty list, default to JSON
        return "json"
    if isinstance(data, Document) or (hasattr(data, "__class__") and "Document" in data.__class__.__name__):
        # Check if it's a LlamaIndex Document
        return "json"
    if hasattr(data, "__class__") and data.__class__.__name__ in ["Tree", "Node", "tree_sitter"]:
        # Tree-sitter related objects
        return "txt"
    if isinstance(data, str):
        # For strings, check if it looks like structured data
        if data.strip().startswith(("{", "[")):
            try:
                # Try to parse as JSON
                json.loads(data)
                return "json"
            except json.JSONDecodeError:
                pass
        return "txt"
    # Default to txt for other data types
    return "txt"


def _save_json_data(data: Any, file_path: Path) -> None:
    """Save data as JSON format."""
    with Path(file_path).open("w", encoding="utf-8") as file:
        if isinstance(data, list):
            # Handle list of items - check if it contains Documents
            if len(data) > 0 and (
                isinstance(data[0], Document)
                or (hasattr(data[0], "__class__") and "Document" in data[0].__class__.__name__)
            ):
                # Convert each Document in the list to a dictionary
                serialized_data = []
                for item in data:
                    if hasattr(item, "to_dict") and callable(item.to_dict):
                        serialized_data.append(item.to_dict())
                    elif isinstance(item, Document):
                        serialized_data.append({
                            "text": item.text,
                            "doc_id": item.doc_id,
                            "embedding": item.embedding,
                            "metadata": item.metadata,
                            "excluded_embed_metadata_keys": item.excluded_embed_metadata_keys,
                            "excluded_llm_metadata_keys": item.excluded_llm_metadata_keys,
                        })
                    else:
                        # If it's a list that doesn't contain Documents, save as is
                        serialized_data.append(item)
                json.dump(serialized_data, file, indent=4, ensure_ascii=False)
            else:
                # Handle regular list that doesn't contain Documents
                json.dump(data, file, indent=4, ensure_ascii=False)
        elif hasattr(data, "to_dict") and callable(data.to_dict):
            # Handle LlamaIndex Document or objects with to_dict method
            json.dump(data.to_dict(), file, indent=4, ensure_ascii=False)
        elif isinstance(data, Document):
            # Handle LlamaIndex Documents specifically
            json.dump(
                {
                    "text": data.text,
                    "doc_id": data.doc_id,
                    "embedding": data.embedding,
                    "metadata": data.metadata,
                    "excluded_embed_metadata_keys": data.excluded_embed_metadata_keys,
                    "excluded_llm_metadata_keys": data.excluded_llm_metadata_keys,
                },
                file,
                indent=4,
                ensure_ascii=False,
            )
        else:
            # Handle regular dict data
            json.dump(data, file, indent=4, ensure_ascii=False)


def _save_csv_data(data: Any, file_path: Path) -> None:
    """Save data as CSV format."""
    with Path(file_path).open("w", encoding="utf-8", newline="") as file:
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                writer = csv.DictWriter(file, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            else:
                writer = csv.writer(file)
                writer.writerows(data)
        else:
            # For non-list data, convert to string and save
            file.write(str(data))


def _save_txt_data(data: Any, file_path: Path) -> None:
    """Save data as text format."""
    if isinstance(data, Node):
        _save_tree_data(data, file_path)
    else:
        with Path(file_path).open("w", encoding="utf-8") as file:
            if isinstance(data, list):
                # Join list items with newlines
                for item in data:
                    file.write(f"{item!s}\n")
            else:
                # Convert to string
                file.write(str(data))


def _save_tree_data(data: Node, file_path: Path) -> None:
    print("came into trees_sitter_save results")
    with Path(file_path).open("w") as f:

        def print_tree_to_file(node: Node, indent=0):
            f.write("  " * indent + node.type + "\n")
            for child in node.children:
                print_tree_to_file(child, indent + 1)

        print_tree_to_file(data)

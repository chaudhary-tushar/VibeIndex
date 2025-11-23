#!/usr/bin/env python3

from src.preprocessing.metadata_extractor import MetadataExtractor
from src.preprocessing.chunk import CodeChunk

extractor = MetadataExtractor()

# Test 1: empty code signature
print("Test 1: empty code signature")
code = ""
signature = extractor.extract_signature(code)
print(f"Result: {repr(signature)}")

# Test 2: complexity with if
print("\nTest 2: complexity with if")
code_if = """if x > 0:
    print("positive")
else:
    print("not positive")"""
complexity = extractor.extract_complexity(code_if)
print(f"Complexity: {complexity}")

# Test 3: complexity with keywords
print("\nTest 3: complexity with keywords")
code_keywords = """if x > 0 and y < 10:
    for item in items:
        if item:
            while condition:
                pass"""
complexity = extractor.extract_complexity(code_keywords)
print(f"Complexity: {complexity}")

# Test 4: enhance_chunk_metadata
print("\nTest 4: enhance_chunk_metadata")
chunk = CodeChunk(
    type="function",
    name="test_function",
    code="def test_function(): pass",
    file_path="test.py",
    language="python",
    start_line=1,
    end_line=1,
    docstring="Original docstring"
)
chunk.tags = ["existing_tag"]

enhanced_chunk = extractor.enhance_chunk_metadata(chunk)
print(f"Enhanced chunk docstring: {enhanced_chunk.docstring}")
print(f"Enhanced chunk tags: {enhanced_chunk.tags}")

# Test 5: line count
print("\nTest 5: line count")
code_10_lines = "\n".join([f"line {i}" for i in range(10)])  # 10 lines
chunk2 = CodeChunk(
    type="function",
    name="multi_line_func",
    code=code_10_lines,
    file_path="test.py",
    language="python",
    start_line=5,
    end_line=14  # This would be 10 lines (14-5+1)
)
enhanced_chunk2 = extractor.enhance_chunk_metadata(chunk2)
print(f"Metadata: {enhanced_chunk2.metadata}")
print(f"Line count in metadata: {'line_count' in enhanced_chunk2.metadata}")

# Test 6: entry point detection
print("\nTest 6: entry point detection")
main_chunk = CodeChunk(
    type="function",
    name="main",
    code="def main(): pass",
    file_path="test.py",
    language="python",
    start_line=1,
    end_line=1
)
enhanced_main = extractor.enhance_chunk_metadata(main_chunk)
print(f"Main metadata: {enhanced_main.metadata}")
print(f"Entry point detection: {'is_entry_point' in enhanced_main.metadata}")
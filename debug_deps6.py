#!/usr/bin/env python3

from src.preprocessing.dependency_mapper import DependencyMapper

mapper = DependencyMapper()

code = """
def helper_function():
    pass

def main_function():
    return helper_function()

class MyClass:
    def method(self):
        return helper_function()
"""

# Test what symbol usage finds without index
deps_no_index = mapper._extract_python_symbol_usage(code, None)
print("Symbol usage without index:", deps_no_index)

# Test what symbol usage finds with index
symbol_index = {
    "helper_function": "path/to/file",
    "MyClass": "path/to/file"
}
deps_with_index = mapper._extract_python_symbol_usage(code, symbol_index)
print("Symbol usage with index:", deps_with_index)

# Test the full extract_dependencies
full_deps = mapper.extract_dependencies(code, "python", symbol_index)
print("Full extract_dependencies with index:", full_deps)
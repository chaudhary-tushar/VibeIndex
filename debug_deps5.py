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

# Define a symbol index with some project symbols
symbol_index = {
    "helper_function": "path/to/file",
    "MyClass": "path/to/file"
}

deps = mapper.extract_dependencies(code, "python", symbol_index)
print("Result with symbol index:", deps)
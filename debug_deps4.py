#!/usr/bin/env python3

from src.preprocessing.dependency_mapper import DependencyMapper

mapper = DependencyMapper()

code = """
class MyClass:
    pass

def my_function():
    return MyClass()

result = my_function()
MyOtherClass = object
"""

# Test the internal function directly
python_imports = mapper._extract_python_imports(code)
print("_extract_python_imports result:", python_imports)

python_symbols = mapper._extract_python_symbol_usage(code, None)
print("_extract_python_symbol_usage result:", python_symbols)

deps = mapper.extract_dependencies(code, "python")
print("Combined extract_dependencies result:", deps)
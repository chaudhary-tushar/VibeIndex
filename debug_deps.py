#!/usr/bin/env python3

from src.preprocessing.dependency_mapper import DependencyMapper

mapper = DependencyMapper()

# Test Python imports
code = """
import os
import sys as system
import collections.abc
from pathlib import Path
from typing import List, Dict
from mymodule.submodule import MyClass
import pandas as pd
"""

deps = mapper.extract_dependencies(code, "python")
print("Python imports:", deps)

# Test Python symbol usage
code2 = """
class MyClass:
    pass

def my_function():
    return MyClass()

result = my_function()
MyOtherClass = object
"""

deps2 = mapper.extract_dependencies(code2, "python")
print("Python symbols:", deps2)

# Test with symbol index
code3 = """
def helper_function():
    pass

def main_function():
    return helper_function()

class MyClass:
    def method(self):
        return helper_function()
"""

symbol_index = {
    "helper_function": "path/to/file",
    "MyClass": "path/to/file"
}

deps3 = mapper.extract_dependencies(code3, "python", symbol_index)
print("Python symbols with index:", deps3)
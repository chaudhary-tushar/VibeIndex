#!/usr/bin/env python3

from src.preprocessing.dependency_mapper import DependencyMapper

mapper = DependencyMapper()

# Test just python imports
code = """
import os
import sys as system
import collections.abc
from pathlib import Path
from typing import List, Dict
from mymodule.submodule import MyClass
import pandas as pd
"""

# Test the internal function directly
python_imports = mapper._extract_python_imports(code)
print("_extract_python_imports result:", python_imports)

# And test the symbol usage
python_symbols = mapper._extract_python_symbol_usage(code)
print("_extract_python_symbol_usage result:", python_symbols)

# Combined result
deps = mapper.extract_dependencies(code, "python")
print("Combined extract_dependencies result:", deps)
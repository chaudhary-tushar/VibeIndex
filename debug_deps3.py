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

for i in range(5):
    python_imports = mapper._extract_python_imports(code)
    print(f"Run {i+1} - _extract_python_imports result:", python_imports)
    
    deps = mapper.extract_dependencies(code, "python")
    print(f"Run {i+1} - Combined result:", deps)
    print()
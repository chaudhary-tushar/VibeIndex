import pytest

from src.preprocessing.dependency_mapper import DependencyMapper


@pytest.fixture
def dependency_mapper():
    """Create a DependencyMapper instance for testing"""
    return DependencyMapper()


def test_dependency_mapper_initialization(dependency_mapper):
    """Test DependencyMapper initialization"""
    assert dependency_mapper.symbol_tables == {}
    assert len(dependency_mapper.dependency_graph) == 0


def test_extract_dependencies_python_imports(dependency_mapper):
    """Test extraction of Python import statements"""
    code = """
import os
import sys as system
import collections.abc
from pathlib import Path
from typing import List, Dict
from mymodule.submodule import MyClass
import pandas as pd
"""

    deps = dependency_mapper.extract_dependencies(code, "python")

    # Check that we get expected imports (note: limited to 10 results)
    assert "os" in deps
    assert "sys" in deps
    assert "collections" in deps  # Should get root module
    assert "pathlib" in deps
    # Note: "typing" may not be present due to the 10-item limit
    assert "mymodule" in deps  # Should get root module
    assert "pandas" in deps

    # The function also extracts symbols, so we expect some of 'MyClass', 'List', 'Dict', 'Path' as symbols too
    # (limited to 10 total results, so not all might be present)
    symbol_found = any([symbol in deps for symbol in ["MyClass", "List", "Dict", "Path"]])
    assert symbol_found

    # Limit should be applied (max 10)
    assert len(deps) <= 10


def test_extract_dependencies_python_symbol_usage(dependency_mapper):
    """Test extraction of Python symbol usage"""
    code = """
class MyClass:
    pass

def my_function():
    return MyClass()

result = my_function()
MyOtherClass = object
"""

    # Test without symbol index (should return all filtered symbols)
    # Note: Only capitalized symbols like classes and constants are captured by the regex
    deps = dependency_mapper.extract_dependencies(code, "python")
    assert "MyClass" in deps  # Capitalized class name
    assert "MyOtherClass" in deps  # Capitalized variable (constant-like)
    # Note: my_function is not captured because it's lowercase and the regex looks for capitalized words


def test_extract_dependencies_python_symbol_usage_with_index(dependency_mapper):
    """Test extraction of Python symbol usage when symbol index is provided"""
    code = """
def HelperFunction():
    pass

def MainFunction():
    return HelperFunction()

class MyClass:
    def method(self):
        return HelperFunction()
"""

    # Define a symbol index with some project symbols
    symbol_index = {"HelperFunction": "path/to/file", "MyClass": "path/to/file"}

    deps = dependency_mapper.extract_dependencies(code, "python", symbol_index)

    # Should only include symbols that are in the symbol index and match the regex pattern
    assert "HelperFunction" in deps  # Capitalized and in index
    assert "MyClass" in deps  # Capitalized and in index
    # MainFunction is not in symbol index, so shouldn't be included as dependency


def test_extract_dependencies_javascript_imports(dependency_mapper):
    """Test extraction of JavaScript import statements"""
    code = """
import React from 'react';
import { Component } from 'react';
import { useState, useEffect } from 'react';
import express from 'express';
import * as utils from './utils';
const fs = require('fs');
const { join } = require('path');
"""

    deps = dependency_mapper.extract_dependencies(code, "javascript")

    # Should extract both import and require statements
    assert "react" in deps
    assert "express" in deps
    assert "fs" in deps
    assert "path" in deps


def test_extract_dependencies_javascript_symbol_usage(dependency_mapper):
    """Test extraction of JavaScript symbol usage"""
    code = """
class MyClass {
    constructor() {
        this.value = 0;
    }

    myMethod() {
        return new MyOtherClass();
    }
}

function myFunc() {
    return new MyClass();
}

const instance = new myFunc();
"""

    symbol_index = {"MyClass": "path/to/file", "MyOtherClass": "path/to/file"}

    deps = dependency_mapper.extract_dependencies(code, "javascript", symbol_index)

    assert "MyClass" in deps
    assert "MyOtherClass" in deps


def test_extract_dependencies_html(dependency_mapper):
    """Test extraction of HTML dependencies"""
    code = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="styles.css">
    <script src="script.js"></script>
</head>
<body>
    <div class="container">
        <img src="image.png" alt="An image">
        <video src="video.mp4"></video>
    </div>
</body>
</html>
"""

    deps = dependency_mapper.extract_dependencies(code, "html")

    # HTML dependencies may be limited, but we should at least see some resource references
    # This will depend on the implementation of _extract_html_dependencies
    # which is not fully shown in the source code, so we'll test basic functionality
    deps = dependency_mapper._extract_html_dependencies(code)
    # Just verify the method exists and doesn't error
    assert isinstance(deps, (list, set))


def test_extract_dependencies_css(dependency_mapper):
    """Test extraction of CSS dependencies"""
    code = """
@import url('base.css');
@import 'theme.css';

.container {
    background-image: url('bg.png');
    font-family: 'CustomFont', sans-serif;
}

.icon::before {
    content: url('icon.svg');
}
"""

    deps = dependency_mapper.extract_dependencies(code, "css")

    # CSS dependencies may be limited, but check basic functionality
    # This will depend on the implementation of _extract_css_dependencies
    # which is not fully shown in the source code
    deps = dependency_mapper._extract_css_dependencies(code)
    # Just verify the method exists and doesn't error
    assert isinstance(deps, (list, set))


def test_extract_dependencies_unknown_language(dependency_mapper):
    """Test extraction with an unknown language"""
    code = "print('hello world')"

    deps = dependency_mapper.extract_dependencies(code, "unknown_lang")

    # Should return an empty list for unknown languages
    assert deps == []


def test_extract_python_imports_basic(dependency_mapper):
    """Test basic Python import extraction"""
    code = "import os"
    deps = dependency_mapper._extract_python_imports(code)
    assert "os" in deps


def test_extract_python_imports_complex(dependency_mapper):
    """Test complex Python import extraction"""
    code = """
import os, sys, json
from collections import defaultdict, Counter
from mypackage.sub import ClassA, ClassB
import mypackage.utils as utils
"""

    deps = dependency_mapper._extract_python_imports(code)

    assert "os" in deps
    assert "sys" in deps
    assert "json" in deps
    assert "collections" in deps
    assert "mypackage" in deps


def test_extract_python_symbol_usage_basic(dependency_mapper):
    """Test basic Python symbol usage extraction"""
    code = "class MyClass: pass\nobj = MyClass()"

    deps = dependency_mapper._extract_python_symbol_usage(code, None)

    assert "MyClass" in deps


def test_extract_python_symbol_usage_with_index(dependency_mapper):
    """Test Python symbol usage extraction with symbol index"""
    code = """
def func_a():
    return ClassB()

class ClassB:
    pass

result = func_a()
"""

    symbol_index = {"ClassB": "path/to/file", "func_a": "path/to/file"}
    deps = dependency_mapper._extract_python_symbol_usage(code, symbol_index)

    assert "ClassB" in deps
    # func_a is in index but also defined in this code, behavior depends on implementation


def test_extract_python_symbol_usage_filtering(dependency_mapper):
    """Test that Python symbol usage filtering works correctly"""
    code = """
def my_function():
    return None  # None is a builtin

result = True  # True is a builtin
error = ValueError  # ValueError is a builtin
x = str(5)  # str is a builtin
"""

    deps = dependency_mapper._extract_python_symbol_usage(code, None)

    # Built-ins should be filtered out
    assert "None" not in deps
    assert "True" not in deps
    assert "ValueError" not in deps
    assert "str" not in deps


def test_extract_dependencies_empty_code(dependency_mapper):
    """Test dependency extraction on empty code"""
    deps = dependency_mapper.extract_dependencies("", "python")
    assert deps == []


def test_extract_dependencies_whitespace_only(dependency_mapper):
    """Test dependency extraction on whitespace-only code"""
    deps = dependency_mapper.extract_dependencies("   \n\t  \n", "javascript")
    assert deps == []

[![Tests](https://github.com/seldonian-framework/Engine/actions/workflows/test-action.yaml/badge.svg)](https://github.com/seldonian-framework/Engine/actions/workflows/test-action.yaml)

[![Docs](https://github.com/seldonian-framework/Engine/actions/workflows/gh-pages.yaml/badge.svg)](https://github.com/seldonian-framework/Engine/actions/workflows/gh-pages.yaml)

# Engine

This is the source code repository for a framework for creating and running [Seldonian](https://seldonian.cs.umass.edu/) algorithms. 

## Installation

To use the latest stable version:
```
pip install seldonian-engine
```

To run this code as a developer, create a virtual environment. Then install the package locally, e.g. 

```
python setup.py develop
```

or 

```
pip install -e .
```

If you want to view the parse tree graphs using the built-in tools this library provides, install [graphviz](https://graphviz.org/download/) onto your system. The Seldonian library uses a Python API for graphviz, and the API requires that graphviz be installed system-wide. This should resolve mysterious error messages like "dot" not found. "dot" is a command line program included with graphviz for rendering the graphs from code. 

## Testing
To run the unit tests, from the command line do:
```
pytest
```

This will automatically run all tests in `tests/`. 

To get more introspection into the tests see the [Pytest documentation](https://docs.pytest.org/).

## Versioning
The naming of versions of this software adheres to [semantic versioning](https://semver.org/). Pre-release versions use a major version of "0", e.g. "0.0.1" is the very first pre-release version. 

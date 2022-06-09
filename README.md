## Disclaimer: This repository is under active development and is not yet suitable for public use. We are currently working on an alpha release and will remove this disclaimer at the time of release. 

# Engine

This is the source code repository for a framework for creating and running [Seldonian](http://aisafety.cs.umass.edu/) algorithms. 

## Installation
To run this code as a developer, create a virtual environment. Then install the package locally, e.g. 

```
python setup.py develop
```

or 

```
pip install -e .
```

If you want to view the parse tree graphs using the built-in tools this library provides, install [graphviz](https://graphviz.org/download/) onto your system. The Seldonian library uses a Python API for graphviz, and the API requires that graphviz be installed system-wide. This should resolve mysterious error messages like "dot" not found. "dot" is a command line program included with graphviz for rendering the graphs from code. 

## Interface

This library is in its infancy, but a lite command line interface exists for parsing string expressions into parse trees. To use this interface, edit the file: `interface.py`. Modify `constraint_str` and `delta` to values of interest. Then run (make sure you have activated the `seldo` conda environment first) from the command line:
```
python interface.py
```
As a quick test, if you use the following variables:
```
constraint_str = 'abs((Mean_Error | [M]) - (Mean_Error | [F])) - 0.1'
delta = 0.05
```
The resulting diagram should match [this one](example_graph.pdf), although the actual intervals may be different due to the random seeding of the base variable intervals. 

## Testing
To run the unit tests, from the command line do:
```
pytest
```

This will automatically run all tests in `tests/`. Currently, there is only one test file in this folder, called `test_parse_tree.py`. Pytest will find that file and run all tests within that file. 

To get more introspection into the tests see the [Pytest documentation](https://docs.pytest.org/).

## Versioning
The naming of versions of this software adheres to [semantic versioning](https://semver.org/). Pre-release versions use a major version of "0", e.g. "0.0.1" is the very first pre-release version. 
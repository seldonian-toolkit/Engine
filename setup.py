import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seldonian_engine",
    version="0.4.1",
    author="Austin Hoag",
    author_email="austinthomashoag@gmail.com",
    description="Core library for Seldonian algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="",
    project_urls={
        "Bug Tracker": "https://github.com/seldonian-framework/Engine/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "autograd>=1.4",
        "cma>=3.2.2",
        "graphviz>=0.19.1",
        "matplotlib==3.5.1", 
        "numpy>=1.21.4",
        "pandas>=1.4.1",
        "pytest>=7.0.1",
        "scikit_learn>=1.1.1",
        "scipy>=1.7.3",
        "tqdm>=4.64.0",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
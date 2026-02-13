"""
Setup script for ScalarGrad package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scalargrad",
    version="1.0.4",
    author="Thamizharasu Saravanan",
    author_email="amizharasu@gmail.com",
    description="A minimal scalar-valued autograd engine for automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/THAMIZH-ARASU/scalargrad",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "viz": ["graphviz>=0.19"],
        "plot": ["matplotlib>=3.3.0", "numpy>=1.19.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "scikit-learn>=0.24.0",
        ],
        "all": ["graphviz>=0.19", "matplotlib>=3.3.0", "numpy>=1.19.0"],
    },
    keywords="autograd automatic-differentiation neural-networks deep-learning machine-learning",
    project_urls={
        "Documentation": "https://github.com/THAMIZH-ARASU/scalargrad#readme",
        "Source": "https://github.com/THAMIZH-ARASU/scalargrad",
        "Bug Tracker": "https://github.com/THAMIZH-ARASU/scalargrad/issues",
    },
)

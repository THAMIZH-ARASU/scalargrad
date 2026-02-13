# ScalarGrad Development & Publishing Guide

This guide will help you develop, test, and publish the ScalarGrad package to PyPI.

## Project Structure

```
scalargrad/
├── scalargrad/              # Main package
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   ├── core.py              # Core Scalar and autograd
│   ├── config.py            # Configuration
│   ├── loss.py              # Loss functions
│   ├── training.py          # Training utilities
│   ├── visualization.py     # Visualization tools
│   ├── nn/                  # Neural network module
│   │   ├── __init__.py
│   │   ├── module.py        # Base Module class
│   │   └── layers.py        # Layer implementations
│   └── optim/               # Optimizers module
│       ├── __init__.py
│       ├── optimizer.py     # Base Optimizer
│       └── optimizers.py    # SGD, Adam
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── test_loss.py
├── examples/                # Example scripts
│   ├── basic_operations.py
│   └── neural_network.py
├── setup.py                 # Setup configuration
├── pyproject.toml           # Modern packaging config
├── requirements.txt         # Optional dependencies
├── requirements-dev.txt     # Dev dependencies
├── README.md                # Documentation
├── LICENSE                  # MIT License
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Contribution guidelines
├── MANIFEST.in              # Package manifest
└── .gitignore               # Git ignore rules
```

## Development Workflow

### 1. Install in Development Mode

```bash
# Navigate to project root
cd scalargrad

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install optional dependencies
pip install -r requirements.txt
```

### 2. Running the Package

```bash
# Run as module
python -m scalargrad

# Run examples
python examples/basic_operations.py
python examples/neural_network.py

# Use in Python
python
>>> from scalargrad import Scalar, MLP, Adam
>>> # Your code here
```

### 3. Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scalargrad --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# View coverage report
# Open htmlcov/index.html in browser
```

### 4. Code Quality

```bash
# Format code with black
black scalargrad/ tests/ examples/

# Check code style with flake8
flake8 scalargrad/ --max-line-length=100

# Type checking with mypy (optional)
mypy scalargrad/
```

### 5. Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes to the code

3. Add tests for new functionality

4. Update documentation if needed

5. Run tests to ensure everything works:
   ```bash
   pytest
   ```

6. Commit and push:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/your-feature
   ```

## Publishing to PyPI

### Prerequisites

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Create TestPyPI account** (for testing): https://test.pypi.org/account/register/
3. **Install build tools**:
   ```bash
   pip install build twine
   ```

### Step 1: Update Version

Update version in:
- `scalargrad/__init__.py`
- `pyproject.toml`
- `setup.py`

### Step 2: Update Changelog

Add your changes to `CHANGELOG.md`:
```markdown
## [<version>] - YYYY-MM-DD

### Added
- New feature X

### Fixed
- Bug in Y
```

### Step 3: Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution packages
python -m build
```

This creates:
- `dist/scalargrad-<version>.tar.gz` (source distribution)
- `dist/scalargrad-<version>-py3-none-any.whl` (wheel)

### Step 4: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ scalargrad
```

### Step 5: Upload to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# You'll be prompted for your PyPI username and password
```

### Step 6: Verify Installation

```bash
# Install from PyPI
pip install scalargrad

# Test it works
python -c "from scalargrad import Scalar; print('Success!')"
```

### Step 7: Create GitHub Release

1. Tag the release:
   ```bash
   git tag -a <version> -m "Release version <version>"
   git push origin <version>
   ```

2. Create release on GitHub:
   - Go to repository → Releases → Create new release
   - Select the tag
   - Add release notes from CHANGELOG.md
   - Attach distribution files

## Using API Tokens (Recommended)

Instead of username/password, use API tokens:

1. **Generate PyPI token**:
   - Go to PyPI Account Settings → API tokens
   - Create token with scope for this project

2. **Configure token**:
   ```bash
   # Create/edit ~/.pypirc
   [pypi]
   username = __token__
   password = pypi-your-token-here

   [testpypi]
   username = __token__
   password = pypi-your-test-token-here
   ```

3. **Upload with token**:
   ```bash
   python -m twine upload dist/*
   ```

## Continuous Integration (Optional)

Consider setting up GitHub Actions for:
- Automatic testing on push/PR
- Code coverage reporting
- Automatic PyPI publishing on release

Example `.github/workflows/test.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest --cov=scalargrad
```

## Troubleshooting

### Import Errors After Installation

```bash
# Reinstall in development mode
pip uninstall scalargrad
pip install -e .
```

### Test Failures

```bash
# Run with verbose output
pytest -v -s

# Run specific test
pytest tests/test_core.py::TestScalarBasicOps::test_addition -v
```

### Build Issues

```bash
# Clear cache and rebuild
rm -rf build/ dist/ *.egg-info __pycache__
python -m build
```

## Useful Commands

```bash
# Check package info
pip show scalargrad

# List package files
python setup.py --version
python setup.py --name
python setup.py --author

# Validate package
twine check dist/*

# Generate requirements from environment
pip freeze > requirements-freeze.txt
```

## Next Steps

After publishing:

1. ✅ Update README badges with PyPI version
2. ✅ Share on social media / communities
3. ✅ Write blog post or tutorial
4. ✅ Set up documentation site (Read the Docs, GitHub Pages)
5. ✅ Add more examples and tutorials
6. ✅ Respond to issues and PRs
7. ✅ Plan next version features

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

Good luck with your package! 🚀

# Contributing to ScalarGrad

Thank you for considering contributing to ScalarGrad! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and inclusive in all interactions with the community.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/THAMIZH-ARASU/scalargrad/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example

### Suggesting Features

1. Check existing [Issues](https://github.com/THAMIZH-ARASU/scalargrad/issues) for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add docstrings to new functions/classes
   - Add type hints where appropriate

4. **Add tests**
   - Write tests for new functionality
   - Ensure all tests pass: `pytest`
   - Maintain or improve code coverage

5. **Update documentation**
   - Update README.md if needed
   - Add docstrings following NumPy style
   - Update CHANGELOG.md

6. **Commit your changes**
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI passes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/THAMIZH-ARASU/scalargrad.git
cd scalargrad

# Install in development mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt
```

## Code Style

- Follow PEP 8 guidelines
- Use `black` for code formatting: `black scalargrad/`
- Use `flake8` for linting: `flake8 scalargrad/`
- Maximum line length: 100 characters

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scalargrad --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestScalarBasicOps::test_addition
```

## Documentation

- Use NumPy-style docstrings
- Include type hints
- Provide examples in docstrings for complex functions

Example:
```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    Brief description.
    
    Longer description providing more details about what
    the function does and how to use it.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg1 is negative
    
    Example:
        >>> result = function_name(5, "test")
        >>> print(result)
        True
    """
    pass
```

## Commit Message Guidelines

Format: `type: brief description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
- `feat: add RMSprop optimizer`
- `fix: correct gradient calculation in sigmoid`
- `docs: update installation instructions`

## Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Questions?

Feel free to open an issue for any questions!

Thank you for contributing! 🎉

# Contributing to warspite-financial

Thank you for your interest in contributing to warspite-financial! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/warspite-financial.git
cd warspite-financial
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e .[dev,providers]
```

4. **Verify the setup:**
```bash
python3 -m pytest tests/unit/ -v
```

### Development Workflow

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes and add tests**

3. **Run the test suite:**
```bash
python3 -m pytest
```

4. **Commit your changes:**
```bash
git commit -m "Add feature: description of your changes"
```

5. **Push and create a pull request**

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints throughout the codebase
- Write clear, descriptive docstrings for all public functions and classes
- Keep functions focused and modular

### Testing Requirements

All contributions must include appropriate tests:

#### Unit Tests
- Test individual functions and classes
- Focus on specific functionality and edge cases
- Use descriptive test names that explain what is being tested

#### Property-Based Tests
- Use Hypothesis for testing universal properties
- Test with randomized inputs to catch edge cases
- Ensure properties hold across all valid inputs

#### Integration Tests
- Test complete workflows and component interactions
- Verify end-to-end functionality
- Test with realistic data and scenarios

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── property/       # Property-based tests using Hypothesis
└── integration/    # End-to-end integration tests
```

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run specific test categories
python3 -m pytest -m unit
python3 -m pytest -m property
python3 -m pytest -m integration

# Run with coverage
python3 -m pytest --cov=warspite_financial

# Run specific test file
python3 -m pytest tests/unit/test_providers.py -v
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Python version and dependency versions
- Minimal code example demonstrating the issue

### Feature Requests

For new features, please:
- Describe the use case and motivation
- Explain how it fits with the library's goals
- Consider backward compatibility
- Provide examples of the proposed API

### Code Contributions

#### New Providers
When adding data providers:
- Inherit from `BaseProvider` or `TradingProvider`
- Implement all required abstract methods
- Add comprehensive error handling
- Include both unit and integration tests
- Update documentation with usage examples

#### New Strategies
When adding trading strategies:
- Inherit from `BaseStrategy`
- Implement `generate_positions()` method
- Add parameter validation and management
- Include property-based tests for signal generation
- Document the strategy's logic and parameters

#### New Visualizations
When adding renderers:
- Inherit from `WarspiteDatasetRenderer`
- Support common visualization options
- Handle edge cases gracefully
- Add tests for various data scenarios
- Update examples to demonstrate usage

### Documentation

- Update README.md for new features
- Add docstrings with examples for new functions
- Update CHANGELOG.md with your changes
- Consider adding example workflows

## Architecture Guidelines

### Design Principles

1. **Modularity**: Keep components loosely coupled and highly cohesive
2. **Extensibility**: Design interfaces that allow easy extension
3. **Performance**: Use numpy arrays for data operations
4. **Error Handling**: Provide clear, actionable error messages
5. **Testing**: Ensure comprehensive test coverage

### Code Organization

```
warspite_financial/
├── providers/      # Data provider implementations
├── strategies/     # Trading strategy implementations
├── datasets/       # Dataset management and serialization
├── emulator/       # Trading emulation engine
├── visualization/  # Chart and report generation
├── utils/          # Utility functions and error handling
└── examples/       # End-to-end workflow examples
```

### Error Handling

- Use custom exception types from `warspite_financial.utils.exceptions`
- Provide clear error messages with actionable guidance
- Handle external API failures gracefully
- Validate inputs at public API boundaries

### Performance Considerations

- Use numpy arrays for numerical operations
- Implement lazy loading where appropriate
- Consider memory usage for large datasets
- Profile performance-critical code paths

## Review Process

### Pull Request Guidelines

1. **Clear Description**: Explain what your PR does and why
2. **Test Coverage**: Include tests for new functionality
3. **Documentation**: Update relevant documentation
4. **Backward Compatibility**: Avoid breaking existing APIs
5. **Small Focused Changes**: Keep PRs focused on a single feature/fix

### Review Criteria

- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance impact
- Backward compatibility
- Security considerations

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Run full test suite
4. Update documentation
5. Create release tag
6. Publish to PyPI

## Getting Help

- **Questions**: Open a GitHub issue with the "question" label
- **Discussions**: Use GitHub Discussions for broader topics
- **Documentation**: Check the README and inline documentation
- **Examples**: Look at the examples in `warspite_financial/examples/`

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

Thank you for contributing to warspite-financial!
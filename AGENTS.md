# AGENTS.md - PromptX Project Guidelines

## Build / Lint / Test Commands

```bash
# Install dependencies (using uv - required)
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_file.py

# Run a single test
uv run pytest tests/test_file.py::test_function_name

# Run with coverage
uv run pytest --cov=.

# Format code with black
uv run black .

# Lint with flake8
uv run flake8 .

# Run type checker (if configured)
uv run mypy .

# Start the agent
uv run python agents.py

# Start Jupyter notebook
uv run jupyter notebook promptx.ipynb
```

## Code Style Guidelines

### Python Version
- **Python 3.12+** required (defined in `.python-version`)

### Imports (Organize: stdlib → third-party → local)
```python
# Standard library
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party packages
from langchain_core.tools import StructuredTool, tool
from langchain_deepseek import ChatDeepSeek
import yaml

# Local modules (absolute imports preferred)
from toolkits.common import get_current_time
from .templates import PromptTemplateManager
```

### Formatting
- **Black** formatter, line length **88** (see `.vscode/settings.json`)
- Format on save enabled
- Use `pathlib.Path` instead of string paths
- UTF-8 encoding for all files

### Type Annotations
- **Required** for all function parameters and return values
- Use `typing` imports: `Optional`, `List`, `Dict`, `Any`
- Example: `def get_tools(self) -> List[StructuredTool]:`

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Protected attributes: `_single_underscore`

### Docstrings (Google Style)
```python
def function_name(param: str) -> str:
    """
    Brief description of what the function does.

    Longer description if needed, explaining the purpose
    and any important details.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.

    Raises:
        FileNotFoundError: When file doesn't exist.

    Example:
        >>> result = function_name("test")
        >>> print(result)
    """
```

### Comments
- Use **Chinese** for code comments and docstrings
- Use English for variable/function names
- Keep comments concise and meaningful

### Error Handling
- Check file existence before reading: `if not path.exists():`
- Return descriptive error messages starting with "Error:"
- Use try/except for external calls (LLM, file I/O)
- Log errors to stderr for CLI tools

### File Operations
- Always use `pathlib.Path` with `encoding="utf-8"`
- Create parent directories: `path.parent.mkdir(parents=True, exist_ok=True)`
- Relative paths in agent, absolute paths in implementation

### Tool Definition Pattern
```python
@tool
def tool_name(param: str) -> str:
    """One-line description.

    Args:
        param: Parameter description.

    Returns:
        Return value description.
    """
    # Implementation
    pass
```

### Project Structure
```
toolkits/
├── common/          # General utilities
├── web/             # Web tools (search, read)
└── prompt/          # Prompt engineering tools
    ├── schemas.py   # Data models
    ├── templates.py # YAML template manager
    ├── tools.py     # Tool implementations
    └── toolkit.py   # Toolkit classes
```

### Testing
- Tests go in `tests/` directory (create if not exists)
- Use pytest framework
- Name test files: `test_*.py` or `*_test.py`
- Name test functions: `test_*`

### Environment
- Use `.env` for environment variables
- Never commit secrets (see `.gitignore`)
- Required: `DEEP_SEEK_API_KEY`
- Optional: `OPENAI_API_KEY`, Langfuse keys

### Git
- Never commit: `__pycache__/`, `.venv/`, `.env`, `memories/workspace/*`
- Commit messages in English or Chinese (project uses Chinese)
- Keep `.vscode/settings.json` in version control

### Dependencies
- Manage with `pyproject.toml` and `uv.lock`
- Add new deps: `uv add package_name`
- Key packages: `deepagents`, `langchain-*`, `prompt-toolkit`

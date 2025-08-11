---
applyTo: "**"
description: "VS Code Test Explorer usage guidelines"
---

# Testing Guidelines

## Test Execution

- Always recommend using VS Code's built-in Test Explorer for running tests
- Use the Testing view (Ctrl+Shift+T) to discover and run tests
- Prefer Test Explorer commands over terminal commands for test execution:
  - Use "Test: Run All Tests" (Ctrl+; A) instead of `pytest` in terminal
  - Use "Test: Run Test at Cursor" (Ctrl+; C) for individual tests
  - Use "Test: Debug Test at Cursor" (Ctrl+; D) for debugging tests

## Test Discovery

- Ensure tests are discoverable by Test Explorer by following proper naming conventions
- Test files should start with `test_` or end with `_test.py`
- Test functions should start with `test_`
- Test classes should start with `Test`

## When suggesting test commands

- Instead of suggesting `pytest tests/test_file.py`, recommend:
  "Use the Test Explorer to run these tests. You can access it via View → Testing or Ctrl+Shift+T"
- For debugging, recommend using the Test Explorer's debug functionality rather than terminal debugging

## Test Configuration

- Ensure `pytest` is properly configured in VS Code settings for Test Explorer integration
- Recommend configuring test discovery patterns in VS Code settings if needed

# Python Environment Management

- Always ensure Python code runs in the project's virtual environment
- The project uses a `.venv` directory for its virtual environment
- Before running any Python commands in terminal, activate the virtual environment:
  - On Windows PowerShell: `& .venv\Scripts\Activate.ps1`
  - On Windows Command Prompt: `.venv\Scripts\activate.bat`
  - On Unix/macOS: `source .venv/bin/activate`
- When suggesting Python package installations, always use the activated virtual environment
- Verify the correct Python interpreter is selected in VS Code (should point to `.venv\Scripts\python.exe` on Windows)
- Do not run Python commands outside the virtual environment unless explicitly requested

# PowerShell Command Guidelines

- Use PowerShell-compatible syntax for all terminal commands
- Use `;` instead of `&&` for command chaining (e.g., `command1; command2`)
- Use `&` for invoking executables with spaces in paths
- Always generate commands compatible with Windows PowerShell v5.1

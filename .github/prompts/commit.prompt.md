# Commit Workflow

Follow this workflow when committing changes:

1. **Identify changed files** - Determine all files edited in the current session
2. **Lint and format** - Run `uv run ruff check --fix` and `uv run ruff format` on changed files
3. **Stage selectively** - Use `git add <file>` for specific files, never `git add -A`
4. **Commit with conventional format**:
   - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
   - Subject: imperative mood, lowercase, no period
   - Body: explain WHY, not just what changed
5. **Fix pre-commit errors** - Iterate until the commit is clean
6. **Push** - Push to remote

## Conventional Commit Format

```
<type>: <subject>

<body>
```

### Example

```
feat: add semantic search for physics domains

Enables users to search across IDS entries using natural language
queries. Uses sentence-transformers for embedding generation and
FAISS for similarity search.

Closes #42
```

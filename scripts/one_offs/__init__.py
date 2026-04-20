"""One-off scripts — research harnesses, data audits, migration helpers.

Scripts here are not part of the main imas-codex package surface. They
are invoked directly (``uv run python scripts/one_offs/<name>.py``) for
one-time or infrequent tasks such as plan-specific audits.

Each script should be self-documenting (module docstring explains why
it exists and how to run it) and read-only against the graph unless
explicitly called out otherwise.
"""

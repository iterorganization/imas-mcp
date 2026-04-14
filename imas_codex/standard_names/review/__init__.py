"""Standard-name review pipeline package.

Layers:
  - **audits** (Layer 1): deterministic pre-flight quality checks
  - **enrichment**: cluster reconstruction, batching, neighborhood context
  - **budget**: concurrent-safe LLM cost budget management
  - **state**: shared state for the review pipeline
  - **pipeline**: EXTRACT → ENRICH → REVIEW → PERSIST orchestrator
"""

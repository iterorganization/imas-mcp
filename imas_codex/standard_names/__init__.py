"""Standard name generation pipeline.

Multi-source pipeline that extracts physics quantities from graph entities
(DD paths, facility signals), composes grammatically valid standard names
via LLM, validates them, and publishes to the catalog.

Pipeline phases: EXTRACT → COMPOSE → VALIDATE
(REVIEW and PUBLISH are future features 06 and 08)
"""

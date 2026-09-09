"""Concepts - "tribal knowledge": which entities/topics a corpus's own
documents refer to, and the terms a person and a filing each use for them.
Distinct from the dictionary, which governs formulas and interpretation
policy, not meaning.

How this gets CONSULTED by resolution or retrieval is TBD - not yet decided,
let alone built. This package is persistence only: `repository.py` mirrors
`dictionary/repository.py`'s shape exactly (same {bucket}.{scope}.concepts
pattern, same `path=` test seam), proving the collection holds real content
before that consuming design lands - the same discipline that already
justified the collection existing, empty, ahead of any content in it.
"""
from .repository import load, save  # noqa: F401

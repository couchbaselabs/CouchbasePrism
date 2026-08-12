"""PRISM — governed retrieval over document corpora on Couchbase.

Three tiers (design/architecture.md):
  catalog     one doc per source file  — "which document is this?"
  retrieval   one doc per chunk        — "what does it say, here?"
  dictionary  one doc per concept      — "how is this defined, and who says so?"
"""

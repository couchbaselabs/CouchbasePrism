"""Corpus adapters.

A corpus module exposes:
    NAME                       str
    questions(company, limit)  -> [{id, question, expected_answer, doc_name, company, evidence}]
    documents(company)         -> [(doc_name, pathlib.Path)]

Each question dict may also carry `formula`, `concept`, `keywords`, `tags`,
`gold_answer` and `lineage` - not every adapter sets all of these, so any
consumer of questions() must treat them as optional and tolerate their
absence rather than assume every corpus provides them.

A corpus is a test suite, not the system - nothing in `prism/` should
import from here.
"""
import importlib

Corpus = object  # marker for type documentation; adapters are plain modules


def load(name: str):
    return importlib.import_module(f"eval.corpora.{name}")

"""Corpus adapters.

A corpus module exposes:
    NAME                       str
    questions(company, limit)  -> [{id, question, expected_answer, doc_name, company}]
    documents(company)         -> [(doc_name, pathlib.Path)]

FinanceBench is the first one. It is a test suite, not the system - nothing in
`prism/` should import from here.
"""
import importlib

Corpus = object  # marker for type documentation; adapters are plain modules


def load(name: str):
    return importlib.import_module(f"eval.corpora.{name}")

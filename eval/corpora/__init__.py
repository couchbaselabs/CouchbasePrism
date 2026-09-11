"""Corpus adapters.

A corpus module exposes:
    NAME                       str
    questions(company, limit)  -> [{id, question, answer, doc_names, company, evidence}]
    documents(company)         -> [(doc_name, pathlib.Path)]

`doc_names` is always a list, even for a single-document question - one
type for every consumer to handle, and a multi-document question needs it
anyway. `answer` is markdown, rendered as-is by a UI. `evidence` is a list
of flat strings ("DOC: Page N · Section · type"), not objects.

Each question dict may also carry `formula`, `number`, `title`, `category`,
`intent`, `verified` and `notes` - not every adapter sets all of these, so
any consumer of questions() must treat them as optional and tolerate their
absence rather than assume every corpus provides them. `intent` is
audience-facing (a UI renders it, unlike author-facing `notes`) - one or
two sentences on what the question would catch in a weaker system, not
what the product is good at.

A corpus is a test suite, not the system - nothing in `prism/` should
import from here.
"""
import importlib

Corpus = object  # marker for type documentation; adapters are plain modules


def load(name: str):
    return importlib.import_module(f"eval.corpora.{name}")

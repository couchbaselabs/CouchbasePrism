"""Short names a reader uses that the filing never prints.

Resolution matches the subject using the company name on the cover page and the
document name's own prefix. Between them those cover abbreviations that are
prefixes of the full name - "JPM" for JPMorgan Chase, "MGM" for MGM Resorts
International. They cannot cover the rest: "JnJ" is not a prefix of JOHNSON &
JOHNSON, and "AMEX" is not a prefix of AMERICAN EXPRESS. Those accounted for 9 of
21 subject-resolution failures measured across a multi-company eval corpus.

This is the one place a model is asked for knowledge rather than for a reading of
a document, so it is deliberately narrow:

  - It runs once per COMPANY, not per document - 40 calls for 354 documents.
  - The result is stored in the catalog as data, with provenance, not compiled
    into a prompt or a code constant.
  - A wrong alias is close to harmless. Aliases only ever ADD candidates, and a
    full-name match always outscores an alias match, so a hallucinated alias can
    only matter for a question that names no real subject at all.

Anything that IS printed on the document is still extracted from the document.
"""
import json

from .. import llm

ALIAS_SYSTEM_PROMPT = """\
You are listing the short forms by which an organisation is commonly known.

Given its full legal name, return the abbreviations, initialisms and short names
that a reader would plausibly use to refer to it in a question.

Return ONLY one valid JSON object:

{
  "aliases": ["<short form>"]
}

GUIDELINES:

- Include only forms in genuine common use. An invented abbreviation is worse
  than a missing one.
- Include the stock ticker only if it is commonly used as the organisation's
  name in prose.
- Do not include the full legal name, and do not include corporate suffixes on
  their own.
- Do not include forms shorter than three characters; they collide with other
  organisations.
- Return an empty array if the organisation has no commonly used short form.
- Return strictly valid JSON without markdown or explanatory text.
"""


def propose_aliases(company: str, model=None) -> list:
    """Short forms for one company. Failure returns nothing rather than raising:
    a catalog without aliases resolves worse, but a catalog build that dies
    halfway is worse still."""
    if not company:
        return []
    try:
        raw = llm.chat_json(ALIAS_SYSTEM_PROMPT, json.dumps({"legal_name": company}),
                            model=model, stage="catalog").get("aliases", [])
    except Exception:
        return []
    seen, out = set(), []
    for alias in raw:
        if not isinstance(alias, str):
            continue
        alias = alias.strip()
        key = alias.upper()
        if len(alias) >= 3 and key not in seen and key != (company or "").upper():
            seen.add(key)
            out.append(alias)
    return out

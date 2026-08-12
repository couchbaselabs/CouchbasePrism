#!/usr/bin/env python
"""Operator commands.

    python manage.py build-catalog --corpus financebench --company 3M
    python manage.py approve "quick ratio" \
        --formula "(total_current_assets - inventory) / total_current_liabilities" \
        --healthy-at-or-above 1.0
    python manage.py show-dictionary

`approve` is the governed correction from ADR-0001 - the entire human step in
the loop. Everything before it is the system proposing; everything after it is
deterministic.
"""
import argparse
import sys

import yaml

from eval.corpora import load as load_corpus
from prism import catalog, config, couchbase_io, dictionary


def build_catalog(args):
    corpus = load_corpus(args.corpus)
    docs = corpus.documents(company=args.company)
    if not docs:
        sys.exit(f"no documents matched company={args.company!r} in {corpus.NAME}")
    couchbase_io.ensure_primary_index(config.CATALOG_COLLECTION)
    print(f"building catalog for {len(docs)} documents -> "
          f"{config.BUCKET}.{config.SCOPE}.{config.CATALOG_COLLECTION}", file=sys.stderr)
    for i, (doc_name, path) in enumerate(docs, 1):
        try:
            document = catalog.build_from_pdf(str(path), doc_name, model=args.model)
            catalog.upsert(document)
            print(f"  [{i}/{len(docs)}] {doc_name}: "
                  f"company={(document['company'] or {}).get('value')!r} "
                  f"type={(document['doc_type'] or {}).get('value')!r} "
                  f"period={document['doc_period']}", file=sys.stderr)
        except Exception as e:
            print(f"  [{i}/{len(docs)}] {doc_name}: ERROR {e}", file=sys.stderr)


def approve(args):
    dictionary.approve(args.concept, args.formula, args.healthy_at_or_above)
    print(f"approved {args.concept!r}: {args.formula}"
          + (f"  (policy: healthy >= {args.healthy_at_or_above})"
             if args.healthy_at_or_above is not None else "")
          + f"\nwritten to {config.DICTIONARY_PATH}", file=sys.stderr)


def show_dictionary(args):
    data = dictionary.load()
    if not data.get("entries"):
        print("(empty — PRISM operates fine like this; entries arrive from use)")
        return
    print(yaml.safe_dump(data, sort_keys=False, width=100))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-catalog")
    b.add_argument("--corpus", default="financebench")
    b.add_argument("--company", default=None)
    b.add_argument("--model", default=None)
    b.set_defaults(func=build_catalog)

    a = sub.add_parser("approve")
    a.add_argument("concept")
    a.add_argument("--formula", required=True)
    a.add_argument("--healthy-at-or-above", type=float, default=None)
    a.set_defaults(func=approve)

    s = sub.add_parser("show-dictionary")
    s.set_defaults(func=show_dictionary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

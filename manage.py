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
    sectors = corpus.sectors() if hasattr(corpus, "sectors") else {}

    if not args.all:
        # Catalog only what is retrievable. Ingestion fails per document, and an
        # entry with no chunks makes resolution point at an empty document.
        ingested = catalog.ingested_doc_names()
        skipped = [n for n, _ in docs if n not in ingested]
        docs = [(n, p) for n, p in docs if n in ingested]
        if skipped:
            print(f"skipping {len(skipped)} document(s) with no chunks ingested "
                  f"(use --all to override): {', '.join(sorted(skipped)[:6])}"
                  + (" ..." if len(skipped) > 6 else ""), file=sys.stderr)
        if not docs:
            sys.exit("no catalogued documents have chunks; is the workflow finished?")
    print(f"building catalog for {len(docs)} documents -> "
          f"{config.BUCKET}.{config.SCOPE}.{config.CATALOG_COLLECTION}", file=sys.stderr)
    for i, (doc_name, path) in enumerate(docs, 1):
        try:
            document = catalog.build_from_pdf(str(path), doc_name, model=args.model,
                                              gics_sector=sectors.get(doc_name))
            catalog.upsert(document)
            print(f"  [{i}/{len(docs)}] {doc_name}: "
                  f"company={(document['company'] or {}).get('value')!r} "
                  f"type={(document['doc_type'] or {}).get('value')!r} "
                  f"period={document['doc_period']} "
                  f"sector={document.get('gics_sector')!r}", file=sys.stderr)
        except Exception as e:
            print(f"  [{i}/{len(docs)}] {doc_name}: ERROR {e}", file=sys.stderr)


def add_aliases(args):
    """Once per company, not per document: 40 model calls cover 354 documents."""
    names = catalog.companies()
    if not names:
        sys.exit("catalog has no companies; build it first")
    print(f"proposing aliases for {len(names)} companies", file=sys.stderr)
    for i, company in enumerate(names, 1):
        aliases = catalog.propose_aliases(company, model=args.model)
        updated = catalog.set_aliases(company, aliases) if aliases else 0
        print(f"  [{i}/{len(names)}] {company}: {aliases or '(none)'} "
              f"-> {updated} document(s)", file=sys.stderr)


def backfill_periods(args):
    """Recompute doc_period from data already in the catalog - no re-extraction.
    Applies the 52/53-week fiscal-year rule and the document-name fallback."""
    fixed = 0
    for row in catalog.all_periods():
        name, current = row["doc_name"], row.get("doc_period")
        iso = row.get("period_end_date_iso")
        wanted, source = catalog.fiscal_year(iso), "fiscal_year_of_period_end"
        if wanted is None:
            wanted, source = catalog.period_from_doc_name(name), "document_name"
        if wanted is not None and wanted != current:
            catalog.set_period(name, wanted, source)
            print(f"  {name}: {current} -> {wanted}  ({source})", file=sys.stderr)
            fixed += 1
    print(f"updated {fixed} catalog document(s)", file=sys.stderr)


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


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def forget(args):
    """Drop one concept, so the ungoverned path can be demonstrated again for
    it without discarding everything else that has been reviewed."""
    entry = dictionary.find_metric(dictionary.load(), args.concept)
    if entry is None:
        sys.exit(f"no approved entry matches {args.concept!r}")
    print(f"will remove {entry['id']}: {entry['interpretation']['formula']}",
          file=sys.stderr)
    if not _confirm("Remove it?", args.yes):
        sys.exit("aborted")
    for removed in dictionary.forget(args.concept):
        print(f"removed {removed}", file=sys.stderr)


def reset_dictionary(args):
    """Empty the dictionary — the cold half of the two-pass demo."""
    entries = dictionary.load().get("entries", [])
    if not entries:
        print("already empty", file=sys.stderr)
        return
    for entry in entries:
        print(f"will remove {entry.get('id')}", file=sys.stderr)
    if not _confirm(f"Remove all {len(entries)} entries?", args.yes):
        sys.exit("aborted")
    dictionary.clear()
    print(f"cleared {config.DICTIONARY_PATH}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-catalog")
    b.add_argument("--all", action="store_true",
                   help="catalog every PDF, including ones with no chunks ingested")
    b.add_argument("--corpus", default="financebench")
    b.add_argument("--company", default=None)
    b.add_argument("--model", default=None)
    b.set_defaults(func=build_catalog)

    bp = sub.add_parser("backfill-periods")
    bp.set_defaults(func=backfill_periods)

    al = sub.add_parser("add-aliases")
    al.add_argument("--model", default=None)
    al.set_defaults(func=add_aliases)

    a = sub.add_parser("approve")
    a.add_argument("concept")
    a.add_argument("--formula", required=True)
    a.add_argument("--healthy-at-or-above", type=float, default=None)
    a.set_defaults(func=approve)

    s = sub.add_parser("show-dictionary")
    s.set_defaults(func=show_dictionary)

    f = sub.add_parser("forget", help="drop one concept's entry and its policy")
    f.add_argument("concept")
    f.add_argument("--yes", action="store_true", help="skip confirmation")
    f.set_defaults(func=forget)

    r = sub.add_parser("reset-dictionary", help="empty the dictionary (cold demo)")
    r.add_argument("--yes", action="store_true", help="skip confirmation")
    r.set_defaults(func=reset_dictionary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Operator commands.

    python manage.py build-catalog --company 3M
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
from prism import initialize as prism_initialize


def build_catalog(args):
    """Reads each cover page from chunks already ingested by the Couchbase AI
    Data Plane workflow, not the PDF - `corpus.documents()` still decides
    WHICH documents belong to this corpus/company and `corpus.sectors()`
    still supplies gics_sector (corpus-external metadata, never extracted
    from the document itself); only the source of the cover-page TEXT
    changed, from PyMuPDF to chunks."""
    corpus = load_corpus(args.corpus)
    docs = corpus.documents(company=args.company)
    if not docs:
        sys.exit(f"no documents matched company={args.company!r} in {corpus.NAME}")
    couchbase_io.ensure_primary_index(config.CATALOG_COLLECTION)
    sectors = corpus.sectors() if hasattr(corpus, "sectors") else {}

    if not args.all:
        # Catalog only what is retrievable. Ingestion fails per document, and an
        # entry with no chunks makes resolution point at an empty document - and
        # now, with nothing at all to read a cover page from.
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
    for i, (doc_name, _path) in enumerate(docs, 1):
        try:
            document = catalog.build_from_chunks(doc_name, model=args.model,
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
    print(f"will remove {entry['id']}: {entry['formula']}", file=sys.stderr)
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
    print(f"cleared {config.BUCKET}.{config.SCOPE}.{config.DICTIONARY_COLLECTION}",
          file=sys.stderr)


def initialize(args):
    """Empty and rebuild the catalog, empty the dictionary, and rebuild the
    search index from design/fts-index.json - the one thing a user runs after
    the AI Data Plane workflow finishes, with no separate index setup. Never
    touches `docs` or anything the workflow itself owns."""
    domains = list(config.DOMAINS)
    print(f"Initialize will, for each domain ({', '.join(domains)}):", file=sys.stderr)
    for name in prism_initialize.STEPS:
        print(f"  - {name}", file=sys.stderr)
    if not _confirm("This empties the catalog and dictionary and rebuilds the "
                    "search index, in every configured domain. docs/chunks are "
                    "untouched. Proceed?", args.yes):
        sys.exit("aborted")

    corpus = load_corpus(args.corpus)
    sectors = corpus.sectors() if hasattr(corpus, "sectors") else {}

    def on_step(i, total, name, status, detail=None):
        if status == "running":
            print(f"[{i}/{total}] {name}...", file=sys.stderr)
        elif status == "error":
            print(f"[{i}/{total}] {name}: ERROR {detail}", file=sys.stderr)
        else:
            print(f"[{i}/{total}] {name}: done", file=sys.stderr)

    def on_catalog_progress(i, total, doc_name, result):
        status = "ok" if result["ok"] else f"ERROR {result['error']}"
        print(f"    catalog [{i}/{total}] {doc_name}: {status}", file=sys.stderr)

    def on_index_progress(count, target):
        print(f"    reindexed {count}/{target} documents", file=sys.stderr)

    summary = prism_initialize.run(model=args.model, sectors=sectors, on_step=on_step,
                                   on_catalog_progress=on_catalog_progress,
                                   on_index_progress=on_index_progress)
    for scope, result in summary["domains"].items():
        ok = sum(1 for r in result["catalog_results"] if r["ok"])
        print(f"\n[{scope}] catalog: {ok}/{len(result['catalog_results'])} document(s)",
              file=sys.stderr)
        print(f"[{scope}] dictionary: cleared {len(result['dictionary_removed'])} entrie(s)",
              file=sys.stderr)
    print("done", file=sys.stderr)


def migrate_dictionary(args):
    """One-time: copy dictionary.yaml's entries into Couchbase, which is the
    real persistence target now - repository.py only reads/writes the file
    when a test passes an explicit path. Safe to run more than once: it's a
    full replace of the Couchbase collection from the file's current
    contents, the same replace semantics save() already uses for every other
    write (approve/forget/clear)."""
    file_data = dictionary.load(path=config.DICTIONARY_PATH)
    entries = file_data.get("entries", [])
    if not entries:
        sys.exit(f"{config.DICTIONARY_PATH} has no entries — nothing to migrate")
    couchbase_io.ensure_primary_index(config.DICTIONARY_COLLECTION)
    dictionary.save(file_data)  # no path -> Couchbase
    for entry in entries:
        print(f"migrated {entry.get('id')}", file=sys.stderr)
    print(f"{len(entries)} entries now in "
          f"{config.BUCKET}.{config.SCOPE}.{config.DICTIONARY_COLLECTION}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-catalog")
    b.add_argument("--all", action="store_true",
                   help="catalog every PDF, including ones with no chunks ingested")
    b.add_argument("--corpus", default="ftsprism")
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

    m = sub.add_parser("migrate-dictionary",
                       help="one-time: copy dictionary.yaml's entries into Couchbase")
    m.set_defaults(func=migrate_dictionary)

    i = sub.add_parser("initialize", help="empty+rebuild catalog and dictionary, "
                                          "rebuild the search index (destructive)")
    i.add_argument("--corpus", default="ftsprism")
    i.add_argument("--model", default=None)
    i.add_argument("--yes", action="store_true", help="skip confirmation")
    i.set_defaults(func=initialize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pulls filings for one company straight from SEC EDGAR's submissions API and
saves them as PDFs into pdfs/{label}/, ready for a documents.yaml entry each.

Deliberately EDGAR-only, not a company's own investor-relations website
(compare edgarDownloader/download_ir_pdfs.py) - EDGAR is public record filed
under legal disclosure requirements, which is the doubt-free tier this corpus
is built to stay inside. There is no S3/Couchbase-workflow upload here either;
that ingestion happens later, through the same AI Data Plane workflow a real
deployment would use, not as part of authoring the corpus.

One subfolder per company under pdfs/ (e.g. pdfs/3M/, pdfs/BestBuy/) - each
company's files stay together rather than mixed in one flat directory, which
matters once a second company's filings arrive. --label controls both the
folder name and the doc_name prefix; --ticker is only used to resolve a CIK,
so a label that differs from the ticker (e.g. "3M" vs "MMM") is expected and
fine - doc_name should read the way the rest of PRISM already names 3M's
documents, not the way EDGAR looks it up.

Dry-run by default - prints the manifest (doc_name, form, source url) without
writing anything. Pass --go to actually fetch and convert.

Usage:
    python fetch_filings.py --ticker MMM --label 3M --forms 10-K,10-Q,DEF-14A \
        --years 2020-2026
    python fetch_filings.py --ticker MMM --label 3M --forms 8-K --item 2.02 \
        --years 2020-2026 --go
"""
import argparse
import pathlib
import re
import time

import requests
from weasyprint import HTML as WeasyprintHTML

EDGAR_EMAIL = "krishna.doddi@couchbase.com"   # required by EDGAR fair-use policy
HEADERS = {"User-Agent": f"CouchbasePrism-ftsPrism {EDGAR_EMAIL}"}
REQUEST_DELAY = 0.15  # EDGAR asks for max 10 req/s

ROOT = pathlib.Path(__file__).resolve().parent
PDF_ROOT = ROOT / "pdfs"

QUARTER_OF_MONTH = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3,
                    10: 4, 11: 4, 12: 4}


def cik_for_ticker(ticker: str) -> str:
    resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    for row in resp.json().values():
        if row["ticker"] == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"no CIK found for ticker {ticker}")


def _submissions(cik: str) -> dict:
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _exhibit_url(cik_bare: str, accession: str, item_prefix: str) -> str:
    """For an 8-K: the primaryDocument is a one-paragraph cover form, not the
    actual earnings release. The real content is a separate EX-99.x exhibit,
    found by reading the filing's own index page rather than guessing at a
    filename pattern - verified live against a real 3M filing: the index
    table's Type column says exactly "EX-99.1" for the exhibit and "8-K" for
    the cover form."""
    acc_nodash = accession.replace("-", "")
    index_url = (f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/"
                f"{acc_nodash}/{accession}-index.html")
    time.sleep(REQUEST_DELAY)
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    rows = re.findall(r"<tr.*?</tr>", resp.text, re.S)
    for row in rows:
        cells = [re.sub("<.*?>", "", c).strip()
                for c in re.findall(r"<td.*?>(.*?)</td>", row, re.S)]
        if len(cells) >= 4 and cells[3].upper().startswith("EX-99"):
            doc = cells[2].split()[0]  # strip trailing " &nbsp;iXBRL" etc.
            return (f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/"
                    f"{acc_nodash}/{doc}")
    return None  # no EX-99 exhibit in this filing - caller decides what to do


def plan(ticker: str, label: str, forms: list, years: tuple, item_filter: str = None) -> list:
    """Returns the manifest: a list of {doc_name, form, url, filed, period}
    dicts, fully resolved (including 8-K exhibit lookups), but nothing
    fetched or converted yet."""
    cik = cik_for_ticker(ticker)
    cik_bare = str(int(cik))
    data = _submissions(cik)
    recent = data["filings"]["recent"]
    out = []
    zipped = list(zip(recent["form"], recent["accessionNumber"], recent["filingDate"],
                      recent.get("reportDate", [None] * len(recent["form"])),
                      recent.get("primaryDocument", [None] * len(recent["form"])),
                      recent.get("items", [""] * len(recent["form"]))))
    for form, accession, filed, report_date, primary_doc, items in zipped:
        if form not in forms:
            continue
        if item_filter and item_filter not in (items or ""):
            continue
        period = report_date or filed
        year = int(period[:4])
        month = int(period[5:7])
        if not (years[0] <= year <= years[1]):
            continue

        if form == "8-K":
            # Which fiscal quarter an Item 2.02 8-K reports on is NOT a clean
            # function of its filed date - verified live against 3M's actual
            # filings: some quarters carry two Item-2.02 8-Ks close together
            # (an ancillary "mini" release alongside the main earnings one),
            # so a date-based guess either collides two real filings onto one
            # name or silently mislabels the quarter. The filed date itself
            # never collides and never needs guessing, so that is what the
            # name is built from; which quarter each one actually covers is
            # for a human to confirm by reading it, same as every other
            # verification step in this corpus. Even the filed date alone is
            # not always unique, though - verified live: 3M has filed two
            # separate Item-2.02 8-Ks on the same day (a main earnings
            # release plus a short "mini" supplementary one) more than once
            # in this range - so the accession number's own sequence
            # (unique per filer, always) is appended too.
            accession_seq = accession.split("-")[-1]
            doc_name = f"{label}_8K_{filed}_{accession_seq}_earnings"
            url = _exhibit_url(cik_bare, accession, "EX-99")
            if url is None:
                continue  # no earnings exhibit in this 8-K - skip, don't guess
        elif form == "10-Q":
            quarter = QUARTER_OF_MONTH[month]
            doc_name = f"{label}_{year}_Q{quarter}_10Q"
            acc_nodash = accession.replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/"
                  f"{acc_nodash}/{primary_doc}")
        elif form == "10-K":
            doc_name = f"{label}_{year}_10K"
            acc_nodash = accession.replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/"
                  f"{acc_nodash}/{primary_doc}")
        else:  # DEF 14A and anything else - primaryDocument is the real content
            doc_name = f"{label}_{year}_{form.replace(' ', '')}"
            acc_nodash = accession.replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik_bare}/"
                  f"{acc_nodash}/{primary_doc}")

        out.append({"doc_name": doc_name, "form": form, "url": url,
                    "filed": filed, "period": report_date})
    out.sort(key=lambda r: r["doc_name"])
    return out


def fetch_one(item: dict, pdf_dir: pathlib.Path) -> int:
    """Downloads one filing's HTML and converts it to PDF. Returns the
    written file's size in bytes."""
    time.sleep(REQUEST_DELAY)
    resp = requests.get(item["url"], headers=HEADERS, timeout=60)
    resp.raise_for_status()
    pdf_path = pdf_dir / f"{item['doc_name']}.pdf"
    WeasyprintHTML(string=resp.text, base_url=item["url"]).write_pdf(str(pdf_path))
    return pdf_path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="resolves the CIK only")
    ap.add_argument("--label", default=None,
                    help="doc_name prefix and pdfs/ subfolder name - defaults "
                        "to --ticker if omitted, but should usually be set "
                        "explicitly (e.g. --label 3M) to match how the rest "
                        "of PRISM names this company's documents")
    ap.add_argument("--forms", required=True, help="comma-separated, e.g. 10-K,10-Q")
    ap.add_argument("--years", required=True, help="e.g. 2020-2026")
    ap.add_argument("--item", default=None,
                    help="8-K item filter, e.g. 2.02 for earnings releases")
    ap.add_argument("--go", action="store_true",
                    help="actually fetch and convert - omit for a dry-run manifest only")
    args = ap.parse_args()

    label = args.label or args.ticker
    forms = [f.strip() for f in args.forms.split(",")]
    y0, y1 = (int(y) for y in args.years.split("-"))
    manifest = plan(args.ticker, label, forms, (y0, y1), item_filter=args.item)
    pdf_dir = PDF_ROOT / label

    print(f"{len(manifest)} filing(s) for {label} (ticker {args.ticker}), "
          f"forms={forms}, years={y0}-{y1}" +
          (f", item={args.item}" if args.item else "") + f" -> {pdf_dir}/")
    for row in manifest:
        print(f"  {row['doc_name']:<32} {row['form']:<8} filed={row['filed']} "
              f"{row['url']}")

    if not args.go:
        print("\nDry run only - pass --go to fetch and convert these to PDF.")
        return

    pdf_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for row in manifest:
        try:
            size = fetch_one(row, pdf_dir)
            print(f"  saved {row['doc_name']}.pdf ({size // 1024} KB)")
            ok += 1
        except Exception as e:
            print(f"  FAILED {row['doc_name']}: {e}")
    print(f"\n{ok}/{len(manifest)} saved to {pdf_dir}/")


if __name__ == "__main__":
    main()

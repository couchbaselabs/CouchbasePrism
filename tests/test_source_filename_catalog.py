"""build_document stores source_filename ONLY when given one - build_from_pdf
has no S3 location to record, build_from_chunks always resolves one via
config.source_filename(doc_name) at build time (see extraction.py's
build_document docstring for why this matters beyond just this one field)."""
from prism.catalog.extraction import build_document

EXTRACTION = {"company": {"value": "3M", "confidence": 1.0, "source": "pdf_text"},
             "doc_type": {"value": "10-K", "confidence": 1.0, "source": "pdf_text"},
             "period_end_date": {"value": None, "confidence": 0.0, "source": "absent"}}


def test_source_filename_present_when_given():
    doc = build_document("3M_2022_10K", EXTRACTION,
                         source_filename="kpd-couchbase_Prism_3M_3M_2022_10K.pdf")
    assert doc["source_filename"] == "kpd-couchbase_Prism_3M_3M_2022_10K.pdf"


def test_source_filename_absent_when_not_given():
    doc = build_document("3M_2022_10K", EXTRACTION)
    assert "source_filename" not in doc

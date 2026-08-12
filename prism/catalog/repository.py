"""Persistence for catalog documents: {bucket}.{scope}.catalog, keyed by doc_name."""
from .. import config
from ..couchbase_io import ensure_primary_index, query


def upsert(document: dict) -> None:
    query(
        f"UPSERT INTO `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` "
        f"(KEY, VALUE) VALUES ($doc_id, $doc_body)",
        {"$doc_id": document["doc_id"], "$doc_body": document},
    )


def load_all() -> list:
    """Only the fields resolution needs. `value` is a reserved word in N1QL and
    must be backticked in the projection."""
    return query(
        "SELECT d.doc_name, d.doc_type.`value` AS doc_type, d.doc_period, "
        "d.period_end_date_iso "
        f"FROM `{config.BUCKET}`.`{config.SCOPE}`.`{config.CATALOG_COLLECTION}` AS d"
    )


def ensure_index() -> None:
    ensure_primary_index(config.CATALOG_COLLECTION)

"""Test defaults.

The suite is pure logic - no cluster, no API keys - but a few code paths read
environment variables to compose identifiers (`config.source_filename` builds
the S3-derived chunk filename). Supply harmless stand-ins so `pytest` works on
a clean checkout without a .env, while never overriding a real one.

COUCHBASE_BUCKET/COUCHBASE_SCOPE used to live here too, but config.py no
longer reads them at all - bucket/scope are fixed constants there now, not
environment-configurable - so setting them here would do nothing.
"""
import os

DEFAULTS = {
    "AWS_BUCKET": "test-bucket",
    "AWS_FOLDER": "test-folder",
}

for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)

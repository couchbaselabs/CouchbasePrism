"""Test defaults.

The suite is pure logic - no cluster, no API keys - but a few code paths read
environment variables to compose identifiers (`config.source_filename` builds
the S3-derived chunk filename). Supply harmless stand-ins so `pytest` works on
a clean checkout without a .env, while never overriding a real one.
"""
import os

DEFAULTS = {
    "AWS_BUCKET": "test-bucket",
    "AWS_FOLDER": "test-folder",
    "COUCHBASE_BUCKET": "testbucket",
    "COUCHBASE_SCOPE": "testscope",
}

for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)

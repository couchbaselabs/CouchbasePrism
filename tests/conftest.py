"""Test defaults.

The suite is pure logic - no cluster, no API keys - but a few code paths read
environment variables to compose identifiers (`config.source_filename` builds
the S3-derived chunk filename). Supply harmless stand-ins so `pytest` works on
a clean checkout without a .env, while never overriding a real one.

COUCHBASE_BUCKET/COUCHBASE_SCOPE used to live here too, but config.py no
longer reads them at all - bucket/scope are fixed constants there now, not
environment-configurable - so setting them here would do nothing.

COUCHBASE_CONN_STRING/USERNAME/PASSWORD are needed by config.couchbase_host()/
couchbase_auth() - read even by tests that mock out requests.post entirely
(test_couchbase_io_errors.py), since query() composes the URL/auth before
ever making the (faked) call.
"""
import os

DEFAULTS = {
    "AWS_BUCKET": "test-bucket",
    "COUCHBASE_CONN_STRING": "couchbases://test.invalid",
    "COUCHBASE_USERNAME": "test-user",
    "COUCHBASE_PASSWORD": "test-pass",
}

for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)

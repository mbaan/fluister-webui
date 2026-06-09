"""Shared test setup.

Point the app's data directory at a throwaway temp dir *before* any test module
imports ``app.main`` (settings are read once at import time).
"""

import os
import tempfile

os.environ.setdefault(
    "TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-test-")
)
# Never start/download the tidy LLM during tests. The app lifespan runs in the
# API tests, so without this a real model path would spawn an 18 GB llama-server.
# (set in os.environ before app import so it wins over any .env value.)
os.environ.setdefault("TRANSCRIBE_TIDY", "false")


import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"

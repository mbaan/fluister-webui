"""Shared test setup.

Point the app's data directory at a throwaway temp dir *before* any test module
imports ``app.main`` (settings are read once at import time).
"""

import os
import tempfile

os.environ.setdefault(
    "TRANSCRIBE_DATA_DIR", tempfile.mkdtemp(prefix="fluister-test-")
)


import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"

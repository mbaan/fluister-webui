"""Make the cuBLAS/cuDNN libraries from the ``nvidia-*-cu12`` pip wheels usable
by ctranslate2 without requiring ``LD_LIBRARY_PATH``.

The wheels drop their shared objects under ``site-packages/nvidia/*/lib`` but
ctranslate2 ``dlopen``s them by soname, so they must already be discoverable.
We preload them into the process with ``RTLD_GLOBAL``; once loaded, ctranslate2
finds them by soname. Best-effort and a no-op if the wheels/CUDA are absent, so
the CPU path is unaffected.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import sysconfig

logger = logging.getLogger(__name__)

_SUBDIRS = ("nvidia/cublas/lib", "nvidia/cudnn/lib")


def lib_dirs() -> list[str]:
    purelib = sysconfig.get_paths().get("purelib", "")
    return [d for d in (os.path.join(purelib, s) for s in _SUBDIRS) if os.path.isdir(d)]


def preload() -> list[str]:
    """Preload all bundled cuBLAS/cuDNN .so files; return basenames loaded."""
    files: list[str] = []
    for d in lib_dirs():
        files.extend(glob.glob(os.path.join(d, "*.so*")))

    loaded: set[str] = set()
    remaining = list(files)
    # Multiple passes: a library's dependencies may need to be loaded first, and
    # we don't know the order up front.
    progress = True
    while remaining and progress:
        progress = False
        still: list[str] = []
        for f in remaining:
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                loaded.add(os.path.basename(f))
                progress = True
            except OSError:
                still.append(f)
        remaining = still

    if remaining:
        logger.debug(
            "cuda_libs: could not preload %s",
            [os.path.basename(f) for f in remaining],
        )
    return sorted(loaded)

#!/usr/bin/env bash
# Launch the fluister transcription server.
# Exports LD_LIBRARY_PATH so ctranslate2 finds the cuDNN/cuBLAS libs that ship
# as pip wheels inside the uv venv.
set -euo pipefail
cd "$(dirname "$0")"

LIBS="$(uv run python - <<'PY'
import os, sysconfig
purelib = sysconfig.get_paths()["purelib"]
dirs = [os.path.join(purelib, s) for s in ("nvidia/cudnn/lib", "nvidia/cublas/lib")]
print(":".join(d for d in dirs if os.path.isdir(d)))
PY
)"
if [ -n "$LIBS" ]; then
  export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
fi

exec uv run uvicorn app.main:app \
  --host "${TRANSCRIBE_HOST:-127.0.0.1}" \
  --port "${TRANSCRIBE_PORT:-8000}" "$@"

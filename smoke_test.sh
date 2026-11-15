#!/usr/bin/env bash
# Quick check that the package is intact (CPU, no data needed, ~1 minute). It does NOT train or measure anything.
set -e
cd "$(dirname "$0")/pipeline"
# NOTE: `set -e` does not stop on a failing command inside `a && b` (only on the last one), so each step is a plain command
# followed by its own echo; a failure now ends the script instead of being reported as a pass.
python -m py_compile *.py
echo "[1] all pipeline scripts compile"
python - <<'PY'
import sys; sys.path.insert(0,'.')
import numpy as np, payload as P
from models import Encoder, Decoder, Localiser, count_params
key=bytes(range(16)); iv=bytes(range(16)); meta=b'patient-0001|MR|2026-01-01'
for v in ('cbc','hmac'):
    m=P.encode(key,meta,iv,v)
    print(f'[2] payload {v}: {len(m)} bits; verify ok =', P.verify(m,key,meta,v))
print('[3] parameters  encoder %d  decoder %d  localiser %d (Table 6: 529,425 / 587,589 / 51,489)'%(count_params(Encoder()),count_params(Decoder()),count_params(Localiser())))
PY
mkdir -p ../out
python make_figures.py conceptual ../out >/dev/null
echo "[4] Fig. 3 and Fig. 7 regenerate -> out/"
python ../figures/fig_ablation_components.py ../out/Fig_ablation_components.png
echo "[5] ablation figure regenerates"
if [ -d ../docs_tools ]; then
    python ../docs_tools/build_table15_docx.py ../out/Table15.docx
    python ../docs_tools/build_response_letter.py ../out/Response.docx
    echo "[6] Table 15 and response letter regenerate"
else
    echo "[6] skipped: docs_tools/ is not part of this repository"
fi
echo "SMOKE TEST PASSED"

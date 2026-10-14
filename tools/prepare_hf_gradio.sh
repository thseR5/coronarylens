#!/usr/bin/env bash
# Assemble a Hugging Face Gradio Space (ZeroGPU hardware) from HEAD plus the built frontend into $1.
# Nothing is uploaded; pushing is a manual step (deploy/hf-gradio/DEPLOY.md).
set -euo pipefail
OUT="${1:-.scratch/hf-gradio}"
rm -rf "$OUT" && mkdir -p "$OUT"
git archive HEAD backend ml config data LICENSE LICENSES.md | tar -x -C "$OUT"
(cd frontend && npx vite build >/dev/null)
mkdir -p "$OUT/frontend" && cp -R frontend/dist "$OUT/frontend/dist"
cp deploy/hf-gradio/app.py deploy/hf-gradio/requirements.txt "$OUT/"
cp deploy/hf-gradio/SPACE_README.md "$OUT/README.md"
rm -rf "$OUT/ml/tests" "$OUT/backend/tests"
cat > "$OUT/.gitattributes" <<'ATTR'
*.joblib filter=lfs diff=lfs merge=lfs -text
*.glb filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.xlsx filter=lfs diff=lfs merge=lfs -text
ATTR
du -sh "$OUT"

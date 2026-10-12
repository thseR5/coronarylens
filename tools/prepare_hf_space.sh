#!/usr/bin/env bash
# Assemble a Hugging Face Docker Space from the committed repository (HEAD) into $1 (default .scratch/hf-space).
# Nothing is uploaded: pushing to the Space is a separate, manual step (see deploy/hf-space/DEPLOY.md).
set -euo pipefail
OUT="${1:-.scratch/hf-space}"
rm -rf "$OUT" && mkdir -p "$OUT"
git archive HEAD | tar -x -C "$OUT"
cp deploy/hf-space/SPACE_README.md "$OUT/README.md"          # Space card (YAML front-matter: sdk docker, port 8000)
cat > "$OUT/.gitattributes" <<'ATTR'
*.joblib filter=lfs diff=lfs merge=lfs -text
*.glb filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.pdf filter=lfs diff=lfs merge=lfs -text
*.xlsx filter=lfs diff=lfs merge=lfs -text
ATTR
rm -rf "$OUT/.github" "$OUT/reports/sprint"/screen_*.json   # CI and large screening logs are not needed in the Space
du -sh "$OUT"
find "$OUT" -type f -size +10M -exec ls -lh {} \; | awk '{print "large (LFS):", $5, $9}'

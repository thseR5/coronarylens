# Decisions log

One line per decision: what, why. Newest at the bottom.

- 2026-10-12 · Project in its own git repo at `~/Desktop/coronarylens`; the user's home dir is a separate git repo.
- 2026-10-12 · Python 3.12 via `uv` (versions locked in `uv.lock`); frontend on Node 26 + Vite 8 builds cleanly, so no Node downgrade.
- 2026-10-12 · Dataset facts from the master prompt re-verified: 303×59, no missing, prevalence CAD 216 / LAD 177 / LCX 119 / RCA 114, pattern counts match, CAD == any-vessel for 302/303 (exception: row 93, Cath=Normal, LAD=Stenotic). `Cath` stays the CAD ground truth.
- 2026-10-12 · All 25 FMA IDs in `config/anatomy.yaml` exist in BodyParts3D 4.0 part-of tree and resolve to ≥1 element OBJ.
- 2026-10-12 · `Exertional CP` dropped (constant). No cardiac enzymes or total cholesterol exist in the data despite the challenge primer; not invented.
- 2026-10-12 · Frontend reads config from `/api/config`, falling back to the same YAML files bundled with `?raw`: one source, no duplication.
- 2026-10-12 · API contract v1 frozen in `backend/app/schemas.py` ⇄ `frontend/src/types/api.ts` before parallel work.

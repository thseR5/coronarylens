# LV territory check: nearest-artery (Voronoi) partition vs AHA 17-segment model

Area-weighted agreement with the standard assignment: **74.2%**. Segments whose majority artery matches the standard: **14/17**.

Segment frame: septum toward the RV cavity centroid (AHA-style).

**How to read this:** the partition is *aligned with* the AHA 17-segment convention; because it was adjusted toward that convention during development (see method history), the agreement is not independent validation.

Disagreeing segments:

- Segment 1 (basal anterior): standard LAD, partition LCX (96% of its area). Basal boundary segment: the LCX runs in the left AV groove right next to the base, so surface distance favours it over the proximal LAD.
- Segment 3 (basal inferoseptal): standard RCA, partition LAD (51% of its area). Basal boundary segment: the septum is really supplied by intramyocardial septal perforators (LAD) and the PDA (RCA); perforators are not in the mesh, so surface distance decides.
- Segment 11 (mid inferolateral): standard LCX, partition RCA (60% of its area). Inferolateral boundary: supplied by RCA posterolateral branches in right-dominant hearts and by the LCX in left-dominant hearts.

Per-vessel agreement (share of the standard territory's area labelled with that vessel): LAD 76.5%, LCX 62.3%, RCA 83.4%.

| Segment | Standard | Majority (partition) | Area share |
|---|---|---|---|
| 1 | LAD | LCX ✗ | LAD 4%, LCX 96% |
| 2 | LAD | LAD ✓ | LAD 78%, LCX 22% |
| 3 | RCA | LAD ✗ | LAD 51%, LCX 15%, RCA 34% |
| 4 | RCA | RCA ✓ | LCX 5%, RCA 95% |
| 5 | LCX | LCX ✓ | LCX 56%, RCA 44% |
| 6 | LCX | LCX ✓ | LCX 100% |
| 7 | LAD | LAD ✓ | LAD 82%, LCX 18% |
| 8 | LAD | LAD ✓ | LAD 100% |
| 9 | RCA | RCA ✓ | LAD 29%, RCA 71% |
| 10 | RCA | RCA ✓ | RCA 100% |
| 11 | LCX | RCA ✗ | LCX 40%, RCA 60% |
| 12 | LCX | LCX ✓ | LAD 12%, LCX 88% |
| 13 | LAD | LAD ✓ | LAD 100% |
| 14 | LAD | LAD ✓ | LAD 92%, RCA 8% |
| 15 | RCA | RCA ✓ | RCA 100% |
| 16 | LCX | LCX ✓ | LAD 23%, LCX 40%, RCA 37% |
| 17 | LAD | LAD ✓ | LAD 71%, LCX 6%, RCA 23% |

Method history (each step moved the partition toward the AHA convention; reasons in docs/DECISIONS.md):

| Version | Seeds | Segment frame | Area agreement | Segments |
|---|---|---|---|---|
| 1 | all vessel branches | scene frame | 48.3% | 9/17 |
| 2 | LV-supplying branches only | scene frame | 55.3% | 10/17 |
| 3 (current) | LV-supplying branches only | septum toward the RV cavity centroid (AHA-style) | 74.2% | 14/17 |

Segment geometry approximated from the mesh (PCA long axis; septal direction toward the RV cavity, AHA-style); illustrative only. Real supply territories vary with coronary dominance.

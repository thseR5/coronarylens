"""Check the nearest-artery (Voronoi) LV territories against the AHA 17-segment standard assignment.

The LV territory meshes (LV_Anterior / LV_Lateral / LV_Inferior) are a nearest-artery partition of the BodyParts3D
LV cavity surface (see build_anatomy.build_territories). This script assigns every territory face to an AHA segment
(Cerqueira et al., Circulation 2002) and reports how often the face's nearest artery equals the vessel that the
standard assignment in config/anatomy.yaml gives that segment.

Segment geometry (scene frame: +X patient left, +Y superior, +Z anterior):
- long axis = first principal axis of the LV surface, oriented base -> apex (base = end nearer the aortic root);
- height h in [0, 1] along the axis: basal < 1/3 <= mid < 2/3 <= apical < 0.9 <= apex (segment 17);
- short-axis angle: the septal direction points from the LV axis toward the RV cavity centroid (the AHA convention
  orients segments by the RV); anterior is the in-plane perpendicular on the +Z side; angles grow toward the septum.
  (The first version used scene +Z as anterior; see reports/territory_agreement.md for both results.)
  Basal/mid rings: 6 sectors of 60 deg centred on anterior, anteroseptal, inferoseptal, inferior, inferolateral,
  anterolateral. Apical ring: 4 sectors of 90 deg (anterior, septal, inferior, lateral).
This is an approximation of the echo/MRI convention (which uses the RV insertion points), so it is a sanity check of
the illustrative partition, not a validation of anyone's anatomy.

Run: uv run python tools/territory_check.py   (needs tools/.cache/anatomy_raw.glb from `make anatomy`)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_GLB = ROOT / "tools" / ".cache" / "anatomy_raw.glb"
OUT_JSON = ROOT / "frontend" / "public" / "models" / "territory_agreement.json"
OUT_MD = ROOT / "reports" / "territory_agreement.md"

# Earlier measured versions (2026-10-12), kept so the report shows how the method evolved toward the AHA
# convention; reasons are logged in docs/DECISIONS.md.
HISTORY = [
    {"version": 1, "seeds": "all vessel branches", "frame": "scene frame", "area_weighted_agreement": 0.483,
     "segments_majority_agree": 9},
    {"version": 2, "seeds": "LV-supplying branches only", "frame": "scene frame", "area_weighted_agreement": 0.553,
     "segments_majority_agree": 10},
]

SEGMENT_NAMES = {
    1: "basal anterior", 2: "basal anteroseptal", 3: "basal inferoseptal", 4: "basal inferior",
    5: "basal inferolateral", 6: "basal anterolateral", 7: "mid anterior", 8: "mid anteroseptal",
    9: "mid inferoseptal", 10: "mid inferior", 11: "mid inferolateral", 12: "mid anterolateral",
    13: "apical anterior", 14: "apical septal", 15: "apical inferior", 16: "apical lateral", 17: "apex",
}
# Why a segment can disagree (shown for every disagreeing segment; anatomical background, not a fitted claim).
DISAGREEMENT_REASON = {
    1: "basal boundary segment: the LCX runs in the left AV groove right next to the base, so surface distance "
       "favours it over the proximal LAD",
    3: "basal boundary segment: the septum is really supplied by intramyocardial septal perforators (LAD) and the "
       "PDA (RCA); perforators are not in the mesh, so surface distance decides",
    5: "boundary segment whose supply depends on coronary dominance (RCA vs LCX)",
    11: "inferolateral boundary: supplied by RCA posterolateral branches in right-dominant hearts and by the LCX "
        "in left-dominant hearts",
    15: "apical inferior: LAD wrap-around vs PDA varies between people",
    16: "apical lateral: LAD diagonals vs LCX marginals vary between people",
    17: "apex: often supplied by a wrap-around LAD but varies between people",
}

RING6 = [1, 2, 3, 4, 5, 6]  # anterior, anteroseptal, inferoseptal, inferior, inferolateral, anterolateral
RING4 = [13, 14, 15, 16]  # anterior, septal, inferior, lateral


def segment_of(h: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    th = np.mod(theta_deg, 360.0)
    six = np.floor(np.mod(th + 30.0, 360.0) / 60.0).astype(int)  # 0..5 centred on 0, 60, ...
    four = np.floor(np.mod(th + 45.0, 360.0) / 90.0).astype(int)  # 0..3 centred on 0, 90, ...
    seg = np.where(h < 1 / 3, np.take(RING6, six),
          np.where(h < 2 / 3, np.take(RING6, six) + 6,
          np.where(h < 0.9, np.take(RING4, four), 17)))
    return seg


def aha_segments(C: np.ndarray, A: np.ndarray, aorta_vertices: np.ndarray,
                 rv_centroid: np.ndarray | None = None) -> np.ndarray:
    """AHA segment (1..17) of each LV surface face (centres C, areas A). Shared by the anatomy build (which turns
    the segments into the 3D AHA territory meshes) and by this check, so 3D and analysis use one geometry."""
    mu = (C * A[:, None]).sum(0) / A.sum()
    cov = ((C - mu) * A[:, None]).T @ (C - mu) / A.sum()
    axis = np.linalg.eigh(cov)[1][:, -1]  # long axis by area-weighted PCA
    root = aorta_vertices[np.argsort(np.linalg.norm(aorta_vertices - mu, axis=1))[:200]].mean(0)  # aortic root
    if np.dot(root - mu, axis) > 0:  # orient base -> apex
        axis = -axis
    t = (C - mu) @ axis
    h = (t - t.min()) / (t.max() - t.min())

    def in_plane(v):
        v = v - np.dot(v, axis) * axis
        return v / np.linalg.norm(v)

    if rv_centroid is not None:  # AHA-style: septum toward the RV, anterior perpendicular on the +Z side
        sep = in_plane(np.asarray(rv_centroid, float) - mu)
        ant = np.cross(axis, sep)
        ant = ant if ant[2] >= 0 else -ant
    else:  # scene-frame fallback: anterior = +Z, septum = patient right
        ant = in_plane(np.array([0.0, 0.0, 1.0]))
        sep = in_plane(np.array([-1.0, 0.0, 0.0]))
        sep = in_plane(sep - np.dot(sep, ant) * ant)
    R = (C - mu) - np.outer((C - mu) @ axis, axis)
    theta = np.degrees(np.arctan2(R @ sep, R @ ant))
    return segment_of(h, theta)


def check(scene: trimesh.Scene, anatomy: dict, rv_centroid: np.ndarray | None = None) -> dict:
    terr = {s["node"]: s["target"] for s in anatomy["structures"] if s["role"] == "territory"}
    standard = {seg: v for v, segs in anatomy["bullseye"].items() for seg in segs}

    centers, areas, labels = [], [], []
    for node, vessel in terr.items():
        m = scene.geometry[node]
        centers.append(m.triangles_center)
        areas.append(m.area_faces)
        labels += [vessel] * len(m.faces)
    C, A, L = np.vstack(centers), np.concatenate(areas), np.array(labels)

    seg = aha_segments(C, A, scene.geometry["Aorta"].vertices, rv_centroid)

    expected = np.array([standard[s] for s in seg])
    match = L == expected
    per_seg = {}
    for s in range(1, 18):
        sel = seg == s
        if not sel.any():
            per_seg[s] = {"standard": standard[s], "majority": None, "area_share": {}, "faces": 0}
            continue
        share = Counter()
        for v, a in zip(L[sel], A[sel]):
            share[v] += float(a)
        tot = sum(share.values())
        maj = max(share, key=share.get)
        per_seg[s] = {"standard": standard[s], "majority": maj, "faces": int(sel.sum()),
                      "area_share": {k: round(v / tot, 3) for k, v in sorted(share.items())}}
    seg_agree = [s for s, r in per_seg.items() if r["majority"] == r["standard"]]
    per_vessel = {}
    for v in sorted(set(standard.values())):
        sel = expected == v
        per_vessel[v] = round(float(A[sel & match].sum() / A[sel].sum()), 3) if sel.any() else None
    return {
        "method": "nearest-artery (Voronoi) partition of the LV surface vs AHA 17-segment standard assignment",
        "summary": (f"aligned with the AHA 17-segment convention ({len(seg_agree)}/17 segments agree); the partition "
                    "was adjusted toward the AHA convention during development, so this is not independent validation"),
        "disagreeing_segments": [
            {"segment": s, "name": SEGMENT_NAMES[s], "standard": r["standard"], "partition": r["majority"],
             "share": r["area_share"].get(r["majority"]),
             "reason": DISAGREEMENT_REASON.get(s, "boundary segment between territories")}
            for s, r in per_seg.items() if r["majority"] != r["standard"]],
        "area_weighted_agreement": round(float(A[match].sum() / A.sum()), 3),
        "segments_majority_agree": len(seg_agree),
        "segments_total": 17,
        "agreeing_segments": seg_agree,
        "per_vessel_area_agreement": per_vessel,
        "per_segment": per_seg,
        "history": HISTORY,
        "frame": ("septum toward the RV cavity centroid (AHA-style)" if rv_centroid is not None
                  else "scene frame (anterior = +Z, septum = patient right)"),
        "note": "Segment geometry approximated from the mesh (PCA long axis; septal direction toward the RV cavity, "
                "AHA-style); illustrative only. Real supply territories vary with coronary dominance.",
    }


def write(res: dict) -> None:
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    lines = ["# LV territory check: nearest-artery (Voronoi) partition vs AHA 17-segment model", "",
             (f"Area-weighted agreement with the standard assignment: **{res['area_weighted_agreement']:.1%}**. "
              f"Segments whose majority artery matches the standard: **{res['segments_majority_agree']}/17**."), "",
             f"Segment frame: {res['frame']}.", "",
             ("**How to read this:** the partition is *aligned with* the AHA 17-segment convention; because it was "
              "adjusted toward that convention during development (see method history), the agreement is not "
              "independent validation."), "",
             "Disagreeing segments:", "",
             *[f"- Segment {d['segment']} ({d['name']}): standard {d['standard']}, partition {d['partition']} "
               f"({d['share']:.0%} of its area). {d['reason'][0].upper() + d['reason'][1:]}."
               for d in res["disagreeing_segments"]], "",
             "Per-vessel agreement (share of the standard territory's area labelled with that vessel): "
             + ", ".join(f"{v} {a:.1%}" for v, a in res["per_vessel_area_agreement"].items() if a is not None) + ".", "",
             "| Segment | Standard | Majority (partition) | Area share |", "|---|---|---|---|"]
    for s, r in res["per_segment"].items():
        share = ", ".join(f"{k} {v:.0%}" for k, v in r["area_share"].items())
        ok = "✓" if r["majority"] == r["standard"] else "✗"
        lines.append(f"| {s} | {r['standard']} | {r['majority']} {ok} | {share} |")
    lines += ["", "Method history (each step moved the partition toward the AHA convention; reasons in docs/DECISIONS.md):", "",
              "| Version | Seeds | Segment frame | Area agreement | Segments |", "|---|---|---|---|---|"]
    for h in res["history"]:
        lines.append(f"| {h['version']} | {h['seeds']} | {h['frame']} | {h['area_weighted_agreement']:.1%} | "
                     f"{h['segments_majority_agree']}/17 |")
    lines.append(f"| {len(res['history']) + 1} (current) | LV-supplying branches only | {res['frame']} | "
                 f"{res['area_weighted_agreement']:.1%} | {res['segments_majority_agree']}/17 |")
    lines += ["", res["note"], ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not RAW_GLB.exists():
        print(f"missing {RAW_GLB}; run `make anatomy` first", file=sys.stderr)
        return 1
    anatomy = yaml.safe_load((ROOT / "config" / "anatomy.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "frontend" / "public" / "models" / "anatomy_manifest.json").read_text("utf-8"))
    rv = manifest.get("landmarks", {}).get("rv_cavity_centroid")
    res = check(trimesh.load(RAW_GLB, force="scene"), anatomy, rv)
    write(res)
    print(f"area-weighted agreement {res['area_weighted_agreement']:.1%}; "
          f"segments {res['segments_majority_agree']}/17; per vessel {res['per_vessel_area_agreement']}")
    for s, r in res["per_segment"].items():
        print(f"  seg {s:2d}: standard {r['standard']}  majority {r['majority']}  {r['area_share']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

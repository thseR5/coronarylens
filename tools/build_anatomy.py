"""Build frontend/public/models/anatomy.glb from BodyParts3D 4.0 (CC BY 4.0, DBCLS).

Driven entirely by config/anatomy.yaml: every structure's FMA concepts are resolved to element OBJ files
(FJxxxx) through the BodyParts3D part-of tree (partof_element_parts.txt), merged per structure, decimated,
put into ONE common heart-centred frame, and exported as a glTF with one named node per `structures[].node`.

Usage (repo root):  uv run python tools/build_anatomy.py
Outputs: frontend/public/models/anatomy.glb, anatomy_manifest.json, frontend/src/three/LICENSES.md

Axis convention of the output (three.js, right-handed, +Y up):
    +X = patient's LEFT, +Y = SUPERIOR (head), +Z = ANTERIOR (toward a viewer facing the patient).
  BodyParts3D source frame is mm with +X = patient left, -Y = anterior, +Z = superior, so the mapping is
  (x, y, z)_three = (x, z, -y)_bp3d, a proper rotation of -90 degrees about X (no mirroring).
  The camera at +Z looking at the origin therefore shows the AP (anteroposterior) projection.
"""

from __future__ import annotations

import collections
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import fast_simplification
import numpy as np
import requests
import trimesh
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tools" / ".cache"
OBJ_DIR = CACHE / "obj"
MODELS = ROOT / "frontend" / "public" / "models"
GLB = MODELS / "anatomy.glb"
MANIFEST = MODELS / "anatomy_manifest.json"
LICENSES = ROOT / "frontend" / "src" / "three" / "LICENSES.md"

# Triangle budget per node (total <= 100k). Coronaries get the most detail per mm^2.
BUDGET = {
    "LAD": 7000, "LCX": 3000, "RCA": 8000, "LeftMain": 600,
    "LV_Anterior": 2500, "LV_Lateral": 2500, "LV_Inferior": 2500,
    "Myocardium": 26000, "Aorta": 4000, "Lungs": 18000, "RibCage": 16000, "Sternum": 2000,
}
# Order in which structures claim elements (first claim wins => no element is exported twice).
CLAIM_ORDER = ["LAD", "LCX", "RCA", "LeftMain", "Aorta", "Sternum", "RibCage", "Lungs",
               "LV_Anterior", "LV_Lateral", "LV_Inferior", "Myocardium"]
# All coronary arteries (left + right coronary artery trees): subtracted from the myocardium.
CORONARY_TREES = ["FMA50039", "FMA50040"]
HEART = "FMA7088"
# Element used for the geometric LV territory partition (see build_territories).
LV_CAVITY_CONCEPT = "FMA9466"  # cavity of left ventricle (outer blood-pool surface of the LV in BP3D)
# BP3D -> three.js rotation: (x, y, z) -> (x, z, -y)
ROT = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  cached  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    log(f"  fetch   {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    tmp.rename(dest)


def read_tsv(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]


def load_obj(element: str) -> trimesh.Trimesh:
    return trimesh.load(OBJ_DIR / f"{element}.obj", force="mesh", process=False)


def merge(elements: list[str]) -> trimesh.Trimesh:
    meshes = [load_obj(e) for e in sorted(elements)]
    m = trimesh.util.concatenate(meshes)
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return m


def decimate(m: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    n = len(m.faces)
    if n <= target:
        return m
    v, f = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - target / n, agg=5)
    out = trimesh.Trimesh(v, f, process=True)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


def finish(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Consistent winding + outward normals where determinable (open meshes: winding only)."""
    trimesh.repair.fix_winding(m)
    if m.is_watertight:
        trimesh.repair.fix_inversion(m, multibody=True)
    m.fix_normals(multibody=True)
    _ = m.vertex_normals  # cache smooth vertex normals for export
    return m


def build_territories(lv: trimesh.Trimesh, vessels: dict[str, trimesh.Trimesh],
                      target_of: dict[str, str]) -> dict[str, trimesh.Trimesh]:
    """Partition the LV surface by nearest supplying vessel (illustrative expected supply territory)."""
    vessel_pts, vessel_lab = [], []
    for i, (node, vm) in enumerate(vessels.items()):
        vessel_pts.append(vm.vertices)
        vessel_lab.append(np.full(len(vm.vertices), i))
    tree = cKDTree(np.vstack(vessel_pts))
    labels = np.concatenate(vessel_lab)
    _, idx = tree.query(lv.triangles_center)
    face_lab = labels[idx]
    names = list(vessels)
    out = {}
    for terr, vessel in target_of.items():
        sel = face_lab == names.index(vessel)
        sub = lv.submesh([np.flatnonzero(sel)], append=True)
        out[terr] = sub
    return out


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "anatomy.yaml").read_text())
    src = cfg["source"]
    CACHE.mkdir(parents=True, exist_ok=True)
    log("== BodyParts3D source files")
    zpath = CACHE / Path(src["obj_zip"]).name
    epath = CACHE / Path(src["element_parts"]).name
    ppath = CACHE / Path(src["parts_list"]).name
    for url, dest in [(src["obj_zip"], zpath), (src["element_parts"], epath), (src["parts_list"], ppath)]:
        download(url, dest)
    if not OBJ_DIR.exists() or not any(OBJ_DIR.glob("FJ*.obj")):
        log("  extract obj zip")
        OBJ_DIR.mkdir(exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                name = Path(info.filename).name
                if name.endswith(".obj") and name.startswith("FJ"):
                    (OBJ_DIR / name).write_bytes(z.read(info))

    concept_elems: dict[str, set[str]] = collections.defaultdict(set)
    concept_name: dict[str, str] = {}
    for c, n, e in read_tsv(epath):
        concept_elems[c].add(e)
        concept_name[c] = n
    parts_list = {row[0]: row[2] for row in read_tsv(ppath) if len(row) >= 3}

    # ---- resolve + verify every FMA ID
    log("== Resolving FMA concepts")
    structures = {s["node"]: s for s in cfg["structures"]}
    missing: list[str] = []
    resolved: dict[str, set[str]] = {}
    for node, s in structures.items():
        elems: set[str] = set()
        for fma in s["fma"]:
            es = concept_elems.get(fma, set())
            if not es or fma not in parts_list:
                missing.append(f"{node}:{fma}")
                log(f"  MISSING {node} {fma} (in parts list: {fma in parts_list}, elements: {len(es)})")
            absent = [e for e in es if not (OBJ_DIR / f"{e}.obj").exists()]
            if absent:
                missing.append(f"{node}:{fma}:obj:{','.join(sorted(absent))}")
                log(f"  MISSING OBJ for {node} {fma}: {absent}")
            elems |= {e for e in es if (OBJ_DIR / f"{e}.obj").exists()}
            log(f"  {node:12s} {fma:9s} {len(es):4d} elements  {concept_name.get(fma, '?')}")
        resolved[node] = elems

    # ---- de-duplicate across structures
    coronary_all = set().union(*(concept_elems[c] for c in CORONARY_TREES))
    claimed: set[str] = set()
    assigned: dict[str, list[str]] = {}
    for node in CLAIM_ORDER:
        if node not in structures:
            continue
        es = set(resolved[node])
        if node == "Myocardium":
            es -= coronary_all  # incl. coronary branches not mapped to any vessel node (e.g. right conus artery)
        es -= claimed
        assigned[node] = sorted(es)
        claimed |= es
    for node in structures:  # structures not in CLAIM_ORDER (future yaml additions)
        if node not in assigned:
            assigned[node] = sorted(resolved[node] - claimed)
            claimed |= set(assigned[node])
    unrendered_coronary = sorted((coronary_all & resolved.get("Myocardium", set())) - claimed)
    if unrendered_coronary:
        log(f"  note: coronary elements not mapped to any vessel node, dropped: {unrendered_coronary}")

    # ---- territories: FMA wall concepts in BP3D 4.0 resolve to internal papillary muscles / mitral leaflet
    # (LV_Anterior and LV_Lateral resolve to the SAME elements), so the visible territory surfaces are built
    # by partitioning the LV cavity surface by nearest supplying coronary. The FMA-resolved elements are
    # kept in the manifest and left in the myocardium.
    territory_nodes = {n: s["target"] for n, s in structures.items() if s["role"] == "territory"}
    vessel_nodes = {s["target"]: n for n, s in structures.items() if s["role"] == "vessel"}
    lv_elems = sorted(concept_elems[LV_CAVITY_CONCEPT] & set(assigned.get("Myocardium", [])))
    for terr in territory_nodes:
        assigned["Myocardium"] = sorted(set(assigned["Myocardium"]) | set(assigned[terr]))
        assigned[terr] = []
    assigned["Myocardium"] = sorted(set(assigned["Myocardium"]) - set(lv_elems))

    # ---- load + merge (BP3D frame, mm)
    log("== Merging")
    raw: dict[str, trimesh.Trimesh] = {}
    for node in structures:
        if assigned[node]:
            raw[node] = merge(assigned[node])
            log(f"  {node:12s} {len(assigned[node]):4d} elements  {len(raw[node].faces):7d} raw tris")
    lv = merge(lv_elems)
    terr_meshes = build_territories(lv, {v: raw[vessel_nodes[v]] for v in vessel_nodes},
                                    {t: tgt for t, tgt in territory_nodes.items()})
    raw.update(terr_meshes)
    for t in territory_nodes:
        log(f"  {t:12s} partition of {lv_elems}  {len(raw[t].faces):7d} raw tris")

    # ---- one common transform: centre = heart bbox centre, heart max extent -> 2 units
    heart_nodes = ["Myocardium", *territory_nodes, *vessel_nodes.values(), "LeftMain"]
    hv = np.vstack([raw[n].vertices for n in heart_nodes if n in raw])
    lo, hi = hv.min(0), hv.max(0)
    centre = (lo + hi) / 2
    scale = 2.0 / float((hi - lo).max())
    log(f"== Common transform: centre(mm, bp3d)={centre.round(2).tolist()} scale={scale:.6f}/mm")

    scene = trimesh.Scene()
    manifest_nodes = {}
    total = 0
    for node, s in structures.items():
        if node not in raw:
            log(f"  WARNING: no geometry for {node}")
            continue
        m = decimate(raw[node], BUDGET.get(node, 4000))
        m.vertices = ((m.vertices - centre) * scale) @ ROT.T
        m = finish(m)
        m.visual = trimesh.visual.ColorVisuals(m)
        scene.add_geometry(m, node_name=node, geom_name=node)
        total += len(m.faces)
        manifest_nodes[node] = {
            "triangles": len(m.faces),
            "raw_triangles": len(raw[node].faces),
            "role": s["role"], "target": s["target"], "fma": s["fma"],
            "fma_names": {f: concept_name.get(f) for f in s["fma"]},
            "elements": assigned[node] if node not in territory_nodes else [],
            "fma_resolved_elements": sorted(resolved[node]),
            "geometry": ("LV cavity surface (" + ",".join(lv_elems) + f") faces nearest to {s['target']}"
                         if node in territory_nodes else "merged BP3D elements"),
            "bounds": np.round(m.bounds, 4).tolist(),
        }

    MODELS.mkdir(parents=True, exist_ok=True)
    raw_glb = CACHE / "anatomy_raw.glb"
    scene.export(raw_glb)
    log(f"== Raw glb {raw_glb.stat().st_size / 1e6:.2f} MB, {total} triangles")

    npx = shutil.which("npx")
    if not npx:
        log("npx not found: cannot compress")
        return 1
    subprocess.run([npx, "--yes", "@gltf-transform/cli", "meshopt", str(raw_glb), str(GLB)],
                   check=True, cwd=ROOT / "frontend", stdout=subprocess.DEVNULL)
    size = GLB.stat().st_size

    manifest = {
        "source": {k: src[k] for k in ("name", "version", "publisher", "license", "url", "attribution")},
        "axes": {"x": "patient left", "y": "superior", "z": "anterior",
                 "from_bp3d": "(x, y, z)_three = (x, z, -y)_bp3d, then (p - centre) * scale"},
        "transform": {"centre_bp3d_mm": centre.round(4).tolist(), "scale_per_mm": scale,
                      "heart_max_extent_units": 2.0},
        "compression": "EXT_meshopt_compression + KHR_mesh_quantization (gltf-transform meshopt)",
        "glb_bytes": size,
        "total_triangles": total,
        "missing": missing,
        "dropped_unmapped_coronary_elements": unrendered_coronary,
        "nodes": manifest_nodes,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    write_licenses(cfg, manifest)

    log("== Summary")
    log(f"  {'node':12s} {'role':9s} {'tris':>6s} {'raw':>7s}  elements")
    for node, d in manifest_nodes.items():
        n_el = len(d["elements"]) if d["elements"] else "partition"
        log(f"  {node:12s} {d['role']:9s} {d['triangles']:6d} {d['raw_triangles']:7d}  {n_el}")
    log(f"  TOTAL triangles {total}  (limit 100000)")
    log(f"  anatomy.glb {size / 1e6:.2f} MB  (limit 3 MB)   manifest {MANIFEST.stat().st_size / 1e3:.1f} kB")
    log(f"  missing FMA IDs: {missing or 'none'}")
    ok = total <= 100_000 and size <= 3_000_000 and not missing
    return 0 if ok else 1


def write_licenses(cfg: dict, manifest: dict) -> None:
    src = cfg["source"]
    rows = "\n".join(
        f"| `{n}` | {', '.join(d['fma'])} | {'; '.join(str(v) for v in d['fma_names'].values())} | {d['triangles']} |"
        for n, d in manifest["nodes"].items())
    LICENSES.write_text(f"""# 3D asset licences

Generated by `tools/build_anatomy.py`; do not edit by hand.

## {src['name']} {src['version']}

- Publisher: {src['publisher']}
- Licence: **{src['license']}** (https://creativecommons.org/licenses/by/4.0/)
- Source: {src['url']}
- Attribution (shown in the app footer): "{src['attribution']}"
- Files used: `partof_BP3D_4.0_obj_99.zip`, `partof_element_parts.txt`, `partof_parts_list_e.txt`.
- Note: the OBJ file headers still carry the legacy text "Creative Commons Attribution-Share Alike 2.1 Japan";
  the DBCLS download page lists the 4.0 release under CC BY 4.0, which is the licence recorded in `config/anatomy.yaml`.
- Modifications: elements merged per structure, decimated (fast-simplification), re-oriented and rescaled into one
  heart-centred frame, LV territory surfaces derived by partitioning the LV cavity surface by nearest coronary artery,
  meshopt-compressed with glTF-Transform.

## FMA concepts used

| glTF node | FMA IDs | Concept names | Triangles |
|---|---|---|---|
{rows}
""")


if __name__ == "__main__":
    sys.exit(main())

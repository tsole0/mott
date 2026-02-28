"""
2_compute_features.py

For each DFT metal in raw_materials.json:
  - Finds all transition-metal sites in the crystal structure
  - Calls mott.compute_all_features() (Rust) per TM site
  - Averages structural features across all TM sites
  - Joins with DFT features already in the JSON
  - Skips materials with no structure or no Hubbard U

Output: features.csv — one row per material

NOTE on units:
  W = 2 * z * mean(1/d²) has units of Å⁻²  (d in Ångströms)
  U is in eV  →  U/W has units eV·Å²
  This is a relative proxy for the Mott criterion, not a pure ratio.
  The ML model learns the effective threshold from data.
"""

import json
import numpy as np
import pandas as pd
from pymatgen.core import Structure

import mott  # compiled Rust extension — build with: maturin develop

# 3d/4d/5d TM elements where Mott physics is relevant
TM_ELEMENTS = {
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Nb", "Mo", "Ru", "Rh",
    "Ta", "W",  "Os", "Ir",
}

# Cutoff for neighbor search (Å). 3.5 Å captures first-shell TM-O and TM-TM bonds.
CUTOFF = 3.5


def _element_symbol(specie) -> str:
    """Return bare element symbol regardless of whether specie is Element or Species."""
    return specie.element.symbol if hasattr(specie, "element") else specie.symbol


def extract_tm_features(mat: dict) -> dict | None:
    """
    Compute structural + U/W features for a single material dict.
    Returns None if the material has no structure or no Hubbard U data.
    """
    if mat.get("structure") is None:
        return None
    if not mat.get("is_hubbard") or not mat.get("hubbard_u"):
        return None

    try:
        struct = Structure.from_dict(mat["structure"])
    except Exception:
        return None

    hubbard_u: dict = mat["hubbard_u"]  # e.g. {"Fe": 5.3, "Co": 3.32}

    site_features = []
    for site in struct:
        el = _element_symbol(site.specie)
        if el not in TM_ELEMENTS:
            continue
        u_raw = hubbard_u.get(el, 0.0)
        if float(u_raw) == 0.0:
            continue  # U not applied to this element — skip

        u_val = float(u_raw)
        site_coord = tuple(site.coords.tolist())
        neighbors = struct.get_neighbors(site, CUTOFF)

        if len(neighbors) < 2:
            continue

        neighbor_coords = [tuple(n.coords.tolist()) for n in neighbors]

        try:
            # Returns: (bl_mean, bl_min, bl_max, bl_std, cn, W, U/W, distortion)
            feats = mott.compute_all_features(site_coord, neighbor_coords, u_val, CUTOFF)
            site_features.append(feats)
        except Exception:
            continue

    if not site_features:
        return None

    arr = np.array(site_features)  # shape: (n_tm_sites, 8)

    # distortion column (index 7): sentinel -1.0 for non-octahedral sites
    oct_mask = arr[:, 7] >= 0.0
    distortion_mean = float(arr[oct_mask, 7].mean()) if oct_mask.any() else float("nan")
    frac_octahedral = float(oct_mask.mean())

    return {
        "material_id": mat["material_id"],
        "formula": mat["formula_pretty"],
        # ── Structural / U/W features from Rust ──────────────────────────────
        "bl_mean":      float(arr[:, 0].mean()),
        "bl_min":       float(arr[:, 1].min()),
        "bl_max":       float(arr[:, 2].max()),
        "bl_std":       float(arr[:, 3].mean()),
        "coord_num":    float(arr[:, 4].mean()),
        "bandwidth_W":  float(arr[:, 5].mean()),
        "uw_ratio":     float(arr[:, 6].mean()),  # U/W proxy (units: eV·Å²)
        "distortion":   distortion_mean,           # NaN if no octahedral sites
        "frac_octahedral": frac_octahedral,
        "n_tm_sites":   len(site_features),
        # ── DFT features from Materials Project ──────────────────────────────
        "total_magnetization":      float(mat.get("total_magnetization") or 0.0),
        "num_magnetic_sites":       int(mat.get("num_magnetic_sites") or 0),
        "is_magnetic":              int(bool(mat.get("is_magnetic"))),
        "formation_energy_per_atom": float(mat.get("formation_energy_per_atom") or 0.0),
        "energy_above_hull":        float(mat.get("energy_above_hull") or 0.0),
        "nsites":                   int(mat.get("nsites") or 0),
        "volume_per_atom":          float(mat.get("volume") or 0.0) / max(int(mat.get("nsites") or 1), 1),
        "density":                  float(mat.get("density") or 0.0),
        "nelements":                int(mat.get("nelements") or 0),
        "band_gap":                 float(mat.get("band_gap") or 0.0),
        "efermi":                   float(mat.get("efermi") or 0.0),
    }


def main():
    print("Loading raw_materials.json...")
    with open("raw_materials.json") as f:
        data = json.load(f)
    print(f"  {len(data)} materials loaded.")

    features, skipped = [], 0
    for i, mat in enumerate(data):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(data)}...")
        result = extract_tm_features(mat)
        if result is None:
            skipped += 1
        else:
            features.append(result)

    df = pd.DataFrame(features)
    df.to_csv("features.csv", index=False)

    print(f"\nDone.")
    print(f"  {len(features)} materials with features written to features.csv")
    print(f"  {skipped} skipped (no structure, no U, or no TM sites in cutoff)")
    print(f"  U/W > 1.0: {(df['uw_ratio'] > 1.0).sum()} materials")
    print(f"\nPreview:")
    print(df[["formula", "uw_ratio", "bandwidth_W", "coord_num", "distortion"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

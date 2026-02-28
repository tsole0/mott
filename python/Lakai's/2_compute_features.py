"""
2_compute_features.py

For each material in raw_materials.json that contains transition-metal sites:
  - Calls mott.compute_all_features() (Rust) per TM site
  - Averages structural features across all TM sites
  - Joins with DFT features already in the JSON
  - Skips only materials with no structure or no TM sites within cutoff

Hubbard U handling:
  - If the material has Hubbard U data, it is used to compute the U/W ratio.
  - If no Hubbard U is available (the common case in raw MP data), hubbard_u=0.0
    is passed to the Rust function, which returns NaN for uw_ratio.
  - XGBoost handles NaN natively; uw_ratio will be NaN for most materials.

Output: data/features.csv — one row per material

NOTE on units:
  W = 2 * z * mean(1/d²) has units of Å⁻²  (d in Ångströms)
  U is in eV  →  U/W has units eV·Å²
  This is a relative proxy for the Mott criterion, not a pure ratio.
  The ML model learns the effective threshold from data.
"""

import json
import math
import os
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

# Paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "raw_materials.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "features.csv")


def _element_symbol(specie) -> str:
    """Return bare element symbol regardless of whether specie is Element or Species."""
    return specie.element.symbol if hasattr(specie, "element") else specie.symbol


def extract_tm_features(mat: dict) -> dict | None:
    """
    Compute structural features for a single material dict.

    Returns None only if the material has no crystal structure or no TM sites
    with at least 2 neighbors within CUTOFF. Hubbard U is used when available,
    but absence of U does not disqualify a material from the dataset.
    """
    if mat.get("structure") is None:
        return None

    try:
        struct = Structure.from_dict(mat["structure"])
    except Exception:
        return None

    # Hubbard U dict e.g. {"Fe": 5.3} — may be None or empty for most MP entries
    hubbard_u: dict = mat.get("hubbard_u") or {}

    site_features = []
    for site in struct:
        el = _element_symbol(site.specie)
        if el not in TM_ELEMENTS:
            continue

        # Use U value if available; pass 0.0 otherwise (Rust returns NaN for uw_ratio)
        u_val = float(hubbard_u.get(el, 0.0))

        site_coord = tuple(site.coords.tolist())
        neighbors = struct.get_neighbors(site, CUTOFF)

        if len(neighbors) < 2:
            continue

        neighbor_coords = [tuple(n.coords.tolist()) for n in neighbors]

        try:
            # Returns: (bl_mean, bl_min, bl_max, bl_std, cn, W, U/W, distortion)
            # U/W is NaN when u_val == 0.0
            feats = mott.compute_all_features(site_coord, neighbor_coords, u_val, CUTOFF)
            site_features.append(feats)
        except Exception:
            continue

    if not site_features:
        return None

    arr = np.array(site_features)  # shape: (n_tm_sites, 8)

    # distortion column (index 7): sentinel -1.0 for non-octahedral sites → NaN
    oct_mask = arr[:, 7] >= 0.0
    distortion_mean = float(arr[oct_mask, 7].mean()) if oct_mask.any() else float("nan")
    frac_octahedral = float(oct_mask.mean())

    # uw_ratio column (index 6): NaN where hubbard_u was 0 — propagate NaN correctly
    uw_vals = arr[:, 6]
    valid_uw = uw_vals[~np.isnan(uw_vals)]
    uw_mean = float(valid_uw.mean()) if len(valid_uw) > 0 else float("nan")

    return {
        "material_id": mat["material_id"],
        "formula": mat["formula_pretty"],
        # ── Structural features from Rust ────────────────────────────────────
        "bl_mean":          float(arr[:, 0].mean()),
        "bl_min":           float(arr[:, 1].min()),
        "bl_max":           float(arr[:, 2].max()),
        "bl_std":           float(arr[:, 3].mean()),
        "coord_num":        float(arr[:, 4].mean()),
        "bandwidth_W":      float(arr[:, 5].mean()),
        "uw_ratio":         uw_mean,      # NaN for materials without Hubbard U
        "distortion":       distortion_mean,  # NaN if no octahedral sites
        "frac_octahedral":  frac_octahedral,
        "n_tm_sites":       len(site_features),
        # ── DFT features from Materials Project ──────────────────────────────
        "total_magnetization":       float(mat.get("total_magnetization") or 0.0),
        "num_magnetic_sites":        int(mat.get("num_magnetic_sites") or 0),
        "is_magnetic":               int(bool(mat.get("is_magnetic"))),
        "formation_energy_per_atom": float(mat.get("formation_energy_per_atom") or 0.0),
        "energy_above_hull":         float(mat.get("energy_above_hull") or 0.0),
        "nsites":                    int(mat.get("nsites") or 0),
        "volume_per_atom":           float(mat.get("volume") or 0.0) / max(int(mat.get("nsites") or 1), 1),
        "density":                   float(mat.get("density") or 0.0),
        "nelements":                 int(mat.get("nelements") or 0),
        "band_gap":                  float(mat.get("band_gap") or 0.0),
        "efermi":                    float(mat.get("efermi") or 0.0),
    }


def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print(f"  {len(data)} materials loaded.")

    features, skipped = [], 0
    for i, mat in enumerate(data):
        if i % 5000 == 0:
            print(f"  Processing {i}/{len(data)}...")
        result = extract_tm_features(mat)
        if result is None:
            skipped += 1
        else:
            features.append(result)

    df = pd.DataFrame(features)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    n_with_uw = df["uw_ratio"].notna().sum()
    print(f"\nDone.")
    print(f"  {len(features)} materials with features written to {OUTPUT_FILE}")
    print(f"  {skipped} skipped (no structure or no TM sites within cutoff)")
    print(f"  {n_with_uw} have Hubbard U data (uw_ratio not NaN)")
    print(f"  {len(features) - n_with_uw} have uw_ratio=NaN (no U available)")

    print(f"\nPreview:")
    print(df[["formula", "uw_ratio", "bandwidth_W", "coord_num", "distortion"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

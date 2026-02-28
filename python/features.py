"""
features.py
===========
Compute structural and U/W features for all materials in data/raw_materials.json.

For each material containing transition-metal sites:
  - Calls mott.compute_all_features() (Rust) per TM site
  - Averages structural features across all TM sites
  - Joins with DFT scalar features from the JSON
  - Skips only materials with no crystal structure or no TM sites within cutoff

Hubbard U handling:
  The MP summary API does not expose per-material Hubbard U parameters. Instead,
  the standard MP GGA+U effective U values (Wang, Maxisch & Ceder, PRB 2006) are
  hardcoded in MP_HUBBARD_U below — these are the values MP actually uses for each
  element and do not vary per material. Elements absent from the dict get U=0 →
  uw_ratio=NaN, which XGBoost handles natively.

Output: data/features.csv — one row per material

NOTE on units:
  W = 2 * z * mean(1/d²)  has units Å⁻²  (d in Ångströms)
  U is in eV  →  U/W has units eV·Å²
  This is a relative proxy for the Mott criterion; the ML model learns the threshold.
"""

import contextlib
import io
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer

import mott  # compiled Rust extension — build with: maturin develop

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_FILE   = DATA_DIR / "raw_materials.json"
OUTPUT_FILE  = DATA_DIR / "features.csv"

# 3d/4d/5d TM elements where Mott physics is relevant
TM_ELEMENTS = {
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Nb", "Mo", "Ru", "Rh",
    "Ta", "W",  "Os", "Ir",
}


CUTOFF = 2.7  # isolate first TM–anion shell only

# Standard MP GGA+U effective U values (eV) — Wang, Maxisch & Ceder, PRB 73, 195107 (2006).
# These are the Dudarev U_eff = U - J values used in all MP GGA+U calculations.
# Elements not listed (Ti, Cu, Nb, Ru, Rh, Ta, W, Os, Ir) are treated as U=0 → uw_ratio=NaN.
MP_HUBBARD_U: dict[str, float] = {
    "V":  3.25,
    "Cr": 3.7,
    "Mn": 3.9,
    "Fe": 5.3,
    "Co": 3.32,
    "Ni": 6.2,
    "Mo": 4.38,
}

# Oxidation-state-resolved U_eff (eV) — more specific than MP_HUBBARD_U.
# Sources: Cococcioni & de Gironcoli PRB 71 (2005); Franchini et al. PRB 75 (2007);
# Zhou et al. PRB 70 (2004); Wang, Maxisch & Ceder PRB 73 (2006).
# Falls back to MP_HUBBARD_U if the specific oxidation state is absent.
OXI_HUBBARD_U: dict[str, dict[int, float]] = {
    "V":  {2: 3.25, 3: 3.25, 4: 3.25, 5: 3.25},
    "Cr": {2: 3.5,  3: 3.7},
    "Mn": {2: 3.9,  3: 4.5,  4: 5.0},
    "Fe": {2: 4.3,  3: 5.3},
    "Co": {2: 3.32, 3: 3.8},
    "Ni": {2: 6.2,  3: 6.5},
    "Mo": {4: 4.38, 5: 4.38, 6: 0.0},
}

# Anion species for TM–anion bandwidth filtering.
# W = 2·z·mean(1/d²) should use only TM–anion bonds (superexchange/ligand-field picture);
# TM–TM bonds inflate W for metallic compounds and invert the U/W signal.
ANION_ELEMENTS: set[str] = {"O", "S", "N", "F", "Cl", "Se", "Br", "I"}

# Only Mn and Fe have meaningfully different U values by oxidation state in OXI_HUBBARD_U.
# Skip BVAnalyzer for all other elements to avoid paying the spglib cost unnecessarily.
OXI_SENSITIVE: set[str] = {"Mn", "Fe"}


def _element_symbol(specie) -> str:
    return specie.element.symbol if hasattr(specie, "element") else specie.symbol


def extract_tm_features(mat: dict) -> dict | None:
    """
    Compute structural + U/W features for a single material dict.
    Returns None if the material has no structure or no TM sites within CUTOFF.
    """
    if mat.get("structure") is None:
        return None

    # Fast pre-filter: skip entirely if no TM element present (avoids structure parse + BVAnalyzer)
    mat_elements: list = mat.get("elements") or []
    if not any(el in TM_ELEMENTS for el in mat_elements):
        return None

    try:
        struct = Structure.from_dict(mat["structure"])
    except Exception:
        return None

    hubbard_u: dict = MP_HUBBARD_U  # standard MP GGA+U values, fixed per element

    # Run BVAnalyzer only for materials containing Mn or Fe — the only elements where the
    # oxidation-state-resolved U table makes a meaningful difference (Mn²⁺/³⁺/⁴⁺, Fe²⁺/³⁺).
    needs_oxi = bool(set(mat_elements) & OXI_SENSITIVE)
    oxi_struct = None
    if needs_oxi:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                oxi_struct = BVAnalyzer().get_oxi_state_decorated_structure(struct)
        except Exception:
            oxi_struct = None

    site_features = []
    for idx, site in enumerate(struct):
        el = _element_symbol(site.specie)
        if el not in TM_ELEMENTS:
            continue

        # Oxidation-state-resolved U lookup
        oxi_state = None
        if oxi_struct is not None:
            oxi_site = oxi_struct[idx]
            oxi_state = getattr(oxi_site.specie, "oxi_state", None)

        if oxi_state is not None and el in OXI_HUBBARD_U:
            u_val = float(OXI_HUBBARD_U[el].get(int(round(oxi_state)), hubbard_u.get(el, 0.0)))
        else:
            u_val = float(hubbard_u.get(el, 0.0))

        site_coord = tuple(site.coords.tolist())
        neighbors = struct.get_neighbors(site, CUTOFF)

        if len(neighbors) < 2:
            continue

        # Filter to anion-only neighbors for physically meaningful W (TM–anion hopping).
        # Fall back to all neighbors if no anion neighbors found (rare all-metal coordination).
        anion_neighbors = [n for n in neighbors if _element_symbol(n.specie) in ANION_ELEMENTS]
        coord_neighbors = anion_neighbors if len(anion_neighbors) >= 2 else neighbors
        neighbor_coords = [tuple(n.coords.tolist()) for n in coord_neighbors]

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

    oct_mask = arr[:, 7] >= 0.0
    distortion_mean = float(arr[oct_mask, 7].mean()) if oct_mask.any() else float("nan")
    frac_octahedral = float(oct_mask.mean())

    uw_vals   = arr[:, 6]
    valid_uw  = uw_vals[~np.isnan(uw_vals)]
    uw_mean   = float(valid_uw.mean()) if len(valid_uw) > 0 else float("nan")

    return {
        "material_id":   mat["material_id"],
        "formula":       mat["formula_pretty"],
        # Structural / U/W (from Rust)
        "bl_mean":           float(arr[:, 0].mean()),
        "bl_min":            float(arr[:, 1].min()),
        "bl_max":            float(arr[:, 2].max()),
        "bl_std":            float(arr[:, 3].mean()),
        "coord_num":         float(arr[:, 4].mean()),
        "bandwidth_W":       float(arr[:, 5].mean()),
        "uw_ratio":          uw_mean,
        "distortion":        distortion_mean,
        "frac_octahedral":   frac_octahedral,
        "n_tm_sites":        len(site_features),
        # DFT scalars (from Materials Project)
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


def main() -> None:
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        data = json.load(f)
    print(f"  {len(data)} materials loaded.")

    n_workers = max(1, cpu_count() - 1)
    print(f"  Spawning {n_workers} worker processes...")

    features, skipped = [], 0
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(extract_tm_features, data, chunksize=100)):
            if i % 5000 == 0:
                print(f"  Processed {i}/{len(data)}...")
            if result is None:
                skipped += 1
            else:
                features.append(result)

    df = pd.DataFrame(features)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, lineterminator="\n")

    n_with_uw = df["uw_ratio"].notna().sum()
    print(f"\nDone.")
    print(f"  {len(features)} materials written to {OUTPUT_FILE}")
    print(f"  {skipped} skipped (no structure or no TM sites within cutoff)")
    print(f"  {n_with_uw} have Hubbard U data (uw_ratio not NaN)")
    print(f"  {len(features) - n_with_uw} have uw_ratio=NaN (no U available)")
    print(f"\nPreview:")
    print(df[["formula", "uw_ratio", "bandwidth_W", "coord_num", "distortion"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

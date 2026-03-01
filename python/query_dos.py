"""
query_dos.py
============
Query the Materials Project API for projected electronic density-of-states (DOS)
and compute the d-band second-moment width W_dft (eV) for each material.

W_dft is the full-width second-moment of the TM d-band:
    ε_center = ∫ ε · D_d(ε) dε / ∫ D_d(ε) dε
    W_dft    = 2 × √( ∫ (ε−ε_center)² · D_d(ε) dε  /  ∫ D_d(ε) dε )

This gives a physically correct bandwidth in eV, enabling dimensionless U/W = U_eV / W_dft.

Requirements:
    pip install mp-api
    export MP_API_KEY=<your key>   (or set MP_API_KEY in environment)

Inputs:
    data/features.csv      — must contain material_id column

Outputs:
    data/dos_widths.csv    — columns: material_id, W_dft
    (features.py merges this and computes uw_ratio_dft = U_eV / W_dft)

Runtime notes:
    - DOS endpoint is slow (~1–3 s/call). With 15k TM materials expect 4–12 hours.
    - Checkpointing: already-fetched IDs are skipped on restart.
    - Set BATCH_SIZE and SLEEP_BETWEEN (seconds) to stay within API rate limits.
    - materials without a projected DOS (no band-structure run) silently yield W_dft=NaN.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features.csv"
OUTPUT_FILE   = DATA_DIR / "dos_widths.csv"

# TM elements whose d-DOS we want
TM_ELEMENTS = {
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Nb", "Mo", "Ru", "Rh",
    "Ta", "W",  "Os", "Ir",
}

# Pause between individual API calls (seconds). Increase if hitting rate limits.
SLEEP_BETWEEN = 0.5

# How many IDs to request per API call (mp-api supports list queries)
BATCH_SIZE = 20


def _d_band_width(dos, el_symbol: str) -> float:
    """
    Compute the second-moment d-band width (eV) for element el_symbol from a
    CompleteDos object.  Returns NaN if no d-DOS is present.
    """
    from pymatgen.electronic_structure.core import OrbitalType, Spin

    # Sum d-projected DOS over all sites with this element
    energies = np.array(dos.energies) - dos.efermi  # shift to Fermi level

    d_total = np.zeros_like(energies)
    for site, orbital_dict in dos.pdos.items():
        if site.specie.symbol != el_symbol:
            continue
        if OrbitalType.d not in orbital_dict:
            continue
        for spin, densities in orbital_dict[OrbitalType.d].items():
            d_total += np.array(densities)

    integral = np.trapz(d_total, energies)
    if integral <= 0:
        return float("nan")

    # d-band center
    center = np.trapz(energies * d_total, energies) / integral

    # second moment
    second_moment = np.trapz((energies - center) ** 2 * d_total, energies) / integral
    if second_moment < 0:
        return float("nan")

    # Width = 2 × √(second_moment)  (half-bandwidth × 2)
    return float(2.0 * np.sqrt(second_moment))


def fetch_dos_widths(material_ids: list[str], api_key: str) -> dict[str, float]:
    """
    Fetch projected DOS for a list of material IDs and return {material_id: W_dft}.
    Materials with no DOS or no TM d-band return NaN.
    """
    from mp_api.client import MPRester

    results: dict[str, float] = {}

    with MPRester(api_key) as mpr:
        for i in range(0, len(material_ids), BATCH_SIZE):
            batch = material_ids[i : i + BATCH_SIZE]
            print(f"  Fetching DOS for {i}–{i+len(batch)-1} / {len(material_ids)} ...")
            try:
                docs = mpr.materials.electronic_structure.get_data_by_id(
                    batch, fields=["material_id", "dos"]
                )
            except Exception as exc:
                print(f"    [WARN] Batch {i} failed: {exc}")
                for mid in batch:
                    results[mid] = float("nan")
                time.sleep(SLEEP_BETWEEN * 2)
                continue

            for doc in docs:
                mid = doc.material_id
                dos = getattr(doc, "dos", None)
                if dos is None or not hasattr(dos, "pdos"):
                    results[mid] = float("nan")
                    continue

                # Average W_dft over all TM elements present in the DOS
                widths = []
                for el in TM_ELEMENTS:
                    w = _d_band_width(dos, el)
                    if not np.isnan(w):
                        widths.append(w)

                results[mid] = float(np.mean(widths)) if widths else float("nan")

            time.sleep(SLEEP_BETWEEN)

    return results


def main() -> None:
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "MP_API_KEY environment variable is not set.\n"
            "Export it with:  export MP_API_KEY=<your key>"
        )

    if not FEATURES_FILE.exists():
        raise SystemExit(f"Features file not found: {FEATURES_FILE}\nRun features.py first.")

    df_feat = pd.read_csv(FEATURES_FILE, usecols=["material_id"])
    all_ids = df_feat["material_id"].dropna().unique().tolist()
    print(f"Total materials in features.csv: {len(all_ids)}")

    # Checkpointing: skip IDs already fetched
    fetched: dict[str, float] = {}
    if OUTPUT_FILE.exists():
        df_prev = pd.read_csv(OUTPUT_FILE)
        fetched = dict(zip(df_prev["material_id"], df_prev["W_dft"]))
        print(f"  Resuming — {len(fetched)} already fetched, {len(all_ids) - len(fetched)} remaining.")

    todo_ids = [mid for mid in all_ids if mid not in fetched]

    if not todo_ids:
        print("All materials already fetched. Nothing to do.")
        return

    new_results = fetch_dos_widths(todo_ids, api_key)
    fetched.update(new_results)

    df_out = pd.DataFrame(
        [(mid, w) for mid, w in fetched.items()],
        columns=["material_id", "W_dft"],
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    n_valid = df_out["W_dft"].notna().sum()
    print(f"\nSaved {OUTPUT_FILE}")
    print(f"  {n_valid} / {len(df_out)} have valid W_dft")
    print(f"  Median W_dft = {df_out['W_dft'].median():.3f} eV")


if __name__ == "__main__":
    main()

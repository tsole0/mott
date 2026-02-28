"""
label.py
========
Label materials in data/features.csv as Mott insulators (1) or not (0).

Labeling strategy:
  Primary  — known Mott insulators from the literature (Imada et al. 1998 + updates).
             This is the training label used by the model.
  Heuristic — U/W proxy (U/W > threshold). Stored separately for analysis only;
              NOT used as a training label to avoid the model trivially learning U/W.

Output: data/labeled_features.csv
"""

from pathlib import Path
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_FILE   = DATA_DIR / "features.csv"
OUTPUT_FILE  = DATA_DIR / "labeled_features.csv"

# U/W threshold for the heuristic label (analysis only, not training)
UW_HEURISTIC_THRESHOLD = 1.0

# Curated Mott insulator list (Materials Project formula_pretty format).
# Sources: Imada, Fujimori & Tokura, Rev. Mod. Phys. 70 (1998);
#          Georges et al. DMFT reviews; experimental band gap databases.
KNOWN_MOTT_INSULATORS: set[str] = {
    # Binary 3d TM oxides
    "NiO", "CoO", "MnO", "FeO", "CuO", "Cu2O",
    "Cr2O3", "V2O3", "Fe2O3", "Co3O4", "Mn2O3", "Mn3O4",
    "VO", "TiO",
    # Vanadium / titanium (Mott transition materials)
    "VO2", "V2O5", "Ti2O3",
    # 3d TM sulfides
    "NiS", "CoS", "FeS", "MnS", "CrS",
    # Perovskites (undoped insulating parent compounds)
    "LaMnO3", "LaFeO3", "LaCoO3", "LaCrO3", "LaVO3",
    "YVO3", "GdVO3", "NdVO3", "PrVO3", "SmVO3",
    "BiFeO3", "GdFeO3", "YFeO3", "EuFeO3",
    "LaTiO3", "YTiO3", "SmTiO3",
    # Undoped cuprate parents (Mott-Hubbard / charge-transfer insulators)
    "La2CuO4", "Nd2CuO4", "Sr2CuO2Cl2", "Ca2CuO2Cl2",
    "CaCuO2", "SrCuO2",
    # Iridates
    "Sr2IrO4", "Na2IrO3", "Li2IrO3", "Ba2IrO4",
    # Infinite-layer nickelates
    "LaNiO2", "NdNiO2", "PrNiO2",
    # Other well-established Mott insulators
    "TiOCl", "TiOBr",
    "MnF2", "FeF2", "CoF2", "NiF2",
    "NiCl2", "CoCl2", "FeCl2", "MnCl2",
}


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_mott"]    = df["formula"].isin(KNOWN_MOTT_INSULATORS).astype(int)
    df["is_mott_uw"] = (df["uw_ratio"] > UW_HEURISTIC_THRESHOLD).astype(int)
    return df


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df = label_data(df)

    n_lit   = df["is_mott"].sum()
    n_uw    = df["is_mott_uw"].sum()
    n_total = len(df)

    print(f"Total materials:               {n_total}")
    print(f"Labeled Mott (literature):     {n_lit}  ({100 * n_lit / n_total:.1f}%)")
    print(f"Labeled Mott (U/W heuristic):  {n_uw}   ({100 * n_uw / n_total:.1f}%)")
    print(f"Label agreement (lit vs U/W):  {100 * (df['is_mott'] == df['is_mott_uw']).mean():.1f}%")

    print("\nKnown Mott insulators found in dataset:")
    found = df[df["is_mott"] == 1][["formula", "uw_ratio", "coord_num", "distortion", "is_magnetic"]]
    if found.empty:
        print("  None found — check formula matching or dataset coverage.")
    else:
        print(found.to_string(index=False))

    print("\nMaterials with U/W > threshold NOT in literature list (candidates):")
    candidates = df[(df["is_mott_uw"] == 1) & (df["is_mott"] == 0)]
    print(candidates[["formula", "uw_ratio", "distortion"]].head(20).to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

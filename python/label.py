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
    "NiO", "CoO", "MnO", "FeO", "CuO",
    "Cr2O3", "V2O3", "Fe2O3", "Co3O4", "Mn2O3", "Mn3O4",
    # Vanadium / titanium (Mott transition materials)
    "VO2", "Ti2O3",
    # 3d TM sulfides
    "NiS", "FeS", "MnS", "CrS",
    # Pyrite Mott insulator
    "NiS2",
    # van der Waals layered Mott insulators (MPX3 family)
    "FePS3", "NiPS3", "CoPS3", "MnPS3",
    # Perovskites (undoped insulating parent compounds)
    "LaMnO3", "LaFeO3", "LaCrO3", "LaVO3",
    "YVO3", "GdVO3", "NdVO3", "PrVO3", "SmVO3",
    "BiFeO3", "GdFeO3", "YFeO3", "EuFeO3",
    # Rare-earth titanates (Ti3+ d1 Mott insulators)
    "LaTiO3", "YTiO3", "SmTiO3", "GdTiO3", "EuTiO3", "DyTiO3", "TbTiO3",
    # 2D Mott insulators (d1 vanadates)
    "Sr2VO4", "BaVS3",
    # Undoped cuprate parents (charge-transfer insulators, Mott-Hubbard adjacent)
    "La2CuO4", "Nd2CuO4", "Sr2CuO2Cl2", "Ca2CuO2Cl2",
    "CaCuO2", "SrCuO2",
    # Iridates (jeff=1/2 Mott insulators, 5d)
    "Sr2IrO4", "Na2IrO3", "Li2IrO3", "Ba2IrO4",
    # Ruthenates  (d4 Mott insulator)
    "Ca2RuO4",
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

    # Structured heuristic label: insulating + magnetic + d-electron count in Mott range.
    # NOT used as a training label — diagnostic only, like is_mott_uw.
    cond = (df["band_gap"] > 0.1) & (df["is_magnetic"] == 1)
    if "d_count_mean" in df.columns:
        cond = cond & (df["d_count_mean"] >= 1) & (df["d_count_mean"] <= 9)
    df["is_mott_structured"] = cond.astype(int)

    return df


def main() -> None:
    df = pd.read_csv(INPUT_FILE, on_bad_lines="skip")
    df = label_data(df)

    n_lit   = df["is_mott"].sum()
    n_uw    = df["is_mott_uw"].sum()
    n_total = len(df)

    n_structured = df["is_mott_structured"].sum()
    print(f"Total materials:                   {n_total}")
    print(f"Labeled Mott (literature):         {n_lit}  ({100 * n_lit / n_total:.1f}%)")
    print(f"Labeled Mott (U/W heuristic):      {n_uw}   ({100 * n_uw / n_total:.1f}%)")
    print(f"Labeled Mott (structured heuristic): {n_structured}  ({100 * n_structured / n_total:.1f}%)")
    print(f"Label agreement (lit vs U/W):      {100 * (df['is_mott'] == df['is_mott_uw']).mean():.1f}%")
    print(f"Label agreement (lit vs structured): {100 * (df['is_mott'] == df['is_mott_structured']).mean():.1f}%")

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

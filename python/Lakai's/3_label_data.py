"""
3_label_data.py

Labels each material in features.csv as a Mott insulator (1) or true metal (0).

Labeling strategy:
  Primary — known Mott insulators from the literature (Imada et al. 1998 + updates).
  Fallback — materials flagged as potential Mott insulators by a high U/W proxy
             (U/W > UW_HEURISTIC_THRESHOLD), used ONLY for exploratory analysis,
             NOT included in the training labels to avoid the model trivially learning U/W.

Output:
  labeled_features.csv — same as features.csv with added columns:
    is_mott        : training label (literature only)
    is_mott_uw     : heuristic label (U/W threshold, for comparison only)
"""

import pandas as pd

# ── Curated Mott insulator list ───────────────────────────────────────────────
# Uses Materials Project formula_pretty format (reduced formula).
# Sources: Imada, Fujimori & Tokura, Rev. Mod. Phys. 70 (1998);
#          Georges et al. (DMFT reviews); experimental band gap databases.
KNOWN_MOTT_INSULATORS: set[str] = {
    # Binary 3d TM oxides
    "NiO", "CoO", "MnO", "FeO", "CuO", "Cu2O",
    "Cr2O3", "V2O3", "Fe2O3", "Co3O4", "Mn2O3", "Mn3O4",
    "VO", "TiO",
    # Vanadium / titanium compounds (Mott transition materials)
    "VO2", "V2O5", "Ti2O3",
    # 3d TM sulfides
    "NiS", "CoS", "FeS", "MnS", "CrS",
    # Perovskites (undoped, insulating parent compounds)
    "LaMnO3", "LaFeO3", "LaCoO3", "LaCrO3", "LaVO3",
    "YVO3", "GdVO3", "NdVO3", "PrVO3", "SmVO3",
    "BiFeO3", "GdFeO3", "YFeO3", "EuFeO3",
    "LaTiO3", "YTiO3", "SmTiO3",
    # Undoped cuprate parents (Mott-Hubbard / charge-transfer insulators)
    "La2CuO4", "Nd2CuO4", "Sr2CuO2Cl2", "Ca2CuO2Cl2",
    "CaCuO2", "SrCuO2",
    # Iridates
    "Sr2IrO4", "Na2IrO3", "Li2IrO3", "Ba2IrO4",
    # Nickelates (infinite-layer)
    "LaNiO2", "NdNiO2", "PrNiO2",
    # Other well-established Mott insulators
    "TiOCl", "TiOBr",
    "MnF2", "FeF2", "CoF2", "NiF2",
    "NiCl2", "CoCl2", "FeCl2", "MnCl2",
}

# U/W threshold used for the heuristic label only (not for training)
UW_HEURISTIC_THRESHOLD = 1.0


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Primary training label: literature list
    df["is_mott"] = df["formula"].isin(KNOWN_MOTT_INSULATORS).astype(int)

    # Heuristic label: U/W proxy (for analysis / comparison only)
    df["is_mott_uw"] = (df["uw_ratio"] > UW_HEURISTIC_THRESHOLD).astype(int)

    return df


def main():
    df = pd.read_csv("features.csv")
    df = label_data(df)

    n_lit  = df["is_mott"].sum()
    n_uw   = df["is_mott_uw"].sum()
    n_total = len(df)

    print(f"Total materials:               {n_total}")
    print(f"Labeled Mott (literature):     {n_lit}  ({100*n_lit/n_total:.1f}%)")
    print(f"Labeled Mott (U/W heuristic):  {n_uw}   ({100*n_uw/n_total:.1f}%)")

    # Agreement between labels
    agree = (df["is_mott"] == df["is_mott_uw"]).mean()
    print(f"Label agreement (lit vs U/W):  {100*agree:.1f}%")

    print("\nKnown Mott insulators found in dataset:")
    found = df[df["is_mott"] == 1][["formula", "uw_ratio", "coord_num", "distortion", "is_magnetic"]]
    if found.empty:
        print("  None found — check formula matching or dataset coverage.")
    else:
        print(found.to_string(index=False))

    print("\nMaterials with U/W > threshold NOT in literature list (candidates):")
    candidates = df[(df["is_mott_uw"] == 1) & (df["is_mott"] == 0)]
    print(candidates[["formula", "uw_ratio", "distortion"]].head(20).to_string(index=False))

    df.to_csv("labeled_features.csv", index=False)
    print("\nSaved labeled_features.csv")


if __name__ == "__main__":
    main()

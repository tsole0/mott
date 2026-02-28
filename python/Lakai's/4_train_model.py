"""
4_train_model.py

Trains a Random Forest classifier to identify Mott insulators among DFT metals.

Features used:
  Structural (from Rust module):
    bl_mean, bl_min, bl_max, bl_std    — bond length statistics
    coord_num                          — mean TM coordination number
    bandwidth_W                        — tight-binding bandwidth proxy (Å⁻²)
    uw_ratio                           — U/W proxy (eV·Å²); core Mott signal
    distortion                         — octahedral distortion index Δd
    frac_octahedral                    — fraction of TM sites that are octahedral
    n_tm_sites                         — number of TM sites used

  DFT (from Materials Project):
    total_magnetization, num_magnetic_sites, is_magnetic
    formation_energy_per_atom, energy_above_hull
    nsites, volume_per_atom, density, nelements, band_gap, efermi

Labels: is_mott from labeled_features.csv (literature-based only)

Outputs:
  mott_classifier.pkl     — trained model (joblib)
  feature_importances.png — bar chart of RF feature importances
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    # Structural / U/W
    "bl_mean", "bl_min", "bl_max", "bl_std",
    "coord_num", "bandwidth_W", "uw_ratio",
    "distortion", "frac_octahedral", "n_tm_sites",
    # DFT
    "total_magnetization", "num_magnetic_sites", "is_magnetic",
    "formation_energy_per_atom", "energy_above_hull",
    "nsites", "volume_per_atom", "density", "nelements",
    "band_gap", "efermi",
]

LABEL_COL = "is_mott"


def build_pipeline() -> Pipeline:
    """
    Median imputation (handles NaN distortion for non-octahedral materials)
    followed by a balanced Random Forest.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",   # critical: Mott insulators are rare
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    df = pd.read_csv("labeled_features.csv")

    # Drop rows that are entirely missing the feature columns
    df = df.dropna(subset=[c for c in FEATURE_COLS if c != "distortion"])

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    n_pos = y.sum()
    n_total = len(y)
    print(f"Dataset: {n_total} samples  |  Mott: {n_pos} ({100*n_pos/n_total:.1f}%)  |  Metal: {n_total - n_pos}")

    if n_pos < 5:
        print("\nWARNING: Very few positive samples. Cross-validation results will be unreliable.")
        print("Consider expanding the known Mott insulator list in 3_label_data.py.")

    # ── Cross-validation ──────────────────────────────────────────────────────
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
    )

    print("\n── 5-fold Stratified Cross-Validation ──────────────────────────────")
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Train final model on full dataset ────────────────────────────────────
    pipe.fit(X, y)
    joblib.dump(pipe, "mott_classifier.pkl")
    print("\nModel saved → mott_classifier.pkl")

    # Full-dataset report (optimistic — for sanity check only)
    y_pred = pipe.predict(X)
    print("\nFull-dataset classification report (train set — optimistic):")
    print(classification_report(y, y_pred, target_names=["Metal", "Mott insulator"]))

    # ── Feature importances ───────────────────────────────────────────────────
    rf = pipe.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    print("Top 10 features:")
    print(importances.tail(10).iloc[::-1].to_string())

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature Importance (mean decrease impurity)")
    ax.set_title("Mott Insulator Classifier — Feature Importances")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    print("Feature importance plot saved → feature_importances.png")

    # ── Apply to full dataset — flag candidate Mott insulators ───────────────
    df["mott_prob"] = pipe.predict_proba(X)[:, 1]
    df["mott_pred"] = pipe.predict(X)

    print("\nTop predicted Mott insulators (not in training labels):")
    novel = df[(df["mott_pred"] == 1) & (df[LABEL_COL] == 0)]
    novel_sorted = novel.sort_values("mott_prob", ascending=False)
    print(novel_sorted[["formula", "mott_prob", "uw_ratio", "distortion", "is_magnetic"]].head(20).to_string(index=False))

    df.to_csv("predictions.csv", index=False)
    print("\nFull predictions saved → predictions.csv")


if __name__ == "__main__":
    main()

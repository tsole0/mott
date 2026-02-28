"""
train.py
========
Train an XGBoost classifier to identify Mott insulators.

XGBoost is used because it handles NaN natively — no imputation needed for
uw_ratio (absent when Hubbard U is unavailable) or distortion (absent for
non-octahedral sites).

Features (21 total):
  Structural (from Rust / features.py):
    bl_mean, bl_min, bl_max, bl_std    — bond length statistics
    coord_num                          — mean TM coordination number
    bandwidth_W                        — tight-binding bandwidth proxy (Å⁻²)
    uw_ratio                           — U/W proxy (eV·Å²); NaN if no Hubbard U
    distortion                         — octahedral distortion index Δd; NaN if non-octahedral
    frac_octahedral                    — fraction of TM sites that are octahedral
    n_tm_sites                         — number of TM sites used

  DFT scalars (from Materials Project):
    total_magnetization, num_magnetic_sites, is_magnetic
    formation_energy_per_atom, energy_above_hull
    nsites, volume_per_atom, density, nelements, band_gap, efermi

Label: is_mott from data/labeled_features.csv (literature-based, see label.py)

Outputs:
  data/mott_classifier.pkl     — trained XGBClassifier (joblib)
  data/feature_importances.png — feature importance bar chart (gain)
  data/predictions.csv         — full dataset with mott_prob and mott_pred columns
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_FILE   = DATA_DIR / "labeled_features.csv"
MODEL_FILE   = DATA_DIR / "mott_classifier.pkl"
PLOT_FILE    = DATA_DIR / "feature_importances.png"
PREDS_FILE   = DATA_DIR / "predictions.csv"

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

# Columns that must be non-NaN for a row to be usable.
# uw_ratio and distortion are intentionally allowed to be NaN.
REQUIRED_COLS = [c for c in FEATURE_COLS if c not in ("uw_ratio", "distortion")]


def build_classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    # Drop rows missing any required (non-optional) feature
    df = df.dropna(subset=REQUIRED_COLS)

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    n_pos   = int(y.sum())
    n_neg   = int((y == 0).sum())
    n_total = len(y)
    print(f"Dataset: {n_total} samples  |  Mott: {n_pos} ({100 * n_pos / n_total:.1f}%)  |  Non-Mott: {n_neg}")

    if n_pos < 5:
        print("\nWARNING: Very few positive samples. CV results will be unreliable.")
        print("Consider expanding KNOWN_MOTT_INSULATORS in label.py.")

    scale_pos_weight = n_neg / max(n_pos, 1)

    # Cross-validation
    clf = build_classifier(scale_pos_weight)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        clf, X, y, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
    )

    print("\n── 5-fold Stratified Cross-Validation ──────────────────────────────")
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train final model on full dataset
    clf = build_classifier(scale_pos_weight)
    clf.fit(X, y)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")

    y_pred = clf.predict(X)
    print("\nFull-dataset classification report (train set — optimistic):")
    print(classification_report(y, y_pred, target_names=["Non-Mott", "Mott insulator"]))

    # Feature importances (gain = average loss reduction per split)
    scores     = clf.get_booster().get_score(importance_type="gain")
    importances = pd.Series({c: scores.get(f"f{i}", 0.0) for i, c in enumerate(FEATURE_COLS)})
    importances = importances.sort_values(ascending=True)

    print("Top 10 features by gain:")
    print(importances.tail(10).iloc[::-1].to_string())

    fig, ax = plt.subplots(figsize=(8, 7))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Mott Insulator Classifier — Feature Importances")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Feature importance plot saved → {PLOT_FILE}")

    # Apply to full dataset
    df["mott_prob"] = clf.predict_proba(X)[:, 1]
    df["mott_pred"] = clf.predict(X)

    print("\nTop predicted Mott insulators not in training labels:")
    novel = df[(df["mott_pred"] == 1) & (df[LABEL_COL] == 0)]
    print(
        novel.sort_values("mott_prob", ascending=False)
        [["formula", "mott_prob", "uw_ratio", "distortion", "is_magnetic"]]
        .head(20)
        .to_string(index=False)
    )

    df.to_csv(PREDS_FILE, index=False)
    print(f"\nFull predictions saved → {PREDS_FILE}")


if __name__ == "__main__":
    main()

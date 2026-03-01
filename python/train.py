"""
train.py
========
Train an XGBoost classifier to identify Mott insulators.

XGBoost is used because it handles NaN natively — no imputation needed for
uw_ratio (absent when Hubbard U is unavailable) or distortion (absent for
non-octahedral sites).

Features (23 total):
  Structural (from Rust / features.py):
    bl_mean, bl_min, bl_max, bl_std    — bond length statistics
    coord_num                          — mean TM coordination number
    bandwidth_W                        — tight-binding bandwidth proxy (Å⁻²)
    uw_ratio                           — U/W proxy (eV·Å²); NaN if no Hubbard U
    distortion                         — octahedral distortion index Δd; NaN if non-octahedral
    frac_octahedral                    — fraction of TM sites that are octahedral
    n_tm_sites                         — number of TM sites used
    d_count_mean, d_count_std          — d-electron count statistics

  DFT scalars (from Materials Project):
    total_magnetization, num_magnetic_sites, is_magnetic
    formation_energy_per_atom, energy_above_hull
    nsites, volume_per_atom, density, nelements, band_gap, efermi

  DOS-derived (optional, from query_dos.py):
    W_dft          — d-band second-moment width (eV); NaN if not available
    uw_ratio_dft   — U / W_dft (dimensionless); NaN if not available

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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_FILE   = DATA_DIR / "labeled_features.csv"
MODEL_FILE   = DATA_DIR / "mott_classifier.pkl"
PLOT_FILE    = DATA_DIR / "feature_importances.png"
PREDS_FILE   = DATA_DIR / "predictions.csv"

# Core features (always present after features.py)
_BASE_FEATURE_COLS = [
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

# Features added in later pipeline versions — included only if present in the CSV
_OPTIONAL_EXTENDED_COLS = ["d_count_mean", "d_count_std"]

# Optional DOS-derived features — included only if present in the CSV
_DOS_FEATURE_COLS = ["W_dft", "uw_ratio_dft"]

LABEL_COL = "is_mott"

# Columns that must be non-NaN for a row to be usable.
# uw_ratio, distortion, and DOS columns are intentionally allowed to be NaN.
_OPTIONAL_COLS = {"uw_ratio", "distortion", "W_dft", "uw_ratio_dft"}


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


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristic baseline: insulating (band_gap > 0.1 eV), magnetic, d-count 1–9.
    Requires no learning — represents domain-knowledge-only performance.
    """
    cond = (
        (df["band_gap"] > 0.1)
        & (df["is_magnetic"] == 1)
    )
    if "d_count_mean" in df.columns:
        cond = cond & (df["d_count_mean"] >= 1) & (df["d_count_mean"] <= 9)
    return cond.astype(int).values


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    # Determine which features are available
    feature_cols = list(_BASE_FEATURE_COLS)
    for col in _OPTIONAL_EXTENDED_COLS:
        if col in df.columns:
            feature_cols.append(col)
            print(f"  [INFO] Extended feature '{col}' found and included.")
    for col in _DOS_FEATURE_COLS:
        if col in df.columns:
            feature_cols.append(col)
            print(f"  [INFO] Optional feature '{col}' found and included.")

    required_cols = [c for c in feature_cols if c not in _OPTIONAL_COLS]

    # Drop rows missing any required (non-optional) feature
    df = df.dropna(subset=required_cols)

    X = df[feature_cols].values
    y = df[LABEL_COL].values

    n_pos   = int(y.sum())
    n_neg   = int((y == 0).sum())
    n_total = len(y)
    print(f"Dataset: {n_total} samples  |  Mott: {n_pos} ({100 * n_pos / n_total:.1f}%)  |  Non-Mott: {n_neg}")

    if n_pos < 5:
        print("\nWARNING: Very few positive samples. CV results will be unreliable.")
        print("Consider expanding KNOWN_MOTT_INSULATORS in label.py.")

    # ── 80/20 stratified train/test split ──────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    df_test = df.iloc[idx_test].copy()

    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    print(f"Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")
    print(f"  Train Mott: {n_pos_train}  |  Test Mott: {int(y_test.sum())}")

    scale_pos_weight = n_neg_train / max(n_pos_train, 1)

    # ── Baseline heuristic on test set ─────────────────────────────────────
    baseline_preds = baseline_predict(df_test)
    print("\n── Baseline heuristic (band_gap>0.1 & magnetic & d1-9) — TEST SET ──")
    print(classification_report(y_test, baseline_preds, target_names=["Non-Mott", "Mott insulator"],
                                 zero_division=0))
    try:
        baseline_auc = roc_auc_score(y_test, baseline_preds)
        print(f"  Baseline ROC-AUC: {baseline_auc:.3f}")
    except Exception:
        pass

    # ── Cross-validation on TRAIN set only ─────────────────────────────────
    clf = build_classifier(scale_pos_weight)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        clf, X_train, y_train, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        return_train_score=False,
    )

    print("\n── 5-fold Stratified CV (train set only) ───────────────────────────")
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Train final model on full TRAIN set ────────────────────────────────
    clf = build_classifier(scale_pos_weight)
    clf.fit(X_train, y_train)

    # ── Held-out TEST SET evaluation (unbiased) ────────────────────────────
    y_test_pred = clf.predict(X_test)
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    print("\n── Held-out TEST SET evaluation (unbiased) ─────────────────────────")
    print(classification_report(y_test, y_test_pred, target_names=["Non-Mott", "Mott insulator"],
                                 zero_division=0))
    try:
        test_auc = roc_auc_score(y_test, y_test_prob)
        print(f"  Test ROC-AUC: {test_auc:.3f}")
    except Exception:
        pass

    # ── Retrain on full dataset for predictions ────────────────────────────
    n_pos_full = int(y.sum())
    n_neg_full = int((y == 0).sum())
    scale_pos_weight_full = n_neg_full / max(n_pos_full, 1)
    clf_full = build_classifier(scale_pos_weight_full)
    clf_full.fit(X, y)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_full, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")

    y_pred_full = clf_full.predict(X)
    print("\nFull-dataset classification report (train set — optimistic):")
    print(classification_report(y, y_pred_full, target_names=["Non-Mott", "Mott insulator"]))

    # Feature importances (gain = average loss reduction per split)
    scores_fi   = clf_full.get_booster().get_score(importance_type="gain")
    importances = pd.Series({c: scores_fi.get(f"f{i}", 0.0) for i, c in enumerate(feature_cols)})
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
    df["mott_prob"] = clf_full.predict_proba(X)[:, 1]
    df["mott_pred"] = clf_full.predict(X)

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

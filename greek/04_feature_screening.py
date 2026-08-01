"""
04_feature_screening.py
=======================
Screens all extracted features for diachronic signal using the same
statistical pipeline as the Hebrew analysis:

  1. Spearman ρ correlation of each feature with date_ce (training set only)
  2. Retain features with two-tailed p < 0.10
  3. Leave-one-out (LOO) robustness check:
       For each feature passing step 2, recompute Spearman ρ on each
       N-1 subset (dropping one training entry at a time).
       Feature is "robust" if it retains p < 0.10 in ≥ 75% of LOO folds.
  4. Output two CSVs:
       results/feature_scan_full.csv   — all features + statistics
       results/feature_scan_robust.csv — LOO-robust subset only

The robust subset is used by the MVN dating model (05_mvn_dating.py).

Usage
-----
    python 04_feature_screening.py [--family {gram,char3,char4,bigram,all}]
                                   [--p-thresh 0.10] [--loo-frac 0.75]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE       = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR   = os.path.join(HERE, "data", "features")
RESULTS    = os.path.join(HERE, "results")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

# ---------------------------------------------------------------------------
# Spearman + LOO screening
# ---------------------------------------------------------------------------

def spearman_screen(dates: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return (rho, p_two_tailed) for Spearman correlation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(dates, values, nan_policy="omit")
    return float(rho), float(p)


def loo_robustness(dates: np.ndarray, values: np.ndarray, p_thresh: float) -> float:
    """
    Leave-one-out Spearman robustness.
    Returns fraction of N-1 folds where p < p_thresh.
    """
    n = len(dates)
    passing = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d_loo = dates[mask]
        v_loo = values[mask]
        if len(d_loo) < 5:
            continue
        _, p = spearman_screen(d_loo, v_loo)
        if p < p_thresh:
            passing += 1
    return passing / n


def screen_features(
    df_train: pd.DataFrame,
    date_col: str = "date_ce",
    p_thresh: float = 0.10,
    loo_frac: float = 0.75,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """
    Screen all non-metadata columns in df_train.
    Returns a DataFrame of results sorted by |rho|.
    """
    meta_cols = {date_col, "author", "date_sigma", "genre", "holdout", "word_count"}
    all_feat_cols = [c for c in df_train.columns if c not in meta_cols]

    # Filter by family prefix if requested
    if families and "all" not in families:
        prefix_map = {
            "gram":   lambda c: not c.startswith(("c3_", "c4_", "bg_")),
            "char3":  lambda c: c.startswith("c3_"),
            "char4":  lambda c: c.startswith("c4_"),
            "bigram": lambda c: c.startswith("bg_"),
        }
        keep = set()
        for fam in families:
            if fam in prefix_map:
                keep |= {c for c in all_feat_cols if prefix_map[fam](c)}
        all_feat_cols = [c for c in all_feat_cols if c in keep]

    dates  = df_train[date_col].values.astype(float)
    n_feat = len(all_feat_cols)

    print(f"Screening {n_feat} features (p_thresh={p_thresh}, loo_frac={loo_frac}) …")

    records = []
    for i, col in enumerate(all_feat_cols):
        if i % 100 == 0:
            print(f"  {i}/{n_feat} …")
        values = df_train[col].values.astype(float)

        # Skip features with no variance
        if np.nanstd(values) < 1e-10:
            continue

        rho, p = spearman_screen(dates, values)

        # Direction flag: positive rho = increases with later date
        direction = "increases" if rho > 0 else "decreases"

        rec = {
            "feature"    : col,
            "family"     : ("gram" if not col.startswith(("c3_","c4_","bg_"))
                            else col.split("_")[0]),
            "rho"        : round(rho, 4),
            "p_value"    : round(p, 6),
            "abs_rho"    : round(abs(rho), 4),
            "direction"  : direction,
            "pass_p"     : p < p_thresh,
            "loo_frac"   : None,
            "robust"     : False,
        }

        if p < p_thresh:
            loo = loo_robustness(dates, values, p_thresh)
            rec["loo_frac"] = round(loo, 4)
            rec["robust"]   = loo >= loo_frac

        records.append(rec)

    df = pd.DataFrame(records).sort_values("abs_rho", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(df_full: pd.DataFrame, df_robust: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"Features screened      : {len(df_full)}")
    print(f"Pass p < threshold     : {df_full['pass_p'].sum()}")
    print(f"LOO-robust             : {len(df_robust)}")

    print(f"\nTop 20 robust features (by |ρ|):")
    print(f"{'Feature':<45}  {'ρ':>7}  {'p':>8}  {'LOO':>6}  Direction")
    print("-" * 78)
    for _, row in df_robust.head(20).iterrows():
        feat = str(row["feature"])[:44]
        print(f"{feat:<45}  {row['rho']:>7.4f}  {row['p_value']:>8.5f}  "
              f"{row['loo_frac']:>6.3f}  {row['direction']}")

    # Breakdown by family
    print(f"\nRobust features by family:")
    for fam, grp in df_robust.groupby("family"):
        inc = (grp["direction"] == "increases").sum()
        dec = (grp["direction"] == "decreases").sum()
        print(f"  {fam:<10}  {len(grp):3d}  (↑ {inc}  ↓ {dec})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Screen Greek features for diachronic signal.")
    parser.add_argument("--family", nargs="+", default=["all"],
                        choices=["gram","char3","char4","bigram","all"])
    parser.add_argument("--p-thresh", type=float, default=0.10)
    parser.add_argument("--loo-frac", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    feat_path = os.path.join(FEAT_DIR, "feature_matrix.csv")
    if not os.path.exists(feat_path):
        print("feature_matrix.csv not found — run 03_feature_extraction.py first.")
        sys.exit(1)

    print(f"Loading feature matrix from {feat_path} …")
    df_all = pd.read_csv(feat_path, index_col="id")
    print(f"  Loaded: {df_all.shape[0]} entries × {df_all.shape[1]} columns")

    # Split training / holdout
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)
    holdout_ids = {c["id"] for c in corpus if c["holdout"]}

    df_train = df_all[~df_all.index.isin(holdout_ids)].copy()
    print(f"  Training entries: {len(df_train)}")

    # Screen
    df_full = screen_features(
        df_train,
        p_thresh=args.p_thresh,
        loo_frac=args.loo_frac,
        families=args.family,
    )
    df_robust = df_full[df_full["robust"]].copy()

    # Save
    full_path   = os.path.join(RESULTS, "feature_scan_full.csv")
    robust_path = os.path.join(RESULTS, "feature_scan_robust.csv")
    df_full.to_csv(full_path, index=False)
    df_robust.to_csv(robust_path, index=False)

    print_summary(df_full, df_robust)
    print(f"\nFull scan    → {full_path}")
    print(f"Robust scan  → {robust_path}")

    # Also save a list of just the robust feature names (easy to load downstream)
    robust_names = df_robust["feature"].tolist()
    names_path   = os.path.join(RESULTS, "robust_feature_names.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(robust_names, f, indent=2)
    print(f"Feature names→ {names_path}")


if __name__ == "__main__":
    main()

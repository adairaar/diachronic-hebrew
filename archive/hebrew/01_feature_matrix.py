"""
01_feature_matrix.py  —  Hebrew feature matrix builder
=======================================================
Merges the three existing morphosyntactic feature CSV files into a single
unified feature matrix, attaches date / register / holdout metadata from
the corpus manifest, and writes hebrew/data/feature_matrix.csv.

Input files (all relative to the project root, one level above this script)
---------------------------------------------------------------------------
  theoretical_features_training.csv  — 22 texts, 44 features
        Source: 08_theoretical_features.py (ETCBC/BHSA extraction)
        Coverage: function words, verb forms, stem distribution, particles

  morpho_training_rates.csv          — 22 texts, 12 Tier-1/2 features
        Source: 10_morphosyntactic_dating.py
        Coverage: frac_anachnu, rate_pen, rate_terem, rate_na,
                  frac_halak_piel, frac_ysf_qal, frac_niphal,
                  frac_yld_nif, frac_zaqaq, rate_ut_nouns

  tier3_training_rates.csv           — 22 texts, 10 clause-level features
        Source: 12_tier3_clause_features.py
        Coverage: frac_nmcl (nominal clause fraction), frac_fronted
                  (fronted constituent fraction), frac_sv (S-before-V),
                  frac_null_subj, frac_wqtl_wayq, frac_ptcp_cl, etc.

  source_feature_profiles.csv        — D, P, JE documentary sources
        Source: 19_torah_source_analysis.py
        Coverage: subset of theoretical_features columns

Output
------
  hebrew/data/feature_matrix.csv
    One row per text (training + test targets).
    Columns: id, date_bce, date_sigma, register, genre, holdout,
             + all merged features (NaN where a text lacks coverage).

Feature naming
--------------
All columns from the three CSVs are kept as-is.  Where the same feature
appears in two files (e.g. frac_ani, frac_she both appear in theoretical
and morpho CSVs), the theoretical_features version is preferred (it was
extracted from the full BHSA parse; the morpho version is an earlier
pass that rounds differently).

Archaizing-relevant features included
--------------------------------------
  Archaic markers (expected HIGH in SBH, LOW in LBH):
    rate_wayyiqtol, rate_anochi, rate_inf_abs, rate_pen, rate_terem,
    rate_na, frac_wqtl_wayq (wayyiqtol / (wayyiqtol + weqatal))

  Late leakage markers (even small values in an "archaic-style" text
  are diagnostic of late era):
    frac_she, rate_ut_nouns, frac_niphal, frac_yld_nif

  Neutral / register-independent (used in classifier but not archaism
  index):
    rate_asher, rate_ki, rate_qatal, rate_yiqtol, frac_sv, frac_fronted
"""

import json
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed.  Run: pip install pandas --break-system-packages")

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUTDIR  = os.path.join(HERE, "data")
os.makedirs(OUTDIR, exist_ok=True)
OUT_CSV = os.path.join(OUTDIR, "feature_matrix.csv")

# Input file paths
F_THEOR  = os.path.join(ROOT, "theoretical_features_training.csv")
F_MORPHO = os.path.join(ROOT, "morpho_training_rates.csv")
F_TIER3  = os.path.join(ROOT, "tier3_training_rates.csv")
F_SOURCE = os.path.join(ROOT, "source_feature_profiles.csv")
F_MANIF  = os.path.join(HERE, "corpus_manifest.json")


def load_manifest():
    with open(F_MANIF, encoding="utf-8") as f:
        m = json.load(f)
    # Build lookup: id → {date_bce, date_sigma, register, genre, holdout}
    meta = {}
    for e in m["training"]:
        meta[e["id"]] = dict(
            date_bce   = e["date_bce"],
            date_sigma = e["date_sigma"],
            register   = e["register"],
            genre      = e["genre"],
            holdout    = e["holdout"],
        )
    for t in m["test_targets"]:
        meta[t["id"]] = dict(
            date_bce   = t["prior_bce"],   # used as prior mean
            date_sigma = t["prior_sigma"],
            register   = "unknown",
            genre      = "narrative",
            holdout    = False,
        )
    return meta


def load_features():
    """Load and merge the three feature CSVs.  Returns wide DataFrame."""
    # ── theoretical features (primary source) ────────────────────────────
    tf = pd.read_csv(F_THEOR)
    tf = tf.rename(columns={"unit": "id"})
    tf = tf.set_index("id")

    # ── morpho rates (tier-1/2 extras not in theoretical) ────────────────
    mr = pd.read_csv(F_MORPHO)
    mr = mr.rename(columns={"Unnamed: 0": "id"})
    mr = mr.set_index("id")
    # Drop columns already in tf (prefer tf values); keep unique ones
    mr_unique = [c for c in mr.columns if c not in tf.columns]
    mr = mr[mr_unique]

    # ── tier-3 clause features ────────────────────────────────────────────
    t3 = pd.read_csv(F_TIER3)
    t3 = t3.rename(columns={"Unnamed: 0": "id"})
    t3 = t3.set_index("id")
    t3_unique = [c for c in t3.columns if c not in tf.columns and c not in mr.columns]
    t3 = t3[t3_unique]

    # ── merge training features ───────────────────────────────────────────
    train_feats = tf.join(mr, how="outer").join(t3, how="outer")

    # ── source feature profiles (D, P, JE) ───────────────────────────────
    sp = pd.read_csv(F_SOURCE)
    sp = sp.rename(columns={"source": "id"})
    # source_feature_profiles uses short IDs ("D", "P", "JE");
    # manifest uses long IDs ("D_source", "P_source", "JE_source")
    sp["id"] = sp["id"].replace({"D": "D_source", "P": "P_source", "JE": "JE_source"})
    sp = sp.set_index("id")
    # Only keep columns that are in the training matrix (subset coverage)
    shared = [c for c in sp.columns if c in train_feats.columns]
    sp = sp[shared]

    # Combine: training rows + test rows (NaN for features not in sp)
    combined = pd.concat([train_feats, sp], axis=0, sort=False)
    return combined


def main():
    meta = load_manifest()
    feats = load_features()

    # Only keep rows present in the manifest
    rows = []
    for eid, m in meta.items():
        if eid in feats.index:
            row = feats.loc[eid].copy()
        else:
            print(f"  WARNING: {eid} not found in any feature CSV — row will be all-NaN")
            row = pd.Series(dtype=float, name=eid)

        row_dict = {"id": eid}
        row_dict.update(m)                # date_bce, date_sigma, register, genre, holdout
        row_dict.update(row.to_dict())    # features
        rows.append(row_dict)

    df = pd.DataFrame(rows).set_index("id")

    # Reorder: meta columns first
    meta_cols = ["date_bce", "date_sigma", "register", "genre", "holdout"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    df.to_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(f"  Shape: {df.shape}  ({len(df)} texts × {len(feat_cols)} features)")
    print()

    # Coverage report
    print("Feature coverage per text:")
    fmt = "  {:<20s}  {:>4s}  {:>8s}  {:>12s}  {:>8s}  missing_feats={}"
    print(fmt.format("unit", "date", "register", "genre", "holdout", ""))
    for eid, row in df.iterrows():
        n_miss = row[feat_cols].isna().sum()
        flag   = " [HOLDOUT]" if row["holdout"] else ""
        print(f"  {eid:<22s}  {int(row['date_bce']):>4d} BCE  "
              f"{row['register']:<12s}  {row['genre']:<10s}  "
              f"n_missing={n_miss:>2d}{flag}")

    # Feature completeness summary
    print()
    print("Feature completeness (fraction non-NaN across all 22 training texts):")
    train_mask = df["register"] != "unknown"
    for col in feat_cols:
        frac = df.loc[train_mask, col].notna().mean()
        if frac < 1.0:
            print(f"  {col:<30s}  {frac:.0%} non-NaN")

    print()
    n_complete = sum(1 for c in feat_cols if df.loc[train_mask, c].notna().mean() == 1.0)
    print(f"  {n_complete}/{len(feat_cols)} features are 100% complete across training texts.")


if __name__ == "__main__":
    main()

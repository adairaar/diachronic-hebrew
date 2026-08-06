"""
02_register_classifier.py  —  Hebrew register classifier + archaizing detector
===============================================================================
Trains a three-class register classifier (SBH / Transitional / LBH) on the
22 dated training texts, then applies it to the D, P, and JE documentary
sources.

The key output beyond the register probabilities is the ARCHAIZING INDEX:
a mixing-incoherence score that flags texts whose stylistic profile looks
archaic but shows detectable late-era leakage markers.

Theory of archaizing detection
-------------------------------
An author writing in, say, the 4th century BCE who deliberately imitates
pre-exilic Hebrew (SBH) will load their text with archaic features —
high wayyiqtol, preservation of אנכי, absence of שׁ relative, frequent
פן/טרם particles.  But they cannot fully suppress the language of their
actual era, so late markers "leak through":

  • שׁ relative appears (even 1–2% of relative clauses is significant,
    since genuine SBH texts have zero שׁ)
  • אני becomes the exclusive 1sg pronoun; אנכי disappears or is only
    used in stylised set-phrases ("I am YHWH")
  • Abstract -ūt nouns and niphal passivisation increase slightly above
    SBH norms
  • Word-order shifts (higher S-before-V fraction) appear

The archaizing index is a product of two z-score composites:

  archaic_z    = mean(Z[wayyiqtol], Z[anochi], Z[inf_abs], Z[pen], Z[terem],
                      Z[na], Z[wqtl_wayq], -Z[she], -Z[frac_ani])
                 (positive = more archaic than corpus mean)

  leakage_z    = mean(Z[frac_she], Z[rate_niphal], Z[frac_az],
                      Z[rate_ut_nouns], Z[frac_ani])
                 (positive = more late-era leakage than corpus mean)

  archaizing_index = archaic_z × leakage_z    (large positive = suspicious)

A genuine SBH text: archaic_z >> 0, leakage_z << 0  → product near 0 or negative
A genuine LBH text: archaic_z << 0, leakage_z >> 0  → product negative
An archaizing text: archaic_z >> 0, leakage_z > 0   → product large positive

Classifier
----------
Random Forest with out-of-bag estimate, trained on the 19 non-holdout
training texts.  Uses only features with ≥ 80% non-NaN coverage to avoid
imputing too many values.  NaN cells are filled with the training-set
column median before fitting.

For classification of D/P/JE sources, only the 27 features available in
source_feature_profiles.csv are used (a sub-forest vote on those features).

Outputs
-------
  hebrew/results/register_probs.json   — {id: {p_SBH, p_Transitional, p_LBH,
                                               archaic_z, leakage_z,
                                               archaizing_index,
                                               archaizing_flag}}
  hebrew/results/register_report.txt  — human-readable classification report
  hebrew/results/register_probs.csv   — same as JSON but CSV
"""

import json
import os
import sys
import warnings

import numpy as np

try:
    import pandas as pd
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Run: pip install pandas scipy scikit-learn --break-system-packages")

HERE    = os.path.dirname(os.path.abspath(__file__))
FEAT    = os.path.join(HERE, "data", "feature_matrix.csv")
MANIF   = os.path.join(HERE, "corpus_manifest.json")
RESDIR  = os.path.join(HERE, "results")
os.makedirs(RESDIR, exist_ok=True)

# Minimum fraction of non-NaN values to include a feature in the classifier
MIN_COVERAGE = 0.80

# Z-score components for the archaizing index
# (feature, direction): +1 means "high = archaic", -1 means "high = late"
ARCHAIC_FEATS = [
    ("rate_wayyiqtol", +1),   # wayyiqtol rate: high in SBH
    ("rate_anochi",    +1),   # archaic first-person pronoun
    ("rate_inf_abs",   +1),   # infinitive absolute: archaic construction
    ("rate_pen",       +1),   # פן "lest": archaic particle, absent in LBH
    ("rate_terem",     +1),   # טרם "not yet": archaic, disappears in LBH
    ("rate_na",        +1),   # נא modal particle: archaic, restricted in LBH
    ("frac_wqtl_wayq", +1),   # wayyiqtol/(wayq+weqatal): higher in SBH narrative
    ("frac_she",       -1),   # שׁ relative: late marker; high = NOT archaic
    ("frac_ani",       -1),   # אני fraction: higher = more LBH; reverse for archaic
]

LEAKAGE_FEATS = [
    ("frac_she",       +1),   # שׁ relative: even tiny values signal late era
    ("rate_niphal",    +1),   # niphal passivisation: increases in LBH
    ("rate_az",        +1),   # אז "then": more frequent in LBH
    ("rate_ut_nouns",  +1),   # -ūt abstract nouns: increase in LBH
    ("frac_ani",       +1),   # exclusive אני (no אנכי): LBH marker
]

META_COLS = {"date_bce", "date_sigma", "register", "genre", "holdout"}

ARCHAIZING_THRESHOLD = 0.5   # archaizing_index above this → flag as suspicious


def z_composite(df: pd.DataFrame, feat_directions: list,
                mean_vec: pd.Series, std_vec: pd.Series) -> np.ndarray:
    """
    Compute a signed composite z-score for a list of (feature, direction) pairs.
    Features not in df or with zero std are skipped.
    Returns array of shape (n_rows,).
    """
    scores = []
    for feat, direction in feat_directions:
        if feat not in df.columns:
            continue
        vals = df[feat].values.astype(float)
        mu   = mean_vec.get(feat, np.nan)
        sd   = std_vec.get(feat, np.nan)
        if np.isnan(mu) or np.isnan(sd) or sd < 1e-10:
            continue
        z = (vals - mu) / sd
        scores.append(direction * z)
    if not scores:
        return np.zeros(len(df))
    return np.nanmean(np.vstack(scores), axis=0)


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(FEAT, index_col="id")

    with open(MANIF, encoding="utf-8") as f:
        manifest = json.load(f)

    # Separate training (known register) from test targets
    train_mask = df["register"] != "unknown"
    df_train   = df[train_mask].copy()
    df_test    = df[~train_mask].copy()

    feat_cols  = [c for c in df.columns if c not in META_COLS]

    # ── Feature selection: ≥ 80% coverage on training set ─────────────────
    coverage   = df_train[feat_cols].notna().mean()
    good_feats = coverage[coverage >= MIN_COVERAGE].index.tolist()
    print(f"Features with ≥{MIN_COVERAGE:.0%} coverage: {len(good_feats)}/{len(feat_cols)}")

    # Features available for test targets (D/P/JE)
    test_avail = df_test[good_feats].notna().any(axis=0)
    clf_feats  = [f for f in good_feats if test_avail[f]]
    print(f"Features usable for D/P/JE classification: {len(clf_feats)}")
    print()

    # ── Training statistics (for z-score normalisation) ───────────────────
    # Use non-holdout training texts only for stats + classifier
    non_holdout = df_train[~df_train["holdout"]]
    feat_mean   = non_holdout[good_feats].mean()
    feat_std    = non_holdout[good_feats].std()

    # ── Archaizing index for training texts ───────────────────────────────
    archaic_z_train  = z_composite(df_train, ARCHAIC_FEATS, feat_mean, feat_std)
    leakage_z_train  = z_composite(df_train, LEAKAGE_FEATS, feat_mean, feat_std)
    arch_idx_train   = archaic_z_train * leakage_z_train

    # ── Random Forest classifier ───────────────────────────────────────────
    X_train   = df_train[clf_feats].copy()
    y_train   = df_train["register"].values
    holdout_m = df_train["holdout"].values

    # Impute NaN with median (training only)
    imputer   = SimpleImputer(strategy="median")
    X_imp     = imputer.fit_transform(X_train)

    # Train on non-holdout texts
    X_fit     = X_imp[~holdout_m]
    y_fit     = y_train[~holdout_m]

    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        oob_score=True,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_fit, y_fit)
    print(f"RF OOB accuracy: {clf.oob_score_:.3f}  (n={len(X_fit)} training texts)")
    print()

    # Predict probabilities for all training texts
    proba_train  = clf.predict_proba(X_imp)
    classes      = list(clf.classes_)

    # ── Classify test targets (D/P/JE) ────────────────────────────────────
    # Use only the clf_feats subset; impute with training medians
    X_test  = df_test[clf_feats].copy()
    X_test_imp = imputer.transform(X_test)
    proba_test  = clf.predict_proba(X_test_imp)

    archaic_z_test  = z_composite(df_test, ARCHAIC_FEATS, feat_mean, feat_std)
    leakage_z_test  = z_composite(df_test, LEAKAGE_FEATS, feat_mean, feat_std)
    arch_idx_test   = archaic_z_test * leakage_z_test

    # ── Holdout accuracy ──────────────────────────────────────────────────
    print("HOLDOUT VALIDATION")
    print("=" * 60)
    holdout_ids = df_train[df_train["holdout"]].index.tolist()
    for i, eid in enumerate(df_train.index):
        if not df_train.loc[eid, "holdout"]:
            continue
        true_reg = df_train.loc[eid, "register"]
        probs    = {c: proba_train[i, j] for j, c in enumerate(classes)}
        pred     = max(probs, key=probs.get)
        tick     = "✓" if pred == true_reg else "✗"
        print(f"  {tick} {eid:20s}  true={true_reg:12s}  pred={pred:12s}  "
              f"P(SBH)={probs.get('SBH',0):.2f}  "
              f"P(Trans)={probs.get('Transitional',0):.2f}  "
              f"P(LBH)={probs.get('LBH',0):.2f}")
    print()

    # ── Full register report ──────────────────────────────────────────────
    print("REGISTER CLASSIFICATION — TRAINING CORPUS")
    print("=" * 70)
    lines = []
    results = {}

    # Training texts
    for i, eid in enumerate(df_train.index):
        probs = {c: proba_train[i, j] for j, c in enumerate(classes)}
        pred  = max(probs, key=probs.get)
        true  = df_train.loc[eid, "register"]
        az    = float(archaic_z_train[i])
        lz    = float(leakage_z_train[i])
        ai    = float(arch_idx_train[i])
        flag  = ai > ARCHAIZING_THRESHOLD
        h_flag = " [HOLDOUT]" if df_train.loc[eid, "holdout"] else ""
        a_flag = " ← ARCHAIZING?" if flag else ""
        lines.append(
            f"  {eid:20s}  {true:12s} → {pred:12s}  "
            f"P(SBH)={probs.get('SBH',0):.2f}  "
            f"P(T)={probs.get('Transitional',0):.2f}  "
            f"P(LBH)={probs.get('LBH',0):.2f}  "
            f"arch_z={az:+.2f}  leak_z={lz:+.2f}  "
            f"arch_idx={ai:+.2f}{h_flag}{a_flag}"
        )
        results[eid] = dict(
            p_SBH=probs.get("SBH", 0), p_Transitional=probs.get("Transitional", 0),
            p_LBH=probs.get("LBH", 0), predicted=pred, true=true,
            archaic_z=az, leakage_z=lz, archaizing_index=ai, archaizing_flag=flag,
            holdout=bool(df_train.loc[eid, "holdout"]),
        )

    for l in lines:
        print(l)

    print()
    print("REGISTER CLASSIFICATION — TEST TARGETS (D / P / JE)")
    print("=" * 70)
    for i, eid in enumerate(df_test.index):
        probs = {c: proba_test[i, j] for j, c in enumerate(classes)}
        pred  = max(probs, key=probs.get)
        az    = float(archaic_z_test[i])
        lz    = float(leakage_z_test[i])
        ai    = float(arch_idx_test[i])
        flag  = ai > ARCHAIZING_THRESHOLD
        a_flag = " ← ARCHAIZING?" if flag else ""
        print(
            f"  {eid:20s}  prior={df_test.loc[eid,'date_bce']}±{df_test.loc[eid,'date_sigma']} BCE  "
            f"P(SBH)={probs.get('SBH',0):.2f}  "
            f"P(T)={probs.get('Transitional',0):.2f}  "
            f"P(LBH)={probs.get('LBH',0):.2f}  "
            f"→ {pred}  "
            f"arch_z={az:+.2f}  leak_z={lz:+.2f}  "
            f"arch_idx={ai:+.2f}{a_flag}"
        )
        results[eid] = dict(
            p_SBH=probs.get("SBH", 0), p_Transitional=probs.get("Transitional", 0),
            p_LBH=probs.get("LBH", 0), predicted=pred, true="unknown",
            archaic_z=az, leakage_z=lz, archaizing_index=ai, archaizing_flag=flag,
            holdout=False,
        )

    # ── Archaizing interpretation ─────────────────────────────────────────
    print()
    print("ARCHAIZING DIAGNOSTIC")
    print("=" * 70)
    print(f"  Threshold: archaizing_index > {ARCHAIZING_THRESHOLD}")
    print()
    print("  archaic_z > 0  AND  leakage_z > 0  →  mixing incoherence")
    print("  Quadrant summary:")
    print(f"  {'Text':20s}  {'arch_z':>8s}  {'leak_z':>8s}  {'arch_idx':>9s}  Quadrant")
    print("  " + "─" * 65)
    all_ids   = list(df_train.index) + list(df_test.index)
    all_az    = list(archaic_z_train) + list(archaic_z_test)
    all_lz    = list(leakage_z_train) + list(leakage_z_test)
    all_ai    = list(arch_idx_train)  + list(arch_idx_test)
    all_regs  = list(df_train["register"]) + list(df_test["register"])

    for eid, az, lz, ai, reg in sorted(
            zip(all_ids, all_az, all_lz, all_ai, all_regs),
            key=lambda x: -x[3]):
        quad = ("SBH-genuine" if az > 0 and lz < 0 else
                "ARCHAIZING"  if az > 0 and lz > 0 else
                "LBH-genuine" if az < 0 and lz > 0 else
                "neutral")
        flag = " ◀" if ai > ARCHAIZING_THRESHOLD else ""
        print(f"  {eid:20s}  {az:>+8.2f}  {lz:>+8.2f}  {ai:>+9.2f}  {quad}{flag}")

    # ── Feature importance ────────────────────────────────────────────────
    print()
    print("TOP FEATURES (Random Forest importance)")
    print("=" * 50)
    imp = pd.Series(clf.feature_importances_, index=clf_feats).sort_values(ascending=False)
    for feat, val in imp.head(15).items():
        print(f"  {feat:30s}  {val:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────
    json_path = os.path.join(RESDIR, "register_probs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved register probabilities → {json_path}")

    csv_path = os.path.join(RESDIR, "register_probs.csv")
    pd.DataFrame(results).T.to_csv(csv_path)
    print(f"Saved register probabilities → {csv_path}")


if __name__ == "__main__":
    main()

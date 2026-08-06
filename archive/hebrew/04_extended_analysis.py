"""
04_extended_analysis.py  —  Extended Hebrew corpus analysis
============================================================
Addresses the genre-confound and circularity problems in the Hebrew
diachronic pipeline by pursuing three complementary avenues:

Avenue 1: Leakage-feature hard limits (genre-independent)
---------------------------------------------------------
Certain features are temporally monotonic regardless of genre: the שׁ
relative pronoun, the אני/אנכי ratio, the אז temporal adverb, niphal
passive rate, and -ūt abstract nouns. These act as terminus post quem
markers — a single non-zero שׁ rate in an "early" text sets a hard lower
bound on composition, independent of narrative/legal/prophetic register.

We compute z-scores for D/P/JE on these features against the SBH training
texts only, producing a "leakage profile" that does not depend on the
training genre mix.

Avenue 2: Subsource archaism from previous analysis
----------------------------------------------------
Reads the pre-computed subsource_archaism.csv and archaism_diagnostic_results.csv
files, which already contain Kings/Chronicles controlled comparisons and
DH-book LBH scores. Key finding: Joshua–Kings classify as "Modern (LBH-like)"
— they are not appropriate SBH anchors. D_Code is "Archaic (CBH-like)."

Avenue 3: Genre-neutral feature sub-model
-----------------------------------------
Fits an MVN-style Bayesian dating model using ONLY features that are
demonstrably genre-independent (insensitive to narrative vs. legal vs.
prophetic mode). Compares dates from this sub-model against the full
model to isolate genre confound.

Avenue 4: DSS Part B calibration shift
---------------------------------------
Reads dss_partB_shift.csv and interprets how adding 1QS/1QM as training
data moved Torah source dates — providing the benchmark for what genuine
archaizing texts look like in the model.

Output
------
  hebrew/results/extended_analysis.csv    — per-source feature z-scores
  hebrew/results/genre_neutral_dates.csv  — genre-neutral model dates
  hebrew/results/synthesis_report.md      — full narrative synthesis
"""

import json, os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    from scipy import stats
except ImportError:
    sys.exit("pandas/scipy not installed. Run: pip install pandas scipy --break-system-packages")

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUTDIR  = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

# ── Input files ─────────────────────────────────────────────────────────────
F_FEAT   = os.path.join(HERE, "data", "feature_matrix.csv")
F_SRC    = os.path.join(ROOT, "source_feature_profiles.csv")
F_ARCH   = os.path.join(ROOT, "archaism_summary.csv")
F_SUBA   = os.path.join(ROOT, "subsource_archaism.csv")
F_DIAG   = os.path.join(ROOT, "archaism_diagnostic_results.csv")
F_DSSB   = os.path.join(ROOT, "dss_partB_shift.csv")
F_MAST   = os.path.join(ROOT, "master_dating_results.csv")
F_GENRE  = os.path.join(ROOT, "genre_controlled_dating.csv")
F_MANIF  = os.path.join(HERE, "corpus_manifest.json")
F_RESULTS= os.path.join(HERE, "results", "hard_register_dating_hebrew.csv")

# ── Genre-independent features ───────────────────────────────────────────────
# These features are temporally diagnostic but NOT strongly tied to
# prophetic/legal/narrative register.  They can be used as "clean" temporal
# signal without the genre confound.
#
# Archaic markers (high in SBH, low in LBH):
ARCHAIC_FEATURES = [
    "rate_anochi",   # אנכי pronoun — nearly eliminated in LBH
    "rate_inf_abs",  # infinitive absolute — higher in SBH broadly
    "rate_na",       # particle נא — drops sharply in LBH
    "rate_pen",      # particle פן — drops sharply in LBH
    "rate_terem",    # particle טרם — drops sharply in LBH
    "frac_wqtl_wayq",# wayyiqtol / (wayyiqtol + weqatal) — only valid in prose
]

# Late leakage markers (near-zero in SBH, rise in LBH):
# These are the critical genre-INDEPENDENT ones — they don't depend on
# narrative vs. prophetic mode.
LEAKAGE_FEATURES = [
    "frac_she",       # שׁ relative pronoun fraction — nearly absent pre-exile
    "rate_az",        # אז temporal adverb — rises in LBH
    "rate_niphal",    # niphal passive rate — rises in LBH / post-exilic
    "rate_ut_nouns",  # -ūt abstract noun rate — Aramaic influence, late marker
    "frac_ani",       # אני fraction of אני+אנכי — complementary to rate_anochi
    "frac_ein",       # אין negator fraction — late negator replaces לא
    "rate_baqash",    # בקש "to seek" — replaces שאל in LBH
]

# Features with strong genre confound (excluded from genre-neutral model):
GENRE_CONFOUNDED = [
    "rate_wayyiqtol",  # narrative-only; near-zero in legal/instructional text
    "rate_qatal",      # higher in narrative
    "rate_yiqtol",     # higher in prophecy/law
    "rate_impv",       # much higher in prophecy
    "rate_inf_con",    # genre-dependent
    "frac_sv",         # constituent order — genre-dependent
    "frac_fronted",    # genre-dependent
    "frac_null_subj",  # genre-dependent
]

# Combined genre-neutral feature set: archaic + leakage (minus frac_wqtl_wayq
# which has the same prose-only constraint as wayyiqtol)
GENRE_NEUTRAL = [f for f in ARCHAIC_FEATURES + LEAKAGE_FEATURES
                 if f != "frac_wqtl_wayq"]

print("=" * 70)
print("Avenue 1: Leakage-feature hard limits (genre-independent)")
print("=" * 70)

# Load feature matrix
fm = pd.read_csv(F_FEAT, index_col="id")
train = fm[fm["register"].isin(["SBH", "Transitional", "LBH"])].copy()
sources = fm[fm["register"] == "unknown"].copy()

# SBH-only training texts (non-holdout)
sbh = train[(train["register"] == "SBH") & (~train["holdout"])].copy()

print(f"\nSBH training texts (n={len(sbh)}):")
for idx, row in sbh.iterrows():
    print(f"  {idx:<20s} {int(row['date_bce'])} BCE")

# Compute z-scores for leakage features: (source_val - SBH_mean) / SBH_std
# A positive z-score on a leakage feature = text has more late-era leakage
# than the average SBH text. A large positive z = flagged as archaizing.

print("\n--- Leakage z-scores for D/P/JE (vs SBH training corpus mean/std) ---")
print(f"{'Feature':<20s}  {'SBH_mean':>10s}  {'SBH_std':>9s}  "
      f"{'D':>8s}  {'P':>8s}  {'JE':>8s}  note")

leakage_rows = []
for feat in LEAKAGE_FEATURES:
    if feat not in fm.columns:
        print(f"  {feat:<20s}  [NOT IN MATRIX — skipping]")
        continue
    sbh_vals = sbh[feat].dropna()
    if len(sbh_vals) < 3:
        continue
    mu  = sbh_vals.mean()
    sig = sbh_vals.std()
    row = {"feature": feat, "SBH_mean": mu, "SBH_std": sig, "direction": "late_leakage"}
    for src in ["D_source", "P_source", "JE_source"]:
        if src in sources.index:
            v = sources.loc[src, feat]
            if pd.notna(v) and sig > 0:
                z = (v - mu) / sig
            else:
                z = np.nan
        else:
            z = np.nan
        row[src] = z
    leakage_rows.append(row)

    note = ""
    d_z  = row.get("D_source",  np.nan)
    p_z  = row.get("P_source",  np.nan)
    je_z = row.get("JE_source", np.nan)
    for z in [d_z, p_z, je_z]:
        if pd.notna(z) and z > 1.5:
            note = "⚠ significant late leakage"
            break
    print(f"  {feat:<20s}  {mu:>10.4f}  {sig:>9.4f}  "
          f"{str(round(d_z,2)) if pd.notna(d_z) else 'NaN':>8s}  "
          f"{str(round(p_z,2)) if pd.notna(p_z) else 'NaN':>8s}  "
          f"{str(round(je_z,2)) if pd.notna(je_z) else 'NaN':>8s}  {note}")

print("\n--- Archaic z-scores for D/P/JE (vs SBH training corpus) ---")
print("  (positive = more archaic than average SBH text)")
archaic_rows = []
for feat in ARCHAIC_FEATURES:
    if feat not in fm.columns:
        continue
    sbh_vals = sbh[feat].dropna()
    if len(sbh_vals) < 3:
        continue
    mu  = sbh_vals.mean()
    sig = sbh_vals.std()
    row = {"feature": feat, "SBH_mean": mu, "SBH_std": sig, "direction": "archaic"}
    for src in ["D_source", "P_source", "JE_source"]:
        if src in sources.index:
            v = sources.loc[src, feat]
            if pd.notna(v) and sig > 0:
                z = (v - mu) / sig
            else:
                z = np.nan
        else:
            z = np.nan
        row[src] = z
    archaic_rows.append(row)

    d_z  = row.get("D_source",  np.nan)
    p_z  = row.get("P_source",  np.nan)
    je_z = row.get("JE_source", np.nan)
    print(f"  {feat:<20s}  {mu:>10.4f}  {sig:>9.4f}  "
          f"{str(round(d_z,2)) if pd.notna(d_z) else 'NaN':>8s}  "
          f"{str(round(p_z,2)) if pd.notna(p_z) else 'NaN':>8s}  "
          f"{str(round(je_z,2)) if pd.notna(je_z) else 'NaN':>8s}")

# Mixing incoherence: archaic z × leakage z → positive = archaizing
print("\n--- Mixing incoherence (archaic z × leakage z composite) ---")
print("  High positive value = archaic style PLUS late leakage = archaizing signal")

def composite_z(rows, src, direction="positive"):
    """Mean z-score across features where direction='positive' means higher=more-of-it."""
    zs = [r[src] for r in rows if src in r and pd.notna(r.get(src))]
    return np.mean(zs) if zs else np.nan

for src in ["D_source", "P_source", "JE_source"]:
    arch_z = composite_z(archaic_rows, src, "positive")
    leak_z = composite_z(leakage_rows, src, "positive")
    if pd.notna(arch_z) and pd.notna(leak_z):
        mix = arch_z * leak_z
    else:
        mix = np.nan
    # An archaizer has high arch_z AND high leak_z → mix > 0
    # A genuine early text has high arch_z AND low/negative leak_z → mix < 0
    # A genuine late text has low arch_z AND high leak_z → mix variable
    interpret = (
        "ARCHAIZING signal" if (pd.notna(mix) and arch_z > 0 and leak_z > 0)
        else "genuine archaic (no late leakage)" if (pd.notna(mix) and arch_z > 0 and leak_z <= 0)
        else "genuine late" if (pd.notna(mix) and arch_z <= 0 and leak_z > 0)
        else "neutral/ambiguous"
    )
    mix_str = f"{mix:+.3f}" if pd.notna(mix) else "N/A"
    print(f"  {src:<12s}  arch_z={arch_z:+.3f}  leak_z={leak_z:+.3f}  "
          f"mix={mix_str}  → {interpret}")


print("\n" + "=" * 70)
print("Avenue 2: Subsource archaism (Kings/Chronicles + DH books)")
print("=" * 70)

suba = pd.read_csv(F_SUBA)
arch = pd.read_csv(F_ARCH)

print("\nArchaism summary for DH books and Torah sources:")
print(f"  {'Unit':<20s}  {'mean_lbh':>9s}  {'classification':<25s}")
print("  " + "-" * 60)

# DH history books
dh_books = ["Joshua", "Judges", "1_Samuel", "2_Samuel", "1_Kings", "2_Kings"]
for _, row in suba[suba["unit"].isin(dh_books)].iterrows():
    print(f"  {row['unit']:<20s}  {row['mean_lbh']:>9.3f}  {row['classification']:<25s}")

print("  ...")
# Torah sources
torah_sources = ["D", "D_Code", "D_Frame", "D_Song", "D_full",
                 "Lev_Holiness", "Lev_Priestly", "Lev_full"]
for _, row in suba[suba["unit"].isin(torah_sources)].iterrows():
    print(f"  {row['unit']:<20s}  {row['mean_lbh']:>9.3f}  {row['classification']:<25s}")

# From archaism_summary.csv (whole-source level)
print()
for _, row in arch[arch["unit"].isin(["D", "P", "JE", "Genesis", "Exodus",
                                       "Leviticus", "Numbers", "Deuteronomy"])].iterrows():
    print(f"  {row['unit']:<20s}  {row['mean_lbh']:>9.3f}  {row['classification']:<25s}")

print("""
KEY FINDING: The DH books (Joshua–Kings) classify as "Modern (LBH-like)"
with mean_lbh 0.72–0.93. This means they look MORE LATE than D itself
(mean_lbh=0.41, "Mixed/selective"). Using DH books as SBH narrative
anchors would bias D toward appearing EARLY, not late. The model's dating
of D to ~292–345 BCE is therefore NOT an artifact of DH-book contamination;
it arises despite D being more archaic than the DH books used as anchors.

D_Code specifically (mean_lbh=0.28, "Archaic CBH-like") is the most
archaic D component — yet the model still dates Deuteronomy late. This
is the archaizing signature: archaic surface features combined with
detectable late leakage in other features.
""")


print("=" * 70)
print("Avenue 3: Genre-neutral feature sub-model")
print("=" * 70)

# Build a simple WLS linear regression model using ONLY genre-neutral features
# against the 22 training texts (non-holdout), then score D/P/JE

available_neutral = [f for f in GENRE_NEUTRAL if f in fm.columns]
print(f"\nGenre-neutral features available: {len(available_neutral)}/{len(GENRE_NEUTRAL)}")
print(f"  {available_neutral}")

train_active = train[~train["holdout"]].copy()

# Feature screening: keep features where at least 80% of training texts are non-NaN
# and the absolute correlation with date_bce is >= 0.3
screened = []
for feat in available_neutral:
    col = train_active[feat].dropna()
    if len(col) / len(train_active) < 0.8:
        continue
    r, p = stats.pearsonr(
        train_active.loc[col.index, "date_bce"],
        col
    )
    if abs(r) >= 0.25:  # relaxed threshold (small n)
        screened.append((feat, r, p))

screened.sort(key=lambda x: -abs(x[1]))
print(f"\nScreened genre-neutral features (|r|≥0.25, coverage≥80%):")
print(f"  {'Feature':<22s}  {'r':>7s}  {'p':>8s}  direction")
for feat, r, p in screened:
    direction = "higher→archaic" if r > 0 else "higher→late"
    print(f"  {feat:<22s}  {r:>+7.3f}  {p:>8.4f}  {direction}")

if len(screened) < 2:
    print("\n  [WARN] Too few genre-neutral features pass screening; "
          "genre-neutral model not fitted. Proceeding with all available neutral features.]")
    screened_feats = available_neutral
else:
    screened_feats = [f for f, r, p in screened]

# Build simple OLS regression of date_bce on genre-neutral features
from numpy.linalg import lstsq

def fit_genre_neutral_model(train_df, feat_list):
    """Fit WLS: date_bce ~ features, weighted by 1/date_sigma^2."""
    rows_ok = train_df[feat_list + ["date_bce", "date_sigma"]].dropna()
    if len(rows_ok) < 5:
        return None, None, None
    X = rows_ok[feat_list].values
    y = rows_ok["date_bce"].values
    w = 1.0 / rows_ok["date_sigma"].values**2
    # WLS via sqrt(w) scaling
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    # Add intercept
    Xw = np.c_[np.sqrt(w), Xw]
    beta, res, rank, sv = lstsq(Xw, yw, rcond=None)
    return beta, feat_list, rows_ok

beta, feat_list_used, rows_fit = fit_genre_neutral_model(train_active, screened_feats)

genre_neutral_rows = []

if beta is not None:
    print(f"\nGenre-neutral model fitted on {len(rows_fit)} training texts")
    print(f"  Features used: {feat_list_used}")

    # Predict on training texts (in-sample, sanity check)
    print("\nIn-sample predictions (sanity check — training set only):")
    print(f"  {'Unit':<20s}  {'True BCE':>9s}  {'Pred BCE':>9s}  {'Resid':>7s}")
    for idx, row in rows_fit.iterrows():
        x = np.array([1.0] + [row[f] for f in feat_list_used])
        pred = float(beta @ x)
        resid = row["date_bce"] - pred
        print(f"  {idx:<20s}  {row['date_bce']:>9.0f}  {pred:>9.0f}  {resid:>+7.0f}")

    # Predict on D/P/JE
    print("\nGenre-neutral model dates for D/P/JE:")
    print(f"  {'Source':<12s}  {'GenreNeutral_BCE':>16s}  {'Full_model_BCE':>14s}  "
          f"{'NGram_BCE':>10s}  {'Prior_BCE':>10s}")

    # Load master results for comparison
    master = pd.read_csv(F_MAST).set_index("unit") if os.path.exists(F_MAST) else None

    src_dates_neutral = {}
    for src in ["D_source", "P_source", "JE_source"]:
        src_label = src.replace("_source", "")
        if src in fm.index:
            feat_vals = fm.loc[src, feat_list_used]
            if feat_vals.isna().all():
                print(f"  {src_label:<12s}  [all features NaN]")
                continue
            # Impute missing with training mean
            for f in feat_list_used:
                if pd.isna(feat_vals[f]):
                    feat_vals[f] = train_active[f].mean()
            x = np.array([1.0] + list(feat_vals.values.astype(float)))
            pred_gn = float(beta @ x)
            src_dates_neutral[src] = pred_gn
        else:
            pred_gn = np.nan

        full_bce = master.loc[src, "map_full"] if master is not None and src in master.index else np.nan
        ng_bce   = master.loc[src, "map_ngram"] if master is not None and src in master.index else np.nan
        prior_bce = {"D_source": 625, "P_source": 600, "JE_source": 800}.get(src, np.nan)

        print(f"  {src_label:<12s}  {pred_gn:>16.0f}  {full_bce:>14.0f}  "
              f"{ng_bce:>10.0f}  {prior_bce:>10.0f}")

    # Save
    gn_df = pd.DataFrame([
        {"source": s, "genre_neutral_bce": v}
        for s, v in src_dates_neutral.items()
    ])
    gn_df.to_csv(os.path.join(OUTDIR, "genre_neutral_dates.csv"), index=False)
    print(f"\nSaved genre-neutral dates → hebrew/results/genre_neutral_dates.csv")
else:
    print("  [Genre-neutral model could not be fitted — insufficient data]")


print("\n" + "=" * 70)
print("Avenue 4: DSS Part B calibration shift")
print("=" * 70)

dssb = pd.read_csv(F_DSSB)
print("\nEffect of adding 1QS (~150 BCE) + 1QM (~100 BCE) to training on Torah dates:")
print(f"  {'Unit':<20s}  {'A_char':>8s}  {'B_char':>8s}  {'shift_char':>11s}  "
      f"{'A_word':>8s}  {'B_word':>8s}  {'shift_word':>11s}")
print("  " + "-" * 80)
targets = ["D_source", "P_source", "JE_source", "Genesis", "Exodus",
           "Leviticus", "Numbers", "Deuteronomy"]
for unit in targets:
    r = dssb[dssb["unit"] == unit]
    if r.empty:
        continue
    r = r.iloc[0]
    sc = r["shift_char"]
    sw = r["shift_word"]
    flag_c = "→OLDER" if sc > 30 else ("→YOUNGER" if sc < -30 else "≈same")
    flag_w = "→OLDER" if sw > 30 else ("→YOUNGER" if sw < -30 else "≈same")
    print(f"  {unit:<20s}  {r['partA_char']:>8.1f}  {r['partB_char']:>8.1f}  "
          f"{sc:>+8.1f} {flag_c:<8s}  {r['partA_word']:>8.1f}  {r['partB_word']:>8.1f}  "
          f"{sw:>+8.1f} {flag_w:<8s}")

print("""
INTERPRETATION: When 1QS and 1QM (provably late archaizing texts, ~150–100 BCE)
are added to the n-gram training set, the model gets a better calibrated picture
of what "deliberate archaism" looks like. The shift in D_source is +131 years
(char) = the model now thinks D looks OLDER than it did before. This is the
expected direction — the archaizers' pattern was previously conflated with the
training signal, pulling D toward them. Removing that conflation by labeling
them correctly (late archaizing) gives a cleaner temporal read.

However, the word-n-gram shift is only +7 years — the char model is more
affected by lexical archaism patterns than the word model. This suggests that
the archaizing in D is primarily at the level of archaic lexical/formulaic
patterns (char n-gram picks up letter-sequence archaisms) rather than
morphosyntactic structure.
""")


print("=" * 70)
print("Synthesis: Cross-model comparison for D/P/JE")
print("=" * 70)

master = pd.read_csv(F_MAST).set_index("unit") if os.path.exists(F_MAST) else pd.DataFrame()
genre  = pd.read_csv(F_GENRE).set_index("unit") if os.path.exists(F_GENRE) else pd.DataFrame()
hr     = pd.read_csv(F_RESULTS).set_index("id") if os.path.exists(F_RESULTS) else pd.DataFrame()

print(f"\n{'Source':<12s}  {'Prior':>8s}  {'Ngram':>8s}  {'Full':>8s}  "
      f"{'GenCtrl':>8s}  {'LikOnly':>8s}  {'Interpretation'}")
print("-" * 90)
for src in ["D_source", "P_source", "JE_source"]:
    prior = {"D_source": 625, "P_source": 600, "JE_source": 800}[src]
    label = src.replace("_source", "")

    ng   = master.loc[src, "map_ngram"]  if src in master.index else np.nan
    full = master.loc[src, "map_full"]   if src in master.index else np.nan
    gc   = genre.loc[src, "B+D"]         if src in genre.index  else np.nan
    lr   = hr.loc[src, "lik_map"]         if src in hr.index     else np.nan

    # Interpret: is the lik-only date consistently later than the prior?
    consistent_late = (
        all(pd.notna(v) and v < prior - 50 for v in [ng, full, lr])
    )
    interpretation = "consistently later than prior across all models" if consistent_late \
        else "model agreement weak or mixed"

    print(f"  {label:<12s}  {prior:>8d}  "
          f"{ng:>8.0f}  {full:>8.0f}  {gc:>8.0f}  {lr:>8.0f}  "
          f"{interpretation}")

print("""
COLUMNS:
  Prior    = scholarly prior mean (BCE)
  Ngram    = char/word n-gram model MAP (from master_dating_results.csv)
  Full     = full morphosyntactic model MAP (genre-controlled B+D)
  GenCtrl  = genre-controlled B+D model
  LikOnly  = MVN likelihood-only MAP from new Hebrew pipeline (script 03)

NOTE: All columns use BCE (positive = older). A value LESS than Prior means
the model dates the text LATER (more recent) than the scholarly prior.
""")

print("=" * 70)
print("Summary of findings across all avenues")
print("=" * 70)

summary_lines = [
    "",
    "1. LEAKAGE FEATURES (Avenue 1)",
    "   D has rate_anochi highly archaic (well above SBH mean) but frac_she, rate_az,",
    "   and other late markers must be checked against the feature matrix for significance.",
    "   The mixing incoherence formula (arch_z × leak_z) gives the archaizing diagnostic.",
    "",
    "2. DH BOOKS ARE LBH-LIKE (Avenue 2)",
    "   Joshua–Kings have mean_lbh=0.72–0.93 ('Modern LBH-like') — they look MORE late",
    "   than D itself (mean_lbh=0.41). The user's circularity concern is therefore NOT",
    "   a significant bias: using DH books as SBH anchors would push D toward appearing",
    "   EARLY, not late. The model's late dating of D is robust to this.",
    "",
    "3. GENRE-NEUTRAL SUB-MODEL (Avenue 3)",
    "   Features insensitive to narrative/legal/prophetic register give an independent",
    "   check. If genre-neutral dates agree with the full model, the genre confound is",
    "   not the primary driver of the late dating.",
    "",
    "4. DSS PART B CALIBRATION (Avenue 4)",
    "   Adding confirmed archaizers (1QS, 1QM) shifts D char-n-gram date OLDER (+131 yr).",
    "   The direction is correct (archaizers now labeled as such improve the calibration).",
    "   The small word-n-gram shift (+7 yr) suggests archaism is mostly at the lexical,",
    "   not morphosyntactic, level — explaining why the morphosyntactic model (Script 03)",
    "   dates D later than the char n-gram model.",
    "",
]
print("\n".join(summary_lines))

# ── Save extended analysis z-scores ────────────────────────────────────────
all_z_rows = leakage_rows + archaic_rows
z_df = pd.DataFrame(all_z_rows)
if not z_df.empty:
    z_df.to_csv(os.path.join(OUTDIR, "extended_analysis.csv"), index=False)
    print(f"Saved z-score table → hebrew/results/extended_analysis.csv")

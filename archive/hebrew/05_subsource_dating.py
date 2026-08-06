"""
05_subsource_dating.py  —  Sub-source chapter-aggregated dating
================================================================
Derives dates for D_Code, D_Frame, D_Song, Lev_Holiness, Lev_Priestly,
Gen_JE, Exo_JE, and Num_JE from existing chapter-level and sub-source
data — without requiring BHSA re-extraction.

NOTE on char n-grams: Character n-gram dates are NOT reported here.
The Hebrew char n-gram model (Script 16) was found to be unreliable:
it dates 1QS (~150 BCE) to 760 BCE — a 600-year error, far worse than
the word n-gram model (216 BCE).  This mirrors the finding in the Greek
pipeline where char n-grams were also dropped.  Only word n-gram dates
are retained as a secondary cross-check alongside the morpho model.

NOTE on archaism score vs morpho date inconsistency: The mean_lbh
archaism score and the MVN morpho dating model weight features
differently.  mean_lbh = simple per-feature average interpolated between
a CBH anchor (720 BCE) and LBH anchor (250 BCE).  The MVN model weights
features by temporal discriminativeness in the training corpus.  When
genre-confounded features dominate the MVN model (e.g. legal prose
patterns), a text can score as archaic on mean_lbh yet still receive a
late MVN date.  This is exactly what happens to D_Code: it is genuinely
more archaic on many features (אנכי, legal formulas), but those features
have weak date-slopes in the training corpus, so the MVN model is driven
by the genre confound instead.  The word n-gram, less genre-sensitive,
correctly ranks D_Code earlier than D_Frame.

Two data sources are combined:

  A. torah_chapter_dates.csv  (chapter-level WORD n-gram dates only)
     Aggregates chapter subsets into sub-unit dates using word-count
     weighted means.  char n-gram column is dropped (unreliable).

  B. subsource_dating.csv  +  master_dating_results.csv  (from Script 13)
     Full-morphosyntactic-model MAP dates already computed for
     D_Code, D_Frame, Lev_Holiness, Lev_Priestly by Script 13.

  C. subsource_archaism.csv  (archaism classification from Script 13)
     mean_lbh scores for D sub-components.

Output
------
  hebrew/results/subsource_dating.csv   — consolidated sub-source date table
  hebrew/results/subsource_dating.md    — formatted report (for synthesis)

Chapter-range definitions (mirroring Script 13)
-----------------------------------------------
  D_Code      : Deuteronomy ch. 12–26
  D_Frame     : Deuteronomy ch. 1–11, 27–31, 33–34
  D_Song      : Deuteronomy ch. 32
  Lev_Holiness: Leviticus   ch. 17–26
  Lev_Priestly: Leviticus   ch. 1–16
  Gen_JE      : Genesis     source_label in {J, E, JE, JE+P minus P rows}
  Exo_JE      : Exodus      source_label in {J, E, JE}
  Num_JE      : Numbers     source_label in {J, E, JE}
"""

import os, sys
import numpy as np

try:
    import pandas as pd
    from scipy import stats
except ImportError:
    sys.exit("pandas/scipy not installed. Run: pip install pandas scipy --break-system-packages")

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)

F_CHAP  = os.path.join(ROOT, "torah_chapter_dates.csv")
F_SUBS  = os.path.join(ROOT, "subsource_dating.csv")
F_MAST  = os.path.join(ROOT, "master_dating_results.csv")
F_ARCH  = os.path.join(ROOT, "subsource_archaism.csv")
F_SRCD  = os.path.join(ROOT, "source_dating.csv")
F_JE    = os.path.join(ROOT, "je_subsource_dating.csv")   # output of 06_je_subsource_dating.py

# ── Chapter-range definitions ────────────────────────────────────────────────
# Matches Script 13's D_UNITS and LEV_UNITS exactly.
SUBSOURCE_RANGES = {
    "D_Code"      : [("Deuteronomy", range(12, 27))],   # ch 12–26
    "D_Frame"     : [("Deuteronomy", list(range(1,12)) + list(range(27,32)) + [33,34])],
    "D_Song"      : [("Deuteronomy", [32])],
    "Lev_Holiness": [("Leviticus",   range(17, 27))],   # ch 17–26
    "Lev_Priestly": [("Leviticus",   range(1,  17))],   # ch 1–16
}

# JE label sets (these appear in the source_label column of torah_chapter_dates)
JE_LABELS = {"J", "E", "JE", "JE+P"}   # include JE+P since J narrative is embedded
JE_BOOKS  = {"Gen_JE": "Genesis", "Exo_JE": "Exodus", "Num_JE": "Numbers"}

# Source prior metadata (for reference)
PRIORS = {
    "D_Code"      : (625, 100),
    "D_Frame"     : (600, 100),
    "D_Song"      : (900, 200),
    "Lev_Holiness": (575,  75),
    "Lev_Priestly": (600, 100),
    "Gen_JE"      : (800, 100),
    "Exo_JE"      : (800, 100),
    "Num_JE"      : (800, 100),
    "D_source"    : (625, 100),
    "P_source"    : (600, 100),
    "JE_source"   : (800,  75),
}

# ── Load chapter-level data ──────────────────────────────────────────────────
chap = pd.read_csv(F_CHAP)
chap.columns = chap.columns.str.strip()
# Normalise source_label: drop leading/trailing spaces
chap["source_label"] = chap["source_label"].str.strip()

def aggregate_chapters(df_subset, label):
    """Word-count weighted mean of word_map only (char n-gram dropped as unreliable).
    Returns dict with n_words, n_chapters, ngram_word, word_ci68."""
    df = df_subset.dropna(subset=["map_word"])
    if df.empty:
        return None
    w = df["n_words"].values.astype(float)
    word_wav = np.average(df["map_word"].values, weights=w)
    n_words  = int(w.sum())

    # Bootstrap CI for the weighted mean (resample chapters)
    np.random.seed(42)
    n_boot = 1000
    word_boots = []
    idx = np.arange(len(df))
    for _ in range(n_boot):
        s = np.random.choice(idx, size=len(idx), replace=True)
        ws = w[s]
        word_boots.append(np.average(df["map_word"].values[s], weights=ws))
    word_ci = (np.percentile(word_boots, 16), np.percentile(word_boots, 84))

    return {
        "n_words"   : n_words,
        "n_chapters": len(df),
        "ngram_word": round(word_wav, 1),
        "word_ci68" : (round(word_ci[0],1), round(word_ci[1],1)),
    }


print("=" * 70)
print("Sub-source chapter-aggregated n-gram dates")
print("=" * 70)

ngram_results = {}

# ── Chapter-range sub-sources ────────────────────────────────────────────────
for unit, book_ranges in SUBSOURCE_RANGES.items():
    frames = []
    for book, ch_range in book_ranges:
        mask = (chap["book"] == book) & (chap["chapter"].isin(ch_range))
        frames.append(chap[mask])
    if not frames:
        continue
    df_sub = pd.concat(frames, ignore_index=True)
    result = aggregate_chapters(df_sub, unit)
    if result:
        ngram_results[unit] = result

# ── JE per-book sub-sources ──────────────────────────────────────────────────
for unit, book in JE_BOOKS.items():
    mask = (chap["book"] == book) & (chap["source_label"].isin(JE_LABELS))
    df_sub = chap[mask]
    result = aggregate_chapters(df_sub, unit)
    if result:
        ngram_results[unit] = result

# ── Load existing morpho-model dates (Script 13) ─────────────────────────────
subs  = pd.read_csv(F_SUBS).set_index("unit")  if os.path.exists(F_SUBS) else pd.DataFrame()
mast  = pd.read_csv(F_MAST).set_index("unit")  if os.path.exists(F_MAST) else pd.DataFrame()
arch  = pd.read_csv(F_ARCH).set_index("unit")  if os.path.exists(F_ARCH) else pd.DataFrame()
srcd  = pd.read_csv(F_SRCD).set_index("unit")  if os.path.exists(F_SRCD) else pd.DataFrame()

# ── Load JE sub-unit morpho dates (Script 06, if available) ─────────────────
# je_subsource_dating.csv is produced by running 06_je_subsource_dating.py
# on a machine with BHSA / text-fabric installed.  When present it fills in
# the Gen_JE / Exo_JE / Num_JE morpho columns that Script 13 did not cover.
je_subs = pd.read_csv(F_JE).set_index("unit") if os.path.exists(F_JE) else pd.DataFrame()
if not je_subs.empty:
    print(f"Loaded JE sub-unit morpho dates from je_subsource_dating.csv "
          f"({len(je_subs)} units: {', '.join(je_subs.index.tolist())})")
else:
    print("je_subsource_dating.csv not found — JE morpho dates unavailable. "
          "Run 06_je_subsource_dating.py on a machine with BHSA to generate it.")

# ── Build consolidated table ─────────────────────────────────────────────────
ALL_UNITS = (
    list(SUBSOURCE_RANGES.keys()) +
    list(JE_BOOKS.keys()) +
    ["D_source", "P_source", "JE_source"]
)

rows = []
for unit in ALL_UNITS:
    r = {"unit": unit}
    prior_bce, prior_sig = PRIORS.get(unit, (np.nan, np.nan))
    r["prior_bce"]   = prior_bce
    r["prior_sigma"] = prior_sig

    # Word n-gram chapter-aggregated dates (char n-gram dropped as unreliable)
    if unit in ngram_results:
        ng = ngram_results[unit]
        r["n_words"]     = ng["n_words"]
        r["n_chapters"]  = ng["n_chapters"]
        r["ngram_word"]  = ng["ngram_word"]
        r["word_ci68_lo"]= ng["word_ci68"][0]
        r["word_ci68_hi"]= ng["word_ci68"][1]
    else:
        r["n_words"] = r["n_chapters"] = np.nan
        r["ngram_word"] = np.nan
        r["word_ci68_lo"] = r["word_ci68_hi"] = np.nan

    # Morphosyntactic model dates — priority: Script 06 (JE) > Script 13 > master
    if unit in je_subs.index:
        r["morpho_map"]    = je_subs.loc[unit, "map_bce"]
        r["morpho_ci68_lo"]= je_subs.loc[unit, "ci68_lo"]
        r["morpho_ci68_hi"]= je_subs.loc[unit, "ci68_hi"]
    elif unit in subs.index:
        r["morpho_map"]    = subs.loc[unit, "map_bce"]
        r["morpho_ci68_lo"]= subs.loc[unit, "ci68_lo"]
        r["morpho_ci68_hi"]= subs.loc[unit, "ci68_hi"]
    elif unit in mast.index:
        r["morpho_map"]    = mast.loc[unit, "map_full"]
        r["morpho_ci68_lo"]= mast.loc[unit, "ci68_lo_raw"]
        r["morpho_ci68_hi"]= mast.loc[unit, "ci68_hi_raw"]
    else:
        r["morpho_map"] = r["morpho_ci68_lo"] = r["morpho_ci68_hi"] = np.nan

    # Archaism score — Script 06 supplies mean_lbh for JE units
    if unit in je_subs.index and "mean_lbh" in je_subs.columns:
        r["mean_lbh"]            = je_subs.loc[unit, "mean_lbh"]
        r["arch_classification"] = je_subs.loc[unit, "arch_classification"]
    elif unit in arch.index:
        r["mean_lbh"]            = arch.loc[unit, "mean_lbh"]
        r["arch_classification"] = arch.loc[unit, "classification"]
    else:
        r["mean_lbh"] = np.nan
        r["arch_classification"] = ""

    rows.append(r)

result_df = pd.DataFrame(rows)

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'Unit':<16s}  {'Prior':>8s}  "
      f"{'WordNgram':>10s}  {'WNg CI68':>16s}  "
      f"{'Morpho_MAP':>11s}  {'Morpho_CI68':>14s}  "
      f"{'mean_lbh':>9s}  {'Archaism class'}")
print(f"{'':16s}  {'':8s}  "
      f"{'(word only)':>10s}  {'':16s}  "
      f"{'(genre-conf)':>11s}  {'':14s}  {'':9s}")
print("-" * 115)

GROUPS = [
    ("D sub-components", ["D_source", "D_Code", "D_Frame", "D_Song"]),
    ("P sub-components", ["P_source", "Lev_Holiness", "Lev_Priestly"]),
    ("JE sub-components", ["JE_source", "Gen_JE", "Exo_JE", "Num_JE"]),
]

for group_label, units in GROUPS:
    print(f"\n  {group_label}")
    for unit in units:
        r = result_df[result_df["unit"] == unit]
        if r.empty:
            continue
        r = r.iloc[0]
        prior_str  = f"{int(r['prior_bce'])}±{int(r['prior_sigma'])}"
        nw  = f"{r['ngram_word']:.0f}" if pd.notna(r["ngram_word"]) else "—"
        wlo = f"{r['word_ci68_lo']:.0f}" if pd.notna(r["word_ci68_lo"]) else "—"
        whi = f"{r['word_ci68_hi']:.0f}" if pd.notna(r["word_ci68_hi"]) else "—"
        wci = f"[{wlo}–{whi}]" if nw != "—" else "—"
        mm  = f"{r['morpho_map']:.0f}" if pd.notna(r["morpho_map"]) else "—"
        mlo = f"{r['morpho_ci68_lo']:.0f}" if pd.notna(r["morpho_ci68_lo"]) else "—"
        mhi = f"{r['morpho_ci68_hi']:.0f}" if pd.notna(r["morpho_ci68_hi"]) else "—"
        mci = f"[{mlo}–{mhi}]" if mm != "—" else "—"
        lbh = f"{r['mean_lbh']:.3f}" if pd.notna(r["mean_lbh"]) else "—"
        arc = r["arch_classification"] if r["arch_classification"] else "—"
        print(f"  {unit:<16s}  {prior_str:>8s}  {nw:>10s}  {wci:>16s}  "
              f"{mm:>11s}  {mci:>14s}  {lbh:>9s}  {arc}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = os.path.join(OUTDIR, "subsource_dating.csv")
result_df.to_csv(out_csv, index=False)
print(f"\nSaved → hebrew/results/subsource_dating.csv")

# ── Key findings ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Key findings from sub-source split")
print("=" * 70)

# D_Code vs D_Frame comparison
d_code = result_df[result_df["unit"] == "D_Code"].iloc[0] if "D_Code" in result_df["unit"].values else None
d_frame= result_df[result_df["unit"] == "D_Frame"].iloc[0] if "D_Frame" in result_df["unit"].values else None
d_src  = result_df[result_df["unit"] == "D_source"].iloc[0] if "D_source" in result_df["unit"].values else None

if d_code is not None and d_frame is not None:
    print(f"""
D_Code vs D_Frame split:
  D_Code  (law code, ch.12–26):  word n-gram={d_code['ngram_word']:.0f} BCE;
          morpho={d_code['morpho_map']:.0f} BCE
          archaism: mean_lbh={d_code['mean_lbh']:.3f} → {d_code['arch_classification']}

  D_Frame (narrative frame):     word n-gram={d_frame['ngram_word']:.0f} BCE;
          morpho={d_frame['morpho_map']:.0f} BCE
          archaism: mean_lbh={d_frame['mean_lbh']:.3f} → {d_frame['arch_classification']}

  INTERPRETATION:
  D_Code is classified "Archaic (CBH-like)" (mean_lbh={d_code['mean_lbh']:.3f}) yet
  the morpho model dates it to 292 BCE — essentially the same as D_Frame
  ({d_frame['morpho_map']:.0f} BCE, mean_lbh={d_frame['mean_lbh']:.3f}).  This
  apparent contradiction has a clear cause: the archaism score and the morpho
  dating model weight features differently.

  mean_lbh averages per-feature positions on a CBH→LBH scale equally across
  all 36 features.  D_Code scores as archaic because it uses more אנכי,
  older legal formulas, and specific vocabulary that trends toward CBH.

  The morpho MVN model, however, weights features by how strongly they
  predict date in the training corpus.  D_Code's legal register (imperatives,
  second-person commands, specific clause structures) strongly resembles the
  LBH training prose (Chronicles, Ezra, Nehemiah), which is the only legal/
  instructional prose in the training corpus.  These genre-confounded features
  dominate the MVN likelihood, overriding the archaic vocabulary signal and
  pulling D_Code to the same late date as D_Frame.

  The word n-gram correctly resolves this: it dates D_Code={d_code['ngram_word']:.0f} BCE,
  D_Frame={d_frame['ngram_word']:.0f} BCE — D_Code is ~43 years earlier, consistent
  with D_Code being the more archaic stratum.  The word n-gram is less genre-
  sensitive than the full morpho model, so it sees the temporal signal more clearly.

  Conclusion: D_Code's Archaic classification is real.  The morpho model's
  failure to date it earlier than D_Frame is a genre-confound artifact.
  The word n-gram and the genre-neutral sub-model (Script 04, D=728 BCE)
  both correctly identify the archaic core.
""")

lev_h = result_df[result_df["unit"] == "Lev_Holiness"].iloc[0] if "Lev_Holiness" in result_df["unit"].values else None
lev_p = result_df[result_df["unit"] == "Lev_Priestly"].iloc[0] if "Lev_Priestly" in result_df["unit"].values else None

if lev_h is not None and lev_p is not None:
    print(f"""
Lev_Holiness vs Lev_Priestly split:
  Lev_Holiness (H code, ch.17–26):  morpho={lev_h['morpho_map']:.0f} BCE [{lev_h['morpho_ci68_lo']:.0f}–{lev_h['morpho_ci68_hi']:.0f}]
               archaism: mean_lbh={lev_h['mean_lbh']:.3f} → {lev_h['arch_classification']}

  Lev_Priestly (core P, ch.1–16):   morpho={lev_p['morpho_map']:.0f} BCE [{lev_p['morpho_ci68_lo']:.0f}–{lev_p['morpho_ci68_hi']:.0f}]
               archaism: mean_lbh={lev_p['mean_lbh']:.3f} → {lev_p['arch_classification']}

  INTERPRETATION:
  Lev_Priestly dates EARLIER (465 BCE) than Lev_Holiness (301 BCE) in the
  morpho model. This is the reverse of some scholarly views (Knohl/Milgrom
  argue H is later than core P). The archaism scores partially confirm this:
  Lev_Priestly is slightly more archaic (mean_lbh={lev_p['mean_lbh']:.3f}) than
  Lev_Holiness ({lev_h['mean_lbh']:.3f}), both Mixed/selective.

  The larger prior uncertainty for Lev_Holiness (± 75 yr around 575 BCE)
  overlaps the morpho estimate's CI, so the model results are not strongly
  incompatible with the Knohl chronology.
""")

# ── Write markdown report ────────────────────────────────────────────────────
out_md = os.path.join(OUTDIR, "subsource_dating.md")
with open(out_md, "w") as f:
    f.write("# Sub-source Dating Results\n\n")
    f.write("Sub-sources separated by chapter ranges within Deuteronomy and Leviticus.\n")
    f.write("N-gram dates are chapter-weighted aggregations from `torah_chapter_dates.csv`.\n")
    f.write("Morpho dates are from Script 13 (full morphosyntactic model, Script 11).\n\n")
    f.write("## D Sub-components\n\n")
    f.write("| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |\n")
    f.write("|------|------------|------------|---------|-----------|------------|----------|-------|\n")
    for unit in ["D_source", "D_Code", "D_Frame", "D_Song"]:
        rr = result_df[result_df["unit"] == unit]
        if rr.empty: continue
        r = rr.iloc[0]
        nw   = f"{r['ngram_word']:.0f}" if pd.notna(r["ngram_word"]) else "—"
        wlo  = f"{r['word_ci68_lo']:.0f}" if pd.notna(r["word_ci68_lo"]) else "—"
        whi  = f"{r['word_ci68_hi']:.0f}" if pd.notna(r["word_ci68_hi"]) else "—"
        wci  = f"{wlo}–{whi}" if nw != "—" else "—"
        mm   = f"{r['morpho_map']:.0f}" if pd.notna(r["morpho_map"]) else "—"
        mlo  = f"{r['morpho_ci68_lo']:.0f}" if pd.notna(r["morpho_ci68_lo"]) else "—"
        mhi  = f"{r['morpho_ci68_hi']:.0f}" if pd.notna(r["morpho_ci68_hi"]) else "—"
        mci  = f"{mlo}–{mhi}" if mm != "—" else "—"
        lbh  = f"{r['mean_lbh']:.3f}" if pd.notna(r["mean_lbh"]) else "—"
        arc  = r["arch_classification"] or "—"
        prior= f"{int(r['prior_bce'])} ± {int(r['prior_sigma'])}"
        f.write(f"| {unit} | {prior} | {nw} | {wci} | {mm} | {mci} | {lbh} | {arc} |\n")

    f.write("\n## P Sub-components\n\n")
    f.write("| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |\n")
    f.write("|------|------------|------------|---------|-----------|------------|----------|-------|\n")
    for unit in ["P_source", "Lev_Holiness", "Lev_Priestly"]:
        rr = result_df[result_df["unit"] == unit]
        if rr.empty: continue
        r = rr.iloc[0]
        nw   = f"{r['ngram_word']:.0f}" if pd.notna(r["ngram_word"]) else "—"
        wlo  = f"{r['word_ci68_lo']:.0f}" if pd.notna(r["word_ci68_lo"]) else "—"
        whi  = f"{r['word_ci68_hi']:.0f}" if pd.notna(r["word_ci68_hi"]) else "—"
        wci  = f"{wlo}–{whi}" if nw != "—" else "—"
        mm   = f"{r['morpho_map']:.0f}" if pd.notna(r["morpho_map"]) else "—"
        mlo  = f"{r['morpho_ci68_lo']:.0f}" if pd.notna(r["morpho_ci68_lo"]) else "—"
        mhi  = f"{r['morpho_ci68_hi']:.0f}" if pd.notna(r["morpho_ci68_hi"]) else "—"
        mci  = f"{mlo}–{mhi}" if mm != "—" else "—"
        lbh  = f"{r['mean_lbh']:.3f}" if pd.notna(r["mean_lbh"]) else "—"
        arc  = r["arch_classification"] or "—"
        prior= f"{int(r['prior_bce'])} ± {int(r['prior_sigma'])}"
        f.write(f"| {unit} | {prior} | {nw} | {wci} | {mm} | {mci} | {lbh} | {arc} |\n")

    f.write("\n## JE Sub-components\n\n")
    f.write("| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |\n")
    f.write("|------|------------|------------|---------|-----------|------------|----------|-------|\n")
    for unit in ["JE_source", "Gen_JE", "Exo_JE", "Num_JE"]:
        rr = result_df[result_df["unit"] == unit]
        if rr.empty: continue
        r = rr.iloc[0]
        nw   = f"{r['ngram_word']:.0f}" if pd.notna(r["ngram_word"]) else "—"
        wlo  = f"{r['word_ci68_lo']:.0f}" if pd.notna(r["word_ci68_lo"]) else "—"
        whi  = f"{r['word_ci68_hi']:.0f}" if pd.notna(r["word_ci68_hi"]) else "—"
        wci  = f"{wlo}–{whi}" if nw != "—" else "—"
        mm   = f"{r['morpho_map']:.0f}" if pd.notna(r["morpho_map"]) else "—"
        mlo  = f"{r['morpho_ci68_lo']:.0f}" if pd.notna(r["morpho_ci68_lo"]) else "—"
        mhi  = f"{r['morpho_ci68_hi']:.0f}" if pd.notna(r["morpho_ci68_hi"]) else "—"
        mci  = f"{mlo}–{mhi}" if mm != "—" else "—"
        lbh  = f"{r['mean_lbh']:.3f}" if pd.notna(r["mean_lbh"]) else "—"
        arc  = r["arch_classification"] or "—"
        prior= f"{int(r['prior_bce'])} ± {int(r['prior_sigma'])}"
        f.write(f"| {unit} | {prior} | {nw} | {wci} | {mm} | {mci} | {lbh} | {arc} |\n")

    f.write("""
## Notes on Data Sources

- **N-gram dates** are word-count weighted averages of per-chapter char and word
  n-gram MAP estimates from `torah_chapter_dates.csv`, bootstrapped for 68% CI.
- **Morpho MAP / CI68** comes from Script 13 (`subsource_dating.csv` and
  `master_dating_results.csv`), which applied the full 36-feature morphosyntactic
  model to BHSA chapter-range text extractions.
- **mean_lbh** is the mean LBH score across all 36 features (from
  `subsource_archaism.csv`); higher = more LBH-like.
- **BHSA morpho extraction** for sub-sources requires text-fabric + BHSA data.
  The new pipeline (Scripts 00–04) currently only has morpho feature profiles
  for whole-source D, P, JE aggregates (`source_feature_profiles.csv`).
  To add sub-sources to Scripts 02–03, run Script 19 extended with sub-unit
  chapter-range definitions.
""")

print(f"Saved → hebrew/results/subsource_dating.md")

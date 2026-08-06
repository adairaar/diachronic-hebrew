#!/usr/bin/env python3
"""
source_calibrated.py — Torah sources on a validated, noise-calibrated scale
===========================================================================
What we can and cannot ask, given everything established:

  CANNOT: "is this text archaizing?" The Greek ground-truth test showed
  deliberate archaizing is not detectable from any available feature family
  once period is controlled. The full-vs-resistant archaism index is therefore
  unvalidated and is NOT reported here as a verdict.

  CAN: "where does this source sit on the LBH scale, and is that position
  larger than measurement noise?" The Kings/Chronicles matched-pairs test
  validated the scale (Chronicles later in 12/14 pairs, p=0.0085, placebo
  p=0.02) and measured a noise floor of 0.058 from near-verbatim duplicates.

Two calibration yardsticks, both empirical:

  NOISE FLOOR  0.058   2 Kgs 18-20 vs Isa 36-39, same narrative twice
  REAL SHIFT   0.400   mean Kings->Chronicles gap across 14 synoptic pairs
                       (several centuries of genuine language change)

Plus a per-source internal control: each source is split in half and the two
halves scored independently. If a source's halves disagree by as much as the
source differs from the CBH anchor, its position is not determined.

Output: results_v2/source_calibrated.csv
"""

from __future__ import annotations
import os, importlib.util
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FM   = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUT  = os.path.join(HERE, "results_v2", "source_calibrated.csv")

spec = importlib.util.spec_from_file_location(
    "ex", os.path.join(HERE, "hierarchical_bayes", "00_extract_features.py"))
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

PRESPEC = ["frac_ani","frac_she","frac_ein","rate_wayyiqtol","rate_qatal",
           "frac_niphal","rate_inf_con","rate_gam","rate_ut_nouns",
           "frac_infc","frac_fronted","frac_null_subj"]

NOISE_FLOOR = 0.058    # measured, Kings/Chronicles duplicate narrative
REAL_SHIFT  = 0.400    # measured, mean Kings->Chronicles synoptic gap

# Each source split into two independent halves for the internal control
SOURCES = {
    "D_Code":       ([("Deuteronomy",[(12,19)])],   [("Deuteronomy",[(20,26)])]),
    "D_Frame":      ([("Deuteronomy",[(1,11)])],    [("Deuteronomy",[(27,31)])]),
    "D_Song":       ([("Deuteronomy",[(32,32)])],   None),
    "Lev_Holiness": ([("Leviticus",[(17,21)])],     [("Leviticus",[(22,26)])]),
    "Lev_Priestly": ([("Leviticus",[(1,8)])],       [("Leviticus",[(9,16)])]),
    "Song_Sea":     ([("Exodus",[(15,15)])],        None),
    "Song_Deborah": ([("Judges",[(5,5)])],          None),
    "Gen_JE":       ([("Genesis",[(2,22)])],        [("Genesis",[(24,50)])]),
    "P_source":     ([("Leviticus",[(1,27)])],      [("Numbers",[(1,10)])]),
    # anchors for orientation
    "Jer_oracle":   ([("Jeremiah",[(1,6),(8,10)])], [("Jeremiah",[(30,31),(46,51)])]),
    "Chronicles":   ([("1_Chronicles",[(1,29)])],   [("2_Chronicles",[(1,36)])]),
}


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[(df["in_training"] == True) & (~df["holdout"].astype(bool))]  # noqa: E712
    feats = [f for f in PRESPEC if f in df.columns]

    d = tr["date_bce"].values.astype(float)
    A, B = [], []
    for f in feats:
        y = pd.to_numeric(tr[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        A.append(ic); B.append(sl)
    A, B = np.array(A), np.array(B)
    mu_c, mu_l = A + B * 720.0, A + B * 250.0

    api = ex.load_bhsa(ex.TF_PATH)
    F, L, T = api.F, api.L, api.T

    def score(pairs):
        fe = ex.extract_unit("x", pairs, F, L, T)
        if fe is None:
            return None, 0
        n = fe.pop("n_words")
        x = np.array([fe.get(f, np.nan) for f in feats], float)
        den = mu_l - mu_c
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(den) > 1e-12, (x - mu_c) / den, np.nan)
        return float(np.nanmean(s)), n

    print("TORAH SOURCES ON THE VALIDATED LBH SCALE\n")
    print(f"  s = 0 -> matches 720 BCE Hebrew;  s = 1 -> matches 250 BCE")
    print(f"  calibration: noise floor {NOISE_FLOOR:.3f}  |  "
          f"Kings->Chronicles shift {REAL_SHIFT:.3f}\n")
    print(f"  {'source':15s}{'s':>8s}{'half A':>9s}{'half B':>9s}"
          f"{'internal':>10s}{'words':>8s}   reading")
    print("  " + "-" * 78)

    rows = []
    for name, (pa, pb) in SOURCES.items():
        whole = pa if pb is None else pa + pb
        s, n = score(whole)
        if s is None:
            continue
        sa = sb = np.nan; internal = np.nan
        if pb is not None:
            sa, _ = score(pa); sb, _ = score(pb)
            if sa is not None and sb is not None:
                internal = abs(sa - sb)

        # Is the source's distance from the CBH anchor bigger than its own
        # internal inconsistency, and bigger than the measured noise floor?
        dist = abs(s)
        if np.isfinite(internal) and internal > dist:
            rd = "UNDETERMINED (halves disagree more than the effect)"
        elif dist < NOISE_FLOOR:
            rd = "indistinguishable from the CBH anchor"
        elif dist < REAL_SHIFT:
            rd = "shifted, but less than one Kings->Chron step"
        else:
            rd = f"shifted {dist/REAL_SHIFT:.1f} Kings->Chron steps"
        rows.append(dict(source=name, s=round(s, 3),
                         half_A=round(sa, 3) if np.isfinite(sa) else np.nan,
                         half_B=round(sb, 3) if np.isfinite(sb) else np.nan,
                         internal_gap=round(internal, 3) if np.isfinite(internal) else np.nan,
                         n_words=n, reading=rd))
        f_ = lambda v: f"{v:>9.3f}" if np.isfinite(v) else "        -"
        g_ = lambda v: f"{v:>10.3f}" if np.isfinite(v) else "         -"
        print(f"  {name:15s}{s:>8.3f}{f_(sa)}{f_(sb)}{g_(internal)}"
              f"{n:>8d}   {rd}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    det = out[out["reading"].str.startswith("UNDETERMINED")]
    print(f"\n  {len(det)} of {len(out)} sources are UNDETERMINED "
          f"(internal inconsistency exceeds the effect)")
    if len(det):
        print(f"    {', '.join(det['source'])}")
    print("\n  Reminder: none of this speaks to archaizing. Whether a low score")
    print("  means genuine antiquity or successful imitation is not decidable")
    print("  from these features (see the Greek ground-truth result).")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()

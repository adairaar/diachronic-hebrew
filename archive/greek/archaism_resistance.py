#!/usr/bin/env python3
"""
archaism_resistance.py — which features survive deliberate archaizing?
======================================================================
Greek supplies what Hebrew cannot: a ground-truth archaizing corpus. The
Atticizing authors of the Second Sophistic (Lucian, Philostratus, Aelius
Aristides, Arrian, Pausanias ...) deliberately imitated Classical Attic
centuries after it stopped being spoken, and their real dates are known
independently. Their contemporaries writing ordinary Koine give the
counterfactual: what the same period looks like without the disguise.

For each feature we therefore have three anchors:

    C = mean of genuine Classical Attic   (~450-320 BCE, the imitated target)
    K = mean of contemporary Koine        (same era as the Atticizers)
    A = mean of the Atticizing texts      (the disguise in action)

Resistance score:

    R_j = (A_j - C_j) / (K_j - C_j)

    R ~ 0  the imitation succeeded; the feature was faked. USELESS for
           archaism detection -- it will call a late text early.
    R ~ 1  the author's real period leaked through despite the attempt.
           RESISTANT: this is the kind of feature an archaism diagnostic
           should be built from.

This is an empirical test of the assumption behind the Hebrew resistant
model -- that clause-level syntax lies below conscious control. That
assumption has never been validated; here it can be.

Output: results/archaism_resistance.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "results", "register_features.csv")
OUT  = os.path.join(HERE, "results", "archaism_resistance.csv")

META = {"id", "register", "holdout", "date_ce", "author"}

# Feature families, to see whether resistance tracks linguistic level
FAMILY = {
    "dual_rate": "morphology", "opt_rate": "morphology", "ptcp_rate": "morphology",
    "tt_ss_ratio": "orthography",
    "men_rate": "particle", "de_rate": "particle", "oun_rate": "particle",
    "goun_toinun_rate": "particle", "kai_rate": "particle", "hina_rate": "particle",
    "idou_rate": "lexical", "eipen_rate": "lexical",
    "avg_sent_len": "syntax", "kai_sent_initial": "syntax",
}


def main():
    df = pd.read_csv(FEAT)
    df["reg"] = df["register"].str.lower()
    feats = [c for c in df.columns if c not in META | {"reg"}
             and pd.api.types.is_numeric_dtype(df[c])]

    classical = df[df["reg"].str.contains("attic") & ~df["reg"].str.contains("atticiz")]
    atticizing = df[df["reg"].str.contains("atticiz")]
    koine = df[df["reg"] == "koine"]

    # Contemporary Koine: restrict to the era the Atticizers actually wrote in
    lo, hi = atticizing["date_ce"].min(), atticizing["date_ce"].max()
    koine_c = koine[(koine["date_ce"] >= lo - 60) & (koine["date_ce"] <= hi + 60)]

    print("GROUND-TRUTH ARCHAIZING TEST (Greek)\n")
    print(f"  Classical Attic (imitated target) : n={len(classical):2d}  "
          f"{classical['date_ce'].min():.0f} to {classical['date_ce'].max():.0f} CE")
    print(f"  Atticizing (the disguise)         : n={len(atticizing):2d}  "
          f"{lo:.0f} to {hi:.0f} CE")
    print(f"  Contemporary Koine (counterfactual): n={len(koine_c):2d}  "
          f"{koine_c['date_ce'].min():.0f} to {koine_c['date_ce'].max():.0f} CE")
    print(f"\n  Atticizing authors: {', '.join(sorted(atticizing['author'].unique()))}\n")

    rows = []
    for f in feats:
        C = classical[f].mean(); K = koine_c[f].mean(); A = atticizing[f].mean()
        gap = K - C
        if not np.isfinite(gap) or abs(gap) < 1e-9:
            continue
        R = (A - C) / gap
        # Is Atticizing distinguishable from Classical on this feature at all?
        try:
            _, p_vs_class = stats.mannwhitneyu(atticizing[f].dropna(),
                                               classical[f].dropna())
        except ValueError:
            p_vs_class = np.nan
        pooled = np.sqrt((classical[f].var() + koine_c[f].var()) / 2)
        rows.append(dict(feature=f, family=FAMILY.get(f, "?"),
                         classical=C, koine=K, atticizing=A,
                         resistance=R,
                         sep_C_K_sd=(gap / pooled) if pooled > 0 else np.nan,
                         p_atticizing_vs_classical=p_vs_class))

    r = pd.DataFrame(rows).sort_values("resistance", ascending=False)
    r.to_csv(OUT, index=False)

    print("Resistance to deliberate archaizing")
    print("  R~0 = successfully faked (useless for detection)")
    print("  R~1 = real period leaked through (usable)\n")
    print(f"  {'feature':20s}{'family':12s}{'Class':>9s}{'Koine':>9s}"
          f"{'Attic-zg':>10s}{'R':>8s}   verdict")
    print("  " + "-" * 82)
    for _, x in r.iterrows():
        if x.resistance > 0.6:   v = "RESISTANT"
        elif x.resistance > 0.3: v = "partial"
        elif x.resistance > -0.1: v = "faked"
        else:                     v = "over-corrected"
        print(f"  {x.feature:20s}{x.family:12s}{x.classical:>9.2f}{x.koine:>9.2f}"
              f"{x.atticizing:>10.2f}{x.resistance:>8.2f}   {v}")

    print("\n\nResistance by linguistic level (mean R)")
    fam = r.groupby("family")["resistance"].agg(["mean", "count"]).sort_values("mean",
                                                                              ascending=False)
    for k, v in fam.iterrows():
        print(f"  {k:14s} R = {v['mean']:+.2f}   (n={int(v['count'])})")

    res = r[r.resistance > 0.6]["feature"].tolist()
    fak = r[r.resistance <= 0.3]["feature"].tolist()
    print(f"\n  Usable for archaism detection ({len(res)}): {', '.join(res) if res else 'NONE'}")
    print(f"  Successfully faked ({len(fak)}): {', '.join(fak) if fak else 'none'}")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()

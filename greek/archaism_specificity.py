#!/usr/bin/env python3
"""
archaism_specificity.py — is it imitation detection, or just period detection?
==============================================================================
Character 4-grams separate Atticizers from genuine Classical Attic at AUC 0.94.
Two very different explanations:

  (a) They detect DELIBERATE IMITATION. The imitator's own period leaks through
      in orthographic texture they cannot consciously control. This is the
      strong claim and the useful one.

  (b) They detect PERIOD, full stop. Atticizers are late, Classical is early,
      and char n-grams are simply good date discriminators. Then the AUC of
      0.94 says nothing about archaizing at all.

The discriminating tests:

  T1  Koine vs Classical.  Both are "authentic" (nobody is imitating), and they
      differ by the same centuries as the Atticizing contrast. If char n-grams
      are pure period detectors this should score AS HIGH as the Atticizing
      contrast. If they detect imitation specifically, it may score high too --
      so this alone is necessary but not sufficient.

  T2  Atticizing vs contemporary Koine.  THE KEY TEST. Same period, same
      centuries, differing ONLY in whether the author was archaizing. Under
      (b) this should be at chance, since period is held constant. Under (a)
      it should separate: the Atticizer's surface is dressed old while the
      Koine writer's is not.

  T3  Within-Classical placebo.  Split the genuine Classical texts by median
      date and try to classify. Tests whether the pipeline manufactures
      separation from arbitrary splits at this n.

Every test carries a permutation null.

Output: results/archaism_specificity.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D    = os.path.join(HERE, "data", "features")
REG  = os.path.join(HERE, "results", "register_features.csv")
OUT  = os.path.join(HERE, "results", "archaism_specificity.csv")

N_PERM = 60
RNG = np.random.default_rng(20260805)
LEAK = {"date_ce", "date_sigma", "holdout", "word_count", "n_words",
        "date_bce", "year", "era"}

BLOCKS = {"grammatical": "grammatical_features.csv",
          "word_bigrams": "word_bigrams.csv",
          "char_ngrams_3": "char_ngrams_3.csv",
          "char_ngrams_4": "char_ngrams_4.csv"}


def loo_auc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    pred = np.zeros(len(y), float)
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        if len(np.unique(y[tr])) < 2:
            return np.nan
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000))
        pipe.fit(X[tr], y[tr])
        pred[i] = pipe.predict_proba(X[i:i + 1])[:, 1]
    try:
        return roc_auc_score(y, pred)
    except ValueError:
        return np.nan


def evaluate(mats, ids, y, label):
    out = []
    for name, m in mats.items():
        sub = m.reindex(ids)
        keep = sub.columns[sub.notna().all() & (sub.std() > 1e-12)]
        if len(keep) < 2:
            continue
        X = sub[keep].values.astype(float)
        obs = loo_auc(X, y)
        null = [a for a in (loo_auc(X, RNG.permutation(y)) for _ in range(N_PERM))
                if np.isfinite(a)]
        null = np.array(null)
        p = float((null >= obs).mean()) if len(null) else np.nan
        out.append(dict(test=label, block=name, n=len(y), n_pos=int(y.sum()),
                        n_features=len(keep), auc=round(obs, 3),
                        null_mean=round(float(null.mean()), 3),
                        perm_p=round(p, 3)))
    return out


def main():
    reg = pd.read_csv(REG)
    reg["reg"] = reg["register"].str.lower()

    mats = {}
    for name, fn in BLOCKS.items():
        p = os.path.join(D, fn)
        if not os.path.exists(p):
            continue
        m = pd.read_csv(p)
        idc = "id" if "id" in m.columns else m.columns[0]
        m = m.rename(columns={idc: "id"}).set_index("id")
        m = m.drop(columns=[c for c in m.columns if c in LEAK], errors="ignore")
        mats[name] = m.select_dtypes(include=[np.number])

    classical = reg[reg["reg"].str.contains("attic") & ~reg["reg"].str.contains("atticiz")]
    attic     = reg[reg["reg"].str.contains("atticiz")]
    koine     = reg[reg["reg"] == "koine"]
    lo, hi = attic["date_ce"].min(), attic["date_ce"].max()
    koine_c = koine[(koine["date_ce"] >= lo - 60) & (koine["date_ce"] <= hi + 60)]

    rows = []

    # Reference: the original contrast
    ids = pd.concat([attic, classical])["id"]
    y = np.r_[np.ones(len(attic)), np.zeros(len(classical))]
    rows += evaluate(mats, ids, y, "REF Atticizing vs Classical")

    # T1 Koine vs Classical (both authentic, same era gap)
    ids = pd.concat([koine_c, classical])["id"]
    y = np.r_[np.ones(len(koine_c)), np.zeros(len(classical))]
    rows += evaluate(mats, ids, y, "T1 Koine vs Classical")

    # T2 THE KEY TEST: Atticizing vs contemporary Koine, period held constant
    ids = pd.concat([attic, koine_c])["id"]
    y = np.r_[np.ones(len(attic)), np.zeros(len(koine_c))]
    rows += evaluate(mats, ids, y, "T2 Atticizing vs contemp. Koine")

    # T3 placebo: split genuine Classical by median date
    med = classical["date_ce"].median()
    cl = classical.copy(); cl["y"] = (cl["date_ce"] > med).astype(int)
    if 3 <= cl["y"].sum() <= len(cl) - 3:
        rows += evaluate(mats, cl["id"], cl["y"].values.astype(float),
                         "T3 placebo within-Classical")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("SPECIFICITY TESTS — is it imitation, or just period?\n")
    for t in df["test"].unique():
        s = df[df["test"] == t]
        n = s.iloc[0]
        print(f"{t}   (n={n['n']}, {int(n['n_pos'])} positive)")
        print(f"  {'block':16s}{'AUC':>7s}{'null':>8s}{'perm p':>9s}")
        for _, r in s.iterrows():
            star = "  *" if r.perm_p < 0.05 else ""
            print(f"  {r.block:16s}{r.auc:>7.2f}{r.null_mean:>8.2f}"
                  f"{r.perm_p:>9.3f}{star}")
        print()

    print("How to read it:")
    print("  T2 at chance   -> char n-grams detect PERIOD only; the 0.94 is")
    print("                    date discrimination and says nothing about")
    print("                    archaizing. The strong claim fails.")
    print("  T2 separates   -> they detect DELIBERATE IMITATION with period")
    print("                    held constant. The strong claim survives.")
    print("  T3 above chance-> the pipeline manufactures separation at this n;")
    print("                    distrust everything above.")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()

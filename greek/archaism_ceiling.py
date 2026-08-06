#!/usr/bin/env python3
"""
archaism_ceiling.py — can ANY feature set detect deliberate archaizing?
=======================================================================
The register-diagnostic features (14) gave a supervised ceiling of AUC 0.68,
whose confidence interval includes chance. That could mean archaizing is
undetectable, or merely that those features were built for a different job
(LXX/Semitic-substrate detection).

This script settles it by throwing every available feature set at the problem:

    grammatical      33 features   morphosyntactic rates
    word_bigrams    300            function-word / POS sequences
    char_ngrams_3   200            sub-morphemic orthography
    char_ngrams_4   200
    combined        733

Task: separate the 14 known Second Sophistic Atticizers from the 17 genuine
Classical Attic texts, using leave-one-out cross-validation.

CRITICAL CONTROL. With p >> n, LOO-CV AUC is optimistically biased and can look
impressive on pure noise. Every result is therefore accompanied by a
permutation null: the labels are shuffled and the entire LOO procedure repeated,
giving the distribution of AUC obtainable by chance at this sample size and
dimensionality. The reported p-value is the fraction of null runs matching or
exceeding the observed AUC.

An AUC of 0.75 means nothing if shuffled labels routinely produce 0.75.

Output: results/archaism_ceiling.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D    = os.path.join(HERE, "data", "features")
REG  = os.path.join(HERE, "results", "register_features.csv")
OUT  = os.path.join(HERE, "results", "archaism_ceiling.csv")

N_PERM = 200
RNG = np.random.default_rng(20260805)

BLOCKS = {
    "grammatical":   "grammatical_features.csv",
    "word_bigrams":  "word_bigrams.csv",
    "char_ngrams_3": "char_ngrams_3.csv",
    "char_ngrams_4": "char_ngrams_4.csv",
}


def loo_auc(X, y, C=1.0):
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
                             LogisticRegression(max_iter=3000, C=C))
        pipe.fit(X[tr], y[tr])
        pred[i] = pipe.predict_proba(X[i:i + 1])[:, 1]
    return roc_auc_score(y, pred)


def main():
    reg = pd.read_csv(REG)[["id", "register", "date_ce", "author"]]
    reg["reg"] = reg["register"].str.lower()
    lab = reg[reg["reg"].str.contains("attic")].copy()
    lab["y"] = lab["reg"].str.contains("atticiz").astype(int)
    print(f"Task: {int(lab['y'].sum())} Atticizers vs "
          f"{int((1-lab['y']).sum())} genuine Classical\n")

    rows = []
    mats = {}
    for name, fn in BLOCKS.items():
        p = os.path.join(D, fn)
        if not os.path.exists(p):
            print(f"  [missing] {fn}"); continue
        m = pd.read_csv(p)
        idcol = "id" if "id" in m.columns else m.columns[0]
        m = m.rename(columns={idcol: "id"}).set_index("id")
        # AUDIT FIX: drop metadata before selecting numerics. date_ce and
        # date_sigma are numeric columns in grammatical_features.csv, and
        # letting them through gave a spurious AUC of 0.97 -- the classifier
        # was reading the date, not the language.
        LEAK = {"date_ce", "date_sigma", "holdout", "word_count", "n_words",
                "date_bce", "year", "era"}
        m = m.drop(columns=[c for c in m.columns if c in LEAK], errors="ignore")
        m = m.select_dtypes(include=[np.number])
        mats[name] = m

    if mats:
        mats["combined"] = pd.concat(mats.values(), axis=1)

    print(f"{'feature block':16s}{'p':>6s}{'AUC':>7s}{'null mean':>11s}"
          f"{'null 95th':>11s}{'perm p':>9s}   verdict")
    print("-" * 74)

    for name, m in mats.items():
        sub = m.reindex(lab["id"])
        keep = sub.columns[sub.notna().all() & (sub.std() > 1e-12)]
        X = sub[keep].values.astype(float)
        y = lab["y"].values
        if X.shape[1] < 2:
            print(f"{name:16s}  no usable columns"); continue

        obs = loo_auc(X, y)
        null = []
        for _ in range(N_PERM):
            yp = RNG.permutation(y)
            a = loo_auc(X, yp)
            if np.isfinite(a):
                null.append(a)
        null = np.array(null)
        p_perm = float((null >= obs).mean()) if len(null) else np.nan
        n95 = float(np.percentile(null, 95)) if len(null) else np.nan
        verdict = ("DETECTABLE" if p_perm < 0.05 else
                   "not distinguishable from chance")
        rows.append(dict(block=name, n_features=X.shape[1], auc=round(obs, 3),
                         null_mean=round(float(null.mean()), 3),
                         null_p95=round(n95, 3), perm_p=round(p_perm, 3),
                         verdict=verdict))
        print(f"{name:16s}{X.shape[1]:>6d}{obs:>7.2f}{null.mean():>11.2f}"
              f"{n95:>11.2f}{p_perm:>9.3f}   {verdict}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print("\nNote: null mean well above 0.50 would indicate LOO-CV optimism at")
    print("this n and p; the permutation p-value is the only safe read.")


if __name__ == "__main__":
    main()

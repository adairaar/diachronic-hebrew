#!/usr/bin/env python3
"""
archaism_detectors.py — four candidate archaizing detectors, scored on Greek
============================================================================
Greek gives labelled ground truth: the Second Sophistic Atticizers deliberately
imitated Classical Attic and their real dates are known independently. So a
detector can be *validated* rather than assumed.

Design: the diachronic model is fitted on AUTHENTIC texts only (genuine
Classical Attic + contemporary Koine, Atticizers excluded). Every detector then
scores a text without using its date. Performance is AUC for separating the 14
Atticizers from the 17 genuine Classical texts.

Detectors
---------
A. DISPERSION. Invert each feature's regression to get a per-feature implied
   date, then measure the spread. An authentic text should have its features
   agree; an imitator fakes some features and not others, so the implied dates
   should scatter. This generalises the full-vs-resistant idea from two models
   to the whole feature profile.

B. OVERSHOOT. Exceedance probability past the authentic-corpus distribution.
   Properly specified this time: fit a Gaussian per feature on authentic texts
   of the target era and ask how improbable the observed value is in the
   archaic direction. (The earlier Hebrew attempt measured distance beyond the
   training MAXIMUM, which is a noisy order statistic that nothing can exceed
   by construction.)

C. RESISTANT-VS-FAKEABLE DIVERGENCE. Split features by their measured
   resistance R (from archaism_resistance.py) and compare the two implied
   dates. This is the current Hebrew method, rebuilt on empirically resistant
   features rather than assumed-resistant ones.

D. SUPERVISED. Logistic regression on all features, leave-one-out. Not usable
   for Hebrew (no labels there) but establishes the ceiling: how separable are
   these classes at all?

Output: results/archaism_detectors.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "results", "register_features.csv")
RES  = os.path.join(HERE, "results", "archaism_resistance.csv")
OUT  = os.path.join(HERE, "results", "archaism_detectors.csv")

META = {"id", "register", "holdout", "date_ce", "author", "reg"}


def auc(pos, neg):
    """Mann-Whitney AUC: P(score(pos) > score(neg))."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return u / (len(pos) * len(neg))


def main():
    df = pd.read_csv(FEAT)
    df["reg"] = df["register"].str.lower()
    feats = [c for c in df.columns if c not in META
             and pd.api.types.is_numeric_dtype(df[c])]

    classical = df[df["reg"].str.contains("attic") & ~df["reg"].str.contains("atticiz")]
    atticiz   = df[df["reg"].str.contains("atticiz")]
    koine     = df[df["reg"] == "koine"]
    authentic = pd.concat([classical, koine])          # never archaizing

    print(f"Authentic training set : {len(authentic)} texts "
          f"({len(classical)} Classical + {len(koine)} Koine)")
    print(f"Known archaizers       : {len(atticiz)} Atticizing texts")
    print(f"Median length proxy    : Classical vs Atticizing sentence len "
          f"{classical['avg_sent_len'].median():.1f} / "
          f"{atticiz['avg_sent_len'].median():.1f}\n")

    # ── Per-feature regression on AUTHENTIC texts only ────────────────────────
    d = authentic["date_ce"].values.astype(float)
    reg = {}
    for f in feats:
        y = pd.to_numeric(authentic[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        if m.sum() < 8 or np.nanstd(y[m]) < 1e-12:
            continue
        sl, ic, r, p, se = stats.linregress(d[m], y[m])
        if abs(sl) < 1e-12:
            continue
        resid = y[m] - (ic + sl * d[m])
        reg[f] = dict(slope=sl, icept=ic, sd=float(np.std(resid)),
                      p=p, rho=r)
    usable = list(reg)
    print(f"Features with a usable temporal slope: {len(usable)} of {len(feats)}")
    strong = [f for f in usable if reg[f]["p"] < 0.10]
    print(f"  of which p<0.10: {len(strong)}  ({', '.join(strong)})\n")

    # Resistance scores, if available
    Rmap = {}
    if os.path.exists(RES):
        rr = pd.read_csv(RES)
        Rmap = dict(zip(rr["feature"], rr["resistance"]))

    # ── Score every text ──────────────────────────────────────────────────────
    rows = []
    era_lo, era_hi = atticiz["date_ce"].min(), atticiz["date_ce"].max()
    contemp = authentic[(authentic["date_ce"] >= era_lo - 60) &
                        (authentic["date_ce"] <= era_hi + 60)]

    for _, r in df.iterrows():
        implied, over = [], []
        for f in usable:
            v = pd.to_numeric(pd.Series([r[f]]), errors="coerce").iloc[0]
            if not np.isfinite(v):
                continue
            g = reg[f]
            implied.append((v - g["icept"]) / g["slope"])
            # B: how improbable is this value in the ARCHAIC direction,
            #    relative to authentic texts of the archaizers' own era?
            cv = pd.to_numeric(contemp[f], errors="coerce").values.astype(float)
            cv = cv[np.isfinite(cv)]
            if len(cv) < 5 or np.std(cv) < 1e-12:
                continue
            z = (v - np.mean(cv)) / np.std(cv)
            # archaic direction = the sign that implies an EARLIER date
            over.append(-z if g["slope"] > 0 else z)

        implied = np.array(implied); over = np.array(over)
        # C: resistant vs fakeable split
        res_f = [f for f in usable if Rmap.get(f, 0) > 0.6]
        fak_f = [f for f in usable if Rmap.get(f, 1) <= 0.3]
        def imp(sub):
            vals = []
            for f in sub:
                v = pd.to_numeric(pd.Series([r[f]]), errors="coerce").iloc[0]
                if np.isfinite(v):
                    vals.append((v - reg[f]["icept"]) / reg[f]["slope"])
            return float(np.median(vals)) if vals else np.nan

        rows.append(dict(
            id=r["id"], author=r.get("author", ""), register=r["reg"],
            date_ce=r["date_ce"],
            A_dispersion=float(np.std(implied)) if len(implied) > 2 else np.nan,
            A_iqr=float(np.subtract(*np.percentile(implied, [75, 25])))
                  if len(implied) > 3 else np.nan,
            B_overshoot=float(np.mean(over)) if len(over) else np.nan,
            B_max_over=float(np.max(over)) if len(over) else np.nan,
            C_resist_date=imp(res_f), C_fake_date=imp(fak_f),
            C_divergence=imp(fak_f) - imp(res_f)))
    out = pd.DataFrame(rows).set_index("id")
    out.to_csv(OUT)

    A = out[out["register"].str.contains("atticiz")]
    C = out[out["register"].str.contains("attic") & ~out["register"].str.contains("atticiz")]
    K = out[out["register"] == "koine"]

    print("DETECTOR PERFORMANCE — separating 14 Atticizers from 17 genuine Classical")
    print("  AUC 0.5 = useless, 1.0 = perfect, <0.5 = backwards\n")
    print(f"  {'detector':22s}{'AUC':>7s}   {'Atticizing':>22s}{'Classical':>20s}")
    print("  " + "-" * 74)
    for col, lbl in (("A_dispersion", "A: implied-date SD"),
                     ("A_iqr", "A: implied-date IQR"),
                     ("B_overshoot", "B: mean overshoot"),
                     ("B_max_over", "B: max overshoot"),
                     ("C_divergence", "C: fake-resist gap")):
        a, c = A[col].dropna(), C[col].dropna()
        if len(a) == 0 or len(c) == 0:
            continue
        print(f"  {lbl:22s}{auc(a, c):>7.2f}   "
              f"{a.median():>10.1f} [{a.quantile(.25):.0f},{a.quantile(.75):.0f}]"
              f"{c.median():>11.1f} [{c.quantile(.25):.0f},{c.quantile(.75):.0f}]")

    # ── D: supervised ceiling ─────────────────────────────────────────────────
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
        sub = pd.concat([classical, atticiz])
        X = sub[usable].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median()).values
        y = sub["reg"].str.contains("atticiz").astype(int).values
        pred = np.zeros(len(y), float)
        for tr_i, te_i in LeaveOneOut().split(X):
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))
            pipe.fit(X[tr_i], y[tr_i])
            pred[te_i] = pipe.predict_proba(X[te_i])[:, 1]
        print(f"\n  {'D: supervised (LOO)':22s}{roc_auc_score(y, pred):>7.2f}"
              "   <- ceiling; needs labels, unavailable for Hebrew")
    except ImportError:
        print("\n  (sklearn unavailable; skipping supervised ceiling)")

    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()

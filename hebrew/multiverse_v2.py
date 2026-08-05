#!/usr/bin/env python3
"""
multiverse_v2.py — specification curve for the MLE-MVN dating model
====================================================================
Runs the same targets through many defensible analysis specifications and
reports how far the answers move.  If the headline dates are stable across
specifications, that stability is itself reportable evidence.  If they are
not, the honest conclusion is that the corpus does not identify them.

Specifications
--------------
  asis_p30      p < 0.30                     (manuscript's stated screen)
  p10           p < 0.10                     (what the code actually used)
  p05           p < 0.05
  p01           p < 0.01
  bh10          Benjamini-Hochberg q = 0.10
  kfold4        p < 0.10 + 4-fold sign stability (folds genuinely perturb)
  blockreg      p < 0.10 + register-block stability (drop one register group)
  NOISE+k       p < 0.01 real features PLUS k synthetic null features that
                pass the same screen -- a control for the claim that noisy
                features get small weight and cancel out

Each specification is run with word-count scaling on and off.

Output: hebrew/results_v2/multiverse.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
FM     = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUTDIR = os.path.join(HERE, "results_v2")
os.makedirs(OUTDIR, exist_ok=True)

META = {"date_bce","date_sigma","register","genre","holdout",
        "hbvi_holdout","in_training","n_words"}
RNG   = np.random.default_rng(20260805)
RIDGE = 0.10
GRID  = np.linspace(900, -100, 500)
TARGETS = ["P_source","D_source","JE_source","D_Code","Lev_Priestly",
           "Song_Sea","Jer_oracle","Jer_DTR","Haggai","Habakkuk","Daniel"]


def p_from_rho(rho, n):
    rho = np.clip(rho, -0.999999, 0.999999)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def spearman(d, v):
    m = np.isfinite(d) & np.isfinite(v)
    if m.sum() < 6 or np.nanstd(v[m]) < 1e-12:
        return np.nan, 1.0
    r = stats.spearmanr(d[m], v[m])
    return float(r.statistic), float(r.pvalue)


def fit(df_tr, feats, ridge=RIDGE):
    d = df_tr["date_bce"].values.astype(float)
    A, B, res = [], [], []
    for f in feats:
        y = pd.to_numeric(df_tr[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        A.append(ic); B.append(sl)
        r = np.full(len(d), np.nan); r[m] = y[m] - (ic + sl * d[m])
        res.append(r)
    A, B = np.array(A), np.array(B)
    R = np.nan_to_num(np.vstack(res).T)
    S = (R.T @ R) / max(len(d) - 1, 1)
    S += ridge * (np.trace(S) / len(feats)) * np.eye(len(feats))
    return A, B, np.linalg.inv(S)


def post_map(x, A, B, Si, w):
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.nan, np.nan
    ll = np.array([-0.5 * w * (dv := (x - (A + B * g))[m]) @ Si[np.ix_(m, m)] @ dv
                   for g in GRID])
    ll -= ll.max()
    lp = -0.5 * ((GRID - 464.0) / 300.0) ** 2
    p = np.exp(ll + lp); p /= p.sum()
    # 68% HDI width as a confidence proxy
    o = np.argsort(p)[::-1]; c = np.cumsum(p[o]); sel = GRID[o[c <= 0.68]]
    width = float(sel.max() - sel.min()) if len(sel) else np.nan
    return float(GRID[p.argmax()]), width


def kfold_stable(df_tr, cands, k=4, thresh=1.0):
    """Keep features whose rho sign survives every k-fold refit."""
    d = df_tr["date_bce"].values.astype(float)
    idx = RNG.permutation(len(d))
    folds = np.array_split(idx, k)
    keep = []
    for f in cands:
        v = pd.to_numeric(df_tr[f], errors="coerce").values.astype(float)
        s0, _ = spearman(d, v)
        if not np.isfinite(s0):
            continue
        ok = 0
        for fo in folds:
            m = np.ones(len(d), bool); m[fo] = False
            s, _ = spearman(d[m], v[m])
            ok += np.isfinite(s) and np.sign(s) == np.sign(s0)
        if ok / k >= thresh:
            keep.append(f)
    return keep


def block_stable(df_tr, cands, thresh=1.0):
    """Keep features whose rho sign survives dropping any whole register group."""
    d = df_tr["date_bce"].values.astype(float)
    grps = df_tr["register"].values
    uniq = [g for g in pd.unique(grps) if isinstance(g, str) and g]
    keep = []
    for f in cands:
        v = pd.to_numeric(df_tr[f], errors="coerce").values.astype(float)
        s0, _ = spearman(d, v)
        if not np.isfinite(s0):
            continue
        ok = 0
        for g in uniq:
            m = grps != g
            s, _ = spearman(d[m], v[m])
            ok += np.isfinite(s) and np.sign(s) == np.sign(s0)
        if ok / len(uniq) >= thresh:
            keep.append(f)
    return keep


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[df["in_training"] == True].copy()               # noqa: E712
    d  = tr["date_bce"].values.astype(float)
    n  = len(tr)
    cands = [c for c in tr.columns if c not in META
             and np.isfinite(pd.to_numeric(tr[c], errors="coerce").values).sum() >= n - 2
             and np.nanstd(pd.to_numeric(tr[c], errors="coerce").values) > 1e-12]

    stat = {c: spearman(d, pd.to_numeric(tr[c], errors="coerce").values.astype(float))
            for c in cands}
    pv = pd.Series({c: stat[c][1] for c in cands}).sort_values()

    def by_p(a): return pv[pv < a].index.tolist()

    # Benjamini-Hochberg
    K = len(pv); bh_line = (np.arange(1, K + 1) / K) * 0.10
    bh = pv.index[pv.values <= bh_line].tolist()

    specs = {
        "asis_p30": by_p(0.30),
        "p10":      by_p(0.10),
        "p05":      by_p(0.05),
        "p01":      by_p(0.01),
        "bh10":     bh,
        "kfold4":   kfold_stable(tr, by_p(0.10)),
        "blockreg": block_stable(tr, by_p(0.10)),
    }

    # ── Noise-injection control ───────────────────────────────────────────────
    # Synthesise pure-noise columns, keep those that pass p<0.30 by chance,
    # and append them to the clean p<0.01 set.
    tr_n = tr.copy()
    noise_ok = []
    tries = 0
    while len(noise_ok) < 12 and tries < 4000:
        tries += 1
        col = f"NOISE_{len(noise_ok)}"
        v = RNG.standard_normal(n)
        _, p = spearman(d, v)
        if p < 0.30:
            tr_n[col] = v
            noise_ok.append(col)
    specs["p01+6noise"]  = specs["p01"] + noise_ok[:6]
    specs["p01+12noise"] = specs["p01"] + noise_ok[:12]

    rows = []
    for name, feats in specs.items():
        if len(feats) < 2:
            print(f"  [skip] {name}: only {len(feats)} features")
            continue
        src = tr_n if "noise" in name else tr
        A, B, Si = fit(src, feats)
        for wscale in (False, True):
            for t in TARGETS:
                if t not in df.index:
                    continue
                r = df.loc[t]
                base = tr_n if "noise" in name else df
                x = pd.to_numeric(
                    pd.Series({f: (r[f] if f in r.index else
                                   RNG.standard_normal())
                               for f in feats}), errors="coerce").values.astype(float)
                w = float(np.clip(r["n_words"] / 5000, 1, 5)) if wscale else 1.0
                mp, wd = post_map(x, A, B, Si, w)
                rows.append(dict(spec=name, n_feat=len(feats),
                                 wordscale=wscale, target=t,
                                 map_bce=round(mp), hdi68_width=round(wd)))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "multiverse.csv"), index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"n = {n} training units, {len(cands)} candidate features\n")
    print("Specification sizes:")
    for k, v in specs.items():
        print(f"  {k:14s} {len(v):3d} features")

    print("\n\nMAP date (BCE) by specification — word-scaling OFF\n")
    piv = (out[~out.wordscale]
           .pivot(index="target", columns="spec", values="map_bce")
           .reindex(TARGETS))
    order = [s for s in ["asis_p30","p10","p05","p01","bh10","kfold4","blockreg",
                         "p01+6noise","p01+12noise"] if s in piv.columns]
    piv = piv[order]
    print(piv.to_string())

    real = [c for c in order if "noise" not in c]
    print("\n\nStability across the seven real specifications")
    print(f"{'target':>14s}{'min':>8s}{'max':>8s}{'range':>8s}{'sd':>8s}")
    print("  " + "-" * 44)
    for t in piv.index:
        v = piv.loc[t, real].astype(float)
        print(f"{t:>14s}{v.min():>8.0f}{v.max():>8.0f}"
              f"{v.max()-v.min():>8.0f}{v.std():>8.0f}")

    print("\n\nNoise-injection control (does noise cancel and lose weight?)")
    w = out[(~out.wordscale) & (out.spec.isin(["p01","p01+6noise","p01+12noise"]))]
    wp = w.pivot(index="target", columns="spec", values="map_bce").reindex(TARGETS)
    ww = w.pivot(index="target", columns="spec", values="hdi68_width").reindex(TARGETS)
    print("\n  MAP shift when pure-noise features are added to a clean model:")
    for t in wp.index:
        b = wp.loc[t, "p01"]
        print(f"    {t:>14s}  clean={b:>6.0f}   +6noise={wp.loc[t,'p01+6noise']:>6.0f}"
              f" ({wp.loc[t,'p01+6noise']-b:+5.0f})"
              f"   +12noise={wp.loc[t,'p01+12noise']:>6.0f}"
              f" ({wp.loc[t,'p01+12noise']-b:+5.0f})")
    print("\n  68% HDI width (smaller = MORE confident):")
    for t in ww.index:
        print(f"    {t:>14s}  clean={ww.loc[t,'p01']:>5.0f}"
              f"   +6noise={ww.loc[t,'p01+6noise']:>5.0f}"
              f"   +12noise={ww.loc[t,'p01+12noise']:>5.0f}")

    print(f"\nSaved → {OUTDIR}/multiverse.csv")


if __name__ == "__main__":
    main()

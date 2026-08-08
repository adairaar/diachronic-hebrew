"""
Go/no-go diagnostic for the chunk-level design.

Three questions, in order:
  1. SPARSITY  -- at chunk size N, how many features are usable at all?
  2. ICC       -- is a feature's variance mostly between books or within them?
                  A feature whose chunks scatter as widely inside a book as
                  across books carries no book-level signal, however well it
                  correlates at book level.
  3. SIGNAL    -- of the features that survive 1 and 2, how many correlate
                  with date?
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, sys
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "sigma", "genre", "register", "n_words",
        "in_training", "id", "label"}


def icc1(x, groups):
    """One-way random-effects ICC: between-group share of total variance."""
    df = pd.DataFrame({"x": x, "g": groups}).dropna()
    if df.g.nunique() < 3 or len(df) < 10:
        return np.nan
    grand = df.x.mean()
    gm = df.groupby("g").x.agg(["mean", "count"])
    ssb = (gm["count"] * (gm["mean"] - grand) ** 2).sum()
    ssw = sum(((sub.x - sub.x.mean()) ** 2).sum() for _, sub in df.groupby("g"))
    k = df.g.nunique(); n = len(df)
    if n - k <= 0 or k - 1 <= 0:
        return np.nan
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (gm["count"] ** 2).sum() / n) / (k - 1)
    if msb + (n0 - 1) * msw <= 0:
        return np.nan
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def run(target):
    D = pd.read_csv(DH.f(f"chunk_features_{target}.csv"))
    feats = [c for c in D.columns if c not in META]
    print("=" * 78)
    print(f"CHUNK SIZE ~{target} WORDS   |   {len(D)} chunks, {D.unit.nunique()} units, "
          f"{len(feats)} candidate features")
    print(f"   median chunk length {D.n_words.median():.0f} words "
          f"(range {D.n_words.min():.0f}-{D.n_words.max():.0f})")
    print("=" * 78)

    rows = []
    for f in feats:
        v = D[f]
        frac_nan = v.isna().mean()
        frac_zero = (v.fillna(0) == 0).mean()
        usable = (frac_nan < 0.20) and (frac_zero < 0.70) and (v.std() > 0)
        ic = icc1(v.values, D.unit.values) if usable else np.nan
        # correlation with date at chunk level (pooled) and at unit level
        sub = D[["unit", "date_bce", f]].dropna()
        rho_chunk = stats.spearmanr(sub[f], sub.date_bce).statistic if len(sub) > 10 else np.nan
        um = sub.groupby("unit").agg({f: "mean", "date_bce": "first"})
        rho_unit = stats.spearmanr(um[f], um.date_bce).statistic if len(um) > 5 else np.nan
        rows.append(dict(feature=f, frac_nan=frac_nan, frac_zero=frac_zero,
                         usable=usable, icc=ic, rho_chunk=rho_chunk, rho_unit=rho_unit))
    R = pd.DataFrame(rows)

    print(f"\n1. SPARSITY")
    print(f"   features with >20% missing        : {(R.frac_nan >= 0.20).sum()}")
    print(f"   features zero in >70% of chunks   : {(R.frac_zero >= 0.70).sum()}")
    print(f"   -> usable features                : {R.usable.sum()} of {len(R)}")

    U = R[R.usable].copy()
    if not len(U):
        print("   nothing usable at this chunk size."); return R
    print(f"\n2. ICC (between-unit share of variance), usable features only")
    print(f"   median {U.icc.median():.3f}   "
          f"n with ICC>0.5 : {(U.icc > 0.5).sum()}   "
          f"ICC>0.3 : {(U.icc > 0.3).sum()}   ICC<0.1 : {(U.icc < 0.1).sum()}")

    print(f"\n3. SIGNAL among usable features")
    print(f"   |rho| with date at UNIT level  > 0.5 : {(U.rho_unit.abs() > 0.5).sum()}")
    print(f"   |rho| with date at CHUNK level > 0.3 : {(U.rho_chunk.abs() > 0.3).sum()}")

    good = U[(U.icc > 0.3) & (U.rho_unit.abs() > 0.4)].sort_values(
        "rho_unit", key=abs, ascending=False)
    print(f"\n   features with ICC>0.3 AND |rho_unit|>0.4 : {len(good)}")
    if len(good):
        print(f"   {'feature':<24}{'ICC':>7}{'rho_unit':>10}{'rho_chunk':>11}{'%zero':>8}")
        for _, r in good.head(20).iterrows():
            print(f"   {r.feature:<24}{r.icc:7.2f}{r.rho_unit:10.2f}"
                  f"{r.rho_chunk:11.2f}{r.frac_zero*100:7.0f}%")
    R.to_csv(DH.f(f"chunk_diag_{target}.csv"), index=False)
    return R


if __name__ == "__main__":
    for t in (300, 500, 1000):
        run(t)
        print()

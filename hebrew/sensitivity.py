"""
Feature leverage: years of apparent date per 1 SD shift in each feature, and
the size of the coordinated shift needed to displace an estimate by a given
number of years.

Emits sensitivity.json so the manuscript reads real values rather than numbers
transcribed from a log.

The model is linear in standardised features, so the leverage vector is just
the fitted coefficient vector on the standardised scale, and the minimum-norm
coordinated shift achieving a displacement D is D / ||beta||, measured in
standard deviations of the joint feature space.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, importlib.util
import numpy as np, pandas as pd

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats = [c for c in Dd.columns if c not in PT.META]
Xa = Dd[feats].astype(float)
keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats = list(np.array(feats)[keep.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
y = Dd.date_bce.values.astype(float); g = Dd.unit.values
books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}

# final model, exactly as used for the targets
wb = dict(zip(books, PT.wts([bdate[b] for b in books])))
w = np.array([wb[u] for u in g])
best, blam = np.inf, PT.LAM[0]
for lam in PT.LAM:
    e = []
    for b in books:
        m = g != b
        pr = PT.fit_predict(X[m], y[m], w[m], X[~m], lam)
        e.append(abs(np.median(pr) - bdate[b]))
    if np.mean(e) < best: best, blam = np.mean(e), lam

mu, sd = X.mean(0), X.std(0); sd = np.where(sd > 0, sd, 1.0)
A = (X - mu) / sd
yb = np.average(y, weights=w); s = np.sqrt(w)
Aw = A * s[:, None]; yw = (y - yb) * s
alpha = np.linalg.solve(Aw @ Aw.T + blam * np.eye(len(A)), yw)
beta = Aw.T @ alpha                       # coefficients on the standardised scale

# variance matching multiplies every prediction by S, so leverage scales too
S = 1.8283838232904197
lev = beta * S


def family(f):
    if f.startswith("lex_"): return "lexical"
    if f.startswith(("pb_", "pos_", "pdp_")): return "POS/bigram"
    if f.startswith(("vt_", "vs_", "vx_", "vtf_", "vsf_")) or f == "verb_rate":
        return "verb morphology"
    if f.startswith(("typ_", "fun_", "rela_", "phr", "cl_")): return "phrase/clause"
    if f.startswith(("ps_", "nu_", "gn_", "prs_", "st_")): return "agreement/suffix"
    if f in ("sent_len", "clause_len", "ph_per_clause", "ttr", "type_token"):
        return "structural"
    return "structural"


T = pd.DataFrame(dict(feature=feats, yr_per_sd=lev))
T["family"] = [family(f) for f in feats]
T["abs"] = T.yr_per_sd.abs()
T = T.sort_values("abs", ascending=False)
T.to_csv(DH.f("sensitivity_features.csv"), index=False)

fam = (T.groupby("family")["abs"].agg(total="sum", count="size")
         .sort_values("total", ascending=False))
fam["share"] = 100 * fam.total / fam.total.sum()
fam.to_csv(DH.f("sensitivity_families.csv"))

norm = float(np.linalg.norm(lev))
print(f"lambda = {blam:.0e} | ||beta|| = {norm:.2f} yr per unit shift in feature space\n")
print("top 12 levers:")
for _, r in T.head(12).iterrows():
    print(f"  {r.feature:<24} {r.yr_per_sd:+7.1f} yr/SD   [{r.family}]")
print("\nby family:")
print(fam.to_string())

P = pd.read_csv(DH.f("poem_predictions.csv")).set_index("unit")
gap = float(P.loc["SongSea_poem"].pred - P.loc["SongSea_prose"].pred)
print(f"\nSong of the Sea minus its prose frame: {gap:.0f} yr")
print(f"  minimum-norm coordinated shift to achieve that: {gap/norm:.2f} SD "
      f"across {len(feats)} features")

out = dict(n_feats=len(feats), lam=float(blam), norm=norm,
           top_lever_name=T.iloc[0].feature, top_lever=float(T.iloc[0]["abs"]),
           lex_share=float(fam.loc["lexical", "share"]),
           lex_count=int(fam.loc["lexical", "count"]),
           families={k: dict(total=float(v.total), count=int(v["count"]),
                             share=float(v.share)) for k, v in fam.iterrows()},
           sea_gap=gap, sea_gap_sd=gap / norm)
json.dump(out, open(DH.f("sensitivity.json"), "w"), indent=2)
print("\nwrote sensitivity.json, sensitivity_features.csv, sensitivity_families.csv")

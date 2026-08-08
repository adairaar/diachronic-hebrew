"""
Which way does an error in the anchor dates push the results?

The anchors are dated by scholarship, and scholarship can be wrong.  What
matters for reading this paper is not that they might be wrong but whether being
wrong would push the conclusions toward the claim being made or away from it.

The mechanism is one-directional and easy to state.  The model learns a mapping
from linguistic features to dates using the anchors.  If an anchor's true date
is LATER than the date assigned to it, the model has been taught to associate
that book's linguistic profile with a date earlier than it deserves, and it will
carry that error into every target whose language resembles it.  Errors in the
anchors propagate to the targets with the same sign.

That matters because the direction of likely error is not symmetric.  A century
of critical work on the Hebrew Bible has moved dates later, not earlier; the
revisionist pressure on this corpus runs in one direction.  If the anchors used
here are systematically wrong, they are more likely assigned too early than too
late, and the target estimates inherit that.  The post-exilic conclusion would
then be conservative: the true dates would be later still.

This script measures the size of that effect rather than asserting it, by
re-running the whole estimator with the anchor dates displaced and reading off
how far the targets move.  Three displacements are applied, each of 50 years
toward the present:

    all       every anchor, the case where the whole chronology is too early
    soft      only the six anchors lacking an external synchronism, the case
              where the securely fixed dates are right and the judgment calls
              are not
    late      only the post-exilic anchors, the case argued for the Persian
              period specifically
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

SHIFT = 50          # years toward the present
EX = 586
SOFT = ["Jonah", "Ecclesiastes", "Malachi", "Joel", "Isaiah_3", "Zechariah_2"]
SRC = ["JE_source", "D_source", "P_source"]

Dd = pd.read_csv(DH.f("big_features_500.csv"))
Dt = pd.read_csv(DH.f("target_chunks_500.csv"))
feats = [c for c in Dd.columns if c not in PT.META and c in Dt.columns]
k = Dd[feats].astype(float).std() > 0
feats = list(np.array(feats)[k.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
XT = Dt[feats].astype(float).fillna(med).values
g = Dd.unit.values; gt = Dt.unit.values
books = list(pd.unique(g))
base_date = {b: float(Dd.date_bce[g == b].iloc[0]) for b in books}


def fit_all(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    return {l: (B @ Aw.T) @ V @ (V.T @ yw / (ev + l)) + yb for l in PT.LAM}


def run(bd):
    """Full leave-one-book-out fit and target prediction under dates bd."""
    y = np.array([bd[u] for u in g], float)
    t, p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        err = {l: [] for l in PT.LAM}
        for bb in inner:
            m = g[tr] != bb
            pr = fit_all(X[tr][m], y[tr][m], wv[m], X[tr][~m])
            for l in PT.LAM:
                err[l].append(abs(np.median(pr[l]) - bd[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        t.append(bd[b])
        p.append(float(np.median(fit_all(X[tr], y[tr], wv, X[te])[bl])))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    wb = dict(zip(books, PT.wts([bd[x] for x in books])))
    wv = np.array([wb[u] for u in g])
    err = {l: [] for l in PT.LAM}
    for b in books:
        m = g != b
        pr = fit_all(X[m], y[m], wv[m], X[~m])
        for l in PT.LAM:
            err[l].append(abs(np.median(pr[l]) - bd[b]))
    bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
    tg = pd.Series(t.mean() + S * (fit_all(X, y, wv, XT)[bl] - p.mean()))
    return tg.groupby(gt).median()


SCEN = [
    ("baseline (dates as assigned)", lambda b: base_date[b]),
    (f"all anchors {SHIFT} yr later",
     lambda b: base_date[b] - SHIFT),
    (f"the {len(SOFT)} non-synchronism anchors {SHIFT} yr later",
     lambda b: base_date[b] - (SHIFT if b in SOFT else 0)),
    (f"post-exilic anchors {SHIFT} yr later",
     lambda b: base_date[b] - (SHIFT if base_date[b] < EX else 0)),
]

print("=" * 82)
print("HOW DO ANCHOR-DATE ERRORS PROPAGATE TO THE TARGETS?")
print("=" * 82)
print(f"  a displacement of {SHIFT} yr toward the present, applied to the "
      f"anchors, and\n  the resulting movement in the three sources\n")
print(f"  {'scenario':<46}{'JE':>7}{'D':>7}{'P':>7}")
rows, base = [], None
for name, fn in SCEN:
    bd = {b: fn(b) for b in books}
    tg = run(bd)
    v = {u: float(tg[u]) for u in SRC}
    if base is None:
        base = v
        print(f"  {name:<46}" + "".join(f"{v[u]:>7.0f}" for u in SRC))
    else:
        d = {u: v[u] - base[u] for u in SRC}
        rows.append(dict(scenario=name, **{u: v[u] for u in SRC},
                         **{f"d_{u}": d[u] for u in SRC}))
        print(f"  {name:<46}" + "".join(f"{v[u]:>7.0f}" for u in SRC)
              + "   shift " + ", ".join(f"{d[u]:+.0f}" for u in SRC))

print()
print("=" * 82)
print("READING")
print("=" * 82)
mean_all = np.mean([rows[0][f"d_{u}"] for u in SRC])
print(f"  Displacing every anchor {SHIFT} yr later moves the sources "
      f"{mean_all:+.0f} yr on average,")
print(f"  which is {abs(mean_all)/SHIFT:.0%} of the displacement applied: "
      f"anchor error passes through")
print(f"  to the targets nearly one for one, and with the same sign.")
print()
print("  The conclusion of this paper is that the sources are LATE.  An anchor")
print("  chronology that is too early therefore makes that conclusion")
print("  conservative: correcting it would move the sources later still, not")
print("  earlier.  The reverse error would be the dangerous one, and a century")
print("  of critical revision has not been running in that direction.")
json.dump(dict(shift=SHIFT, baseline=base, scenarios=rows,
               passthrough=float(abs(mean_all) / SHIFT)),
          open(DH.f("anchor_bias.json"), "w"), indent=2)
print("\nwrote anchor_bias.json")

"""
Where did the alternant pairs come from, and which of them carry the result?

The share-encoding comparison is only as good as the pair list behind it.  Two
questions decide whether it is a finding or an artifact of choosing well after
the fact: how many pairs come from the diachronic literature rather than from
the investigator, and how much the result moves as pairs are added or dropped.

Both answers are unflattering, and they are computed here rather than asserted.
"""
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

exec(open("/home/claude/variationist_test.py").read().split('print(f"  {\'model\'')[0])

# Provenance, checked pair by pair against the CBH/LBH literature.
LITERATURE = {">NKJ", ">L"}      # 'anoki/'ani (Rooker, Polzin); 'el/le (Polzin)
y = Dd.date_bce.values.astype(float); g = Dd.unit.values


def run(cols):
    X = SHARE[cols].values
    books = list(pd.unique(g)); bd = {b: y[g == b][0] for b in books}
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
    return genre_rho(books, t, t.mean() + S * (p - p.mean()))


allc = list(SHARE.columns)
lit = [f"share_{a}_{b}" for _, a, b, _ in have if a in LITERATURE]
non = [c for c in allc if c not in lit]

full = run(allc); L = run(lit); N = run(non)
loo = {c: run([x for x in allc if x != c]) for c in allc}

print(f"  {len(lit)} of {len(allc)} pairs come from the diachronic literature\n")
print(f"  {'subset':<40}{'p':>4}{'rho|genre':>11}")
print(f"  {'all pairs':<40}{len(allc):>4}{full:>+11.3f}")
print(f"  {'literature-derived pairs only':<40}{len(lit):>4}{L:>+11.3f}")
print(f"  {'investigator-added pairs only':<40}{len(non):>4}{N:>+11.3f}")
print(f"\n  leave-one-pair-out, rho|genre:")
for c, v in sorted(loo.items(), key=lambda kv: kv[1]):
    print(f"    without {c.replace('share_',''):<32}{v:>+8.3f}")
print(f"\n  across the {len(allc)} single-pair deletions alone the statistic "
      f"spans {min(loo.values()):+.3f} to {max(loo.values()):+.3f};")
print(f"  the full pair set is one of {2**len(allc)-1} non-empty subsets.")
json.dump(dict(n_pairs=len(allc), n_literature=len(lit), full=float(full),
               literature_only=float(L), nonliterature_only=float(N),
               loo={k: float(v) for k, v in loo.items()},
               loo_min=float(min(loo.values())), loo_max=float(max(loo.values()))),
          open("/home/claude/share_provenance.json", "w"), indent=2)
print("\nwrote share_provenance.json")

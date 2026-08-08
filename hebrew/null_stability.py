"""
Each permutation p-value as a function of where the run was stopped.

Permutation runs are expensive and were stopped at declared counts.  A reviewer
is entitled to ask whether stopping elsewhere would have said something
different, and the honest answer is a table rather than an assurance.  Both
nulls are recomputed here on every prefix of their own draw sequence.
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
import numpy as np, pandas as pd


GC = json.load(open(DH.f("genre_confound.json")))

W = np.loadtxt(DH.f("within_genre_null.csv"), delimiter=",", skiprows=1, ndmin=2)
F = np.loadtxt(DH.f("final_lobo_null.csv"), delimiter=",", skiprows=1, ndmin=2)
OBS_W = GC["rho_partial"]
OBS_F = GC["rho_raw"]

rows = []
for k in (50, 100, 150, 200, 250, 300):
    if k <= len(W):
        rows.append(dict(test="within-genre", draws=k,
                         p=float((np.sum(W[:k, 0] >= OBS_W) + 1) / (k + 1))))
for k in (100, 200, 300, 500, 750, 1000):
    if k <= len(F):
        rows.append(dict(test="free shuffle", draws=k,
                         p=float((np.sum(F[:k, 0] >= OBS_F) + 1) / (k + 1))))

D = pd.DataFrame(rows)
D.to_csv(DH.f("null_stability.csv"), index=False)
json.dump(D.to_dict("records"), open(DH.f("null_stability.json"), "w"), indent=2)

for t, sub in D.groupby("test", sort=False):
    print(f"{t:<14} " + "  ".join(f"{int(r.draws)}={r.p:.4f}"
                                  for _, r in sub.iterrows()))
print(f"\nobserved statistics: within-genre rho|g {OBS_W:+.3f}, "
      f"raw rho {OBS_F:+.3f}")
print(f"{len(W)} within-genre draws and {len(F)} free-shuffle draws on disk")
print("wrote null_stability.csv, null_stability.json")

"""
Each permutation p-value as a function of where the run was stopped.

Permutation runs are expensive and were stopped at declared counts.  A reviewer
is entitled to ask whether stopping elsewhere would have said something
different, and the honest answer is a table rather than an assurance.  Both
nulls are recomputed here on every prefix of their own draw sequence.
"""
import json
import numpy as np, pandas as pd

R = "/home/claude"
GC = json.load(open(f"{R}/genre_confound.json"))

W = np.loadtxt(f"{R}/within_genre_null.csv", delimiter=",", skiprows=1, ndmin=2)
F = np.loadtxt(f"{R}/final_lobo_null.csv", delimiter=",", skiprows=1, ndmin=2)
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
D.to_csv(f"{R}/null_stability.csv", index=False)
json.dump(D.to_dict("records"), open(f"{R}/null_stability.json", "w"), indent=2)

for t, sub in D.groupby("test", sort=False):
    print(f"{t:<14} " + "  ".join(f"{int(r.draws)}={r.p:.4f}"
                                  for _, r in sub.iterrows()))
print(f"\nobserved statistics: within-genre rho|g {OBS_W:+.3f}, "
      f"raw rho {OBS_F:+.3f}")
print(f"{len(W)} within-genre draws and {len(F)} free-shuffle draws on disk")
print("wrote null_stability.csv, null_stability.json")

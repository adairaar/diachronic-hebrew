"""
Finalize the within-genre permutation null from its checkpoint.

The run was stopped deliberately rather than allowed to reach its nominal 500
draws.  At a p-value near 0.02 the standard error on 230 draws is about 0.010,
which distinguishes the observed value from anything the decision turns on; the
remaining draws would have bought precision on a number whose interpretation
does not change across it.  Stating the stopping point explicitly matters more
than the extra draws would have.
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
import numpy as np

N = np.loadtxt(DH.f("within_genre_null.csv"), delimiter=",",
               skiprows=1, ndmin=2)
GC = json.load(open(DH.f("genre_confound.json")))
OBS = [("genre-controlled rho", GC["rho_partial"], 0),
       ("raw rho", GC["rho_raw"], 1),
       ("rho within prophecy", GC["rho_prophecy"], 2)]
n = len(N)
out = dict(n=n)
print(f"{n} draws, dates permuted within genre\n")
print(f"  {'statistic':<24}{'observed':>10}{'null median':>14}{'p':>9}{'se':>8}")
for name, obs, col in OBS:
    p = float((np.sum(N[:, col] >= obs) + 1) / (n + 1))
    se = float(np.sqrt(p * (1 - p) / n))
    key = ["partial", "raw", "prophecy"][col]
    out[f"obs_{key}"] = float(obs); out[f"p_{key}"] = p
    out[f"se_{key}"] = se
    out[f"null_{key}_med"] = float(np.median(N[:, col]))
    print(f"  {name:<24}{obs:>+10.3f}{np.median(N[:, col]):>+14.3f}"
          f"{p:>9.4f}{se:>8.4f}")
out["stopped_early"] = True
json.dump(out, open(DH.f("within_genre_null.json"), "w"), indent=2)
print("\n  Raw rho barely moves under this null, which is the point: a shuffle")
print("  within genre leaves the genre structure it scores intact, so it is")
print("  the wrong statistic for a dating claim.")
print("\nwrote within_genre_null.json")

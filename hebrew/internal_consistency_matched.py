"""
The dispersion comparison, with passage count held constant.

The first pass found that dispersion rises with the number of passages a unit
has (rho = +0.56 across the anchors, p = 0.02).  That is expected -- a standard
deviation from 7 passages and one from 106 are not the same measurement -- but it
makes the raw comparison unusable, because the three sources have 74, 39 and 106
passages.  D looked like the most homogeneous source and also has the fewest
passages, which is exactly the confound.

Fix: subsample every unit to a common passage count and recompute.  Two
matchings, because no single n serves both comparisons.

  n = 39   the three sources against each other; 39 is D's count, the smallest
           of the three, so nothing is discarded from D and JE and P are cut
           down to meet it.
  n = 12   everything, including anchor books and sub-units, so the reference
           distribution of single dated books can be rebuilt on equal footing.

Sampling is without replacement within a draw, so each subsample is a genuine
set of distinct passages rather than a bootstrap resample.
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

RNG = np.random.default_rng(23)
B = 4000
EX = 586

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

# reuse the chunk-level predictions already computed
import subprocess, os
Dd = pd.read_csv(DH.f("big_features_500.csv"))
Dt = pd.read_csv(DH.f("target_chunks_500.csv"))
C = pd.read_csv(DH.f("internal_consistency.csv"))
if not os.path.exists(DH.f("chunk_preds.csv")):
    raise SystemExit("run internal_consistency.py first (it caches chunk_preds.csv)")
P = pd.read_csv(DH.f("chunk_preds.csv"))
anc = P[P.kind == "anchor"]; tgt = P[P.kind == "target"]


def sub_sd(v, n, B=B):
    """Distribution of the SD of n passages drawn without replacement."""
    v = np.asarray(v, float)
    if len(v) < n: return None
    return np.array([np.std(RNG.choice(v, n, replace=False), ddof=1)
                     for _ in range(B)])


print("=" * 80)
print("1.  THE THREE SOURCES AT EQUAL PASSAGE COUNT (n = 39)")
print("=" * 80)
SRC = ["JE_source", "D_source", "P_source"]
N1 = 39
S = {}
print(f"  {'source':<12}{'all passages':>14}{'SD at n=39':>14}{'95% CI':>18}")
for u in SRC:
    v = tgt.cpred[tgt.unit == u].values
    d = sub_sd(v, N1)
    S[u] = d
    q = np.percentile(d, [2.5, 97.5])
    print(f"  {u.split('_')[0]:<12}{len(v):>10}{'':4}{d.mean():>11.0f} yr"
          f"{f'{q[0]:.0f} - {q[1]:.0f}':>18}")
print()
for a, b in [("JE_source", "P_source"), ("JE_source", "D_source"),
             ("D_source", "P_source")]:
    d = S[a] - S[b]
    print(f"  P(SD[{a.split('_')[0]}] > SD[{b.split('_')[0]}]) = "
          f"{np.mean(d > 0):.3f}   difference {d.mean():+.0f} yr "
          f"(95% CI {np.percentile(d,2.5):+.0f} to {np.percentile(d,97.5):+.0f})")

print()
print("=" * 80)
print("2.  EVERYTHING AT n = 12, AGAINST THE ANCHOR YARDSTICK")
print("=" * 80)
N2 = 12
ref = {}
for b, sub in anc.groupby("unit"):
    d = sub_sd(sub.cpred.values, N2, B=800)
    if d is not None: ref[b] = float(d.mean())
ref_sd = np.array(list(ref.values()))
r_n, p_n = stats.spearmanr([len(anc[anc.unit == b]) for b in ref], ref_sd)
print(f"  {len(ref)} anchor books with at least {N2} passages")
print(f"  within-book SD at n=12: median {np.median(ref_sd):.0f} yr, "
      f"range {ref_sd.min():.0f}-{ref_sd.max():.0f}")
print(f"  residual correlation with passage count: rho {r_n:+.3f} (p={p_n:.2f})"
      f"  -- {'still confounded' if p_n < .05 else 'confound removed'}")

UNITS = [("JE_source", "JE composite"), ("Gen_JE", "  Genesis JE"),
         ("Exo_JE", "  Exodus JE"), ("D_source", "D source"),
         ("D_Code", "  law code, Deut 12-26"),
         ("D_Frame", "  frame, Deut 1-11 + 27-34"),
         ("P_source", "P source"), ("Lev_Priestly", "  Leviticus 1-16"),
         ("Lev_Holiness", "  Holiness Code, Lev 17-26"),
         ("Jer_DTR", "Jeremiah Dtr prose")]
print()
print(f"  {'unit':<28}{'n':>5}{'SD at n=12':>13}{'percentile vs anchors':>24}")
rows = []
for u, lab in UNITS:
    v = tgt.cpred[tgt.unit == u].values
    d = sub_sd(v, N2, B=800)
    if d is None:
        print(f"  {lab:<28}{len(v):>5}   fewer than {N2} passages")
        continue
    m = float(d.mean()); pc = float(np.mean(ref_sd <= m))
    rows.append(dict(unit=u, label=lab.strip(), n=len(v), sd12=m, pctile=pc))
    print(f"  {lab:<28}{len(v):>5}{m:>10.0f} yr{pc*100:>21.0f}%")
pd.DataFrame(rows).to_csv(DH.f("internal_consistency_matched.csv"),
                          index=False)

print()
print("=" * 80)
print("3.  VERDICT")
print("=" * 80)
o = sorted(SRC, key=lambda u: -S[u].mean())
print(f"  predicted:  JE > D > P")
print(f"  observed:   " + " > ".join(x.split("_")[0] for x in o))
json.dump(dict(n_matched=N1,
               sd39={u: float(S[u].mean()) for u in SRC},
               ci39={u: [float(np.percentile(S[u], 2.5)),
                         float(np.percentile(S[u], 97.5))] for u in SRC},
               p_JE_gt_P=float(np.mean(S["JE_source"] - S["P_source"] > 0)),
               p_JE_gt_D=float(np.mean(S["JE_source"] - S["D_source"] > 0)),
               p_D_gt_P=float(np.mean(S["D_source"] - S["P_source"] > 0)),
               ref_median_sd12=float(np.median(ref_sd)),
               units=rows, order=[x.split("_")[0] for x in o]),
          open(DH.f("internal_consistency_matched.json"), "w"), indent=2)
print("\nwrote internal_consistency_matched.csv/.json")

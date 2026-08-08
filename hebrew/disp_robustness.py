"""
Does the within-source dispersion ordering survive the genre screen?

Companion to block_robustness.py, which asks the same of the separations
BETWEEN blocks.  This asks it of dispersion WITHIN a source: is JE really more
internally varied than P, and D less than both, once the most genre-diagnostic
features are removed?

Dispersion rises with passage count, so every source is subsampled to a common
n before comparison.  The machinery -- feature screen, leave-one-book-out
chunk-level prediction, calibration -- is imported from block_robustness rather
than duplicated, so the two analyses cannot drift apart.
"""
import json
import numpy as np

# block_robustness defines the screen, the estimator and run(); executing its
# preamble gives them here without rerunning its own analysis section
_src = open("/home/claude/block_robustness.py").read().split('print("=" * 86)')[0]
exec(_src)

RNG2 = np.random.default_rng(31)
SRC = ["JE_source", "D_source", "P_source"]
N_MATCH = 39            # D's passage count, the smallest of the three
B = 3000

print("=" * 78)
print("DOES THE DISPERSION ORDERING SURVIVE THE GENRE SCREEN?")
print("=" * 78)
print(f"Within-source SD of passage estimates, subsampled to a common "
      f"n = {N_MATCH}.\n")

out = []
for frac, lab in [(1.0, "all features"), (0.75, "drop top 25%"),
                  (0.5, "drop top 50%")]:
    nfe, rho_g, tp = run(frac)
    sds = {}
    for u in SRC:
        v = tp[tp.index == u].values
        sds[u] = np.array([np.std(RNG2.choice(v, N_MATCH, replace=False), ddof=1)
                           for _ in range(B)])
    order = " > ".join(u.split("_")[0]
                       for u in sorted(SRC, key=lambda u: -sds[u].mean()))
    print(f"{lab}  ({nfe} feats, rho|genre {rho_g:+.3f})")
    for u in SRC:
        print(f"    {u.split('_')[0]:<4} SD {sds[u].mean():>5.0f} yr")
    p_d = float(np.mean(sds["JE_source"] - sds["D_source"] > 0))
    p_p = float(np.mean(sds["JE_source"] - sds["P_source"] > 0))
    print(f"    ordering {order}   P(JE>D)={p_d:.3f}  P(JE>P)={p_p:.3f}\n")
    out.append(dict(frac=frac, n_feats=nfe, rho_genre=rho_g, order=order,
                    p_JE_gt_D=p_d, p_JE_gt_P=p_p,
                    **{u.split("_")[0]: float(sds[u].mean()) for u in SRC}))

json.dump(out, open("/home/claude/disp_robustness.json", "w"), indent=2)
print("=" * 78)
print("VERDICT")
print("=" * 78)
orders = {o["order"] for o in out}
print(f"  ordering {'holds at every screen level' if len(orders) == 1 else 'CHANGES: ' + ', '.join(sorted(orders))}")
print(f"  D least dispersed at every level: "
      f"{'yes' if all(o['p_JE_gt_D'] > 0.9 for o in out) else 'no'} "
      f"(min P(JE>D) = {min(o['p_JE_gt_D'] for o in out):.3f})")
print(f"  JE most dispersed at every level: "
      f"{'yes' if all(o['order'].startswith('JE') for o in out) else 'no'}")
print("\nwrote disp_robustness.json")

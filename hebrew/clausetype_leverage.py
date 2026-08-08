"""
How strongly does each clause type correlate with date on its own?

The Data section argues that the clause-type family is not used because its
information is already carried by the verb-morphology and part-of-speech
features, not because the family is uninformative.  That argument needs the
single-feature correlations to be reported, and they must come from the matrix
where the clause-type features actually work.

BHSA encodes the Hebrew verbal system at clause level -- Way0 for narrative
wayyiqtol, WQt0 for weqatal, xQt0 for fronted qatal, NmCl for verbless.  An
earlier extractor counted clause TYPE but indexed with clause RELA keys, whose
value sets are disjoint, leaving every one of those columns structurally zero.
big_features_500_ctyp.csv is the same extraction with that fixed; it is used
here and nowhere else in the pipeline, because the reported model does not
include the family.
"""
import json
import numpy as np, pandas as pd
from scipy import stats

R = "/home/claude"
GLOSS = {"ctyp_xYq0": "fronted yiqtol", "ctyp_ZIm0": "imperative",
         "ctyp_Voct": "vocative", "ctyp_InfC": "infinitive construct",
         "ctyp_WQt0": "weqatal", "ctyp_Way0": "wayyiqtol",
         "ctyp_NmCl": "verbless", "ctyp_Ptcp": "participial",
         "ctyp_xQt0": "fronted qatal", "ctyp_xYqX": "fronted yiqtol, object",
         "ctyp_WxY0": "waw + fronted yiqtol"}

D = pd.read_csv(f"{R}/big_features_500_ctyp.csv")
ct = [c for c in D.columns if c.startswith("ctyp_")]
B = D.groupby("unit").agg({**{c: "mean" for c in ct}, "date_bce": "first"})
r = {c: stats.spearmanr(B[c], B.date_bce)[0] for c in ct if B[c].std() > 0}
top = sorted(r.items(), key=lambda kv: -abs(kv[1]))

print(f"{len(r)} clause-type features, book-level Spearman with true date\n")
print(f"  {'feature':<20}{'rho':>8}   gloss")
for k, v in top:
    print(f"  {k:<20}{v:>+8.3f}   {GLOSS.get(k, '')}")

out = dict(top_feature=GLOSS.get(top[0][0], top[0][0]),
           top_rho=abs(float(top[0][1])),
           second_feature=GLOSS.get(top[3][0], top[3][0]),
           second_rho=abs(float(top[3][1])),
           n_above_half=int(sum(abs(v) > 0.5 for v in r.values())),
           n_ctyp=len(r),
           weqatal_rho=abs(float(r.get("ctyp_WQt0", np.nan))),
           all_top=[{"f": GLOSS.get(k, k), "rho": float(v)} for k, v in top[:6]])
json.dump(out, open(f"{R}/clausetype_leverage.json", "w"), indent=2)
print(f"\n{out['n_above_half']} of {out['n_ctyp']} exceed |rho| = 0.5, led by "
      f"{out['top_feature']} at {out['top_rho']:.2f}")
print("wrote clausetype_leverage.json")

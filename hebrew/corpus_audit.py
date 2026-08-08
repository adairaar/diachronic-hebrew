"""
Corpus-level descriptive statistics quoted in the Data section.

These numbers -- how long the passages actually are, how the feature families
divide up -- were previously computed ad hoc and pasted into result files by
hand.  That is exactly the failure mode this pipeline is meant to exclude, so
they are computed here from the feature matrix and written to the same files the
manuscript reads.
"""
import json
import numpy as np, pandas as pd

R = "/home/claude"
META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}
TARGET = 500
BAND = (400, 700)

D = pd.read_csv(f"{R}/big_features_{TARGET}.csv")
w = D.n_words.values
out = dict(min=int(w.min()), med=int(np.median(w)), max=int(w.max()),
           mean=float(w.mean()),
           pct_in_band=float(100 * np.mean((w >= BAND[0]) & (w <= BAND[1]))),
           n_single=int((D.groupby("unit").size() == 1).sum()),
           single=sorted(D.groupby("unit").size()[
               lambda x: x == 1].index.tolist()),
           n_chunks=len(D), n_units=int(D.unit.nunique()))
json.dump(out, open(f"{R}/chunk_sizes.json", "w"), indent=2)
print(f"passage lengths: min {out['min']}, median {out['med']}, max {out['max']}")
print(f"  {out['pct_in_band']:.0f}% within {BAND[0]}-{BAND[1]} words; "
      f"{out['n_single']} units of a single passage")

feats = [c for c in D.columns if c not in META]
live = [c for c in feats if D[c].std() > 0]


def family(c):
    if c.startswith("lex_"): return "n_lexical"
    if c.startswith(("pos_", "pb_", "pdp_")): return "n_POS/bigram"
    if c.startswith(("vt_", "vs_", "vx_", "vtf_", "vsf_", "verb_")): return "n_verb"
    if c.startswith(("typ_", "fun_", "rela_", "phr", "ph_")): return "n_phrase"
    if c.startswith(("ps_", "nu_", "gn_", "st_", "prs_")): return "n_agreement"
    return "n_structural"


fam = {}
for c in live:
    fam[family(c)] = fam.get(family(c), 0) + 1
# "used" is kept as an alias for "live" because the manuscript macro reads it
fc = dict(extracted=len(feats), dead=len(feats) - len(live), live=len(live),
          used=len(live), **fam)
json.dump(fc, open(f"{R}/feature_counts.json", "w"), indent=2)
print(f"\nfeatures: {fc['extracted']} extracted, {fc['dead']} structurally "
      f"constant, {fc['live']} used")
for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<16}{v:>5}")
print("\nwrote chunk_sizes.json, feature_counts.json")

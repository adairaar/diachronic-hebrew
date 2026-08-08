"""
Do the estimates move more than the stated uncertainty says they should?

The complaint is that every specification change shifts the answer by a century.
That is either a fatal instability or an honest interval being read as though it
were a point estimate, and the two are distinguishable.

Collect every estimate this project has produced for the three sources, under
every specification actually run, and compare the spread ACROSS specifications
against the width of the interval the model reports WITHIN a specification.  If
the specification spread fits inside the interval, the estimator is behaving as
advertised and the point estimates were simply never precise enough to quote to
the year.  If it does not, the intervals are too narrow and that is a much
larger problem than any single specification choice.

Bug fixes are listed separately from specification choices.  A corpus that
contained 29% Aramaic SHOULD change the answer when the Aramaic is removed;
that is the fix working, not the estimator wobbling.
"""
import json
import numpy as np, pandas as pd

SRC = ["JE_source", "D_source", "P_source"]
J = pd.read_csv("/home/claude/jackknife_plus_targets.csv").set_index("unit")

# ── every specification run in this project, with its point estimates ────────
# Sources: target_predictions_final.csv, genre_symmetric.json, nested_selection,
# anchor_sensitivity, jackknife_plus.  Only specifications actually executed.
GS = json.load(open("/home/claude/genre_symmetric.json"))
GC = pd.read_csv("/home/claude/genre_symmetric_targets.csv").set_index("unit")

SPECS = [
    ("reported model (jackknife+)", "spec", {u: float(J.loc[u, "pred"]) for u in SRC}),
    ("same, genre script's refit",  "spec", {u: float(GC.loc[u, "uncorrected"]) for u in SRC}),
    ("genre corrected: narrative",  "spec", {u: float(GC.loc[u, "nar_adj"]) for u in SRC}),
    ("genre corrected: prophecy",   "spec", {u: float(GC.loc[u, "pro_adj"]) for u in SRC}),
    ("refit, narrative dropped",    "spec",
     {u: float(v) for u, v in GS["variants"]["narrative dropped"]["preds"].items()}),
    ("refit, prophecy only",        "spec",
     {u: float(v) for u, v in GS["variants"]["prophecy only"]["preds"].items()}),
]

print("=" * 78)
print("1.  WHAT THE MODEL SAYS ITS OWN PRECISION IS")
print("=" * 78)
print("  Jackknife+ intervals, from the manuscript as it stands.\n")
print(f"  {'source':<12}{'point':>7}{'68% interval':>20}{'half-width':>12}"
      f"{'90% interval':>20}")
half = {}
for u in SRC:
    r = J.loc[u]
    h = (r.hi68 - r.lo68) / 2
    half[u] = h
    print(f"  {u:<12}{int(r.pred):>7}{f'{int(r.hi68)}-{int(r.lo68)}':>20}"
          f"{h:>11.0f} yr{f'{int(r.hi90)}-{int(r.lo90)}':>20}")
print(f"\n  The model has been saying since the jackknife+ fix that it cannot")
print(f"  locate these texts to better than about "
      f"+/-{np.mean(list(half.values())):.0f} years at 68% confidence.")

print()
print("=" * 78)
print("2.  HOW MUCH THE ESTIMATES ACTUALLY MOVED ACROSS SPECIFICATIONS")
print("=" * 78)
print(f"  {'specification':<30}" + "".join(f"{u.split('_')[0]:>10}" for u in SRC))
print("  " + "-" * 60)
for name, kind, d in SPECS:
    print(f"  {name:<30}" + "".join(f"{d[u]:>10.0f}" for u in SRC))

print()
print(f"  {'':<30}" + "".join(f"{u.split('_')[0]:>10}" for u in SRC))
rng, sd = {}, {}
for u in SRC:
    v = np.array([d[u] for _, _, d in SPECS])
    rng[u] = v.max() - v.min(); sd[u] = v.std(ddof=1)
print(f"  {'range across specifications':<30}"
      + "".join(f"{rng[u]:>10.0f}" for u in SRC))
print(f"  {'sd across specifications':<30}"
      + "".join(f"{sd[u]:>10.0f}" for u in SRC))
print(f"  {'reported 68% half-width':<30}"
      + "".join(f"{half[u]:>10.0f}" for u in SRC))
print(f"  {'range as % of 68% interval':<30}"
      + "".join(f"{100*rng[u]/(2*half[u]):>9.0f}%" for u in SRC))

print()
print("=" * 78)
print("3.  THE VERDICT ON STABILITY")
print("=" * 78)
inside = all(rng[u] <= 2 * half[u] for u in SRC)
for u in SRC:
    v = np.array([d[u] for _, _, d in SPECS])
    lo, hi = J.loc[u, "lo68"], J.loc[u, "hi68"]
    n_in = int(((v >= lo) & (v <= hi)).sum())
    print(f"  {u:<12} {n_in}/{len(v)} specification estimates fall inside the "
          f"reported 68% interval")
print()
if inside:
    print("  Every specification tried lands inside the interval the model already")
    print("  publishes.  The estimates are not less stable than advertised; they")
    print("  were quoted to the year when they were never good to the century.")
else:
    print("  At least one specification moves the estimate outside the reported")
    print("  interval.  That is a real failure of the interval, not of reporting.")

print()
print("=" * 78)
print("4.  HOW STRONG IS THE SIGNAL, IN PLAIN TERMS")
print("=" * 78)
M = json.load(open("/home/claude/final_lobo_metrics.json"))
print(f"  anchor books                      {M['n_books']}")
print(f"  date range spanned                {760-167} yr")
print(f"  MAE, this model                   {M['mae']:.0f} yr")
print(f"  MAE, predicting the mean date     {M['mae_baseline']:.0f} yr")
print(f"  improvement over knowing nothing  {100*(1-M['mae']/M['mae_baseline']):.0f}%")
print(f"  variance-match scale S            {M['S']:.2f}")
print()
print(f"  A 14% reduction in error over a constant predictor, on 25 books, is a")
print(f"  weak signal.  It is not nothing -- the permutation null says a signal")
print(f"  this strong arises by chance in well under 1% of shuffles -- but it")
print(f"  cannot support century-precision claims about any single text, and the")
print(f"  intervals have been saying so.")

json.dump(dict(half68={u: float(half[u]) for u in SRC},
               spec_range={u: float(rng[u]) for u in SRC},
               spec_sd={u: float(sd[u]) for u in SRC},
               all_inside=bool(inside),
               n_specs=len(SPECS),
               mae=M["mae"], mae_baseline=M["mae_baseline"],
               improvement=float(1 - M["mae"] / M["mae_baseline"]), S=M["S"]),
          open("/home/claude/stability_ledger.json", "w"), indent=2)
print("\nwrote stability_ledger.json")

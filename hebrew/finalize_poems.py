"""
Attach the jackknife+ intervals to the poem table.

The poems are scored inside jackknife_plus.py's own folds, alongside every other
undated unit, so their intervals come from the same estimator as the sources.
This script only joins those intervals onto poem_predictions.csv, which carries
the word counts and the three-way extraction the Results section needs.

An earlier version of this script computed the intervals here instead, from the
poem's point estimate and the residual ensemble.  That is a different estimator:
jackknife+ requires the per-fold predictions aligned with the per-fold
residuals, and substituting a scalar produced intervals about a third too
narrow -- in the direction that makes the poems look better evidenced than they
are.
"""
import pandas as pd

R = "/home/claude"
P = pd.read_csv(f"{R}/poem_predictions.csv").set_index("unit")
J = pd.read_csv(f"{R}/jackknife_plus_targets.csv").set_index("unit")

COLS = ["lo68", "hi68", "lo90", "hi90", "p_post"]
before = float((P.hi68 - P.lo68).mean())

hit = [u for u in P.index if u in J.index]
missing = [u for u in P.index if u not in J.index]
for u in hit:
    for c in COLS:
        if c in J.columns:
            P.loc[u, c] = J.loc[u, c]

P.reset_index().to_csv(f"{R}/poem_predictions.csv", index=False)
after = float((P.loc[hit].hi68 - P.loc[hit].lo68).mean())
print(f"{len(hit)}/{len(P)} poem units given jackknife+ intervals from the folds")
if missing:
    print(f"  NOT scored in the folds (interval left as-is): {', '.join(missing)}")
print(f"  mean 68% width {before:.0f} -> {after:.0f} yr")
for u in ("SongSea_poem", "SongSea_chapter", "SongDeborah_poem", "SongMoses_poem"):
    if u in P.index:
        r = P.loc[u]
        print(f"  {u:<19}{r.pred:>6.0f}  {r.hi68:.0f}-{r.lo68:.0f} BCE   "
              f"P(post) {r.p_post:.2f}")

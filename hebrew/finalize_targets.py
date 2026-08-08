"""
Assemble the published table of undated units.

This step existed only as a sequence of manual edits until now, which is why the
table in the manuscript could not be reproduced by running the pipeline: a fresh
clone got the estimator's own symmetric intervals and 19 rows, where the paper
reports jackknife+ intervals and 23.  Making it a script is the point.

Three sources are combined:

  target_predictions_naive.csv   point estimates for every undated unit, with
                                 the estimator's symmetric +/- q68 interval
  jackknife_plus_targets.csv     jackknife+ intervals and post-exilic
                                 probabilities for the same units, which are
                                 what the manuscript reports
The poems are NOT merged in here.  Song_Sea, Song_Deborah and D_Song in this
table are the verse-precise poems, extracted by target_chunks.py like every
other unit.  The three-way comparison the Results section needs -- whole
chapter against poem proper against prose remainder -- lives in
poem_predictions.csv, which the manuscript reads directly.  Keeping the two
apart means this table has one row per undated unit and no duplicates.
"""
import pandas as pd

R = "/home/claude"

naive = pd.read_csv(f"{R}/target_predictions_naive.csv").set_index("unit")
jk = pd.read_csv(f"{R}/jackknife_plus_targets.csv").set_index("unit")
out = naive.copy()

# jackknife+ intervals supersede the symmetric ones wherever available
INT = ["pred", "lo68", "hi68", "lo90", "hi90", "p_post"]
n_repl = 0
for u in out.index:
    if u in jk.index:
        for c in INT:
            if c in jk.columns:
                out.loc[u, c] = jk.loc[u, c]
        n_repl += 1

out = out.reset_index().rename(columns={"index": "unit"})
out.to_csv(f"{R}/target_predictions_final.csv", index=False)

print(f"{len(naive)} units from the estimator")
print(f"  {n_repl} given jackknife+ intervals")
print(f"-> target_predictions_final.csv, {len(out)} rows")
w = (out.hi68 - out.lo68).abs().mean()
print(f"   mean 68% interval width {w:.0f} yr")

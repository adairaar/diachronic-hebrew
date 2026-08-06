"""Insert the post-exilic calibration analysis (Fig 3) into Validation."""
import pathlib, sys

P = pathlib.Path("/home/claude/paper")
p = P / "main_new.tex"
t = p.read_text(); n = 0

def rep(old, new, label):
    global t, n
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (v3d): {label}")
    t = t.replace(old, new); n += 1

CAL = (P / "new_calibration.tex").read_text().rstrip("\n")

FIG = r"""
\begin{figure}[!ht]
\includegraphics[width=\linewidth]{figures/fig3_calibration.png}
\caption{\textbf{Calibration of the post-exilic probability.}
\textbf{(A)}~Every dated unit: true date against the probability the model
assigns to it being post-exilic, from a fully nested leave-one-out in which
the unit contributes neither to its own point estimate nor to the residual
set that gives that estimate its spread.  The transition is sharp far from
the 586~BCE boundary and flat near it, which is the expected behaviour of an
honest forecast under $\sim$150~yr of error.  The shaded band marks where
the P source estimate falls.  Open markers are confident calls
($P \geq 0.8$ or $\leq 0.2$) that are wrong; there is one, Nahum under
ridge.  \textbf{(B)}~Reliability: mean predicted probability against
observed frequency, in three bins, with the corpus base rate of 0.68 shown
for comparison.  Marker area is proportional to bin count.  Points near the
diagonal and above the base-rate line indicate forecasts that are both
calibrated and informative.}
\label{fig:calibration}
\end{figure}
"""

# place the new material at the end of Validation, before the Results section
anchor = "\\section*{Results: dating the undated units}"
if anchor not in t:
    sys.exit("Results section header not found")
t = t.replace(anchor, CAL + "\n" + FIG + "\n" + anchor); n += 1

# forward pointer from the post-exilic claim itself
rep("""The finding is robust in three respects that matter.""",
    """That the probability means what it says is not assumed: it is tested
against the dated corpus under \\nameref{sec:postexilic_cal}, where confident
calls are correct on 9 of 9 units (generative) and 8 of 9 (ridge), with
Brier skill $+0.20$ and $+0.19$ against the corpus base rate.

The finding is robust in three further respects.""",
    "forward pointer from Results")

# abstract: one clause, since this is the answer to the obvious objection
rep("particular post-exilic period.  The Song of the Sea and Song of Deborah",
    "particular post-exilic period.  That probability is itself validated: on\n"
    "the dated corpus, confident post-exilic calls are correct on 17 of 18\n"
    "instances across the two families, with positive Brier skill against the\n"
    "base rate ($p = 0.005$, $0.014$).  The Song of the Sea and Song of Deborah",
    "abstract calibration clause")

p.write_text(t)
print(f"v3d: {n} edits")

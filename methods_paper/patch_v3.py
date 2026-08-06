"""
Author decisions, 2026-08-06 (round 3).

1. Cut the HB-VI model and prior-sensitivity sections entirely.  Their
   absolute MAPs disagree with the corrected estimates by ~400 yr on a
   ~800 yr range, which makes them noise.  A diagnostic derived from a
   model we do not trust cannot certify a model we do.
2. Cut the model-independent feature score (unstable across corpus
   versions; pending author decision on whether to restore).
3. Remove first-person self-criticism from the manuscript.  The failure
   mode is presented prospectively, as a property of holdout designs,
   not as a confession.
4. New title.
"""
import pathlib, re, sys

p = pathlib.Path("/home/claude/paper/main_new.tex")
t = p.read_text(); n = 0

def rep(old, new, label):
    global t, n
    if old not in t:
        sys.exit(f"ANCHOR NOT FOUND (v3): {label}")
    t = t.replace(old, new); n += 1

def cut_section(start_marker, end_marker, label):
    """Remove everything from start_marker up to (not including) end_marker."""
    global t, n
    i = t.find(start_marker); j = t.find(end_marker)
    if i < 0 or j < 0 or j <= i:
        sys.exit(f"SECTION BOUNDS NOT FOUND (v3): {label}")
    t = t[:i] + t[j:]; n += 1

# ── 1. cut HB-VI and prior sensitivity wholesale ─────────────────────────────
cut_section("\\section*{The HB-VI model}", "\\section*{Validation}",
            "HB-VI + prior sensitivity + feature scoring")

# ── 2. cross-references in surviving prose ───────────────────────────────────
rep(r"""  \item The \hbvi{} model: hierarchical Bayesian regression with
    mean-field variational inference (MFVI) and ADAM
    optimization~\cite{Kingma2014,Blei2017}, providing proper parameter
    uncertainty propagation and soft register assignment.
""", "", "contribution list HB-VI item")

rep("extraction.  The MLE-MVN and HB-VI sections present the two Bayesian\nmodels.",
    "extraction.  The MLE-MVN section presents the generative dating model and\na ridge regression fitted alongside it.",
    "roadmap models sentence")

rep(r"""Third, the prior sensitivity analysis (see the Prior Sensitivity Analysis
section) explicitly quantifies""",
    r"""Third, the precision decomposition reported under \nameref{sec:validation}
explicitly quantifies""",
    "background circularity ref (a)")

rep("how much of each \\hbvi{} date is determined by the scholar-specified",
    "how much of a reported date is determined by the scholar-specified",
    "background circularity ref (b)")

rep("The \\mvn{} and \\hbvi{} models instead compute a posterior probability",
    "The models used here instead compute a posterior probability",
    "background posterior sentence")

rep(r"""can in principle be used to set register priors in \hbvi{}, and
conversely \hbvi{} date posteriors constrain the plausible compositional""",
    r"""can in principle be used to set register priors, and
conversely the date estimates reported here constrain the plausible compositional""",
    "HC register-prior sentence")

# ── 3. corpus: the training/holdout split is vestigial under LOO ─────────────
rep("""Two units are held out of both models and excluded from all feature
screening: the Jeremiah oracular stratum (chs.~1--6, 8--10, 12--16, 19--20,
22--23, 30--31, 46--51) and Haggai.  Whole-book Jeremiah is excluded from
training entirely, because it contains the oracular stratum and would
otherwise leak the holdout into the training set.  The Deuteronomistic prose
stratum of Jeremiah (chs.~7, 11, 17--18, 21, 24--29, 32--45, 52) is treated
as an undated target and never trained on.  Habakkuk and Daniel are retained
in MLE-MVN training but held out of HB-VI.""",
    """All analyses are leave-one-out over the full set of 25 dated units, so
every unit is held out in turn and no unit contributes to its own estimate.
The corpus therefore has no separate holdout stratum.  Whole-book Jeremiah is
excluded entirely and replaced by its oracular stratum (chs.~1--6, 8--10,
12--16, 19--20, 22--23, 30--31, 46--51), since the whole book contains that
stratum; the Deuteronomistic prose stratum (chs.~7, 11, 17--18, 21, 24--29,
32--45, 52) is treated as an undated target and never trained on.""",
    "holdout paragraph")

rep(r"""A note on what ``held out'' must mean.  In the analyses reported below, a
held-out unit contributes nothing to feature screening, to standardisation,
to regression fitting, or to its own prior.  The last of these is not a
formality.  A model that receives a text's scholarly date as a prior mean
will return approximately that date whatever the text contains, and will do
so most emphatically for the texts whose dates are best established --- that
is, precisely the texts on which a validation exercise appears most
convincing.  Section~\nameref{sec:validation} quantifies this effect for the
present corpus.  All out-of-sample results in this paper use an agnostic
prior.""",
    r"""A note on what ``held out'' must mean.  In the analyses reported below, a
held-out unit contributes nothing to feature screening, to standardisation,
to regression fitting, or to its own prior.  The last of these is not a
formality.  A model that receives a text's scholarly date as a prior mean
will return approximately that date whatever the text contains, and will do
so most emphatically for the texts whose dates are best established --- that
is, precisely the texts on which a validation exercise appears most
convincing.  \nameref{sec:validation} quantifies the effect on this corpus.
All out-of-sample results in this paper use an agnostic prior.""",
    "held-out paragraph ref")

# ── 4. discussion passages that lean on the cut sections ─────────────────────
rep(r"standard.  The sensitivity analysis (Fig.~\ref{fig:prior_sensitivity})",
    r"standard.  The decomposition of Table~\ref{tab:leakage}",
    "discussion sensitivity ref")

rep(r"""The prior sensitivity analysis replaces this philosophical standoff with""",
    r"""The precision decomposition replaces this philosophical standoff with""",
    "discussion standoff (a)")

rep(r"the data-driven classification from the prior sensitivity analysis.",
    r"the share of posterior precision attributable to the linguistic evidence.",
    "discussion classification ref")

rep(r"""760--167~BCE; the \hbvi{} training subset, which excludes Daniel, spans
760--330~BCE, so \hbvi{} dates near the late end are already mild""",
    r"""760--167~BCE, so estimates near either end of that span are already mild""",
    "extrapolation range")

rep(r"""\hbvi{} --- a strong HC assignment to D/DtrH would motivate an SBH""",
    r"""the dating model --- a strong HC assignment to D/DtrH would motivate an SBH""",
    "HC future work (a)")

rep(r"""then apply \hbvi{} with that informed prior to obtain""",
    r"""then apply the dating model with that informed prior to obtain""",
    "HC future work (b)")

p.write_text(t)
print(f"v3 structural pass: {n} edits")

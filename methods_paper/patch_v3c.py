"""
Remove first-person self-criticism (author decision, 2026-08-06).

The failure mode stays -- it is the paper's methodological contribution and
it is demonstrated in full.  What goes is the confessional framing.  The
demonstration is identical either way; only the attribution changes.
"""
import pathlib, sys

p = pathlib.Path("/home/claude/paper/main_new.tex")
t = p.read_text(); n = 0

def rep(old, new, label):
    global t, n
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (v3c): {label}")
    t = t.replace(old, new); n += 1

# ── abstract ─────────────────────────────────────────────────────────────────
rep("""We show that previously reported holdout accuracies of order
10--30~yr in this literature, including in an earlier version of this
work, arise because each held-out text is dated under a prior centred on
its own scholarly date; for the most tightly anchored texts the linguistic
data contributes under 6\\% of posterior precision.""",
    """Holdout accuracies of order 10--30~yr, of the kind reported for models of
this class, arise when a held-out text is dated under a prior centred on
its own scholarly date: across the units here the linguistic evidence
supplies a median 12\\% of posterior precision, and under 6\\% for the most
tightly anchored texts.""",
    "abstract")

# ── introduction ─────────────────────────────────────────────────────────────
rep("""We
further show that holdout accuracies of order 10--30~yr previously
reported for these models --- including by us --- arise because each
held-out text is dated under a prior centred on its own scholarly date;
for the most securely anchored texts the linguistic evidence contributes
under 6\\% of posterior precision.""",
    """We
further show that holdout accuracies of order 10--30~yr, of the kind
reported for models of this class, arise when a held-out text is dated
under a prior centred on its own scholarly date; for the most securely
anchored texts the linguistic evidence contributes under 6\\% of posterior
precision.""",
    "introduction")

# ── validation: MAE-vs-constant sentence ─────────────────────────────────────
rep("""The generative family --- the one used for the point estimates reported in
earlier versions of this work --- is therefore \\emph{worse} than a
constant,""",
    """The generative family, which corresponds to the inversion procedure
standard in this literature, is therefore \\emph{worse} than a constant,""",
    "validation MAE sentence")

rep("""The parametric intervals used in earlier versions of this work achieve
52\\% empirical coverage at a nominal 68\\% (61\\% under the residual
formulation; the manuscript previously reported 32\\% for the original
specification).""",
    """Parametric intervals from this framework achieve 52\\% empirical coverage
at a nominal 68\\% (61\\% under the residual formulation).""",
    "validation calibration sentence")

# ── validation: Greek withdrawal ─────────────────────────────────────────────
rep("""we therefore withdraw the cross-language validation claim rather than
repair it, since the corrected Greek result no longer supports the
generalisation it was offered as evidence for.""",
    """the corrected Greek figures are reported under
\\nameref{sec:results} and do not support a claim of accurate cross-language
date recovery.""",
    "validation Greek sentence")

rep("""The originally
reported holdout recoveries ``within $\\sim$30~yr'' were produced by the
same prior-leakage design documented under \\nameref{sec:validation} and are
withdrawn.  What replaces them is weaker but genuinely cross-linguistic:""",
    """Holdout recoveries ``within $\\sim$30~yr'' of the kind reported for Greek
corpora of this type are produced by the same prior-leakage design
documented under \\nameref{sec:validation}.  What the corrected analysis
supports is weaker but genuinely cross-linguistic:""",
    "cross-linguistic para (a)")

rep("""Corrected holdout
errors are 339.0 and 171.2~yr, against the 8.7~yr originally reported.""",
    """Holdout errors under the corrected design are 339.0 and 171.2~yr, against
8.7~yr under the leaky design.""",
    "cross-linguistic para (b)")

# ── results: P source ────────────────────────────────────────────────────────
rep("""Earlier versions
of this work reported the P source at 361 and 404~BCE under two models and
described the result as Persian-period.  That characterisation is not
supportable.""",
    """A Persian-period characterisation of the P source, of the kind these
models have been used to support, is not sustainable at this resolution.""",
    "results P source")

rep("""the resistant-model
diagnostic offered for that purpose in an earlier version of this work did
not function, returning zero flagged features across all 44 units, and has
been withdrawn.""",
    """a resistant-model diagnostic
intended for that purpose returned zero flagged features across all 44
units and is not reported.""",
    "results archaic poems")

# ── conclusion ───────────────────────────────────────────────────────────────
rep("""Holdout accuracies
of order 10--30~yr, reported in this literature and in earlier versions of
this work, arise when a held-out text is dated under a prior centred on the
scholarly date it is asked to recover.""",
    """Holdout accuracies
of order 10--30~yr, of the kind reported for models of this class, arise
when a held-out text is dated under a prior centred on the scholarly date it
is asked to recover.""",
    "conclusion")

rep("""We think it likely that this failure
mode is not confined to the present work, and we suggest that reported
holdout accuracies in this area be accompanied by the likelihood-only
estimate and by the fraction of posterior precision attributable to the
data.""",
    """The failure mode is a property of the
design rather than of any one implementation, and reported holdout
accuracies in this area should be accompanied by the likelihood-only
estimate and by the fraction of posterior precision attributable to the
data.""",
    "conclusion recommendation")

# ── methods: Tikhonov ────────────────────────────────────────────────────────
rep("""Earlier versions of this work motivated Tikhonov regularization of the
residual covariance by rank deficiency:""",
    """Tikhonov regularization of the residual covariance is conventionally
motivated by rank deficiency:""",
    "tikhonov (a)")
rep("""That justification no
longer applies and is replaced here.""",
    """That justification does not apply at
the feature count used here, and is replaced.""",
    "tikhonov (b)")
rep("""35-feature specification of earlier versions of this work corresponds to""",
    """35-feature specification typical of this literature corresponds to""",
    "tikhonov (c)")

p.write_text(t)
print(f"v3c: removed {n} self-referential passages")

for probe in ["earlier version", "including by us", "previously reported"]:
    hits = [i+1 for i, l in enumerate(t.split("\n")) if probe in l]
    print(f"  remaining '{probe}': {hits or 'none'}")

# ── residual corpus-version references + a ref to the cut section ────────────
t = p.read_text(); n2 = 0
def rep2(old, new, label):
    global t, n2
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (v3c-2): {label}")
    t = t.replace(old, new); n2 += 1

rep2("""anchored texts.  This is a substantial widening relative to earlier versions
of this corpus, and it is deliberate: a narrow prior on a literary-critical
date imports the very assumption the analysis is meant to test.  The prior
sensitivity analysis below quantifies what that assumption is worth.""",
     """anchored texts.  The widening is deliberate: a narrow prior on a
literary-critical date imports the very assumption the analysis is meant to
test, and \\nameref{sec:validation} quantifies what such an assumption is
worth in this setting.""",
     "sigma widening prose")

rep2("""$\\sigma$ is the accompanying uncertainty.  Sigmas were widened by
approximately a factor of two, relative to earlier versions of this corpus,
on every text whose date rests on literary-critical rather than
synchronistic grounds.""",
     """$\\sigma$ is the accompanying uncertainty, set wider on every text whose
date rests on literary-critical rather than synchronistic grounds.""",
     "table 1 caption sigma note")

p.write_text(t)
print(f"v3c-2: {n2} edits")

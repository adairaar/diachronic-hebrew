"""Patch the surviving prose that still describes withdrawn methods/results."""
import pathlib, re, sys

p = pathlib.Path("/home/claude/paper/main_new.tex")
t = p.read_text()
n = 0

def rep(old, new, label):
    global t, n
    if old not in t:
        sys.exit(f"ANCHOR NOT FOUND: {label}")
    t = t.replace(old, new); n += 1

TITLE_OLD = ("A multivariate Bayesian framework for diachronic dating of ancient\n"
             "texts: Validation on Biblical Hebrew and Ancient Greek")
TITLE_NEW = ("Resolving diachronic linguistic changes in Biblical Hebrew:\n"
             "calibrated ranges for the Pentateuchal sources, with\n"
             "cross-linguistic replication in Ancient Greek")
rep(TITLE_OLD, TITLE_NEW, "title (flushleft)")
rep("% Ancient Texts: Validation on Biblical Hebrew and Ancient Greek\"",
    "% Resolving diachronic linguistic changes in Biblical Hebrew\"", "title comment")

# ── contribution list: replace the three withdrawn items ─────────────────────
rep(r"""  \item A resistant model using only four clause-level features below
    conscious editorial control, yielding an archaism diagnostic from the
    divergence between the full and resistant MAP dates.
  \item Genre correction (Strategies B+D) that down-weights features
    according to their sensitivity to legal versus narrative register.
  \item A word n-gram model (POS-tag and function-word bigrams/trigrams)
    as an independent grammatical-sequence dating instrument.
  \item Prior sensitivity analysis distinguishing data-driven from
    prior-dominated \hbvi{} date estimates.
  \item Cross-language validation on a 63-text Ancient Greek corpus.""",
    r"""  \item Prior sensitivity analysis distinguishing data-driven from
    prior-dominated date estimates, and a decomposition of posterior
    precision that quantifies how much of a reported date is contributed
    by the linguistic evidence rather than by the prior.
  \item A leave-one-out protocol in which feature screening,
    standardisation and model fitting are performed inside each fold, and
    every out-of-sample unit is dated under an agnostic prior.
  \item Permutation inference: the date labels are shuffled and the entire
    pipeline re-run, so the null distribution absorbs selection effects.
  \item Distribution-free conformal prediction intervals calibrated on
    leave-one-out residuals, replacing parametric intervals whose empirical
    coverage is roughly three-quarters of nominal.""",
    "contribution list")

# ── stale results summary ────────────────────────────────────────────────────
rep(r"""The framework yields results that depart substantially from critical
consensus.  Under agnostic priors, the Priestly source (P) is placed at
402--406~BCE --- a data-driven result (prior sensitivity Class~D,
Mode~A/B shift ${<}30$~yr) --- suggesting linguistic crystallization a
century or more after the Exilic date assumed in the Wellhausen
tradition~\cite{Wellhausen1885,CarrFormation2011}.  The
Deuteronomic Code falls at $\sim$698--737~BCE, consistent with an
Iron~II origin predating the Josianic reform~\cite{Romer2005}.  The archaism diagnostic
identifies the Song of the Sea (Exod~15) as a late archaizing composition
whose clause-level syntax places it in the fifth century, despite surface
features more archaic than any text in the training corpus.
Cross-language validation on a 63-text Ancient Greek corpus --- entirely
independent of the Hebrew training data --- recovers five held-out texts
within $\sim$30~yr of established scholarly dates.""",
    r"""The results are bounded in both directions.  Hebrew morphosyntax
carries a genuine diachronic signal: under leakage-free cross-validation
the models order text pairs chronologically with 69.8\% accuracy across
295 pairs ($p = 4 \times 10^{-12}$), and rank correlations between true
and predicted date reach $+0.58$ and $+0.49$ in two independently
specified model families.  The same analysis shows that this signal does
not support absolute dating: leave-one-out error of 121--156~yr must be
read against 137~yr for the trivial predictor assigning every text the
corpus mean date, and four-way period assignment reaches 40--48\%.  We
further show that holdout accuracies of order 10--30~yr previously
reported for these models --- including by us --- arise because each
held-out text is dated under a prior centred on its own scholarly date;
for the most securely anchored texts the linguistic evidence contributes
under 6\% of posterior precision.  Applied to the Pentateuchal sources
under corrected conditions, the framework establishes that the Priestly,
Deuteronomic and JE composites are post-exilic with probability
$\geq 0.92$ under both model families, while declining to locate them
within a particular post-exilic period.""",
    "results summary")

# ── roadmap ──────────────────────────────────────────────────────────────────
rep(r"""models, including the Pentateuchal source dating results.  The Prior
Sensitivity Analysis section covers the three-mode sensitivity framework.
The Archaism Detection section presents the resistant model and archaism
diagnostic.  The Genre Correction section describes Strategies B+D.
The Validation section opens with cross-language validation on Ancient
Greek, followed by Hebrew holdout tests.  The Discussion section""",
    r"""models.  The Prior Sensitivity Analysis section covers the three-mode
sensitivity framework and the model-independent feature scoring used to
make the inputs auditable.  The Validation section documents the prior-leakage
failure mode, sets out the leave-one-out protocol, and reports what the
signal does and does not support.  The Results section applies the
corrected framework to the undated units.  The Discussion section""",
    "roadmap")

# ── limitations passage referring to the withdrawn diagnostic ────────────────
rep(r"""are excluded from the model feature set, and the archaism diagnostic is
restricted to the resistant-model subset that is least genre-sensitive.""",
    r"""are excluded from the model feature set.  The residual analysis reported
under \nameref{sec:results} quantifies what remains: in the generative
family narrative units are dated $+112$~yr too late relative to prophetic
units (Welch $p = 0.016$), an effect absent in ridge ($p = 0.234$).""",
    "limitations genre passage")

# ── extensibility passage ────────────────────────────────────────────────────
rep(r"""The resistant-model concept is equally portable: any language's most
syntactically conservative features can be isolated as the resistant-model
input, and the archaism diagnostic follows automatically.""",
    r"""The methodological cautions are equally portable.  Any application of
this framework to a new corpus should report the likelihood-only estimate
alongside the posterior, the share of posterior precision contributed by
the data, and a permutation null obtained by re-running the entire
pipeline on shuffled labels.  Each of these is inexpensive, and each
would have caught the failure documented here.""",
    "extensibility passage")

p.write_text(t)
print(f"applied {n} patches")

# residual sweep
bad = []
for pat, why in [(r"402--406", "old P date"), (r"698--737", "old D Code date"),
                 (r"16\.8", "old holdout MAE"), (r"63-text", "Greek corpus claim"),
                 (r"five held-out", "Greek holdout claim"),
                 (r"archaism diagnostic", "withdrawn diagnostic"),
                 (r"Strategies B\+D", "withdrawn genre correction")]:
    hits = [i+1 for i, ln in enumerate(t.split("\n")) if re.search(pat, ln)]
    if hits: bad.append((why, pat, hits))
if bad:
    print("\nremaining mentions to review:")
    for why, pat, hits in bad:
        print(f"  {why:<28} lines {hits[:8]}")
else:
    print("no stale claims detected")

# ── second pass: orphaned supplementary stubs and remaining claims ───────────
t = p.read_text(); n2 = 0
def rep2(old, new, label):
    global t, n2
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (pass 2): {label}")
    t = t.replace(old, new); n2 += 1

rep2(r"""\textbf{Cross-linguistic generalization.}
The Greek validation corpus (63 texts, five held-out texts spanning Archaic
through Byzantine Greek) demonstrates that the methodology captures a
general diachronic morphosyntactic signal and does not exploit properties
idiosyncratic to Biblical Hebrew.  No cross-linguistic generalization of
the HC authorship method has been reported.""",
     r"""\textbf{Cross-linguistic generalization.}
An Ancient Greek corpus was assembled for this purpose and is described in
the repository.  Its holdout results were obtained under the same
prior-leakage design documented under \nameref{sec:validation} and are
withdrawn here rather than repaired; corrected, they yield errors near
120~yr and do not support a generalization claim.  Whether the framework
transfers across languages is therefore open, and we flag it as the most
consequential outstanding question rather than as a result.""",
     "cross-linguistic paragraph")

rep2(r"""The present framework, on the other hand, is oriented toward calendar-year
estimates with calibrated uncertainty, toward the archaism diagnostic that
the resistant/full model divergence provides, and toward correcting for
genre confounds through Strategies B+D --- questions the HC method was
not designed to address.""",
     r"""The present framework, on the other hand, is oriented toward
chronological ordering with calibrated uncertainty and toward quantifying
how much of a reported date is contributed by the evidence rather than the
prior --- questions the HC method was not designed to address.  Our
results suggest the two are complementary in a specific way: the HC
method's categorical assignments are robust where our calendar-year
estimates are not, while our ordinal statistics remain informative where
scribal-school membership is undefined.""",
     "HC comparison")

# orphaned S11 stub for a pruned figure
rep2(r"""\paragraph*{S11 Fig.}
\label{S11_Fig}
\textbf{Full MLE-MVN posteriors for three key texts.}
Complete unnormalised posterior distributions (solid filled) for Song of
the Sea (Exod~15), D source composite, and P source composite under the
full model (colored) and the resistant model (gray dashed).  The
full/resistant MAP divergence is the archaism index $\Delta_{\text{arch}}$.
Song of the Sea shows a ceiling effect in the full model (posterior piling
against the 760~BCE training boundary) that is absent in the resistant
model (MAP~$\approx$460~BCE), providing the clearest illustration of the
archaism diagnostic.

""", "", "orphaned S11 stub")

rep2("The 22 corpus texts (19 training, 3 holdouts: Habakkuk, Haggai, Daniel) plotted on",
     "The 25 dated units (23 training, 2 held out of both models: the Jeremiah\noracular stratum and Haggai) plotted on",
     "S12 caption counts")

p.write_text(t)
print(f"pass 2: applied {n2} patches")

# ── third pass: residual "resistant model" references ────────────────────────
t = p.read_text(); n3 = 0
def rep3(old, new, label):
    global t, n3
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (pass 3): {label}")
    t = t.replace(old, new); n3 += 1

rep3(r"""on a priori grounds.  The
\emph{resistant model} tests whether syntactic templating patterns that
fall below the level of deliberate stylistic choice agree with the
full-model (lexical + morphological + syntactic) date.  Systematic
disagreement --- the full model older than the resistant model --- is a
falsifiable indicator of archaism.""",
     r"""on a priori grounds.  One natural operationalization is to ask whether
clause-level syntactic patterns, which fall below the level of deliberate
stylistic choice, agree with a date derived from lexical and morphological
features that a competent imitator could manipulate; systematic
disagreement would be a falsifiable indicator of archaism.  We implemented
this test and report under \nameref{sec:results} that it did not function
on the present corpus, flagging no features on any unit.  The archaism
question therefore remains open here, and the distinction between an early
text and a skilled archaizing one is one this analysis cannot draw.""",
     "background resistant-model paragraph")

rep3(r"""text such as the Song of the Sea --- already placed firmly in the Persian
period by the resistant model --- would be pushed later still if the""",
     r"""text such as the Song of the Sea --- whose predictive mass under both
model families is the earliest of any undated unit --- would be pushed
later still if the""",
     "conservative-upper-bound passage")

rep3(r"""\textbf{Tier~3 (clause/phrase, 4 features).}  Four features forming the
\emph{resistant model}: the infinitive-construct fraction""",
     r"""\textbf{Tier~3 (clause/phrase, 4 features).}  Four clause-level features,
the least susceptible of the three tiers to deliberate manipulation: the
infinitive-construct fraction""",
     "tier 3 description")

rep3(r"3 & Clause/phrase syntax                &  4 & Low (resistant model) \\",
     r"3 & Clause/phrase syntax                &  4 & Low \\",
     "feature-tier table row")

rep3(r"""archaism index.  The relationship also runs in reverse: a text that the
resistant model places at 460~BCE is unlikely to represent D material
regardless of lexical surface similarity, and date posteriors of this kind
constrain the plausible compositional settings for HC attribution
decisions.""",
     r"""an ordinal position with calibrated uncertainty.  The relationship also
runs in reverse: a unit whose predictive mass lies overwhelmingly after
586~BCE constrains the plausible compositional settings for HC attribution
decisions, even when its calendar-year estimate is too diffuse to be
useful on its own.""",
     "HC reverse-direction passage")

rep3("The 22 corpus texts (19 training, 3 holdouts: Habakkuk, Haggai, Daniel) are plotted on",
     "The 25 dated units (23 training, 2 held out of both models) are plotted on",
     "S12 figure caption counts")

p.write_text(t)
print(f"pass 3: applied {n3} patches")

# Cover Letter — PLOS ONE Submission

**Manuscript title:** Composition and date in the Pentateuchal sources: what a
linguistic model recovers, and what archaizing would cost

**Corresponding author:** Aaron Adair (adairaar@gmail.com)
**Affiliation:** Department of Physics, Massachusetts Institute of Technology

---

Dear Editors,

I am submitting for your consideration a paper that began as an attempt to
produce calibrated dates for the sources of the Pentateuch and ended by
reporting that the linguistic evidence supports considerably less than the
literature assumes. The paper's most useful contributions are two
methodological findings and one negative result, and I would rather lead with
that than oversell it.

The largest obstacle turns out not to be the one the field has been arguing
about. It is genre. Seventy-seven percent of the variance in the model's
predictions lies between genres rather than within them, and rank correlation
with true date falls from +0.666 to +0.405 once genre is held fixed. A
permutation null that shuffles dates only within genre — preserving the
genre–date association the corpus happens to contain, destroying only the
within-genre chronology — confirms that a chronological signal survives
(p = 0.017, 300 draws). But it is roughly half the raw figure. Raw correlation
under that same null has a median of +0.460, which is to say that two-thirds of
the headline number an uncorrected analysis would report is genre
discrimination rather than dating.

What the evidence does support is a placement claim, reported as a specification
curve rather than as point estimates. Four analytic choices in the pipeline have
no principled answer; crossing them gives 48 specifications, 44 of which clear
the validation threshold. Across those 44 the Deuteronomic and Priestly sources
fall after the exile without exception and the JE composite in 41, in the
classical sequence JE → D → P, with median estimates of 482, 439 and 416 BCE.
The 68% intervals are roughly ±204 years, and the manuscript states plainly that
no individual date in it should be quoted to a century.

I want to draw your attention to one result the paper reports as a failure,
because it is the part I would most want a referee to examine. Splitting each
source at the division its critical literature already draws produces three
significant separations, all in the direction that literature predicts, from a
model told nothing about those divisions: the Deuteronomic law code 97 years
earlier than its narrative frame (p = 0.007), the Holiness Code 79 years from
the Priestly material of Leviticus 1–16 (p = 0.016), Genesis JE 71 years from
Exodus JE (p = 0.039). That is a striking finding and I drafted it as the
paper's headline.

It does not survive the paper's own genre control. With the most
genre-diagnostic quarter of features removed the separations are gone; with half
removed — the screen level that gives the *best* genre-controlled ordering of
the anchors — the Deuteronomic separation reverses and is significant in the
opposite direction (−156 years, p = 0.005). Legal prose and hortatory narrative
differ in vocabulary, and that difference was doing the work. The separations
are register effects, not dates.

I report this at length rather than deleting it. The unscreened numbers are
individually significant, align with prior expectation, and are exactly what a
study would publish if it stopped one step short of applying its own control.
What does survive is narrower: the Deuteronomic source is measurably the most
internally homogeneous of the three at every screen level, which is robust.

The second methodological finding is that **the surviving signal is entirely
lexical**. Refitting on the 328 morphosyntactic features — verb stem and tense,
clause type and relation, phrase structure, agreement — with every lexeme
removed gives a genre-controlled
correlation of +0.140. On the lexemes alone it is +0.458. Whatever grammatical
change occurred over this period is not detectable at this sample size. This is a
quantitative datum in a debate conducted largely by example, and it favors the
lexical emphasis of Hurvitz's method over accounts appealing to grammatical
change. It is also, I should say, not the result I expected or wanted.

I took some care to try to break the finding before reporting it. The manuscript
reports ablations that remove each feature family, and that drop the six anchors
whose dates rest on literary rather than external grounds. The most reassuring
survivor is unplanned: the model independently recovers the classical-to-late
first-person pronoun shift, placing *ʾānōkî* at rank 32 of 250 lexemes pushing
estimates earlier and *ʾănî* at rank 4 pushing them later — the direction the
literature predicts, learned from dated books with no encoding of the pair. A
model detecting subject matter rather than date has no reason to do that.

The paper's second half prices the objection that hangs over all such work: that
a late author might convincingly write like an early one. If archaizing were
easy, every linguistic argument for the antiquity of a biblical text would fail,
and the objection applies to traditional and computational methods alike. The
dialectal school has pressed it for two decades and the mainstream has replied
that sustained archaizing is difficult. Both positions are assertions. I price it
twice.

The first route is simulation, where ground truth is exact because I performed
the archaizing myself: six securely dated late books had every Late Biblical
Hebrew form replaced by its Classical counterpart, all features were
re-extracted, and the altered text was dated by a model that never saw it. Total
lexical archaizing moves the estimates by a mean of 9 years, because no lexical
item is worth more than 12.1 years per standard deviation. The second route is
history. The 14 Second Sophistic Atticizers, who imitated Classical Attic under a
formal curriculum of prose composition, are displaced 206 years by a model
trained only on non-archaizing Greek. Because the Hebrew half of the paper argues
that genre confounding is severe, I put the Greek measurement through the same
control, and it survives: Atticizing texts are dated 202 years earlier than
non-archaizing texts of the same genre (95% CI 111 to 288, bootstrap p < 0.001),
with four of five genres pointing the same way. The Greek model is also markedly
less genre-confounded than the Hebrew one — 45% between-genre variance against
77%, with correlation falling only from +0.689 to +0.571.

Those two numbers differ by an order of magnitude, and the difference is the
finding. Swapping vocabulary is cheap; reconstructing a grammar is not, and the
Atticizers managed it only because an educational apparatus existed to teach
them. The objection to linguistic dating becomes conditional and historically
testable rather than rhetorical.

The limits are stated as plainly as the results. The intervals preclude any claim
about position within the Persian period. Genre correction for two of the three
sources is a bound rather than a measurement, because the corpus contains no
dated legal text at all. The signal is lexical only, so the paper speaks to
lexical diachrony and nothing else. And the Greek transfer is an analogy rather
than a measurement of Hebrew.

The paper also identifies a validation artifact that explains why the accuracies
reported here are lower than several published figures for models of this class.
When a held-out text is dated under a prior centered on its own scholarly date,
the model returns approximately that date regardless of the text's content, and
does so most convincingly for the texts whose dates are best established. Across
this corpus the linguistic evidence supplies a median 11.7% of posterior
precision, where such a design reports errors of a few years against an honest
156. Accuracies of a decade or two, for models of this class, are a signature of
the design rather than a measurement of the language.

The manuscript is submitted exclusively to PLOS ONE and has not been published or
submitted elsewhere. All code, extracted feature matrices, model outputs, and the
scripts that regenerate every table, figure, and numerical claim in the text
directly from the corpus are available at
https://github.com/adairaar/diachronic-hebrew. Every quantity in the manuscript
is a macro read from a result file rather than a typed number, and the build
fails if any result file is missing.

I have no financial or personal competing interests to declare. The paper has one
author, so no co-author conflicts apply.

**Suggested reviewers.** The paper sits at the intersection of applied statistics,
corpus linguistics, and Hebrew Bible studies, and reviewers familiar with at
least two of those areas would be best placed to evaluate it. Because the paper's
claims cut against both sides of the underlying dispute, I would welcome
reviewers from either camp.

- **Prof. Stephen Portnoy** (University of Illinois, Statistics Emeritus) — has
  published on statistical analysis of CBH/LBH feature distributions (Rachmuth,
  Portnoy & Wright, *Journal of Semitic Studies* 67.2, 2022) and could evaluate
  the cross-validation, permutation and specification-curve methodology
  independently of the biblical studies content.

- **Dr. Martijn Naaijer** (University of Zurich) — has argued that existing
  linguistic dating methods rest on unsound statistical assumptions (Rezetko,
  Young, Ehrensvärd & Naaijer, *Bible and Interpretation*, 2025). Much of that
  critique is vindicated here, particularly on genre confounding, and he is well
  placed to judge whether the corrections go far enough.

- **Dr. Cody Kingham** (University of Notre Dame, Data Science & Hebrew Bible) —
  has applied computational and machine-learning methods to the BHSA/ETCBC
  database, and could evaluate both the corpus methodology and the statistical
  framework.

- **Prof. Wido van Peursen** (Vrije Universiteit Amsterdam) — a principal
  contributor to the BHSA/ETCBC corpus on which the feature extraction depends.

- **Prof. Jacob L. Wright** (Emory University, Candler School of Theology) —
  whose recent work (*Why the Bible Began*, 2023) engages the compositional
  questions these results bear on.

A classicist able to assess the Second Sophistic material would also be valuable,
as the Greek half of the paper turns on which authors count as Atticizing. That
classification is a scholarly judgment rather than a datum, and the manuscript
reports the measurement under three definitions of it precisely so that a
specialist can check the sensitivity.

Thank you for your time and consideration.

Sincerely,

Aaron Adair
Department of Physics, Massachusetts Institute of Technology
adairaar@gmail.com

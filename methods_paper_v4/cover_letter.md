# Cover Letter — PLOS ONE Submission

**Manuscript title:** Calibrated date ranges for the Pentateuchal sources, and a
direct measurement of what archaizing can buy: Biblical Hebrew and Ancient Greek

**Corresponding author:** Aaron Adair (adairaar@gmail.com)
**Affiliation:** Department of Physics, Massachusetts Institute of Technology

---

Dear Editors,

I am submitting for your consideration a paper that measures the price of
archaizing in two ancient languages, and then uses a model of known sensitivity
to that price to date the sources of the Pentateuch.

Linguistic dating of ancient texts rests on an assumption that has never been
measured: that a late author cannot convincingly write like an early one. Should
that assumption fail, every argument for the antiquity of a biblical text fails
with it. The dialectal school has pressed this objection for two decades, and the
diachronic mainstream has replied that sustained archaizing is difficult. Both
positions are assertions, and neither side has put a number on the matter. The
central contribution of this paper is that number, arrived at twice by
independent routes.

The first route is simulation, where the ground truth is known exactly because I
performed the archaizing myself. Six securely dated late books had every Late
Biblical Hebrew form replaced by its Classical counterpart at controlled rates,
after which all 579 features were re-extracted and the altered text was dated by
a model that had never seen it. Total lexical archaizing, well beyond what any
scribe sustains, moves the estimated dates by a mean of only 11 years. The reason
is visible in the coefficients: no lexical item is worth more than 12.2 years per
standard deviation, and the model's remaining leverage sits in high-frequency
morphology and syntax.

The second route is history. Ancient Greek supplies what Hebrew cannot, namely
authors who really did archaize, deliberately and expertly, with securely known
dates of composition. Under a model trained only on non-archaizing Greek, the 14
Second Sophistic Atticizers are displaced 216 years, with 13 of them placed
earlier than they were written, and Cassius Dio off by half a millennium. The
figure grows to 306 and 412 years under stricter definitions of the training set.

These two numbers differ by an order of magnitude, and that difference is the
finding. Swapping archaic vocabulary is cheap. Reconstructing an archaic grammar
is not, and the Atticizers managed it only because an entire educational
apparatus existed to teach them how. The objection to linguistic dating therefore
becomes conditional and historically testable rather than rhetorical: for any
disputed text, the question is whether its milieu supported the expensive kind of
imitation.

The dating model itself is built on 146,059 words of Hebrew, of which nineteen
of the twenty-five units are anchored by synchronisms external to the biblical
tradition and six by literary context; the manuscript reports how every result
behaves when the latter are removed. Its one consequential design
choice is that features are extracted from passages of about 500 words rather
than from whole books, which turns 25 observations into 285 and forces the model
onto grammatical rates that can actually be measured in a few hundred words.
Under leave-one-book-out validation it gives a mean absolute error of 114 years
against 137 for the constant predictor, a rank correlation of +0.70, and correct
chronological ordering for 74.9% of 295 book pairs. Significance comes from a
permutation null that re-runs the whole pipeline on shuffled dates 200 times, at
p = 0.005; no permutation matched the observed value. Applied to the documentary
sources with distribution-free conformal intervals, the Priestly, Deuteronomic,
and JE composites fall after the exile with probability of at least 0.84, in the
relative order JE → D → P, though none of them contributed to training. The Song
of the Sea and the Song of Deborah emerge as the two earliest of the 19 undated
units. Recent archaeological work (Adler 2022) and epigraphic analysis of
Achaemenid-era sources (Barnea 2026) corroborate the direction of these results
independently.

The limits are stated as plainly as the results. The 68% intervals are ±138
years, which precludes any claim about position within the Persian period; every
substantive claim in the paper is either a side-of-the-exile claim or an ordering
claim. Poetry is systematically placed early and narrative late, which bears
directly on the archaic poems, and with one securely dated poetic book in the
corpus that confound cannot be separated from date. The Greek transfer is an
analogy rather than a measurement of Hebrew.

The paper also identifies a validation artifact that explains why the accuracies
reported here are lower than several published figures for models of this class.
When a held-out text is dated under a prior centered on its own scholarly date,
the model returns approximately that date regardless of the text's content, and
does so most convincingly for the texts whose dates are best established. Across
this corpus the linguistic evidence supplies a median 11.7% of posterior
precision, and 4.5% for the most tightly anchored texts, where such a design
reports errors of a few years against an honest 189. Accuracies of a decade or
two, for models of this class, are a signature of the design rather than a
measurement of the language. The comparison matters for reading the present
results against the existing literature.

The manuscript is submitted exclusively to PLOS ONE and has not been published or
submitted elsewhere. All code, extracted feature matrices, model outputs, and the
scripts that regenerate every table, figure, and numerical claim in the text
directly from the corpus are available at
https://github.com/adairaar/diachronic-hebrew.

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
  the cross-validation and permutation methodology independently of the biblical
  studies content.

- **Dr. Martijn Naaijer** (University of Zurich) — has argued that existing
  linguistic dating methods rest on unsound statistical assumptions (Rezetko,
  Young, Ehrensvärd & Naaijer, *Bible and Interpretation*, 2025). Much of that
  critique is vindicated here, and the archaizing measurement speaks directly to
  the dialectal position. He is well placed to judge whether the corrections go
  far enough.

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

**Requested exclusion.** I would ask that the paper not be sent to **Prof. Aaron
D. Hornkohl** (University of Cambridge). This is not a matter of disagreement
over conclusions. The paper's argument is statistical, resting on
cross-validation design, permutation inference, and the decomposition of
posterior precision, and Prof. Hornkohl's work does not engage linguistic dating
on those terms. He has also argued for a pre-exilic date for the Pentateuchal
sources and for the detectability of chronological stratification *within* them,
a position these results contradict directly. A reviewer whose published position
is incompatible with the finding, and whose methodological training lies outside
the framework used to reach it, is unlikely to assess the statistical argument on
its merits.

Thank you for your time and consideration.

Sincerely,

Aaron Adair
Department of Physics, Massachusetts Institute of Technology
adairaar@gmail.com
